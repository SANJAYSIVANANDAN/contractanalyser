import uuid

import openai
from sqlalchemy import delete

from app.db.database import SessionLocal, init_db
from app.db.models import COMPLETED, FAILED, PENDING, PROCESSING, Chunk, Clause, Document
from app.services.cache import acquire_lock, invalidate_document, release_lock
from app.services.chunker import chunk_blocks
from app.services.contract_analyzer import analyze_document
from app.services.document_parser import parse_pdf
from app.services.embeddings import embed_texts
from app.services.observability import flush as flush_traces
from app.workers.celery_app import celery_app

init_db()  # the worker writes to the same tables as the API; make sure they exist here too

# Temporary problems worth retrying. Anything else (bad PDF, bug) fails immediately.
RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def mark_failed(document_id: uuid.UUID, message: str):
    """Use a fresh session so a broken transaction can't stop us recording the failure."""
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc:
            doc.status = FAILED
            doc.error_message = message[:1000]
            db.commit()
    finally:
        db.close()


@celery_app.task(name="process_document", bind=True, max_retries=3)
def process_document(self, document_id: str):
    doc_id = uuid.UUID(document_id)

    # Idempotency guard 1: Redis lock. If another worker is already on this document, skip.
    lock_key = f"lock:process_document:{document_id}"
    if not acquire_lock(lock_key):
        return "skipped: already being processed"

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is None:
            return "skipped: document not found"
        # Idempotency guard 2: database state. A redelivered message for a finished doc is a no-op.
        if doc.status == COMPLETED:
            return "skipped: already completed"

        doc.status = PROCESSING
        doc.error_message = None
        db.commit()

        # --- slow work happens OUTSIDE any open write transaction ---
        blocks = parse_pdf(doc.file_path)           # Docling
        chunks = chunk_blocks(blocks)                # structure-aware, overlapping chunks
        # Embed "section title + content" so headings help retrieval; we store only the content.
        texts = [f"{c['section_title'] or ''}\n{c['content']}".strip() for c in chunks]
        vectors = embed_texts(texts)                # OpenAI

        # --- ONE transaction: replace old chunks, insert new, mark COMPLETED ---
        # If anything fails, rollback leaves no half-written chunks; a retry starts clean.
        db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        for chunk, vector in zip(chunks, vectors):
            db.add(
                Chunk(
                    document_id=doc_id,
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    page_number=chunk["page_number"],
                    section_title=chunk["section_title"],
                    is_table=chunk["is_table"],
                    embedding=vector,
                )
            )
        doc.status = COMPLETED
        db.commit()
        # Chunk ids just changed, so any cached retrieval that pointed at the old ones is stale.
        invalidate_document(doc_id)
        return f"ok: {len(chunks)} chunks"

    except RETRYABLE as e:
        db.rollback()
        if self.request.retries >= self.max_retries:
            mark_failed(doc_id, f"Gave up after retries: {e}")
            raise
        # Put it back to PENDING while waiting for the retry, then retry with backoff: 20s, 40s, 80s
        _set_status(doc_id, PENDING)
        raise self.retry(exc=e, countdown=20 * 2**self.request.retries)

    except Exception as e:
        db.rollback()
        mark_failed(doc_id, str(e))
        raise

    finally:
        db.close()
        release_lock(lock_key)


def _set_status(document_id: uuid.UUID, status: str):
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc:
            doc.status = status
            db.commit()
    finally:
        db.close()


def mark_analysis_failed(document_id: uuid.UUID, message: str):
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc:
            doc.analysis_status = FAILED
            doc.analysis_error_message = message[:1000]
            db.commit()
    finally:
        db.close()


@celery_app.task(name="analyze_contract", bind=True, max_retries=3)
def analyze_contract(self, document_id: str):
    doc_id = uuid.UUID(document_id)

    lock_key = f"lock:analyze_contract:{document_id}"
    if not acquire_lock(lock_key):
        return "skipped: already being analyzed"

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is None:
            return "skipped: document not found"
        if doc.status != COMPLETED:
            return "skipped: document not yet ingested"
        if doc.analysis_status == COMPLETED:
            return "skipped: already analyzed"

        doc.analysis_status = PROCESSING
        doc.analysis_error_message = None
        db.commit()

        # --- slow work (several LLM calls) happens OUTSIDE any open write transaction ---
        findings = analyze_document(db, doc_id)

        # --- ONE transaction: replace old clauses, insert new, mark COMPLETED ---
        db.execute(delete(Clause).where(Clause.document_id == doc_id))
        for finding in findings:
            db.add(
                Clause(
                    document_id=doc_id,
                    clause_type=finding.clause_type,
                    found=finding.found,
                    title=finding.title,
                    summary=finding.summary,
                    supporting_text=finding.supporting_text,
                    risk_level=finding.risk_level,
                    explanation=finding.explanation,
                    page_number=finding.page_number,
                )
            )
        doc.analysis_status = COMPLETED
        db.commit()
        return f"ok: {len(findings)} clauses"

    except RETRYABLE as e:
        db.rollback()
        if self.request.retries >= self.max_retries:
            mark_analysis_failed(doc_id, f"Gave up after retries: {e}")
            raise
        _set_analysis_status(doc_id, PENDING)
        raise self.retry(exc=e, countdown=20 * 2**self.request.retries)

    except Exception as e:
        db.rollback()
        mark_analysis_failed(doc_id, str(e))
        raise

    finally:
        db.close()
        release_lock(lock_key)
        flush_traces()   # worker process may idle a while before the next task; don't lose traces


def _set_analysis_status(document_id: uuid.UUID, status: str):
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc:
            doc.analysis_status = status
            db.commit()
    finally:
        db.close()
