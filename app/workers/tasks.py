import uuid

import openai
from sqlalchemy import delete, func

from app.db.database import SessionLocal
from app.db.models import COMPLETED, FAILED, PENDING, PROCESSING, Chunk, Document
from app.services.cache import acquire_lock, release_lock
from app.services.chunker import chunk_blocks
from app.services.document_parser import parse_pdf
from app.services.embeddings import embed_texts
from app.workers.celery_app import celery_app

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
        chunks = chunk_blocks(blocks)               # page-aware chunks
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
                    embedding=vector,
                    search_vector=func.to_tsvector("english", chunk["content"]),
                )
            )
        doc.status = COMPLETED
        db.commit()
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
