import time
import uuid

from app.db.database import SessionLocal
from app.db.models import COMPLETED, FAILED, PROCESSING, Document
from app.workers.celery_app import celery_app


@celery_app.task(name="process_document")
def process_document(document_id: str):
    """PHASE 2 STUB: only moves the status along. Phase 3 adds Docling + embeddings."""
    db = SessionLocal()
    try:
        doc = db.get(Document, uuid.UUID(document_id))
        if doc is None:
            return "document not found"

        doc.status = PROCESSING
        db.commit()

        time.sleep(3)  # pretend to do slow work (so you can watch PROCESSING)

        doc.status = COMPLETED
        db.commit()
        return "ok"
    except Exception as e:
        db.rollback()
        doc = db.get(Document, uuid.UUID(document_id))
        if doc:
            doc.status = FAILED
            doc.error_message = str(e)
            db.commit()
        raise
    finally:
        db.close()
