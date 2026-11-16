import hashlib
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import Document
from app.schemas.documents import DocumentOut, DocumentStatusOut
from app.services.cache import invalidate_document
from app.workers.tasks import process_document

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    # 1. Validate: extension, size, and the PDF "magic bytes"
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are allowed")
    data = await file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large (max 20 MB)")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a valid PDF")

    # 2. SHA-256 hash. Same file uploaded twice -> return the existing record.
    file_hash = hashlib.sha256(data).hexdigest()
    existing = db.scalars(select(Document).where(Document.file_hash == file_hash)).first()
    if existing:
        return existing

    # 3. Save to shared storage (the worker reads the same volume)
    doc_id = uuid.uuid4()
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(data)

    # 4. Create the DB row (status PENDING) and COMMIT before publishing the task,
    #    otherwise the worker could look for a row that doesn't exist yet.
    doc = Document(id=doc_id, filename=file.filename, file_path=file_path, file_hash=file_hash)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 5. Publish the task to RabbitMQ. Only the ID travels, not the PDF.
    process_document.delay(str(doc_id))

    # 6. Return immediately
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.created_at.desc())).all()


def get_document_or_404(doc_id: uuid.UUID, db: Session) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    return get_document_or_404(doc_id, db)


@router.get("/{doc_id}/status", response_model=DocumentStatusOut)
def get_document_status(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    return get_document_or_404(doc_id, db)


@router.delete("/{doc_id}")
def delete_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = get_document_or_404(doc_id, db)
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    db.delete(doc)  # cascades to chunks and clauses
    db.commit()
    invalidate_document(doc_id)  # drop cached retrievals that included this document
    return {"deleted": str(doc_id)}
