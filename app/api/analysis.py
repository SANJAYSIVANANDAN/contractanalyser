import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.db.models import COMPLETED, PENDING, Clause, Document, User
from app.schemas.clauses import ClauseOut
from app.schemas.documents import DocumentOut
from app.services.contract_analyzer import CLAUSE_TYPE_NAMES
from app.workers.tasks import analyze_contract

router = APIRouter(prefix="/documents", tags=["analysis"])


def get_document_or_404(doc_id: uuid.UUID, db: Session, user: User) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None or doc.user_id != user.id:
        raise HTTPException(404, "Document not found")
    return doc


@router.post("/{doc_id}/analyze", response_model=DocumentOut)
def analyze(doc_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    doc = get_document_or_404(doc_id, db, user)
    if doc.status != COMPLETED:
        raise HTTPException(400, "Document must finish processing before it can be analyzed")

    doc.analysis_status = PENDING
    doc.analysis_error_message = None
    db.commit()
    db.refresh(doc)

    analyze_contract.delay(str(doc_id))
    return doc


@router.get("/{doc_id}/clauses", response_model=list[ClauseOut])
def get_clauses(doc_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    get_document_or_404(doc_id, db, user)
    clauses = {c.clause_type: c for c in db.scalars(select(Clause).where(Clause.document_id == doc_id))}
    # Stable order matching the clause taxonomy, regardless of insertion order.
    return [clauses[name] for name in CLAUSE_TYPE_NAMES if name in clauses]
