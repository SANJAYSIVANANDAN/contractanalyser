from fastapi import Depends, FastAPI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import documents
from app.db.database import get_db

app = FastAPI(title="DocuPilot AI")
app.include_router(documents.router)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Proves the API can talk to PostgreSQL."""
    db.execute(select(1))
    return {"status": "ok", "database": "up"}
