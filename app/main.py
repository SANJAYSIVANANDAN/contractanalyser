from fastapi import Depends, FastAPI
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.database import get_db

app = FastAPI(title="DocuPilot AI")


@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Proves the API can talk to PostgreSQL and that pgvector is installed."""
    db.execute(select(1))
    # Enable pgvector (idempotent). Alembic will own this once models exist.
    db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    db.commit()
    version = db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar()
    return {"status": "ok", "database": "up", "pgvector": version}
