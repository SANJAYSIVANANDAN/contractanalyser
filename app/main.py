import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import analysis, chat, documents
from app.db.database import get_db
from app.services.reranker import _get_model

logging.basicConfig(level=logging.INFO)  # so app.* loggers (cache HIT/MISS) show in docker logs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load the reranker now, so the first user question
    # doesn't pay the ~15 s model-loading cost.
    _get_model()
    yield


app = FastAPI(title="DocuPilot AI", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(analysis.router)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    """Proves the API can talk to PostgreSQL."""
    db.execute(select(1))
    return {"status": "ok", "database": "up"}
