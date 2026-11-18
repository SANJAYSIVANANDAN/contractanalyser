import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

# Engine = the connection pool to PostgreSQL. Created once per process.
engine = create_engine(settings.database_url, pool_pre_ping=True)

# SessionLocal() gives a new Session = one unit of work / transaction.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """All ORM models (models.py) inherit from this."""


def init_db() -> None:
    """Creates the pgvector extension and every table directly from the models,
    no migration tool involved. Safe to call on every process startup (api, worker):
    CREATE EXTENSION/TABLE IF NOT EXISTS is idempotent."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    from app.db import models  # noqa: F401  (registers tables on Base.metadata)

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready (no Alembic)")


def get_db():
    """FastAPI dependency. `yield` hands the session to the endpoint,
    then the `finally` block always closes it, even if the endpoint raised."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
