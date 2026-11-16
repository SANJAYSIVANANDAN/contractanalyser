from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# Engine = the connection pool to PostgreSQL. Created once per process.
engine = create_engine(settings.database_url, pool_pre_ping=True)

# SessionLocal() gives a new Session = one unit of work / transaction.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """All ORM models (models.py, Phase 2) inherit from this."""


def get_db():
    """FastAPI dependency. `yield` hands the session to the endpoint,
    then the `finally` block always closes it, even if the endpoint raised."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
