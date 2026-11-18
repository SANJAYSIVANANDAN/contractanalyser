import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    """What the API returns for a document. from_attributes lets Pydantic read a SQLAlchemy object."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: str
    error_message: str | None
    analysis_status: str | None
    analysis_error_message: str | None
    created_at: datetime


class DocumentStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    error_message: str | None
