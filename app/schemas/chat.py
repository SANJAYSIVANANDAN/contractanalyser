import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Which documents to search. Empty list = search all COMPLETED documents.
    document_ids: list[uuid.UUID] = []
    # Omit to start a new conversation.
    conversation_id: uuid.UUID | None = None


class Citation(BaseModel):
    """Built ONLY from database rows, never from text written by the LLM."""
    source_number: int                 # the [n] label the answer text uses
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    section_title: str | None
    supporting_excerpt: str            # real text copied from the stored chunk


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list | None
    created_at: datetime
