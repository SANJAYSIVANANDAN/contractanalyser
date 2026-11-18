import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

# Document processing statuses (plain strings stored in the DB).
PENDING = "PENDING"
PROCESSING = "PROCESSING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(64), index=True)  # SHA-256 hex
    status: Mapped[str] = mapped_column(String(20), default=PENDING)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    # Contract analysis (clause extraction) is a separate pipeline from ingestion above.
    # None = never analysed. Reuses the same PENDING/PROCESSING/COMPLETED/FAILED strings.
    analysis_status: Mapped[str | None] = mapped_column(String(20), default=None)
    analysis_error_message: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # cascade: deleting a document also deletes its chunks and clauses
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    clauses: Mapped[list["Clause"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)  # None if Docling can't tell
    section_title: Mapped[str | None] = mapped_column(String(500))
    embedding = mapped_column(Vector(1536))          # OpenAI text-embedding-3-small
    search_vector = mapped_column(TSVECTOR)          # for keyword search

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        # GIN index makes full-text search fast
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
        # HNSW index makes cosine nearest-neighbour search fast
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Clause(Base):
    __tablename__ = "clauses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    clause_type: Mapped[str] = mapped_column(String(50))
    title: Mapped[str | None] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(Text)
    supporting_text: Mapped[str | None] = mapped_column(Text)
    risk_level: Mapped[str | None] = mapped_column(String(20))
    explanation: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    found: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")

    document: Mapped["Document"] = relationship(back_populates="clauses")

    __table_args__ = (
        # One row per clause type per document; re-analysis deletes-and-reinserts, like chunks.
        UniqueConstraint("document_id", "clause_type", name="ux_clauses_document_clause_type"),
    )

    @property
    def content(self) -> str:
        """Lets a Clause be dropped into rag.py's chunk-shaped passage formatting unchanged."""
        if not self.found:
            return f"No such clause was found in this contract. {self.explanation or ''}".strip()
        parts = [
            self.summary or "",
            f'"{self.supporting_text}"' if self.supporting_text else "",
            f"Risk: {self.risk_level}." if self.risk_level else "",
            self.explanation or "",
        ]
        return "\n".join(part for part in parts if part)

    @property
    def section_title(self) -> str:
        return self.title or self.clause_type.replace("_", " ").title()
