"""Scans a whole contract for a fixed set of standard clause types.

Plain Python/LangChain, not LangGraph: this is a batch extraction job run once
from the Celery worker (see workers/tasks.py), the same shape as chunking/embedding,
not an interactive multi-step flow. LangGraph stays scoped to the Q&A chat in rag.py.

A contract can have more chunks than fit in one LLM call, so the document's chunks
(ordered, same source rag.py's summary path uses) are packed into batches under a
char budget and scanned one batch at a time. A clause type counts as "found" if ANY
batch found it; it is only reported "missing" after every batch has been scanned.
"""
import uuid
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Chunk
from app.services.observability import get_langfuse_handler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

BATCH_CHAR_LIMIT = 12000  # chars of contract text per LLM call

CLAUSE_TYPES: list[tuple[str, str]] = [
    ("PARTIES", "Who the contracting parties are and their legal names/roles"),
    ("TERM", "The contract's start date, duration, and effective period"),
    ("TERMINATION", "Conditions and process for ending the contract early"),
    ("PAYMENT", "Payment amounts, schedule, invoicing, and late-payment terms"),
    ("LIABILITY", "Limitation of liability, caps on damages"),
    ("INDEMNIFICATION", "Who indemnifies whom, and for what"),
    ("CONFIDENTIALITY", "Non-disclosure / confidentiality obligations"),
    ("INTELLECTUAL_PROPERTY", "Ownership or licensing of IP created or used under the contract"),
    ("GOVERNING_LAW", "Which jurisdiction's law governs the contract"),
    ("RENEWAL", "Auto-renewal or renewal terms"),
    ("DISPUTE_RESOLUTION", "Arbitration, mediation, or litigation venue for disputes"),
]
CLAUSE_TYPE_NAMES = [name for name, _description in CLAUSE_TYPES]


class ClauseFinding(BaseModel):
    clause_type: Literal[*CLAUSE_TYPE_NAMES]
    found: bool
    title: str | None = None
    summary: str | None = Field(None, description="Plain-language summary of what the clause says")
    supporting_text: str | None = Field(None, description="Verbatim quote from the contract")
    risk_level: Literal["low", "medium", "high"] | None = None
    explanation: str = Field(description="Why this risk level, or why the clause was not found")
    page_number: int | None = None


class BatchExtraction(BaseModel):
    findings: list[ClauseFinding]


SYSTEM_PROMPT = """You are a contract analysis assistant extracting standard clauses from a contract.

You will be shown a batch of passages (not necessarily the whole contract). For EVERY one of the
following clause types, decide if this batch contains it:

{clause_list}

Rules:
- Only use text that actually appears in the supplied passages. Never invent clause content.
- If a clause type is present, set found=true, quote the relevant text in supporting_text,
  write a plain-language summary, assign a risk_level (low/medium/high) with a one-sentence
  explanation (e.g. a liability clause with no cap is higher risk), and give its page_number.
- If a clause type is NOT present in this batch, still return it with found=false and a short
  explanation like "Not present in this batch" - do not omit any clause type.
- Return exactly one finding per clause type listed above, no more, no fewer."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Contract passages:\n{passages}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=settings.openai_api_key,
    max_retries=3,
    timeout=60,
)
structured_chain = prompt | llm.with_structured_output(BatchExtraction)

CLAUSE_LIST_TEXT = "\n".join(f"- {name}: {description}" for name, description in CLAUSE_TYPES)


def make_batches(chunks: list[Chunk], char_limit: int = BATCH_CHAR_LIMIT) -> list[list[Chunk]]:
    """Greedily packs chunks (already in reading order) under a char budget per batch,
    so a long contract is scanned in full across several LLM calls instead of truncated."""
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_chars = 0
    for chunk in chunks:
        chunk_chars = len(chunk.content)
        if current and current_chars + chunk_chars > char_limit:
            batches.append(current)
            current, current_chars = [], 0
        current.append(chunk)
        current_chars += chunk_chars
    if current:
        batches.append(current)
    return batches


def format_passages(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"Page: {chunk.page_number} | Section: {chunk.section_title or 'n/a'}\n{chunk.content}"
        for chunk in chunks
    )


def analyze_document(db: Session, document_id: uuid.UUID) -> list[ClauseFinding]:
    chunks = list(
        db.scalars(select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index))
    )
    config = {}
    handler = get_langfuse_handler()
    if handler:
        config = {"callbacks": [handler], "metadata": {"langfuse_session_id": str(document_id), "langfuse_tags": ["contract-analysis"]}}

    found_so_far: dict[str, ClauseFinding] = {}
    for batch in make_batches(chunks):
        result = structured_chain.invoke(
            {"clause_list": CLAUSE_LIST_TEXT, "passages": format_passages(batch)},
            config=config,
        )
        for finding in result.findings:
            if finding.found:
                found_so_far[finding.clause_type] = finding

    return [
        found_so_far.get(
            clause_type,
            ClauseFinding(
                clause_type=clause_type,
                found=False,
                explanation="Not found in any passage scanned.",
            ),
        )
        for clause_type, _description in CLAUSE_TYPES
    ]
