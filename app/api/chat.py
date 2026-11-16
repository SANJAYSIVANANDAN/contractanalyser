import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, get_db
from app.db.models import COMPLETED, Conversation, Document, Message
from app.schemas.chat import ChatRequest, ChatResponse, MessageOut
from app.services.rag import rag_graph, run_rag

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def resolve_document_ids(db: Session, requested: list[uuid.UUID]) -> list[uuid.UUID]:
    """Only COMPLETED documents can be searched. Empty request = all completed documents."""
    stmt = select(Document.id).where(Document.status == COMPLETED)
    if requested:
        stmt = stmt.where(Document.id.in_(requested))
    return list(db.scalars(stmt))


def get_or_create_conversation(db: Session, request: ChatRequest) -> Conversation:
    if request.conversation_id:
        conversation = db.get(Conversation, request.conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        return conversation
    conversation = Conversation(title=request.question[:60])
    db.add(conversation)
    db.flush()   # assigns the id without committing yet
    return conversation


def save_exchange(db: Session, conversation: Conversation, question: str, state: dict) -> Message:
    """Store the user's question and the assistant's validated answer in ONE transaction."""
    db.add(Message(conversation_id=conversation.id, role="user", content=question))
    assistant = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=state["answer"],
        citations=[c.model_dump(mode="json") for c in state["citations"]],
    )
    db.add(assistant)
    db.commit()
    return assistant


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    document_ids = resolve_document_ids(db, request.document_ids)
    conversation = get_or_create_conversation(db, request)

    try:
        state = run_rag(db, request.question, document_ids) if document_ids else None
    except Exception as e:
        logger.exception("RAG pipeline failed")
        raise HTTPException(502, f"Answer generation failed: {type(e).__name__}")

    if state is None:   # no searchable documents at all
        state = {"answer": "No processed documents are available to search.", "citations": [], "insufficient_evidence": True}

    message = save_exchange(db, conversation, request.question, state)
    return ChatResponse(
        conversation_id=conversation.id,
        message_id=message.id,
        answer=state["answer"],
        citations=state["citations"],
        insufficient_evidence=state["insufficient_evidence"],
    )


def sse(event: str, data: dict) -> str:
    """One Server-Sent Event: 'event:' line, 'data:' line, blank line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streams the answer. NOTE: no Depends(get_db) here. FastAPI closes dependency
    sessions before a streaming body finishes, so the generator owns its own session."""

    def event_stream():
        db = SessionLocal()
        try:
            document_ids = resolve_document_ids(db, request.document_ids)
            conversation = get_or_create_conversation(db, request)
            if not document_ids:
                yield sse("error", {"message": "No processed documents are available to search."})
                return

            streamed_any_token = False
            final_state = None
            # stream_mode "messages" = LLM tokens as they arrive; "values" = full state after each node
            for mode, payload in rag_graph.stream(
                {"question": request.question, "document_ids": document_ids},
                config={"configurable": {"db": db}},
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    token, metadata = payload
                    if metadata.get("langgraph_node") == "generate" and token.content:
                        streamed_any_token = True
                        yield sse("token", {"text": token.content})
                else:
                    final_state = payload

            if final_state is None or "answer" not in final_state:
                yield sse("error", {"message": "The pipeline produced no answer."})
                return

            if not streamed_any_token:   # e.g. the "insufficient evidence" path has no LLM tokens
                yield sse("token", {"text": final_state["answer"]})

            # Citations are sent only AFTER backend validation (validate_citations node)
            yield sse(
                "citations",
                {
                    "sources": [c.model_dump(mode="json") for c in final_state["citations"]],
                    "insufficient_evidence": final_state["insufficient_evidence"],
                    "rejected_sources": final_state.get("rejected_sources", []),
                },
            )

            try:
                message = save_exchange(db, conversation, request.question, final_state)
            except Exception:
                db.rollback()
                logger.exception("Could not save the chat exchange")
                yield sse("error", {"message": "Answer generated but could not be saved."})
                return

            yield sse("done", {"message_id": str(message.id), "conversation_id": str(conversation.id)})

        except GeneratorExit:
            # Client disconnected mid-stream: stop quietly, store nothing half-finished.
            logger.info("Client disconnected during /chat/stream")
            raise
        except Exception as e:
            logger.exception("Streaming failed")
            yield sse("error", {"message": f"Answer generation failed: {type(e).__name__}"})
        finally:
            db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def get_messages(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    if db.get(Conversation, conversation_id) is None:
        raise HTTPException(404, "Conversation not found")
    stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    return db.scalars(stmt).all()
