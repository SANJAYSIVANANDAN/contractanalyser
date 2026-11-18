"""Langfuse tracing for the LangChain/LangGraph pipelines (rag.py, contract_analyzer.py).

Optional: if no keys are configured, get_langfuse_handler() returns None and nothing
else in the app changes - observability must never break answering, same principle
as the Redis cache (see services/cache.py).
"""
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_handler = None
_initialized = False


def get_langfuse_handler():
    """Returns a LangChain CallbackHandler, or None if Langfuse isn't configured."""
    global _handler, _initialized
    if not _initialized:
        _initialized = True
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse.langchain import CallbackHandler
                _handler = CallbackHandler()
            except Exception:
                logger.warning("Langfuse is configured but failed to initialize; tracing disabled", exc_info=True)
    return _handler


def flush() -> None:
    """Call at the end of a short-lived process (a Celery task) so traces reach Langfuse
    even if the process exits right after, instead of waiting for the SDK's background batching."""
    if get_langfuse_handler() is None:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        logger.warning("Langfuse flush failed", exc_info=True)
