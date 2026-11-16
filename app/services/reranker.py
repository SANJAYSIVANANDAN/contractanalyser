"""Local CrossEncoder reranker (no external API).

Why it exists: embeddings compare the question and each chunk SEPARATELY (fast, rough).
A cross-encoder reads question + chunk TOGETHER in one model pass (slow, much more precise).
So we use the cheap search to get ~15 candidates, then the cross-encoder picks the best 5.
"""
from app.db.models import Chunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None  # loaded once per process, on first use


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # imported lazily: slow import
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(question: str, candidates: list[Chunk], top_n: int = 5) -> list[tuple[Chunk, float]]:
    """Returns the top_n [(chunk, relevance_score)] best-first. Higher score = more relevant."""
    if not candidates:
        return []
    # Same text we embedded at ingestion: section title + content
    pairs = [(question, f"{c.section_title or ''}\n{c.content}".strip()) for c in candidates]
    scores = _get_model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:top_n]]
