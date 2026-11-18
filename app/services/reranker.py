"""Local CrossEncoder reranker (no external API).

Why it exists: embeddings compare the question and each chunk SEPARATELY (fast, rough).
A cross-encoder reads question + chunk TOGETHER in one model pass (slow, much more precise).
So we use the cheap hybrid search to get a wide candidate pool, then the cross-encoder
picks the best few.

Two fixes versus the first version of this reranker, which made RAGAS scores WORSE
(see app/evaluation/evaluate.py's A-D baseline): that version (a) used
cross-encoder/ms-marco-MiniLM-L-6-v2, trained on short web/news search queries, not
dense contract/financial text, and (b) let the cross-encoder fully override the fused
rank order even when both legs of hybrid search had already agreed on a chunk.
Here the reranker score is BLENDED with the RRF rank it arrived with, so a chunk both
searches agreed on can't be bumped out purely on one noisy cross-encoder opinion.
"""
import uuid

from app.db.models import Chunk

# A stronger, more general-purpose cross-encoder than MS-MARCO MiniLM. Swap this constant
# (and re-run app/evaluation/evaluate.py) to try alternatives before committing to one.
MODEL_NAME = "BAAI/bge-reranker-base"

# How much weight the cross-encoder's own opinion gets vs. the fused RRF rank it came in
# with. 1.0 = old behaviour (cross-encoder fully overrides fusion); 0.0 = ignore it entirely.
BLEND_WEIGHT = 0.7

_model = None  # loaded once per process, on first use


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # imported lazily: slow import
        _model = CrossEncoder(MODEL_NAME)
    return _model


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rerank(
    question: str,
    candidates: list[Chunk],
    top_n: int = 5,
    rrf_scores: dict[uuid.UUID, float] | None = None,
) -> list[tuple[Chunk, float]]:
    """Returns the top_n [(chunk, blended_score)] best-first. Higher score = more relevant.

    `rrf_scores`, if given, maps chunk.id -> its fused RRF score; the final ranking blends
    the cross-encoder's normalized score with the normalized RRF score (BLEND_WEIGHT favors
    the cross-encoder). Omit it to use the cross-encoder score alone."""
    if not candidates:
        return []
    # Same text we embedded at ingestion: section title + content
    pairs = [(question, f"{c.section_title or ''}\n{c.content}".strip()) for c in candidates]
    raw_scores = [float(s) for s in _get_model().predict(pairs)]

    if rrf_scores:
        ce_norm = _minmax(raw_scores)
        rrf_norm = _minmax([rrf_scores.get(c.id, 0.0) for c in candidates])
        blended = [BLEND_WEIGHT * ce + (1 - BLEND_WEIGHT) * rr for ce, rr in zip(ce_norm, rrf_norm)]
    else:
        blended = raw_scores

    ranked = sorted(zip(candidates, blended), key=lambda pair: pair[1], reverse=True)
    return [(chunk, score) for chunk, score in ranked[:top_n]]
