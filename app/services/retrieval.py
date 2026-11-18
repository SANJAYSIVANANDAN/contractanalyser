"""Hybrid retrieval: semantic (pgvector) search + real BM25 search, merged with
Reciprocal Rank Fusion.

Every search function returns a list of Chunk objects ORDERED best-first. RRF only
looks at the ORDER (rank), never at the raw scores, because cosine distance and
BM25 scores live on different scales and can't be added together.

Why BM25 instead of Postgres full-text search: contract jargon (defined terms like
"O&M Baseline", numeric tokens, section references) doesn't stem the way English
prose does, and Postgres's websearch_to_tsquery/ts_rank are tuned for that prose.
BM25 is computed in-process, per request, over just the candidate document(s)'
chunks (a document's corpus here is at most a few hundred chunks, so this is a
few milliseconds) and gives proper IDF weighting of rare/important terms.
"""
import logging
import re
import uuid

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.services.cache import get_cached_retrieval, make_retrieval_key, set_cached_retrieval
from app.services.embeddings import embed_query
from app.services.query_rewrite import rewrite_query

logger = logging.getLogger(__name__)

TOP_K_EACH = 40   # candidates taken from each leg before fusion (widened so reranking
                  # downstream has real material to recover, not just reshuffle a narrow set)
RRF_K = 60        # standard RRF constant: dampens the advantage of rank 1 over rank 2
FUSION_TOP_N = 30 # how many fused candidates survive to the rerank stage

# Preserves hyphenated/alphanumeric contract tokens ("O&M", "P1", "24x7") instead of
# BM25's naive whitespace split, which would shred them.
_TOKEN = re.compile(r"[a-z0-9&]+(?:[-'][a-z0-9&]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def semantic_search(db: Session, query_vector: list[float], document_ids: list[uuid.UUID], limit: int = TOP_K_EACH) -> list[Chunk]:
    """Nearest chunks by cosine distance (smaller = more similar). Uses the HNSW index."""
    stmt = (
        select(Chunk)
        .where(Chunk.document_id.in_(document_ids), Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(limit)
    )
    return list(db.scalars(stmt))


def bm25_search(db: Session, question: str, document_ids: list[uuid.UUID], limit: int = TOP_K_EACH) -> list[Chunk]:
    """Real BM25 (Okapi), built fresh per request over this document set's chunks.
    Cheap enough at this corpus size that there's no separate index to maintain/invalidate."""
    chunks = list(db.scalars(select(Chunk).where(Chunk.document_id.in_(document_ids))))
    if not chunks:
        return []
    corpus = [_tokenize(f"{c.section_title or ''} {c.content}") for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, score in ranked[:limit] if score > 0]


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int = RRF_K, top_n: int = FUSION_TOP_N) -> list[tuple[Chunk, float]]:
    """RRF(d) = sum over every ranking of 1 / (k + rank(d)), rank starting at 1.
    A chunk found by BOTH searches gets two terms added, so it floats to the top.
    Returns [(chunk, rrf_score), ...] best-first, with duplicates merged."""
    scores: dict[uuid.UUID, float] = {}
    chunks_by_id: dict[uuid.UUID, Chunk] = {}
    for ranking in rankings:
        for position, chunk in enumerate(ranking):
            rank = position + 1
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.id] = chunk
    best_first = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [(chunks_by_id[chunk_id], scores[chunk_id]) for chunk_id in best_first]


def load_chunks_in_order(db: Session, chunk_ids: list[str]) -> list[Chunk] | None:
    """Fetch chunks by id, keeping the cached order. Returns None if any chunk no longer
    exists (document was deleted/reprocessed), which makes the caller treat it as a miss."""
    ids = [uuid.UUID(chunk_id) for chunk_id in chunk_ids]
    by_id = {chunk.id: chunk for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(ids)))}
    if len(by_id) != len(set(ids)):
        return None
    return [by_id[chunk_id] for chunk_id in ids]


def hybrid_retrieve(
    db: Session, question: str, document_ids: list[uuid.UUID], top_n: int = FUSION_TOP_N, use_query_rewrite: bool = True
) -> list[tuple[Chunk, float]]:
    """Full retrieval step with a cache-aside Redis cache:
    hit  -> reload the cached chunk ids from Postgres (skips the OpenAI embedding call and both searches)
    miss -> (optionally rewrite jargon) -> embed -> semantic + BM25 -> RRF -> store in Redis.

    The cache key is based on the ORIGINAL question (rewriting is a deterministic
    function of it), so a cache hit skips the rewrite step too."""
    if not document_ids:
        return []

    key = make_retrieval_key(question, document_ids)
    cached = get_cached_retrieval(key)
    if cached is not None:
        chunks = load_chunks_in_order(db, [chunk_id for chunk_id, _score in cached])
        if chunks is not None:
            logger.info("retrieval cache HIT")
            return [(chunk, score) for chunk, (_id, score) in zip(chunks, cached)]

    logger.info("retrieval cache MISS")
    retrieval_question = rewrite_query(question) if use_query_rewrite else question
    query_vector = embed_query(retrieval_question)
    semantic = semantic_search(db, query_vector, document_ids)
    sparse = bm25_search(db, retrieval_question, document_ids)
    fused = reciprocal_rank_fusion([semantic, sparse], top_n=top_n)

    if fused:  # never cache an empty result
        set_cached_retrieval(key, document_ids, [[str(chunk.id), score] for chunk, score in fused])
    return fused
