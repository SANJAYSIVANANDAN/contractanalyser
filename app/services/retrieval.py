"""Hybrid retrieval: semantic search + keyword search, merged with Reciprocal Rank Fusion.

Every search function returns a list of Chunk objects ORDERED best-first.
RRF only looks at the ORDER (rank), never at the raw scores, because cosine
distances and ts_rank values live on different scales and can't be added together.
"""
import logging
import re
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.services.cache import get_cached_retrieval, make_retrieval_key, set_cached_retrieval
from app.services.embeddings import embed_query

logger = logging.getLogger(__name__)

TOP_K_EACH = 15   # candidates taken from each search
RRF_K = 60        # standard RRF constant: dampens the advantage of rank 1 over rank 2


def semantic_search(db: Session, query_vector: list[float], document_ids: list[uuid.UUID], limit: int = TOP_K_EACH) -> list[Chunk]:
    """Nearest chunks by cosine distance (smaller = more similar). Uses the HNSW index."""
    stmt = (
        select(Chunk)
        .where(Chunk.document_id.in_(document_ids), Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(limit)
    )
    return list(db.scalars(stmt))


def keyword_search(db: Session, question: str, document_ids: list[uuid.UUID], limit: int = TOP_K_EACH) -> list[Chunk]:
    """PostgreSQL full-text search. Uses the GIN index on chunks.search_vector.

    websearch_to_tsquery ANDs all words by default, which is too strict for a
    full sentence ("what are the termination conditions"). We join the words with
    'or' so a chunk matching SOME words can still be found; ts_rank puts chunks
    matching MORE words first. Words are reduced to letters/digits so user input
    can't inject query operators."""
    words = re.findall(r"\w+", question)
    if not words:
        return []
    tsquery = func.websearch_to_tsquery("english", " or ".join(words))
    rank = func.ts_rank(Chunk.search_vector, tsquery)
    stmt = (
        select(Chunk)
        .where(Chunk.document_id.in_(document_ids), Chunk.search_vector.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt))


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int = RRF_K, top_n: int = 15) -> list[tuple[Chunk, float]]:
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


def hybrid_retrieve(db: Session, question: str, document_ids: list[uuid.UUID], top_n: int = 15) -> list[tuple[Chunk, float]]:
    """Full retrieval step with a cache-aside Redis cache:
    hit  -> reload the cached chunk ids from Postgres (skips the OpenAI embedding call and both searches)
    miss -> embed question -> semantic + keyword -> RRF -> store in Redis."""
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
    query_vector = embed_query(question)
    semantic = semantic_search(db, query_vector, document_ids)
    keyword = keyword_search(db, question, document_ids)
    fused = reciprocal_rank_fusion([semantic, keyword], top_n=top_n)

    if fused:   # never cache an empty result
        set_cached_retrieval(key, document_ids, [[str(chunk.id), score] for chunk, score in fused])
    return fused
