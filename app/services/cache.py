import hashlib
import json
import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)

# One Redis client per process. decode_responses=True gives us str instead of bytes.
# Short timeouts: if Redis is slow or down we want to skip the cache, not hang the request.
redis_client = redis.Redis.from_url(
    settings.redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
)

RETRIEVAL_CACHE_TTL = 600  # seconds
# Part of every cache key. Bump it whenever retrieval logic changes (RRF k, top-k, search
# queries...) so old cached results are never served for the new logic.
RETRIEVAL_VERSION = "v1-rrf60-top15"


# ------------------------------------------------------------------ locks (used by the worker)
def acquire_lock(key: str, ttl_seconds: int = 1800) -> bool:
    """SET key value NX EX ttl  -> atomic "create only if it doesn't exist".
    Returns True for exactly ONE caller; everyone else gets False.
    The TTL means a crashed worker can't hold the lock forever."""
    return bool(redis_client.set(key, "1", nx=True, ex=ttl_seconds))


def release_lock(key: str) -> None:
    redis_client.delete(key)


# ------------------------------------------------------------------ retrieval cache (cache-aside)
def make_retrieval_key(question: str, document_ids: list) -> str:
    """Key = version + hash(normalized question + sorted document ids).
    Normalizing (lowercase, collapse spaces) makes 'Payment  terms?' and 'payment terms?' share an entry.
    Sorting the ids makes [A, B] and [B, A] the same search."""
    normalized = " ".join(question.lower().split())
    ids = ",".join(sorted(str(d) for d in document_ids))
    digest = hashlib.sha256(f"{normalized}|{ids}".encode()).hexdigest()
    return f"retrieval:{RETRIEVAL_VERSION}:{digest}"


def get_cached_retrieval(key: str):
    """Returns [[chunk_id, rrf_score], ...] or None on a miss.
    Any Redis problem is treated as a miss: the cache must never break answering."""
    try:
        value = redis_client.get(key)
        return json.loads(value) if value else None
    except redis.RedisError:
        logger.warning("Redis unavailable: skipping retrieval cache read")
        return None


def set_cached_retrieval(key: str, document_ids: list, items: list) -> None:
    """Store the result for 600 s, and remember which documents it depends on,
    so deleting or reprocessing a document can find and remove its entries."""
    try:
        pipe = redis_client.pipeline()
        pipe.setex(key, RETRIEVAL_CACHE_TTL, json.dumps(items))
        for doc_id in document_ids:
            index_key = f"retrieval_keys:{doc_id}"
            pipe.sadd(index_key, key)
            pipe.expire(index_key, RETRIEVAL_CACHE_TTL + 60)
        pipe.execute()
    except redis.RedisError:
        logger.warning("Redis unavailable: skipping retrieval cache write")


def invalidate_document(doc_id) -> int:
    """Remove every cached retrieval that included this document. Returns how many were removed."""
    try:
        index_key = f"retrieval_keys:{doc_id}"
        keys = redis_client.smembers(index_key)
        if keys:
            redis_client.delete(*keys)
        redis_client.delete(index_key)
        return len(keys)
    except redis.RedisError:
        logger.warning("Redis unavailable: could not invalidate cache for %s", doc_id)
        return 0
