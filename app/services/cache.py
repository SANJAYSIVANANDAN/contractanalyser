import redis

from app.config import settings

# One Redis client per process. decode_responses=True gives us str instead of bytes.
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def acquire_lock(key: str, ttl_seconds: int = 1800) -> bool:
    """SET key value NX EX ttl  -> atomic "create only if it doesn't exist".
    Returns True for exactly ONE caller; everyone else gets False.
    The TTL means a crashed worker can't hold the lock forever."""
    return bool(redis_client.set(key, "1", nx=True, ex=ttl_seconds))


def release_lock(key: str) -> None:
    redis_client.delete(key)

# Phase 6 adds the retrieval cache (cache-aside) to this file.
