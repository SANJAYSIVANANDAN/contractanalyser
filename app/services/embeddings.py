from openai import OpenAI

from app.config import settings

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
BATCH_SIZE = 64

# max_retries=3: the OpenAI client itself retries short network/429 hiccups.
client = OpenAI(api_key=settings.openai_api_key, max_retries=3)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts, 64 per API call. Returns one vector per input, same order.
    Errors (rate limit, network) are NOT caught here: the Celery task decides whether to retry."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(question: str) -> list[float]:
    """Same model for queries as for chunks, otherwise the vectors aren't comparable."""
    return embed_texts([question])[0]
