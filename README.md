# ContractAnalyser

An AI-powered contract analysis platform: upload a PDF contract, ask questions about it
with cited answers, and get an automated clause/risk breakdown (termination, liability,
indemnity, IP, confidentiality, and more).

## Architecture

- **FastAPI** backend (`app/`), **Streamlit** frontend (`frontend/`), **PostgreSQL + pgvector**
  for storage, **Redis** for caching and as the Celery broker/backend, **Celery** for background
  document ingestion and clause extraction.
- **Hybrid retrieval**: dense search (pgvector cosine similarity) + real BM25 (computed
  in-process per document with `rank_bm25`), fused with Reciprocal Rank Fusion. A cross-encoder
  reranker (blended with the fused rank, not overriding it) narrows the final candidates — see
  `app/services/retrieval.py` and `app/services/reranker.py`.
- **LangGraph** orchestrates the Q&A pipeline (`app/services/rag.py`): retrieve → rerank →
  generate → validate citations. The graph is compiled with a **Postgres-backed checkpointer**,
  so a conversation's state (including recent turn history) is persisted per
  `user_id:conversation_id` and survives logins and restarts — this is the app's conversation
  memory, not just a replayed message log.
- **Auth**: a minimal email/password login (JWT) ties documents and conversations to a
  returning user. No Alembic — tables are created directly from the SQLAlchemy models
  (`app/db/database.py::init_db`) on process startup.
- **RAGAS evaluation** (`app/evaluation/`) benchmarks multiple retrieval configurations
  (vector-only, BM25-only, hybrid RRF, hybrid+rerank, hybrid+rerank+query-rewrite) against a
  hand-written gold Q&A set, so retrieval/reranking changes are measured, not assumed.



## Running it

```
cp .env.example .env     # fill in OPENAI_API_KEY at least
docker compose up --build
```

- API: http://localhost:8001 (docs at `/docs`)
- Frontend: http://localhost:8502
- `postgres` and `redis` are internal services (ports 5433/6380 on the host, for debugging)

The `api`, `worker`, and `frontend` services all build from the same single `Dockerfile` and
differ only by their `command:` in `docker-compose.yml`. The first build is slow (PyTorch +
Docling + Sentence-Transformers + Ragas all land in one image) — later builds are fast thanks
to Docker's layer cache as long as `requirements.txt` doesn't change.

Everything behind `/documents`, `/chat`, `/conversations` requires a login — register once via
the Streamlit UI, or directly against the API:

```
curl -X POST http://localhost:8001/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "a-strong-password"}'
# -> {"access_token": "...", "token_type": "bearer", "user_id": "...", "email": "..."}

curl -X POST http://localhost:8001/documents/upload \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@contract.pdf"
```

## Evaluating retrieval quality

```
docker compose run --rm api python -m app.evaluation.evaluate
```

Scores go to `app/evaluation/results/` (JSON + CSV), comparing vector-only, BM25-only, hybrid
RRF, hybrid+rerank, and hybrid+rerank+query-rewrite against the gold dataset in
`app/evaluation/dataset.py`. Reranking is a config flag (`RERANK_ENABLED` in
`app/services/rag.py`) — flip it off if a run shows hybrid RRF alone still wins.

## Manual checks

```
docker compose exec api python -m tests.test_citations
docker compose exec api python -m tests.try_retrieval "your question"
```
