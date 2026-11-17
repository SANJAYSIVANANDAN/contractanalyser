# DocuPilot AI — Build Plan

## Context
Build a citation-grounded contract-intelligence chatbot (FastAPI + Celery/RabbitMQ + Docling + PostgreSQL/pgvector + Redis + OpenAI + Streamlit) in ~2 hours. The user is a beginner who must understand and explain every file in an interview, so: explicit code, no LangChain, no abstractions, build → explain → trace → test → modify after each phase. Never claim an acceptance criterion is done without testing it; defer rather than fake.

## Current state
Repo `c:\Users\sanja\projects\contractanalyser` was empty (only `sample.txt`). An interrupted command already wrote the Phase 1 skeleton (untested, nothing committed, Docker NOT yet started):
`requirements.txt`, `.env.example`, `.env`, `.gitignore`, `Dockerfile`, `docker-compose.yml` (postgres/pgvector, rabbitmq-management, redis, api), `app/config.py`, `app/db/database.py`, `app/main.py` (`/health`), empty `__init__.py` package files, `uploads/.gitkeep`, empty dirs (`frontend/`, `tests/`, `app/evaluation/`).
Phase 1 execution therefore = review those files, run them, verify.

## Phases (each ends with: show code, explain, trace, test commands + expected output, likely errors, modification exercise, 2-3 interview questions)

**Phase 1 (0–15 min) — skeleton + infra** *(files written, needs verification)*
- `docker compose up -d --build`; `docker compose ps` shows all healthy.
- `curl localhost:8000/health` → `{"status":"ok","database":"up","pgvector":"0.x"}`; RabbitMQ UI at :15672.
- Known risk: Windows Docker bind-mount `.:/app` + `--reload`; `.env` hostnames are compose service names (won't work for host-run uvicorn).

**Phase 2 (15–35) — upload + Celery**
- `db/models.py` (documents, chunks, conversations, messages, clauses; pgvector `Vector(1536)`, `TSVECTOR`), Alembic init + first migration (vector extension, GIN + vector indexes), remove extension creation from `/health`.
- `schemas/documents.py`, `api/documents.py`: `POST /documents/upload` (validate PDF, SHA-256, save to shared volume, row PENDING, `process_document.delay(doc_id)`, return id), list/get/status/delete.
- `workers/celery_app.py` (RabbitMQ broker, Redis result backend optional), `workers/tasks.py` stub that flips PENDING→PROCESSING→COMPLETED; add `worker` service to compose sharing the `uploads` volume.
- Verify: upload returns instantly; message visible in RabbitMQ UI; worker logs show the task.

**Phase 3 (35–60) — ingestion**
- `services/document_parser.py` (Docling `DocumentConverter`, iterate items with `prov[].page_no`, section headings), `chunker.py` (page-aware, never merges across pages; no fabricated page numbers), `embeddings.py` (batched `text-embedding-3-small`).
- `tasks.py`: idempotency (Redis `SET NX` lock + delete-old-chunks-in-one-transaction), `autoretry_for` on temporary errors, FAILED + `error_message` on unrecoverable. Load Docling converter once per worker process (lazy module-level); bake model download into worker image/volume.
- Verify: chunks in Postgres with correct page numbers (spot-check vs PDF); retry creates no duplicates.

**Phase 4 (60–80) — retrieval**
- `retrieval.py`: pgvector cosine search, `websearch_to_tsquery` + `ts_rank` FTS (search_vector populated at insert via `to_tsvector`), separate `rrf()` function (k=60), doc-id filtering, dedupe. `reranker.py`: CrossEncoder `ms-marco-MiniLM-L-6-v2` loaded once, top 5.
- Verify: three small scripts/tests print ranked lists for a known question.

**Phase 5 (80–105) — grounded answers + SSE**
- **CHANGE (user request, Phase 4): use LangChain + LangGraph.** Retrieval/RRF/reranker stay custom (tested). `rag.py` becomes a LangGraph `StateGraph` with explicit nodes: `retrieve` (hybrid_retrieve) → `rerank` → `generate` (LangChain `ChatOpenAI` + `ChatPromptTemplate`, structured output of answer + cited chunk_ids) → `validate_citations` (backend rebuilds citations from DB). Conditional edge: no chunks → straight to an insufficient-evidence node. Streaming uses `graph.astream(stream_mode="messages")`. Embeddings stay on the OpenAI SDK.
- `rag.py`: chunks passed with their UUIDs; LLM returns JSON (answer + cited chunk_ids); backend validates IDs ⊆ retrieved set, rebuilds citations (filename, page, section, excerpt) from DB rows only; empty retrieval → `insufficient_evidence=true`, no citations. Strict system prompt from spec.
- `schemas/chat.py`, `api/chat.py`: `POST /chat`, `POST /chat/stream` (events token / citations / done), persist messages, handle disconnect + OpenAI errors, `GET /conversations/{id}/messages`.
- Streaming design decision: stream tokens of the answer, then validate citation IDs and send `citations` event after (citations never come from raw model text).

**Phase 6 (105–120) — cache + UI + end-to-end test**
- `cache.py` (cache-aside, TTL 600, key = normalized question + sorted doc ids + config version; invalidate on delete/reprocess); wire into retrieval.
- `frontend/streamlit_app.py` (upload, doc list/status, doc selector, streaming chat, citation cards) + compose service.
- Test with a real PDF: cited answer, page number check, unanswerable question → insufficient evidence, cache hit on repeat.

**Next milestone (deferred unless time remains):** `contract_analyzer.py` + clauses endpoints (only report "missing" after scanning all chunks), RAGAS `evaluation/dataset.py` + `evaluate.py` (4 configs: vector / keyword / hybrid / hybrid+rerank; CSV/JSON output), README, interview-prep Q&A.

## Key design decisions
- Worker receives only `document_id`; PDFs on shared Docker volume.
- Citations built in backend from DB metadata; LLM only picks chunk IDs.
- LangChain/LangGraph (added at user's request, reversing the original "avoid" instruction) run the question-answering flow only (one file, `rag.py`); the search logic itself stays plain Python/SQL.
- Alembic owns schema; no `create_all` in production path.
- Pin dependency versions; add heavy deps (celery, docling, sentence-transformers, openai, redis, streamlit, ragas) only in the phase that needs them. Worker image is separate/heavier than API image (torch).

## Verification (overall)
Walk the 19 acceptance criteria in the spec; mark each only after an actual run with observed output; list any as "deferred" explicitly.

## Open questions for the user
- Do you have an `OPENAI_API_KEY` ready (needed from Phase 3) and a sample contract PDF with real page numbers for testing?
- Is Docker Desktop running on this machine (Phase 1 needs it)?
