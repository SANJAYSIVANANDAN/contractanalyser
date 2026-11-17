# DocuPilot AI — System Design

A contract-intelligence chatbot. Users upload contract PDFs, then ask questions. Every factual answer is backed by citations (filename, real PDF page, section, excerpt) that the **backend** builds from database rows — never from text the LLM wrote.

> **Status as of Phase 6.** Built and tested: upload, async ingestion, hybrid retrieval, Redis retrieval cache, reranking, LangGraph answer pipeline, citation validation, `/chat`, SSE `/chat/stream`, conversation history, Streamlit UI, RAGAS evaluation (section 7b). **Not built yet:** clause extraction + `/clauses` `/analysis` endpoints, OCR. See [section 13](#13-status-against-the-acceptance-criteria).

---

## 1. Architecture

```
                         ┌──────────────┐   HTTP / SSE   ┌─────────────────────────┐
                         │  Client      │ ─────────────► │  FastAPI  (Process A)   │
                         │ (Streamlit UI / curl)         │  upload, status, chat   │
                         └──────────────┘                └──┬────────┬─────────┬───┘
                                                            │        │         │
                                         publish doc_id     │        │ SQL     │ retrieval cache
                                                            ▼        ▼         ▼
                                                     ┌──────────┐ ┌────────────────┐ ┌───────┐
                                                     │ RabbitMQ │ │ PostgreSQL     │ │ Redis │
                                                     │ (queue)  │ │ + pgvector     │ │       │
                                                     └────┬─────┘ └───────▲────────┘ └───▲───┘
                                                          │ doc_id        │ chunks,      │ lock
                                                          ▼               │ vectors      │
                                                 ┌──────────────────┐     │              │
                                                 │ Celery worker    │ ────┴──────────────┘
                                                 │ (Process B)      │
                                                 │ Docling → chunk  │ ──► OpenAI embeddings
                                                 └────────▲─────────┘
                                                          │ reads PDF
                                              shared Docker volume  /app/uploads
                                              (API writes, worker reads)
```

Two separate processes, one codebase (a *modular monolith*):

| Process | Does | Never does |
|---|---|---|
| **A: FastAPI** | validate + save uploads, create DB rows, publish tasks, answer chat requests, stream SSE | convert PDFs or create embeddings in a request |
| **B: Celery worker** | convert PDF with Docling, chunk, embed, store | talk to end users |

The message on the queue contains **only the document ID**. The worker loads the PDF from the shared volume and the row from Postgres.

---

## 2. Why each technology

| Tech | Role | Why this and not something else |
|---|---|---|
| **FastAPI** | HTTP API | Type-hint based validation (Pydantic), dependency injection, native `StreamingResponse` for SSE. |
| **PostgreSQL** | System of record | Relational data (documents ↔ chunks ↔ messages), transactions, and — with extensions — vector + full-text search in one place. |
| **pgvector** | Vector similarity | One database instead of a second vector store: chunk metadata and vectors are joined in a single query and committed in a single transaction. HNSW index for fast cosine search. |
| **RabbitMQ** | Message broker | Durable queue between API and worker. If the worker is down, tasks wait. Acks give at-least-once delivery. |
| **Celery** | Task framework | Worker lifecycle, retries with backoff, acks — so we don't hand-write a consumer loop. |
| **Redis** | Processing lock + retrieval cache | Fast, atomic `SET NX EX` for the processing lock; cache-aside store for retrieval results with a 600 s TTL. Disposable by design — no permanent contract data lives there. |
| **Streamlit** | UI | Minimal Python-only front end: upload, document status, document selection, streaming chat, citation blocks. Talks to the API over HTTP/SSE. |
| **Docling** | PDF → structured document | Preserves layout, headings, tables, and per-item page provenance (`prov[].page_no`), which citations depend on. |
| **OpenAI `text-embedding-3-small`** | Embeddings (1536-dim) | Cheap, strong baseline. The *same* model embeds chunks and queries. |
| **CrossEncoder `ms-marco-MiniLM-L-6-v2`** | Local reranker | Reads question + passage together → much more precise than embeddings; runs locally, no API call. |
| **LangGraph** | Orchestrates the answer pipeline | Makes the pipeline an explicit graph of named steps with a conditional branch. Added at the user's request. |
| **LangChain** (`ChatOpenAI`, `ChatPromptTemplate`) | Model + prompt wrapper | Standard interface; token streaming integrates with LangGraph's `stream_mode="messages"`. |
| **SQLAlchemy 2.0 + Alembic** | ORM + migrations | Explicit models; schema changes are versioned files, not ad-hoc `create_all`. |
| **Docker Compose** | Local infrastructure | One command starts all services with health checks and shared volumes. |

Retrieval itself (semantic search, keyword search, RRF, reranking) is **plain Python + SQL** — not LangChain — so each step can be read and tested directly.

---

## 3. Data model

```
documents 1───* chunks            documents 1───* clauses   (table exists, not populated yet)
conversations 1───* messages
```

| Table | Key columns |
|---|---|
| `documents` | `id` (UUID), `filename`, `file_path`, `file_hash` (SHA-256), `status` (PENDING/PROCESSING/COMPLETED/FAILED), `error_message`, `created_at` |
| `chunks` | `id`, `document_id` (FK, cascade), `chunk_index`, `content`, `page_number` (nullable), `section_title`, `embedding VECTOR(1536)`, `search_vector TSVECTOR` |
| `conversations` | `id`, `title`, `created_at` |
| `messages` | `id`, `conversation_id` (FK), `role`, `content`, `citations` (JSON), `created_at` |
| `clauses` | `id`, `document_id`, `clause_type`, `title`, `summary`, `supporting_text`, `risk_level`, `explanation`, `page_number` |

Indexes (declared in [models.py](app/db/models.py), created by the Alembic migration):
- **HNSW** on `chunks.embedding` with `vector_cosine_ops` — approximate nearest-neighbour search.
- **GIN** on `chunks.search_vector` — inverted index for full-text search.
- B-tree on foreign keys and `documents.file_hash`.

`page_number` is nullable on purpose: if Docling can't establish a page, we store `NULL` rather than guess.

---

## 4. Ingestion flow (upload → searchable)

1. `POST /documents/upload` ([api/documents.py](app/api/documents.py)) validates: `.pdf` extension, ≤ 20 MB, starts with `%PDF`.
2. SHA-256 of the bytes. If a document with the same hash exists, return it (no duplicate work).
3. Save to `/app/uploads/<uuid>.pdf` (shared volume).
4. Insert `documents` row, status `PENDING`, **commit**.
5. `process_document.delay(doc_id)` → RabbitMQ. (Commit comes first, otherwise the worker could look for a row that doesn't exist yet.)
6. API returns the document immediately (measured ≈ 0.26 s).
7. Worker ([workers/tasks.py](app/workers/tasks.py)) takes a Redis lock `lock:process_document:<id>` (`SET NX EX 1800`); if it can't, it skips.
8. Loads the row; if already `COMPLETED`, skips (redelivered message).
9. Sets `PROCESSING`.
10. [document_parser.py](app/services/document_parser.py): Docling converts the PDF; we walk `iterate_items()`, skip page headers/footers, track the current section from headings, read the page from `item.prov[0].page_no`, export tables as Markdown.
11. [chunker.py](app/services/chunker.py): group blocks into ≈1000-character chunks. **A chunk never spans two pages or two sections**, so each chunk has exactly one page and one section.
12. [embeddings.py](app/services/embeddings.py): embed `"<section title>\n<content>"` in batches of 64.
13. **One transaction:** delete any old chunks for the document, insert the new ones (with `to_tsvector('english', content)`), set `COMPLETED`, commit. Rollback leaves no half-written data.
14. `finally`: release the lock.

Slow work (Docling, OpenAI) happens *before* the write transaction opens, so no DB transaction is held during a 40–100 s conversion.

Measured on the sample 15-page PDF: 74 chunks, pages 1–15, all with vectors; 14 of 15 randomly sampled chunks matched the exact PDF page by text search (the 15th was a table, which the text matcher can't verify).

---

## 5. Query flow (chat)

`POST /chat` and `POST /chat/stream` ([api/chat.py](app/api/chat.py)) run the LangGraph pipeline in [services/rag.py](app/services/rag.py):

```
START ──(specific question)──► retrieve → rerank ──(no candidates at all)──► insufficient ──► END
  │                                           │
  │                                           └──(candidates)──► generate → validate_citations ──► END
  │                                                                  ▲
  └──(summary / overview question)──► summary_retrieve ──────────────┘
```

| Node | What it does |
|---|---|
| `retrieve` | `hybrid_retrieve`: embed the question, run semantic + keyword search over the selected documents, merge with RRF → up to 15 candidates. |
| `rerank` | CrossEncoder scores each (question, passage); keep the best 5. **No score cutoff** (see section 6). |
| start edge | Questions matching "summarize / summary / overview / what is this document about…" go to `summary_retrieve`; everything else to `retrieve`. |
| `summary_retrieve` | Instead of search hits, take up to 16 passages spread evenly across each document in reading order (skipping tiny heading chunks), so a summary covers the whole document. |
| conditional edge | Only if retrieval returned nothing at all → `insufficient`; otherwise → `generate`. |
| `generate` | LangChain `prompt | ChatOpenAI(gpt-4o-mini, temperature 0)`. Passages are numbered `[1]..[5]`. |
| `validate_citations` | Backend check of every `[n]` in the answer (see section 7). |
| `insufficient` | Fixed "Insufficient evidence…" answer, empty citations. |

The DB session is passed to nodes through `config["configurable"]["db"]`, so nodes stay plain functions.

---

## 6. Retrieval details

All in [retrieval.py](app/services/retrieval.py) and [reranker.py](app/services/reranker.py).

**Semantic search.** Embed the question, order chunks by `embedding <=> query` (cosine distance), top 15, filtered by `document_id IN (…)`.

**Keyword search.** PostgreSQL full-text search with `websearch_to_tsquery('english', …)` and `ts_rank`, top 15. The default query ANDs all words, which is too strict for a sentence, so words are reduced to `\w+` and joined with `or` (this also prevents operator injection from user input). Chunks matching more words rank higher.

**Reciprocal Rank Fusion.**

```
RRF(d) = Σ over rankings  1 / (k + rank(d))      k = 60, rank starts at 1
```

Only ranks are used, never raw scores — cosine distance and `ts_rank` are on different scales and can't be added. A chunk found by both searches gets two terms and rises. Verified: the top payment chunk scored `0.0328 = 1/61 + 1/61`.

**Retrieval cache (cache-aside).** In `hybrid_retrieve` ([retrieval.py](app/services/retrieval.py), keys and helpers in [cache.py](app/services/cache.py)):

1. Key = `retrieval:<version>:<sha256(normalized question | sorted document ids)>`. Normalizing (lowercase, collapsed spaces) and sorting ids means equivalent searches share an entry; `RETRIEVAL_VERSION` is bumped when retrieval logic changes.
2. **Hit:** Redis holds `[[chunk_id, rrf_score], ...]`; the chunks are reloaded from Postgres by id (the cache stores ids and scores only, never contract text). This skips the OpenAI query embedding and both searches.
3. **Miss:** run the searches + RRF, store the result with a 600 s TTL. Empty results are never cached.
4. If any cached chunk no longer exists, the entry is treated as a miss.
5. **Invalidation:** each entry is also registered in a per-document set (`retrieval_keys:<doc_id>`); `DELETE /documents/{id}` and a finished (re)processing in the worker remove every entry that included that document.
6. **Redis down:** reads/writes are skipped with a warning and chat still works (tested: HTTP 200, but slower).

Measured: cold lookup 0.605 s → cached 0.019 s. Over HTTP the total request is ≈ 3–4 s either way, because the LLM call dominates and is intentionally not cached. The cache saves ≈ 0.6 s and an embedding call per repeated question.

**Reranking.** Embeddings encode question and passage separately (fast, rough). A cross-encoder reads them together (slow, precise), so it only runs on the ~15 RRF candidates and keeps the best 5. Its scores are raw logits, not probabilities.

**Why there is no relevance-score cutoff (a bug we hit and fixed).** Vector search *always* returns its nearest neighbours, so "no results" can't signal "no evidence". We first added a rerank-score floor (−9.0), calibrated on specific factual questions. It then rejected **every broad question** a user would naturally ask, because those score as low as questions the document truly can't answer:

| Question | Best rerank score |
|---|---|
| Summarize this agreement | −10.3 |
| Who are the parties to this agreement | −10.1 |
| Is there a confidentiality clause (not in the document) | −11.2 |
| What are the payment milestones | +5.7 |

No threshold separates "broad but answerable" from "absent", so the floor was removed. Instead: (1) the **model** decides whether the passages are enough, using the strict system prompt ("Insufficient evidence: …" plus one sentence saying what wasn't found); (2) **citation validation** guarantees any claim points at a passage we actually sent; (3) broad questions get the **summary path** above. Re-tested after the change: summarize → 9 citations across pages 3–14; parties, "what is this about" → answered with citations; termination, governing law, indemnification, "capital of France" → refused with an explanation.

Trade-off: with no hard gate, a borderline question can be answered or refused depending on the run (observed: "key risks" was refused once and answered with 3 citations another time). Answers are still limited to cited passages, but the refusal decision is the model's, not a measurable threshold.

---

## 7. Citation integrity (the core guarantee)

1. The LLM sees passages labelled `[1]…[5]` with filename, page, section. It is told to put the number after each claim.
2. After generation, `validate_citations_node` extracts every `[n]` from the text.
3. Numbers not among the passages we sent are **rejected**: removed from the answer text and reported in `rejected_sources`.
4. For valid numbers, the `Citation` object is built from the retrieved `Chunk` row: `chunk_id`, `document_id`, `filename` (from `documents`), `page_number`, `section_title`, `supporting_excerpt` (first 600 characters of the stored chunk text).
5. If no valid citation remains, or the answer begins "Insufficient evidence", the response is flagged `insufficient_evidence = true` (and "insufficient" answers carry no citations).
6. In streaming, tokens go out first; the `citations` event is sent only after validation and carries the final `insufficient_evidence` flag.

Tested with fabricated model outputs ([tests/test_citations.py](tests/test_citations.py)): valid citation resolved from DB; invented `[7]` rejected; only-invalid and uncited answers flagged insufficient; "Insufficient evidence" answers carry no citations; duplicates merged.

**Prompt-injection stance.** The system prompt marks document text as untrusted, passages are wrapped in `<passage>` tags, and — more importantly — the model has no way to alter citations or page numbers, because those never come from its text.

---

## 7b. Evaluation (RAGAS)

Code: [dataset.py](app/evaluation/dataset.py) (12 hand-written questions, each reference answer checked against the contract text and tagged with the gold page) and [evaluate.py](app/evaluation/evaluate.py). Run offline only: `docker compose --profile eval run --rm eval` (separate image, ~20 min, small OpenAI cost). Results are saved to `app/evaluation/results/` as JSON and CSV.

For each question, four configurations each retrieve 5 passages: **A** vector only, **B** keyword only, **C** hybrid (RRF), **D** hybrid + CrossEncoder rerank (judged on ranking quality alone). The same prompt and model generate each answer; RAGAS (judge: gpt-4o-mini) scores it.

| Metric | What it measures | Low value points to |
|---|---|---|
| **Faithfulness** | Share of the answer's claims supported by the retrieved passages | Generation problem (model added unsupported content) |
| **Answer relevancy** | Whether the answer addresses the question (judged from questions the answer implies) | Rambling, off-topic or hedged answers |
| **Context precision** | Whether the relevant passages are ranked near the top of what was retrieved | Ranking/noise problem in retrieval |
| **Context recall** | Whether the retrieved passages contain what the reference answer needs | Retrieval missed information |

**Results** (12 questions, one contract, 0 missing scores; file `ragas_20261116_115547`):

| Config | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| A vector only | 0.972 | 0.821 | 0.790 | 0.896 |
| B keyword only | 0.942 | 0.722 | 0.874 | 0.979 |
| C hybrid (RRF) | 0.972 | 0.807 | **0.881** | 0.958 |
| D hybrid + rerank | 0.962 | 0.774 | 0.769 | 0.958 |

**What the results do and don't show**
- No configuration wins on every metric. Hybrid (C) has the best precision, ties for the best faithfulness, and has near-best recall. Keyword (B) has the best recall but the worst answer relevancy and faithfulness.
- **The reranker is not shown to help.** D has the lowest precision and no recall gain over C, so the claim "reranking improves quality" is **not supported** by this data. (This evaluation only measures answer quality for questions the contract can answer.)
- Hybrid's value shows up per question: on "Why is Phase-1 export-driven?" vector search had context recall 0.0 (it retrieved headings containing "Export-Driven" but not the passage about Sell.Do's missing APIs), while keyword and hybrid scored 1.0.
- **Small sample.** With 12 questions, one question flipping from 0 to 1 moves a mean by 0.083. Most gaps above are within that noise; treat the table as a baseline, not a ranking.

**Caveats**
- Answer relevancy is unreliable here: RAGAS scored 0.00 on "What is the total duration?" for all four configs although the answers were correct (it appears to treat the hedge "depending on client validation turnaround" as non-committal), and RAGAS warned it could generate only 1 of the 3 questions it normally uses.
- The judge and the generator are the same model family (gpt-4o-mini), which can flatter faithfulness.
- Our own `page_hit` check (did any retrieved chunk come from the gold page?) was 1.0 for all configs, so it is too coarse to discriminate.
- A single run; repeat runs were not done, so run-to-run variance is unknown. An earlier run with RAGAS's default concurrency produced many timeouts (missing scores) and was discarded; the final run uses 4 workers and a 300 s timeout.
- Unanswerable questions are excluded (RAGAS needs a reference); that behaviour is covered by the chat tests.

---

## 8. API

| Endpoint | State |
|---|---|
| `GET /health` | ✅ |
| `POST /documents/upload`, `GET /documents`, `GET /documents/{id}`, `GET /documents/{id}/status`, `DELETE /documents/{id}` | ✅ |
| `POST /chat` (JSON answer + citations) | ✅ |
| `POST /chat/stream` (SSE: `token`, `citations`, `done`, `error`) | ✅ |
| `GET /conversations/{id}/messages` | ✅ |
| `GET /documents/{id}/clauses`, `GET /documents/{id}/analysis` | ⏳ deferred (clause extraction) |

SSE choice: the stream is one-directional (server → client), works over plain HTTP, reconnects easily, and needs no handshake protocol. WebSockets would add bidirectional capability we don't need.

---

## 9. Failure handling and idempotency

| Situation | Behaviour |
|---|---|
| Same file uploaded twice | Hash match → existing document returned |
| Duplicate / redelivered task | Redis lock + `status == COMPLETED` check → skipped |
| Worker crashes mid-task | `acks_late=True` → message redelivered; reprocessing deletes old chunks first |
| Retry creates duplicates? | No — delete + insert + status update are one transaction |
| OpenAI rate-limit / timeout / 5xx / connection error | Task retried with backoff 20 s, 40 s, 80 s (max 3), then `FAILED` |
| Corrupt or unreadable PDF | Immediate `FAILED` with the error message (tested) |
| Scanned PDF (no text layer) | `FAILED: no text extracted` — OCR is off (deferred) |
| Lock holder crashes | Lock expires after 1800 s (TTL) |
| Client disconnects mid-stream | Generator stops; nothing half-finished is stored |
| OpenAI connection drops mid-answer (`httpx.RemoteProtocolError: peer closed connection … incomplete chunked read`) | Seen once in testing (it produced a 502). `/chat` now retries the whole pipeline up to 3× (1 s, 2 s pauses). `/chat/stream` retries only while no text has been sent (retrying after would repeat text on screen); otherwise it sends `event: error`. The retry logic was tested with a simulated drop; a real drop has not recurred since. 40/40 direct OpenAI calls and 8/8 streamed chats succeeded afterwards. |
| Redis down during chat | Cache skipped with a warning; answer still returned (tested). Ingestion is **not** protected: the worker's lock call fails before its `try` block, leaving the document `PENDING` (known gap) |
| Stale cache after delete/reprocess | Entries for that document are removed (tested: 3 keys → 0 after reprocess) |
| OpenAI failure during chat | `/chat` → 502; stream → `event: error` |
| DB failure when saving the answer | Rollback, `event: error` |

**Not yet exercised:** the OpenAI retry path (no outage was simulated) and a true concurrent race on the lock (the worker runs one task at a time; the lock itself was tested directly).

---

## 10. Scaling and large PDFs

- **More throughput:** run more worker containers (`--scale worker=N`) — they compete for messages on the same queue; the Redis lock and DB status check keep them from doing the same document twice.
- **Large PDFs:** the heavy step is Docling (CPU-bound; ≈ 40 s for 15 pages warm, +~60 s first-run model load). Options: raise worker concurrency/CPU, split the PDF by page ranges into sub-tasks, raise the 20 MB cap with streaming upload, add a task time limit.
- **Larger corpora:** HNSW keeps vector search sub-linear; filter by `document_id` keeps queries scoped. Beyond one Postgres node you'd partition by tenant or move vectors to a dedicated store.
- **API scaling:** the API is stateless (state is in Postgres/Redis), so replicate it behind a load balancer. The reranker model loads once per API process (memory cost per replica).

---

## 11. Design decisions and trade-offs

- **Numbered sources instead of raw UUIDs in the prompt.** The model cites `[1]..[5]`, which is easier to copy correctly than 36-character UUIDs; the backend maps numbers to real chunk UUIDs, which appear in the final `Citation` objects. This deviates slightly from "pass chunk IDs to the LLM"; the security property (model can only reference what we sent) is the same.
- **LangChain/LangGraph for generation only.** Reversed the original "no frameworks" instruction at the user's request. Embeddings still use the OpenAI SDK directly (the worker uses them and gained nothing from a wrapper).
- **Sync SQLAlchemy, sync LangGraph `stream()`.** Simpler to read than async; FastAPI runs sync endpoints in a thread pool. The streaming endpoint opens its own session because FastAPI (0.115) closes `Depends(get_db)` sessions before a streaming body ends.
- **Single-turn answers.** Conversation messages are stored, but earlier turns are not fed back into the prompt. Follow-ups like "what about the second one?" won't work.
- **Excerpt = start of the cited chunk,** not the exact sentence. It is real stored text (never LLM-written) but may include neighbouring text; sentence-level excerpts are a possible improvement.
- **No OCR** in the MVP to keep ingestion fast and the worker image simple.

---

## 12. Known limitations

1. No relevance cutoff: whether the passages suffice is decided by the model (section 6), so borderline questions can vary between runs. The summary trigger is a keyword regex, not real intent detection.
2. Compare-two-contracts works only as "retrieve across both documents"; top-5 may come mostly from one document. No per-document balancing yet.
3. Many small chunks (one per section) — safe for citations, noisy for retrieval; merging small neighbours is a possible tweak.
4. Table chunks are Markdown with `|` characters.
5. No auth, no per-user document ownership, no rate limiting.
6. Uploaded files are validated by extension and magic bytes only (no malware scan).
7. Output is an informational aid, not legal advice.
8. The API's `--reload` does not reliably pick up file changes on Windows bind mounts — restart the container after editing code.
9. The reranker is preloaded at API startup (first request ≈ 3.5 s instead of ≈ 17 s), which makes API start-up slower.

---

## 13. Status against the acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | PDF upload works | ✅ tested |
| 2 | Document ID returned without waiting for conversion | ✅ tested (≈0.26 s) |
| 3 | RabbitMQ receives a task | ✅ worker received it via the broker (management UI not inspected) |
| 4 | Separate Celery worker executes it | ✅ tested |
| 5 | Docling converts the PDF | ✅ tested |
| 6 | Page metadata retained | ✅ tested (14/15 sampled exact; 1 table unverifiable) |
| 7 | OpenAI embeddings generated | ✅ tested |
| 8 | PostgreSQL stores chunks and vectors | ✅ tested (74 chunks, 1536 dims) |
| 9 | Vector search works | ✅ tested |
| 10 | Keyword search works | ✅ tested |
| 11 | RRF works | ✅ tested |
| 12 | Local reranking works | ✅ tested |
| 13 | GPT-4o-mini generates grounded answers | ✅ tested |
| 14 | Every factual answer has validated citations | ✅ tested (live + fabricated-output unit tests) |
| 15 | Citation filename/page come from DB metadata | ✅ tested |
| 16 | Unsupported questions → insufficient evidence | ✅ re-tested after removing the score floor (termination, governing law, indemnification, off-topic all refused with an explanation). Note: "key risks" varied between runs. |
| 17 | SSE streams the response | ✅ tested (51 token events → citations → done) |
| 18 | Redis caches retrieval results | ✅ tested (MISS → HIT → HIT for a repeated/normalized question, TTL ≈ 600 s, invalidation on reprocess) |
| 19 | Streamlit displays answer and citations | ✅ tested headlessly with Streamlit `AppTest` against the live API (answer, source blocks with filename and page, insufficient-evidence warning). **Not checked visually in a browser.** |

Deferred: clause extraction and `/clauses` `/analysis` endpoints (so the UI has a "Summarize" button that uses chat, but no extracted-clauses panel), OCR.

---

## 14. Repository map and how to run

```
app/main.py                  FastAPI app, routers
app/config.py                settings from environment
app/db/database.py           engine, SessionLocal, Base, get_db
app/db/models.py             5 ORM tables + indexes
app/schemas/                 Pydantic request/response models
app/api/documents.py         upload, list, status, delete
app/api/chat.py              /chat, /chat/stream, messages
app/services/document_parser.py   Docling → blocks with page + section
app/services/chunker.py           blocks → page-safe chunks
app/services/embeddings.py        OpenAI embeddings
app/services/retrieval.py         semantic, keyword, RRF
app/services/reranker.py          CrossEncoder
app/services/rag.py               LangGraph pipeline + citation validation
app/services/cache.py             Redis client, processing lock, retrieval cache
frontend/streamlit_app.py         Streamlit UI
app/workers/                      Celery app + process_document task
alembic/                          migrations
app/evaluation/                   RAGAS dataset, evaluate.py, results/
Dockerfile.eval, requirements-eval.txt   separate image for evaluation (RAGAS)
tests/                            try_retrieval.py, test_citations.py, sample PDF
```

```bash
docker compose up -d --build                 # UI :8502, API :8001, RabbitMQ UI :15673, Postgres :5433, Redis :6380
docker compose exec api alembic upgrade head # create tables (first run)
curl -F "file=@tests/Masked_apeXaudit-Proposal_v8.pdf" localhost:8001/documents/upload
curl localhost:8001/documents/<id>/status    # wait for COMPLETED
curl -X POST localhost:8001/chat -H 'Content-Type: application/json' \
     -d '{"question": "What are the payment milestones?"}'
docker compose exec api python -m tests.test_citations
docker compose --profile eval run --rm eval  # RAGAS evaluation (~20 min, uses OpenAI)
```

Open http://localhost:8502 for the UI. Host ports are shifted (8502/8001/5433/5673/15673/6380) because another project was using the defaults; containers talk to each other on the standard ports.

---

## 15. 60-second interview pitch

"DocuPilot is a contract Q&A system where every answer is cited and the citations can't be faked. Uploads go to a FastAPI service that saves the PDF, writes a row, and publishes just the document ID to RabbitMQ, returning in a fraction of a second. A separate Celery worker converts the PDF with Docling, which keeps real page numbers, chunks it so no chunk crosses a page or section, embeds the chunks with OpenAI, and stores them in PostgreSQL with pgvector — all in one transaction, guarded by a Redis lock so retries never duplicate data. At question time a LangGraph pipeline runs hybrid retrieval — pgvector semantic search plus PostgreSQL full-text search merged with Reciprocal Rank Fusion — then a local cross-encoder reranks the candidates, and summary-style questions take a separate path that samples the whole document. GPT-4o-mini answers using numbered passages, and then the backend validates every citation number and builds the citation objects from database rows, so the model can't invent a page or a filename. Answers stream to the client over SSE, and the citations arrive only after validation."
