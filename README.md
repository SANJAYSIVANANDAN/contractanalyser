# DocuPilot AI

A contract-intelligence chatbot. Upload contract PDFs, then ask questions about them. Every factual answer comes with citations (filename, real PDF page number, section title, supporting excerpt) that the **backend** builds from database records, never from text the language model wrote.

> Informational aid only. It is not legal advice.

## What it does

- Upload PDFs (employment contracts, vendor agreements, NDAs, proposals, ...).
- Ask questions: "What are the payment milestones?", "Summarize this agreement.", and so on.
- Get a streamed answer plus validated source citations (filename, page, section, excerpt).
- Say "insufficient evidence" instead of guessing when the document doesn't contain the answer.
- Choose which documents to search.

## How it works

```
Streamlit UI ──HTTP / SSE──► FastAPI ──publishes document id──► RabbitMQ ──► Celery worker
                               │  │                                            │
                               │  └─ Redis (retrieval cache, processing lock)  │ Docling → chunks → OpenAI embeddings
                               ▼                                               ▼
                       PostgreSQL + pgvector  ◄──────────────────────  chunks + vectors
```

1. **Upload:** the API validates the PDF, saves it to a shared volume, records it in PostgreSQL and queues a task. It returns immediately.
2. **Ingestion (worker):** Docling converts the PDF and keeps real page numbers. The text is split into chunks that never cross a page or section. OpenAI embeds each chunk and PostgreSQL stores it.
3. **Question answering:** a LangGraph pipeline runs hybrid retrieval (pgvector semantic search plus PostgreSQL full-text search, merged with Reciprocal Rank Fusion), reranks locally with a cross-encoder, and has GPT-4o-mini answer from the numbered passages.
4. **Citations:** the model only writes source numbers like `[1]`. The backend rejects any number it didn't send and builds each citation from the stored chunk record.

For the full design, trade-offs and test results, see [system design.md](system%20design.md).

## Tech stack

| Area | Technology |
|---|---|
| API | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0, Alembic |
| Database | PostgreSQL with pgvector (HNSW index) and full-text search (GIN index) |
| Background jobs | Celery with RabbitMQ |
| Cache and locks | Redis |
| PDF parsing | Docling |
| AI | OpenAI `text-embedding-3-small`, `gpt-4o-mini`; local cross-encoder `ms-marco-MiniLM-L-6-v2` |
| Question-answering flow | LangGraph (the step-by-step flow) and LangChain (`ChatOpenAI` and the prompt), all in `app/services/rag.py`. The search logic itself is plain Python and SQL. |
| UI | Streamlit |
| Evaluation | RAGAS |
| Infrastructure | Docker Compose |

## Quick start

**Prerequisites:** Docker Desktop (running), an OpenAI API key, and a few GB of free disk space (PyTorch and Docling are large).

1. **Create your environment file**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set `OPENAI_API_KEY=` to your key. Leave the other values alone. `.env` is git-ignored.

2. **Build and start everything**
   ```bash
   docker compose up -d --build
   docker compose ps
   ```
   The first build takes about 5–10 minutes. All six services should show `Up`.

3. **Create the database tables (first run only)**
   ```bash
   docker compose exec api alembic upgrade head
   ```

4. **Wait for the API** (it loads the reranker model at startup; up to about a minute the first time)
   ```bash
   curl http://localhost:8001/health
   ```
   Expected: `{"status":"ok","database":"up"}`. On PowerShell use `curl.exe`.

5. **Open the app**

   | What | URL |
   |---|---|
   | Chat UI | http://localhost:8502 |
   | API docs (Swagger) | http://localhost:8001/docs |
   | RabbitMQ dashboard | http://localhost:15673 (guest / guest) |

   Host ports are shifted from the defaults (8001, 8502, 5433, 5673, 15673, 6380) to avoid clashing with other local projects. Containers talk to each other on the standard ports. Change the left-hand numbers in `docker-compose.yml` if you need different ones.

## Using the app

1. In the sidebar, upload a PDF (`tests/Masked_apeXaudit-Proposal_v8.pdf` is included as a sample) and click **Upload**.
2. Click **Refresh status** until it shows ✅ COMPLETED. About 40 seconds for 15 pages; the very first document is slower because Docling downloads its models once.
3. Make sure the document is ticked under "Search in", then ask a question.

Try:
- `What are the payment milestones?` returns an answer with citations (page 10 in the sample).
- `Summarize this agreement.` returns a summary citing passages across the document.
- `What are the termination conditions?` returns an insufficient-evidence message, because the sample has no such clause.

### Stop and restart
```bash
docker compose down       # stop; data is kept
docker compose up -d      # start again
docker compose down -v    # DESTRUCTIVE: also deletes the database and uploads
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and database check |
| `POST /documents/upload` | Upload a PDF; returns its id immediately |
| `GET /documents`, `GET /documents/{id}` | List and fetch documents |
| `GET /documents/{id}/status` | `PENDING`, `PROCESSING`, `COMPLETED` or `FAILED` |
| `DELETE /documents/{id}` | Delete a document, its chunks and cached results |
| `POST /chat` | Answer a question; JSON with answer, citations and an `insufficient_evidence` flag |
| `POST /chat/stream` | Same, streamed over Server-Sent Events (`token`, `citations`, `done`, `error`) |
| `GET /conversations/{id}/messages` | Stored conversation history |

Example:
```bash
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" \
     -d '{"question": "What are the payment milestones?"}'
```
Not built yet: `GET /documents/{id}/clauses` and `GET /documents/{id}/analysis`.

## Project structure

```
app/
  main.py, config.py            FastAPI app and settings
  api/                          documents.py, chat.py (endpoints)
  db/                           database.py (engine, sessions), models.py (5 tables)
  schemas/                      Pydantic request/response models
  services/
    document_parser.py          Docling to text blocks with page and section
    chunker.py                  page-safe chunking
    embeddings.py               OpenAI embeddings
    retrieval.py                semantic search, keyword search, RRF, cache-aside
    reranker.py                 local cross-encoder
    rag.py                      LangGraph pipeline and citation validation
    cache.py                    Redis client, locks, retrieval cache
  workers/                      Celery app and the process_document task
  evaluation/                   RAGAS dataset, evaluate.py, saved results
frontend/streamlit_app.py       the UI
alembic/                        database migrations
tests/                          citation tests, retrieval check script, sample PDF
Dockerfile, Dockerfile.worker, Dockerfile.frontend, Dockerfile.eval
docker-compose.yml
```

## Tests and evaluation

```bash
# Citation validator (no OpenAI calls): invented source numbers are rejected, etc.
docker compose exec api python -m tests.test_citations

# See what each retrieval stage returns for a question
docker compose exec api python -m tests.try_retrieval "What are the payment terms?"

# RAGAS evaluation: ~20 minutes and some OpenAI cost. Run offline, not per question.
docker compose --profile eval run --rm eval
```

The RAGAS run compares four retrieval setups on 12 hand-verified questions about the sample contract (saved in `app/evaluation/results/`):

| Config | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Vector only | 0.972 | 0.821 | 0.790 | 0.896 |
| Keyword only | 0.942 | 0.722 | 0.874 | 0.979 |
| Hybrid (RRF) | 0.972 | 0.807 | 0.881 | 0.958 |
| Hybrid + rerank | 0.962 | 0.774 | 0.769 | 0.958 |

No setup wins on every metric, and reranking did not improve these scores. With only 12 questions and one run, small differences are within noise. See section 7b of [system design.md](system%20design.md) for what each metric means and the caveats.

## Known limitations

- Clause extraction and risk analysis (the `/clauses` and `/analysis` endpoints) are not built.
- No OCR: scanned (image-only) PDFs fail with "no text extracted".
- Single-turn answers: earlier messages in a conversation are stored but not used as context.
- There is no relevance cutoff. The model decides whether the retrieved passages are enough, so borderline questions can be answered in one run and refused in another.
- "Compare two contracts" searches across both documents but doesn't balance passages between them.
- No authentication, per-user document ownership or rate limiting.
- The ingestion worker isn't protected against Redis being down (the document would stay `PENDING`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "port is already allocated" | Another program uses that port; change the left-hand number in `docker-compose.yml`. |
| Docker commands or the app hang | Restart Docker Desktop, then `docker compose up -d`. |
| Document stays ⏳ PENDING | Check `docker compose logs worker`; try `docker compose restart worker`. |
| Document shows ❌ FAILED | The error is shown under it in the sidebar (scanned PDFs fail). |
| UI says "API unreachable" / "Could not reach the API" | The UI can't get a reply from the API. Most often the API is still starting after `docker compose up` (it takes about 25 seconds, up to a minute the first time, because it loads the reranker model): wait and click **Refresh status**. Otherwise check `docker compose ps` (is `api` Up?), `docker compose logs api` for a crash, and that Docker Desktop is running. |
| Code edits don't take effect | `docker compose restart api worker` (auto-reload is unreliable on Windows bind mounts). |
| Occasional OpenAI connection error | The app retries automatically; ask again. If it persists, check your network and API key. |

## Documentation

- [system design.md](system%20design.md): architecture, data model, flows, design decisions, test results.
- [plan.md](plan.md): the original phased build plan.
