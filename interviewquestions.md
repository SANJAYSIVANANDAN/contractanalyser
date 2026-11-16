# DocuPilot AI — Interview Questions

Each question has a short model answer. Try answering out loud first. Answers refer to this project's actual code ([system design.md](system%20design.md) has the full design). Questions marked **(not built yet)** concern Phase 6 features.

---

## A. Backend: FastAPI, SQLAlchemy, PostgreSQL

**1. Walk me through the lifecycle of `GET /health` in this app.**
uvicorn receives the request → FastAPI matches the route → sees `Depends(get_db)` and calls `get_db`, which opens a `SessionLocal()` and `yield`s it → the endpoint runs with that session → the response is built (validated against the response model) → the `finally` in `get_db` closes the session.

**2. What is dependency injection in FastAPI, and why use it for the DB session?**
Functions declare what they need (`db: Session = Depends(get_db)`) and FastAPI supplies it per request. Benefits: every endpoint gets its own session, cleanup is guaranteed by the `finally` after `yield`, and tests can swap the dependency.

**3. Why does `get_db` use `yield` instead of `return`?**
Code before `yield` is setup, code after is teardown. With `return` there would be no moment to close the session after the endpoint finishes.

**4. Engine vs Session in SQLAlchemy?**
The Engine owns the connection pool (one per process). A Session is a unit of work: it tracks objects, holds a transaction, and borrows a connection from the pool when needed.

**5. What does `db.commit()` do, and what happens if you forget it?**
It ends the transaction and makes writes durable and visible to other connections. Without it, the changes are rolled back when the session closes.

**6. Why does the upload endpoint commit the row *before* calling `.delay()`?**
If the task were published first, the worker might run before the row is visible and find nothing. Commit first, then publish.

**7. Why did the streaming endpoint not use `Depends(get_db)`?**
In FastAPI 0.115 the dependency's teardown runs before a streaming body finishes, so the session would be closed mid-stream. The generator creates its own `SessionLocal()` and closes it in `finally`.

**8. What does `cascade="all, delete-orphan"` plus `ondelete="CASCADE"` give you?**
Deleting a document removes its chunks and clauses — in the ORM (cascade) and in the database itself (FK `ON DELETE CASCADE`).

**9. Which indexes exist on `chunks` and why?**
HNSW on `embedding` (fast approximate cosine nearest-neighbour), GIN on `search_vector` (fast full-text matching), B-tree on `document_id` (filtering by document).

**10. Why Alembic instead of `Base.metadata.create_all()`?**
`create_all` only creates missing tables; it can't alter existing ones or record history. Alembic keeps versioned, reviewable migrations (including the hand-added `CREATE EXTENSION vector`, which autogenerate can't detect).

**11. Why is `chunks.page_number` nullable?**
If Docling can't establish a page, storing `NULL` is honest. A guessed page number would produce a false citation.

---

## B. Distributed processing: RabbitMQ, Celery, Redis

**12. Why does PDF processing run outside the API process?**
Docling takes ~40–100 s per document. Doing it in the request would block the connection, tie up API capacity, and lose the work if the client disconnects. The API returns an ID immediately; a worker does the work.

**13. What goes in the queue message, and why only that?**
Just the document ID. Messages stay tiny, the PDF has one source of truth (shared volume), and the DB row is the authority on state.

**14. What is `task_acks_late=True`?**
The message is acknowledged after the task finishes, not when it starts. If the worker dies mid-task, RabbitMQ redelivers it. The trade-off is at-least-once delivery, so tasks must be idempotent.

**15. How is `process_document` idempotent?**
Three layers: (1) Redis lock `SET NX EX` so only one worker processes a document at a time; (2) skip if `status == COMPLETED`; (3) one transaction that deletes old chunks, inserts new ones, and sets `COMPLETED` — so a retry never leaves duplicates.

**16. What does `SET key value NX EX 1800` do?**
Atomically sets the key only if it doesn't exist (`NX`) and expires it after 1800 s (`EX`). Exactly one caller succeeds; the TTL stops a crashed worker from holding the lock forever.

**17. Which errors are retried and which aren't?**
Temporary OpenAI errors (connection, timeout, rate limit, 5xx) are retried with backoff 20/40/80 s, up to 3 times, then marked `FAILED`. Deterministic errors (corrupt PDF, no text) fail immediately — retrying can't fix them.

**18. Why is the slow work done *before* opening the write transaction?**
So the database isn't holding a transaction (and locks/connections) open for a minute of Docling + network time.

**19. How would you scale ingestion?**
Run more worker replicas on the same queue. The lock + DB check prevent duplicate processing. For one huge PDF, split into page-range sub-tasks.

**20. Why Redis and not Postgres for the lock?**
Redis gives an atomic, expiring key with one command and very low latency. (A Postgres advisory lock would also work; Redis is already in the stack for the cache.)

**21. Why is the worker image separate and bigger?**
Docling pulls in PyTorch and model files. The API image doesn't need Docling, so keeping them separate keeps the API image smaller and faster to build. Docling is imported lazily so the API can still import the shared modules.

**22. Where do Docling's models come from and how do we avoid re-downloading?**
They download from Hugging Face on first use into `/root/.cache`; a named Docker volume (`model_cache`) persists them, and the converter object is created once per worker process.

---

## C. AI / RAG

**23. What does Docling give you that plain text extraction doesn't?**
Structure: headings, tables, reading order, and per-item provenance with page numbers. Page provenance is what makes correct citations possible.

**24. Why not use Markdown line numbers as page numbers?**
Markdown has no page concept; line numbers say nothing about the PDF page. We read `item.prov[0].page_no` from the structured document instead.

**25. How does the chunker protect citation accuracy?**
A chunk never spans a page or section boundary, so each chunk has exactly one page number and one section title. Cost: many small chunks.

**26. What is an embedding, and what is cosine similarity?**
An embedding is a vector (1536 numbers here) representing meaning. Cosine similarity compares the *angle* between two vectors, ignoring length; pgvector's cosine *distance* is `1 − similarity`, so smaller means more similar.

**27. Why must queries and chunks use the same embedding model?**
Different models produce incompatible vector spaces; distances between them are meaningless.

**28. Why hybrid search instead of only vectors?**
Vectors capture meaning but can miss exact terms (clause numbers, names, "Section 12"); keyword search catches exact terms but misses paraphrases. Together they cover both.

**29. Why join the question words with `or` in keyword search?**
`websearch_to_tsquery` ANDs words by default, so a full sentence rarely matches any single chunk (our termination question returned nothing). With `or`, chunks matching some words qualify and `ts_rank` puts the best first. Reducing words to `\w+` also stops users injecting query operators.

**30. Explain Reciprocal Rank Fusion with the formula.**
`RRF(d) = Σ 1/(k + rank(d))` over each ranking, `k = 60`. A document found by both searches gets two terms and rises. In our run the top chunk scored `1/61 + 1/61 = 0.0328`.

**31. Why not add the cosine score and the `ts_rank` score directly?**
They're on different scales and distributions, so the sum would be dominated by one. RRF uses only ranks, which are comparable.

**32. What does `k = 60` do?**
It dampens the gap between rank 1 and rank 2 so no single list dominates; 60 is the commonly used default.

**33. Bi-encoder vs cross-encoder — why both?**
Embeddings (bi-encoder) encode query and passage separately: fast, precomputable, rough. A cross-encoder reads them together: much more precise but must run per pair, so we only run it on ~15 candidates.

**34. Why can't "no results" signal "no evidence"?**
Vector search always returns its nearest neighbours, even for "capital of France". So we added a rerank-score floor (−9.0): answerable questions scored ≥ −7.34, unanswerable ≤ −10.21 in our test.

**35. What are the weaknesses of that floor?**
It was calibrated on 15 questions and one document. Scores are raw logits and can shift with document type, so it needs recalibration — ideally against an evaluation set.

**36. How do you prevent fabricated citations?**
The LLM only writes `[n]` markers. The backend rejects numbers it didn't send, then builds every citation field (filename, page, section, excerpt, chunk ID) from database rows. The model never supplies a page number or excerpt.

**37. What happens if the model cites `[7]` when we sent five passages?**
It's rejected: removed from the answer text and reported in `rejected_sources`. If nothing valid remains, the response is flagged `insufficient_evidence`. (Covered by `tests/test_citations.py`.)

**38. How do you handle prompt injection from contract text?**
The system prompt says document text is untrusted data; passages are fenced in `<passage>` tags; and the structural defence is that the model can't change citations. It can still be *influenced* in its wording — the design limits blast radius, it doesn't eliminate the risk.

**39. Why `temperature=0`?**
For legal-style answers we want the most deterministic, least creative output.

**40. Why show "Interpretation:" separately from quotes?**
Users must be able to tell direct contract wording from the model's reading of it — and it's not legal advice.

**41. Why numbered sources `[1]..[5]` instead of UUIDs in the prompt?**
Short numbers are easier for the model to copy correctly than 36-character UUIDs. The mapping back to real chunk UUIDs is done by the backend. The security property (model can only reference what we sent) is unchanged.

**42. What are RAGAS's four metrics?**
*Faithfulness*: are the answer's claims supported by the retrieved context (hallucination check). *Answer relevancy*: does the answer address the question. *Context precision*: are the relevant chunks ranked near the top of what was retrieved. *Context recall*: did retrieval fetch everything needed to answer. Low faithfulness → generation problem; low recall/precision → retrieval problem.

**43. How would you prove the reranker helps?**
Run the same evaluation set under four configs — vector only, keyword only, hybrid, hybrid + rerank — and compare context precision/recall and faithfulness. Don't claim improvement without those numbers. We did this: on 12 questions the reranked config had the *lowest* context precision (0.769 vs 0.881 for plain hybrid) and no recall gain, so the data does **not** support "the reranker improves quality" for this contract.

**43b. What did your RAGAS evaluation show overall?**
Hybrid (RRF) was the most balanced: best precision (0.881), top faithfulness (0.972, tied with vector), recall 0.958. Keyword had the best recall (0.979) but the worst relevancy. Vector alone had the lowest recall (0.896) — e.g. it missed the passage about Sell.Do's missing APIs. Reranking did not help. All gaps are small relative to a 12-question sample.

**43c. Why can't you trust the differences much?**
Small sample (one question flipping 0→1 moves a mean by 0.083), one run (no variance estimate), same model family as judge and generator, and an answer-relevancy metric that scored correct-but-hedged answers 0.00 and could only generate 1 of its usual 3 questions.

**43d. How did you handle RAGAS returning NaN scores?**
The first run used RAGAS's default 16 parallel workers and many jobs timed out (6–9 of 12 context-precision scores missing per config), so averages were meaningless. I lowered concurrency to 4 workers with a 300 s timeout and retries; the rerun had zero missing scores. The script also reports a `_missing` count so NaNs can't be silently averaged away.

**43e. Why evaluate offline and not on every user question?**
Metrics need a reference answer and several LLM-judge calls per sample (it took ~20 minutes and real API cost for 48 answers). It's a regression test for changes to chunking, retrieval or prompts, not a runtime check.

---

## D. LangChain and LangGraph

**44. What does LangGraph add here, and what doesn't it do?**
It expresses the pipeline as a graph of named nodes with shared state and a conditional edge (rerank → generate *or* insufficient). It doesn't do retrieval for us — that's plain SQL/Python — and it doesn't guarantee citation correctness; our validation node does.

**45. What is the "state" in the graph?**
A `TypedDict` (`RagState`) that each node reads and partially updates: question, candidates, top chunks, raw answer, final answer, citations, rejected sources, insufficient flag.

**46. How does token streaming work through the graph?**
`rag_graph.stream(..., stream_mode=["messages", "values"])` yields LLM tokens (filtered to the `generate` node) and the full state after each node. We forward tokens as SSE `token` events and use the last state for the `citations` event.

**47. How do the nodes get the DB session?**
Through `config={"configurable": {"db": db}}`, read in nodes via `config["configurable"]["db"]`.

**48. What does `prompt | llm` mean?**
LangChain's pipe operator composes runnables: the output of the prompt template (formatted messages) feeds the chat model.

**49. Why keep embeddings on the OpenAI SDK rather than LangChain?**
The worker uses them for ingestion and gains nothing from a wrapper; fewer dependencies in the heavy path.

---

## E. System design

**50. Why SSE instead of WebSockets?**
Answer streaming is one-way (server → client). SSE is plain HTTP, auto-reconnecting in browsers, simple to implement in FastAPI. WebSockets add a bidirectional protocol we don't need.

**51. What do you send, in what order, over SSE?**
`token` events as text arrives → one `citations` event after backend validation (with `insufficient_evidence`) → `done` with the saved message ID. `error` on failure.

**52. What if the client disconnects mid-stream?**
The generator receives `GeneratorExit`, stops, and nothing half-finished is stored.

**53. Why is Redis in this stack, and how does the cache work?**
Two jobs: the processing lock (`SET NX EX`) and a cache-aside store for retrieval results with a 600 s TTL. The key is a hash of the normalized question + sorted document IDs, prefixed with a retrieval version. On a hit we get cached chunk IDs and reload the chunks from Postgres (skipping the query embedding and both searches); on a miss we compute and store. Entries are indexed per document so deleting or reprocessing a document removes them. It holds IDs and scores only — nothing permanent. Measured: 0.605 s cold vs 0.019 s cached.

**53b. Why cache chunk IDs instead of the chunk text?**
Contract text stays in Postgres as the single source of truth, the cache stays small, and a missing chunk (deleted or reprocessed) is detected on reload and treated as a miss, so stale data can't be cited.

**53c. What happens to the cache if Redis is down?**
Reads and writes catch Redis errors, log a warning, and behave like a miss, so chat keeps working (tested; it was slower, ~11 s, while connections failed). The ingestion worker is *not* protected the same way yet.

**53d. Why is the overall request barely faster with a cache hit?**
Retrieval was only ~0.6 s of a ~4 s request; the LLM call dominates and isn't cached. The cache mainly saves an OpenAI embedding call and database work for repeated questions.

**54. Why RabbitMQ and not just Redis as the broker?**
RabbitMQ has durable queues and explicit acknowledgements built for reliable delivery; Redis would work but is better suited as a cache here, and keeping roles separate makes failure modes clearer.

**55. Why PostgreSQL + pgvector instead of a dedicated vector DB?**
One system of record: transactional writes of chunk + vector + metadata, SQL filters and joins, full-text search in the same database. At this scale it's simpler and cheaper; a dedicated store becomes attractive at much larger vector counts.

**56. How do you handle large PDFs?**
Raise worker resources, split into page-range subtasks, add task time limits, stream uploads, and raise the size cap deliberately. Today: 20 MB cap, one task per document.

**57. What fails if RabbitMQ is down? If Redis is down? If OpenAI is down?**
RabbitMQ down: uploads can save the file/row but publishing fails (the row stays `PENDING`; it needs a re-publish mechanism — a real gap). Redis down: the worker's lock call raises *before* the task's `try` block, so the task errors without retrying and the document stays `PENDING` (another gap — the lock acquisition should be handled; later the cache should degrade gracefully). OpenAI down: ingestion retries then `FAILED`; chat returns 502 / `event: error`.

**58. What are this system's main limitations?**
Single-turn answers; floor calibrated on one document; compare-two-contracts isn't balanced per document; no OCR; no auth or per-user isolation; excerpts are chunk starts, not exact sentences; the retry path for OpenAI errors hasn't been exercised under a real outage.

---

## F. 60-second pitch (practise this)

"DocuPilot is a contract Q&A system where every answer is cited and the citations can't be faked. Uploads go to FastAPI, which saves the PDF, writes a row, and publishes only the document ID to RabbitMQ. A Celery worker converts the PDF with Docling, keeping real page numbers, chunks it without crossing pages or sections, embeds with OpenAI, and stores everything in PostgreSQL with pgvector in one transaction, guarded by a Redis lock so retries don't duplicate data. At question time a LangGraph pipeline runs hybrid retrieval — vectors plus full-text search merged with Reciprocal Rank Fusion — reranks locally with a cross-encoder, and uses a score floor to decide whether there's any evidence. GPT-4o-mini answers from numbered passages, and the backend validates each citation and builds it from database rows, so the model can't invent a page or filename. Answers stream over SSE, with citations arriving only after validation."

---

## Practice mode

Ask me to quiz you: I'll ask these one at a time (in any order or by section) and evaluate your answers.
