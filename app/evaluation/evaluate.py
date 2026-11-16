"""Offline RAGAS evaluation of four retrieval configurations.

Run (costs a small amount of OpenAI usage; do NOT run per user question):
    docker compose --profile eval run --rm eval

What it does, for every question in dataset.py:
    1. Retrieve 5 passages with each configuration:
         A. vector search only
         B. keyword search only
         C. hybrid search with RRF
         D. hybrid search + local CrossEncoder reranking
    2. Generate an answer from those passages with the SAME prompt/model the app uses.
    3. Score question / passages / answer / reference with RAGAS.
Results go to app/evaluation/results/ as JSON (summary + details) and CSV (one row per question).

Evaluation note: the production score floor (MIN_RERANK_SCORE) is NOT applied here, so that
configuration D is judged on ranking quality alone.
"""
import csv
import json
import math
import os
import re
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.db.database import SessionLocal
from app.db.models import COMPLETED, Document
from app.evaluation.dataset import EVAL_SET
from app.services.embeddings import embed_query
from app.services.rag import NO_EVIDENCE_ANSWER, generate_node
from app.services.reranker import rerank
from app.services.retrieval import keyword_search, reciprocal_rank_fusion, semantic_search

TOP_N = 5
CONFIG_NAMES = ["A_vector_only", "B_keyword_only", "C_hybrid_rrf", "D_hybrid_rerank"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
NO_PASSAGES = "(no passages were retrieved)"


# ------------------------------------------------------------------ step 1: retrieval
def retrieve_for_all_configs(db, question, document_ids):
    """Returns {config_name: [Chunk, ...]} with at most TOP_N chunks each."""
    semantic = semantic_search(db, embed_query(question), document_ids)   # 15 candidates
    keyword = keyword_search(db, question, document_ids)                  # 15 candidates
    fused = reciprocal_rank_fusion([semantic, keyword], top_n=15)
    fused_chunks = [chunk for chunk, _score in fused]
    return {
        "A_vector_only": semantic[:TOP_N],
        "B_keyword_only": keyword[:TOP_N],
        "C_hybrid_rrf": fused_chunks[:TOP_N],
        "D_hybrid_rerank": [chunk for chunk, _s in rerank(question, fused_chunks, top_n=TOP_N)],
    }


# ------------------------------------------------------------------ step 2: generation
def generate_answer(question, chunks):
    """Same generate step as the app. No passages -> the app would say 'insufficient evidence'."""
    if not chunks:
        return NO_EVIDENCE_ANSWER
    state = {"question": question, "top_chunks": [(chunk, 0.0) for chunk in chunks]}
    raw = generate_node(state, config={})["raw_answer"]
    return re.sub(r"\s*\[\d+\]", "", raw).strip()   # drop [1] markers; RAGAS judges plain text


# ------------------------------------------------------------------ our own extra metric
def page_hit(chunks, gold_pages):
    """1.0 if any retrieved chunk comes from a page that holds the answer, else 0.0."""
    return 1.0 if any(chunk.page_number in gold_pages for chunk in chunks) else 0.0


# ------------------------------------------------------------------ step 3: RAGAS
def run_ragas(samples):
    """samples: list of dicts with user_input, retrieved_contexts, response, reference.
    Returns a list of dicts (one per sample) with the four metric scores."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key))
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    )
    metrics = [
        Faithfulness(),                       # are the answer's claims supported by the passages?
        ResponseRelevancy(),                  # does the answer address the question?
        LLMContextPrecisionWithReference(),   # are relevant passages ranked near the top?
        LLMContextRecall(),                   # did the passages contain what the reference says?
    ]
    # RAGAS defaults to 16 parallel requests and a 180 s timeout. On our connection that caused
    # many TimeoutError/APIConnectionError jobs, i.e. missing (NaN) scores. Fewer workers = reliable.
    run_config = RunConfig(max_workers=4, timeout=300, max_retries=5)
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        show_progress=False,
    )
    return result.to_pandas().to_dict(orient="records")


METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_with_reference",
    "context_recall": "context_recall",
}


def mean_ignoring_nan(values):
    good = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return (round(sum(good) / len(good), 3) if good else None), len(values) - len(good)


# ------------------------------------------------------------------ main
def main():
    db = SessionLocal()
    document_ids = list(db.scalars(select(Document.id).where(Document.status == COMPLETED)))
    if not document_ids:
        raise SystemExit("No COMPLETED documents to evaluate against.")

    rows = []   # one entry per (config, question)
    for number, item in enumerate(EVAL_SET, start=1):
        print(f"[{number}/{len(EVAL_SET)}] {item['question']}", flush=True)
        retrieved = retrieve_for_all_configs(db, item["question"], document_ids)
        for config in CONFIG_NAMES:
            chunks = retrieved[config]
            rows.append(
                {
                    "config": config,
                    "question": item["question"],
                    "reference": item["reference"],
                    "response": generate_answer(item["question"], chunks),
                    "contexts": [c.content for c in chunks] or [NO_PASSAGES],
                    "pages_retrieved": [c.page_number for c in chunks],
                    "gold_pages": item["gold_pages"],
                    "page_hit": page_hit(chunks, item["gold_pages"]),
                }
            )

    print(f"Scoring {len(rows)} answers with RAGAS (this takes a few minutes)...", flush=True)
    scores = run_ragas(
        [
            {
                "user_input": r["question"],
                "retrieved_contexts": r["contexts"],
                "response": r["response"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )
    for row, score in zip(rows, scores):
        for name, column in METRIC_COLUMNS.items():
            row[name] = score.get(column)

    # ---- summary per configuration
    summary = {}
    for config in CONFIG_NAMES:
        config_rows = [r for r in rows if r["config"] == config]
        entry = {"questions": len(config_rows)}
        for name in list(METRIC_COLUMNS) + ["page_hit"]:
            mean, missing = mean_ignoring_nan([r[name] for r in config_rows])
            entry[name] = mean
            if missing:
                entry[f"{name}_missing"] = missing   # RAGAS returned NaN for these samples
        summary[config] = entry

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"ragas_{stamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"ragas_{stamp}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False, default=str)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        columns = ["config", "question", "faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "page_hit", "pages_retrieved", "gold_pages", "response"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ---- print table
    header = ["config", "faithful", "ans_relev", "ctx_prec", "ctx_recall", "page_hit"]
    print("\n" + "".join(f"{h:<18}" for h in header))
    for config, e in summary.items():
        values = [e["faithfulness"], e["answer_relevancy"], e["context_precision"], e["context_recall"], e["page_hit"]]
        print(f"{config:<18}" + "".join(f"{('n/a' if v is None else v):<18}" for v in values))
    print(f"\nSaved: {json_path}\n       {csv_path}")


if __name__ == "__main__":
    main()
