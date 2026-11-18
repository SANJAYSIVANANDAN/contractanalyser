"""Manual check: prints what each retrieval stage returns.
Run inside the api container:  python -m tests.try_retrieval "your question"
(run as a module, not a bare script path, so `app.*` imports resolve)
"""
import sys

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import COMPLETED, Document
from app.services.embeddings import embed_query
from app.services.reranker import rerank
from app.services.retrieval import bm25_search, reciprocal_rank_fusion, semantic_search


def show(title, rows):
    print(f"\n=== {title}")
    if not rows:
        print("  (nothing)")
    for chunk, score in rows:
        text = chunk.content.replace("\n", " ")[:80]
        score_text = "" if score is None else f"{score:.4f}  "
        print(f"  {score_text}p{chunk.page_number} #{chunk.chunk_index:<3} [{(chunk.section_title or '-')[:28]}] {text}")


question = sys.argv[1] if len(sys.argv) > 1 else "What are the payment terms?"
print("QUESTION:", question)

db = SessionLocal()
doc_ids = list(db.scalars(select(Document.id).where(Document.status == COMPLETED)))

semantic = semantic_search(db, embed_query(question), doc_ids)
sparse = bm25_search(db, question, doc_ids)
fused = reciprocal_rank_fusion([semantic, sparse])
rrf_scores = {c.id: s for c, s in fused}

show("A. semantic (top 5 of 40)", [(c, None) for c in semantic[:5]])
show("B. BM25     (top 5 of 40)", [(c, None) for c in sparse[:5]])
show("C. RRF fused (top 5)", fused[:5])
show("D. after reranking (blended with RRF)", rerank(question, [c for c, _ in fused], top_n=5, rrf_scores=rrf_scores))
