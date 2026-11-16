"""Checks the citation validator with FAKE llm answers (no OpenAI call).
Run inside the api container:  python -m tests.test_citations
"""
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import Chunk
from app.services.rag import validate_citations_node

db = SessionLocal()
chunks = list(db.scalars(select(Chunk).order_by(Chunk.chunk_index).limit(2)))
state_base = {"top_chunks": [(chunks[0], 0.0), (chunks[1], 0.0)]}   # we "sent" passages 1 and 2


def run(raw):
    return validate_citations_node({**state_base, "raw_answer": raw})


# 1. valid citation -> citation built from DB row
r = run("The fee is X [1].")
assert len(r["citations"]) == 1 and r["citations"][0].chunk_id == chunks[0].id
assert r["citations"][0].page_number == chunks[0].page_number
assert r["citations"][0].filename == chunks[0].document.filename
assert not r["insufficient_evidence"]
print("PASS valid citation resolved from database")

# 2. invented source number [7] -> rejected, removed from text, valid one kept
r = run("A [1]. B [7].")
assert r["rejected_sources"] == [7] and "[7]" not in r["answer"] and len(r["citations"]) == 1
print("PASS invented source [7] rejected")

# 3. only invented numbers -> no valid citation -> insufficient evidence
r = run("Made-up claim [9].")
assert r["citations"] == [] and r["insufficient_evidence"] is True
print("PASS answer with only invalid citations flagged insufficient")

# 4. factual-looking answer with NO citation -> insufficient evidence
r = run("The fee is 5000 rupees.")
assert r["citations"] == [] and r["insufficient_evidence"] is True
print("PASS uncited answer flagged insufficient")

# 5. model says insufficient -> no citations attached
r = run("Insufficient evidence: not stated. [1]")
assert r["citations"] == [] and r["insufficient_evidence"] is True
print("PASS 'Insufficient evidence' answer carries no citations")

# 6. same source cited twice -> one citation object
r = run("A [1]. B [1]. C [2].")
assert [c.source_number for c in r["citations"]] == [1, 2]
print("PASS duplicate citations merged")
print("ALL PASSED")
