"""Lightweight query expansion for contract jargon/abbreviations.

Dense and sparse retrieval both suffer when the question uses an abbreviation
("O&M", "P1") but the contract spells it out ("operations and maintenance",
"Priority 1"), or vice versa. Rather than an extra LLM call per question (cost +
latency for a small corpus), this expands known abbreviations with a static map
built from the same vocabulary contract_analyzer.py already extracts clauses for.
Retrieval-only: the ORIGINAL question is still what's shown to the user and sent
to the generation prompt (see rag.py), so citations/answers read naturally."""
import re

# key: abbreviation/synonym (matched case-insensitively, whole word) -> expansion(s) to append
SYNONYMS: dict[str, str] = {
    "o&m": "operations and maintenance",
    "sla": "service level agreement",
    "ip": "intellectual property",
    "nda": "non-disclosure confidentiality agreement",
    "poc": "point of contact proof of concept",
    "p1": "priority 1 severity 1",
    "p2": "priority 2 severity 2",
    "opex": "operating expense operational cost",
    "capex": "capital expenditure",
    "kpi": "key performance indicator",
    "rbac": "role-based access control",
    "sso": "single sign-on",
    "api": "application programming interface",
    "tos": "terms of service",
    "ind": "indian rupees",
    "inr": "indian rupees",
}

_WORD = re.compile(r"[a-zA-Z&]+")


def rewrite_query(question: str) -> str:
    """Appends known expansions for any abbreviation found in the question.
    Returns the question unchanged if nothing matched (the common case)."""
    words = {w.lower() for w in _WORD.findall(question)}
    extra = [SYNONYMS[w] for w in words if w in SYNONYMS]
    if not extra:
        return question
    return f"{question} ({'; '.join(extra)})"
