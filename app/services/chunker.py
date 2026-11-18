"""Structure-aware chunking.

Rule that protects citations: a chunk NEVER spans two pages or two sections,
so each chunk has exactly one page_number and one section_title.

Beyond that rule, the previous version of this chunker just cut every ~1000 chars
regardless of where a sentence or clause ended, with zero overlap between chunks —
a known cause of lost context when the answer straddles a cut. This version:
  - keeps a table block atomic: a table is always its own chunk, never split or
    merged with surrounding prose, since it carries dense numeric data.
  - prefers to break at a numbered-clause boundary ("1.2", "(a)", "Article 3", ...)
    once a chunk is already big enough, instead of always bulldozing through.
  - never cuts mid-sentence: long lines are split on sentence boundaries.
  - carries a small overlap of the previous chunk's tail into the next chunk
    (same page/section only), so an answer spanning a cut is still retrievable
    from either chunk.
"""
import re

MIN_CHARS = 300   # don't flush a chunk before it reaches roughly this size...
MAX_CHARS = 800   # ...unless it would exceed this size
OVERLAP_CHARS = 120  # ~15-20% of a MIN_CHARS chunk, carried into the next one

CLAUSE_START = re.compile(
    r"^\s*(\d+(\.\d+)*[.)]\s|\(?[a-z]\)\s|\(?[ivxIVX]+\)\s|Article\s+\d+\b|Section\s+\d+\b)",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT.split(text) if s]


def _split_long_piece(piece: str, max_chars: int) -> list[str]:
    """A single sentence/line longer than max_chars: break on whitespace, never mid-word."""
    pieces = []
    while len(piece) > max_chars:
        cut = piece.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        pieces.append(piece)
    return pieces


def _pieces_for_block(block: dict) -> list[str]:
    """Breaks one parser block into sentence-safe pieces, splitting on newlines first
    (Docling sometimes packs several list items/lines into one block)."""
    pieces = []
    for line in block["text"].split("\n"):
        line = line.strip()
        if not line:
            continue
        for sentence in _split_sentences(line):
            pieces.extend(_split_long_piece(sentence, MAX_CHARS))
    return pieces or [block["text"]]


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    """Returns [{"chunk_index", "content", "page_number", "section_title", "is_table"}, ...]"""
    chunks: list[dict] = []
    current: dict | None = None
    last_tail: str | None = None          # overlap carried from the previous non-table chunk
    last_tail_key: tuple | None = None    # (page_number, section_title) that tail belongs to

    def flush():
        nonlocal current, last_tail, last_tail_key
        if current and current["content"].strip():
            current["chunk_index"] = len(chunks)
            chunks.append(current)
            if not current["is_table"]:
                tail = current["content"][-OVERLAP_CHARS:]
                cut = tail.find(" ")  # don't start the overlap mid-word
                last_tail = tail[cut:].strip() if cut > 0 else tail
                last_tail_key = (current["page_number"], current["section_title"])
            else:
                last_tail, last_tail_key = None, None
        current = None

    for block in blocks:
        key = (block["page_number"], block["section_title"])

        if block["is_table"]:
            flush()
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "content": block["text"],
                    "page_number": block["page_number"],
                    "section_title": block["section_title"],
                    "is_table": True,
                }
            )
            last_tail, last_tail_key = None, None
            continue

        for piece_index, piece in enumerate(_pieces_for_block(block)):
            is_clause_start = bool(CLAUSE_START.match(piece))
            same_place = current is not None and (current["page_number"], current["section_title"]) == key
            fits = current is not None and len(current["content"]) + len(piece) + 1 <= MAX_CHARS
            big_enough_to_break = current is not None and len(current["content"]) >= MIN_CHARS

            # Start a new chunk if: different page/section, chunk is full, or (chunk is
            # already big enough AND this piece starts a new numbered clause).
            if same_place and fits and not (is_clause_start and big_enough_to_break and piece_index > 0):
                current["content"] += ("\n" if current["content"] else "") + piece
            else:
                flush()
                seed = ""
                if last_tail_key == key and last_tail:
                    seed = last_tail + "\n"
                current = {
                    "content": seed + piece,
                    "page_number": block["page_number"],
                    "section_title": block["section_title"],
                    "is_table": False,
                }
    flush()
    return chunks
