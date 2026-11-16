"""Groups parser blocks into chunks of roughly MAX_CHARS characters.

Rule that protects citations: a chunk NEVER spans two pages or two sections,
so each chunk has exactly one page_number and one section_title.
"""

MAX_CHARS = 1000


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split one very long block into pieces, breaking on whitespace."""
    pieces = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    """Returns [{"chunk_index", "content", "page_number", "section_title"}, ...]"""
    chunks = []
    current = None  # the chunk we are currently filling

    def flush():
        if current and current["content"].strip():
            current["chunk_index"] = len(chunks)
            chunks.append(current)

    for block in blocks:
        for piece in _split_long_text(block["text"], MAX_CHARS):
            same_place = (
                current is not None
                and current["page_number"] == block["page_number"]
                and current["section_title"] == block["section_title"]
            )
            fits = current is not None and len(current["content"]) + len(piece) + 1 <= MAX_CHARS

            if same_place and fits:
                current["content"] += "\n" + piece
            else:
                flush()
                current = {
                    "content": piece,
                    "page_number": block["page_number"],
                    "section_title": block["section_title"],
                }
    flush()
    return chunks
