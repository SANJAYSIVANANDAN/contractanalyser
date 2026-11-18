"""Turns a PDF into a list of "blocks": text + the REAL page number + section title.

We use Docling's structured document (not Markdown) because only the structured
document knows which PDF page each piece of text came from (item.prov[].page_no).
"""
import html

# Labels we skip: running headers/footers/page numbers add noise, not contract content.
SKIP_LABELS = {"page_header", "page_footer"}

_converter = None  # created once per worker process, then reused (loading models is slow)


def _get_converter():
    global _converter
    if _converter is None:
        # Imported here (not at the top) so a process that doesn't need parsing
        # doesn't pay Docling's import cost.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = False  # MVP: digital PDFs only. Scanned PDFs need OCR (deferred).
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
    return _converter


def parse_pdf(file_path: str) -> list[dict]:
    """Returns [{"text", "page_number", "section_title", "is_table"}, ...] in reading order.
    Raises ValueError if no text could be extracted."""
    result = _get_converter().convert(file_path)
    document = result.document

    blocks = []
    current_section = None

    for item, _level in document.iterate_items():
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        if label in SKIP_LABELS:
            continue

        is_table = label == "table"
        # Tables: export as markdown so rows/columns stay readable
        if is_table:
            text = item.export_to_markdown(doc=document)
        else:
            text = getattr(item, "text", "") or ""
        text = html.unescape(text).strip()  # Docling's table export leaves "&amp;" in the text
        if not text:
            continue

        # Headings update the "current section" for everything after them
        if label in ("section_header", "title"):
            current_section = text[:500]

        # Page number comes from provenance. If Docling gave none, we store None (never guess).
        page_number = item.prov[0].page_no if item.prov else None

        blocks.append(
            {"text": text, "page_number": page_number, "section_title": current_section, "is_table": is_table}
        )

    if not blocks:
        raise ValueError("No text could be extracted from this PDF (is it a scanned image?)")
    return blocks
