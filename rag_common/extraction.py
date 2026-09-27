"""Text extraction for the Bronze layer, with a fallback for PDFs whose
embedded font lacks a proper ToUnicode CMap. When that happens, pypdf emits
literal glyph-name placeholders (`/uni000001fb`, ...) instead of characters --
seen in practice on a LaTeX-generated arXiv PDF where certain figure/table
pages used such a font. Falls back to pdfplumber, which uses a different
text-extraction path and often recovers cleanly from exactly this failure.

pypdf/pdfplumber are imported lazily (inside the functions that need them),
matching rag_common's "no hard PDF/Databricks dependency to just import this
module" pattern (see embeddings.py's lazy sentence-transformers import).
"""
from __future__ import annotations

from io import BytesIO

_GARBLED_MARKER = "/uni"
_GARBLED_MARKER_LEN = len("/uni00000000")
_GARBLED_THRESHOLD = 0.02  # fraction of chars belonging to a /uniXXXXXXXX marker


def _looks_garbled(text: str) -> bool:
    """True if `text` is dominated by pypdf's /uniXXXXXXXX glyph fallback,
    or is empty (extraction failed outright)."""
    if not text:
        return True
    marker_chars = text.count(_GARBLED_MARKER) * _GARBLED_MARKER_LEN
    return marker_chars / len(text) > _GARBLED_THRESHOLD


def _extract_with_pypdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_with_pdfplumber(content: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(BytesIO(content)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def extract_pdf_text(content: bytes) -> str:
    """Extract text from a PDF's raw bytes, falling back to pdfplumber if
    pypdf's output looks like it hit the font-encoding failure above."""
    try:
        text = _extract_with_pypdf(content)
    except Exception:
        text = ""

    if not _looks_garbled(text):
        return text

    try:
        fallback_text = _extract_with_pdfplumber(content)
    except Exception:
        fallback_text = ""

    if fallback_text and not _looks_garbled(fallback_text):
        return fallback_text
    # Neither extractor produced clean text -- keep whichever is longer.
    return fallback_text if len(fallback_text) > len(text) else text


def extract_text(content: bytes, path: str) -> str:
    if path.lower().endswith(".pdf"):
        return extract_pdf_text(content)
    return content.decode("utf-8", errors="ignore")
