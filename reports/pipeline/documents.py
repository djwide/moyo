"""Turn an uploaded note, PDF, or Word file into plain text."""

from __future__ import annotations

from io import BytesIO

MAX_DOCUMENT_BYTES = 4_000_000
MAX_NOTE_CHARS = 400_000


def document_to_text(raw: bytes, suffix: str) -> str:
    """Recover UTF-8 text. Empty and image-only files raise ValueError."""
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document is empty or exceeds 4 MB.")
    ext = suffix.lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    if ext in {".md", ".txt"}:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("That note is not valid UTF-8.") from exc
    elif ext == ".pdf":
        text = _pdf_text(raw)
    elif ext == ".docx":
        text = _docx_text(raw)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("No text could be recovered from that document.")
    if len(text) > MAX_NOTE_CHARS:
        raise ValueError("Recovered text exceeds the note limit.")
    return text


def _pdf_text(raw: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:
        raise ValueError("PDF text extraction is unavailable.") from exc
    try:
        return extract_text(BytesIO(raw)) or ""
    except Exception as exc:
        raise ValueError("No text could be recovered from that document.") from exc


def _docx_text(raw: bytes) -> str:
    try:
        import docx
    except ImportError as exc:
        raise ValueError("Word text extraction is unavailable.") from exc
    try:
        document = docx.Document(BytesIO(raw))
    except Exception as exc:
        raise ValueError("No text could be recovered from that document.") from exc
    parts: list[str] = []
    for paragraph in document.paragraphs:
        line = paragraph.text.strip()
        if line:
            parts.append(line)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)
