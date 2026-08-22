"""Load a local document, chunk it, and keep valuable phrases."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from moyo.privateside.phrases.filter import extract_sensitive_phrases, normalize_phrase
from moyo.privateside.phrases.schema import PhraseRecord, phrase_id
from moyo.privateside.phrases.store import PhraseStore


def load_document_text(path: Path | str) -> str:
    """Extract text from a local file (txt/md/pdf/docx/pptx/xlsx/html/csv)."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Document not found: {src}")
    data = src.read_bytes()
    mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
    try:
        from shared_utils.ingest.loaders import load_text_from_bytes
        from shared_utils.ingest.validators import ValidationConfig

        doc = load_text_from_bytes(str(src), data, mime, ValidationConfig())
        text = (doc.text or "").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        return src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return src.read_text(encoding="latin-1")


def chunk_for_phrases(text: str) -> list[str]:
    """Sentence and list-item chunks; skip bulky section copies."""
    from shared_utils.chunking import chunk_text, chunk_text_multi_granularity

    granular = chunk_text_multi_granularity(
        text,
        chunk_size=360,
        overlap=0,
        max_tokens=120,
        include_sentences=True,
        include_items=True,
        min_sentence_chars=12,
        min_item_chars=3,
    )
    pieces = [
        normalize_phrase(c.text)
        for c in granular
        if c.level in {"sentence", "item"} and normalize_phrase(c.text)
    ]
    if pieces:
        return _dedupe(pieces)
    # Short notes with no sentence punctuation still need a pass.
    fallback = [normalize_phrase(c) for c in chunk_text(text, chunk_size=220, overlap=0)]
    return _dedupe([p for p in fallback if p])


def ingest_text(
    text: str,
    store: PhraseStore,
    *,
    source_path: str | None = None,
    direction: str | None = None,
    complete=None,
    progress=None,
) -> dict:
    """Ask Kimi for sensitive phrases and queue them for review."""
    extracted = extract_sensitive_phrases(
        text, direction=direction, complete=complete, progress=progress
    )
    kept = [
        PhraseRecord(
            id=phrase_id(row["text"]),
            text=row["text"],
            label=row["label"],
            status="pending",
            source="document" if source_path else "manual",
            source_path=source_path,
            reason=row.get("reason") or "kimi",
            score=float(row.get("score") or 1.0),
        )
        for row in extracted
    ]
    added = store.enqueue(kept)
    return {
        "path": source_path or "",
        "chunks": 0,
        "candidates": len(kept),
        "queued": len(added),
        "dropped": 0,
        "duplicates": len(kept) - len(added),
        "pending": added,
    }


def ingest_document(
    path: Path | str,
    store: PhraseStore,
    *,
    direction: str | None = None,
    complete=None,
    progress=None,
) -> dict:
    """Load a document, extract with Kimi, queue phrases for review."""
    src = Path(path)
    result = ingest_text(
        load_document_text(src),
        store,
        source_path=str(src),
        direction=direction,
        complete=complete,
        progress=progress,
    )
    result["path"] = str(src)
    return result


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
