"""Normalization: encoding → UTF-8, Unicode NFKC, LF newlines, strip controls."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class NormalizedDoc:
    src_key: str
    mime: str
    bytes_sha256: str
    size_bytes: int
    meta: Dict[str, Any]
    text_norm: str
    text_sha256: str


def normalize_text(text: str) -> str:
    # Unicode NFKC
    text = unicodedata.normalize("NFKC", text)
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip control characters except tab/newline
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text


def _sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_normalized_doc(loaded: "LoadedDoc") -> NormalizedDoc:
    from .loaders import LoadedDoc  # avoid cycle in type checkers
    norm = normalize_text(loaded.text or "")
    return NormalizedDoc(
        src_key=loaded.src_key,
        mime=loaded.mime,
        bytes_sha256=loaded.bytes_sha256,
        size_bytes=loaded.size_bytes,
        meta=loaded.meta,
        text_norm=norm,
        text_sha256=_sha256_text(norm),
    )


