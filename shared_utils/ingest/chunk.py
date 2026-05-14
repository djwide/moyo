"""Deterministic chunking strategies: sentences, paragraphs, fixed."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class Chunk:
    id: str
    text: str
    start_offset: int
    end_offset: int
    meta: Dict[str, Any] = field(default_factory=dict)


def _short_sha256(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def chunk_sentences(text: str, overlap: int = 1) -> List[Chunk]:
    # Simple heuristic: split on sentence enders followed by whitespace
    spans: List[tuple[int, int]] = []
    start = 0
    pattern = re.compile(r"(?<=[.!?])\s+")
    parts = pattern.split(text)
    offsets: List[int] = []
    idx = 0
    for part in parts:
        end = start + len(part)
        spans.append((start, end))
        offsets.append(start)
        start = end + 1  # account for removed space in split
    # Build chunks with overlap
    chunks: List[Chunk] = []
    for i in range(len(spans)):
        s = max(0, i - overlap)
        s_off = spans[s][0]
        e_off = spans[i][1]
        text_slice = text[s_off:e_off]
        cid = f"{_short_sha256(text_slice)}-sentences-{i:04d}"
        chunks.append(Chunk(cid, text_slice, s_off, e_off, {"strategy": "sentences", "idx": i}))
    return chunks


def chunk_paragraphs(text: str, overlap: int = 0) -> List[Chunk]:
    parts = text.split("\n\n")
    offsets: List[int] = []
    cursor = 0
    for part in parts:
        offsets.append(cursor)
        cursor += len(part) + 2
    chunks: List[Chunk] = []
    for i in range(len(parts)):
        s = max(0, i - overlap)
        s_off = offsets[s]
        e_off = offsets[i] + len(parts[i])
        text_slice = text[s_off:e_off]
        cid = f"{_short_sha256(text_slice)}-paragraphs-{i:04d}"
        chunks.append(Chunk(cid, text_slice, s_off, e_off, {"strategy": "paragraphs", "idx": i}))
    return chunks


def chunk_fixed(text: str, size: int = 1000, overlap: int = 100) -> List[Chunk]:
    chunks: List[Chunk] = []
    i = 0
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        text_slice = text[start:end]
        cid = f"{_short_sha256(text_slice)}-fixed-{i:04d}"
        chunks.append(Chunk(cid, text_slice, start, end, {"strategy": "fixed", "idx": i}))
        if end == len(text):
            break
        start = end - overlap if overlap < size else end
        i += 1
    return chunks


def chunk_text(text: str, *, strategy: str = "sentences", overlap: int = 1, size: int = 1000) -> List[Chunk]:
    if strategy == "sentences":
        return chunk_sentences(text, overlap)
    if strategy == "paragraphs":
        return chunk_paragraphs(text, overlap)
    if strategy == "fixed":
        return chunk_fixed(text, size=size, overlap=overlap)
    raise ValueError(f"Unknown chunk strategy: {strategy}")


