"""Shared utilities for sente and moyo projects.

Embeddings and FAISS are imported lazily so Cloud Run explore/report can
load this package without torch / sentence-transformers / faiss.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

from .chunking import chunk_lines, chunk_text, chunk_text_multi_granularity
from .document_schema import DocumentCollection, NormalizedDocument
from .ids import generate_content_hash, generate_id, generate_stable_document_id
from .logging import get_logger
from .storage import ensure_directory
from .text_processing import (
    DeduplicationConfig,
    TextNormalizationConfig,
    deduplicate_texts,
    normalize_text,
)

_LAZY_ATTRS = {
    "embed": (".embeddings", "embed"),
    "get_embedding_dimension": (".embeddings", "get_embedding_dimension"),
    "get_embedding_model": (".embeddings", "get_embedding_model"),
    "FAISSIndex": (".faiss_index", "FAISSIndex"),
    "resolve_index_directory": (".faiss_index", "resolve_index_directory"),
}

__all__ = [
    "chunk_lines",
    "chunk_text",
    "chunk_text_multi_granularity",
    "DocumentCollection",
    "NormalizedDocument",
    "embed",
    "ensure_directory",
    "generate_content_hash",
    "generate_id",
    "generate_stable_document_id",
    "get_embedding_dimension",
    "get_embedding_model",
    "get_logger",
    "FAISSIndex",
    "resolve_index_directory",
    "deduplicate_texts",
    "normalize_text",
    "TextNormalizationConfig",
    "DeduplicationConfig",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(__all__))
