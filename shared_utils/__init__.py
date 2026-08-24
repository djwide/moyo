"""Shared utilities for sente and moyo projects."""

__version__ = "0.1.0"

from .chunking import chunk_lines, chunk_text, chunk_text_multi_granularity
from .document_schema import DocumentCollection, NormalizedDocument
from .embeddings import embed, get_embedding_dimension, get_embedding_model
from .faiss_index import FAISSIndex, resolve_index_directory
from .ids import generate_content_hash, generate_id, generate_stable_document_id
from .logging import get_logger
from .storage import ensure_directory
from .text_processing import (
    DeduplicationConfig,
    TextNormalizationConfig,
    deduplicate_texts,
    normalize_text,
)

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
