"""Reusable ingestion pipeline for moyo and sente projects.

Modules:
- validators: MIME/type/size/archive guards
- loaders: type-specific text extractors
- normalize: encoding and Unicode normalization
- chunk: deterministic chunking strategies
- manifest: JSONL schema and writer (idempotent)
- pipeline: high-level orchestration API
"""

from .pipeline import ingest_paths, IngestConfig
from .manifest import ManifestWriter

__all__ = [
    "ingest_paths",
    "IngestConfig",
    "ManifestWriter",
]


