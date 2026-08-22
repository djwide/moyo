"""Build-spec fingerprint so public and private indexes can be matched.

Barrier analysis is only valid when both sides share the embedding model,
L2-normalization, and (for document corpora) the same chunk packing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from shared_utils.chunking import default_chunk_overlap, resolve_chunk_max_tokens
from shared_utils.model_config import resolve_model_name


GRANULARITY_MULTI = "multi"
GRANULARITY_PHRASES = "phrases"
GRANULARITY_UNKNOWN = "unknown"


@dataclass
class IndexBuildSpec:
    """Settings that must agree across a public/private index pair."""

    embedding_model: str
    normalize_embeddings: bool = True
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    max_tokens: Optional[int] = None
    min_chunk_length: Optional[int] = None
    deduplication_enabled: bool = True
    granularity: str = GRANULARITY_MULTI

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "IndexBuildSpec":
        data = dict(data or {})
        model = data.get("embedding_model") or data.get("model_name") or ""
        return cls(
            embedding_model=resolve_model_name(model) if model else "",
            normalize_embeddings=bool(data.get("normalize_embeddings", True)),
            chunk_size=data.get("chunk_size"),
            chunk_overlap=data.get("chunk_overlap"),
            max_tokens=data.get("max_tokens"),
            min_chunk_length=data.get("min_chunk_length"),
            deduplication_enabled=bool(data.get("deduplication_enabled", True)),
            granularity=data.get("granularity") or GRANULARITY_UNKNOWN,
        )


def spec_from_config(
    config: Any,
    *,
    granularity: str = GRANULARITY_MULTI,
) -> IndexBuildSpec:
    """Build a spec from CorpusConfig, IndexConfig, or ProcessingConfig."""
    model = getattr(config, "embedding_model", "") or ""
    max_tokens = getattr(config, "max_tokens", None)
    chunk_size = getattr(config, "chunk_size", None)
    overlap = getattr(config, "chunk_overlap", None)
    if overlap is None:
        overlap = default_chunk_overlap(int(chunk_size or 512))
    return IndexBuildSpec(
        embedding_model=resolve_model_name(model) if model else "",
        normalize_embeddings=bool(getattr(config, "normalize_embeddings", True)),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        max_tokens=resolve_chunk_max_tokens(model, max_tokens) if model else max_tokens,
        min_chunk_length=getattr(config, "min_chunk_length", None),
        deduplication_enabled=bool(getattr(config, "deduplication_enabled", True)),
        granularity=granularity,
    )


def load_index_spec(path: Union[str, Path]) -> Optional[IndexBuildSpec]:
    """Load a spec from an index directory or ``.faiss`` file."""
    path = Path(path)
    directory = path.parent if path.is_file() and path.suffix == ".faiss" else path
    if not directory.exists():
        return None

    index_info = _read_json(directory / "index_info.json")
    if index_info and (
        index_info.get("embedding_model") or index_info.get("granularity")
    ):
        return IndexBuildSpec.from_dict(index_info)

    corpus_info = _read_json(directory / "corpus_info.json")
    if corpus_info:
        cfg = (corpus_info.get("metadata") or {}).get("config") or {}
        if cfg:
            spec = IndexBuildSpec.from_dict(cfg)
            spec.granularity = cfg.get("granularity") or GRANULARITY_MULTI
            if corpus_info.get("embedding_model") and not spec.embedding_model:
                spec.embedding_model = resolve_model_name(corpus_info["embedding_model"])
            return spec

    public_meta = _read_json(directory / "metadata.json")
    if public_meta:
        cfg = public_meta.get("config") or {}
        if cfg:
            return spec_from_config(type("Cfg", (), cfg)(), granularity=GRANULARITY_MULTI)
        nested = (public_meta.get("metadata") or {})
        if nested.get("embedding_model"):
            return IndexBuildSpec.from_dict({**nested, "granularity": GRANULARITY_MULTI})

    return None


def compare_index_specs(
    private: IndexBuildSpec,
    public: IndexBuildSpec,
) -> List[str]:
    """Return hard-error messages if the pair cannot be compared.

    Phrase-level private indexes are allowed to differ in chunk size from a
    multi-granular public index (public already emits sentence/item vectors).
    Embedding model and L2-normalization must always match.
    """
    errors: List[str] = []
    priv_model = resolve_model_name(private.embedding_model) if private.embedding_model else ""
    pub_model = resolve_model_name(public.embedding_model) if public.embedding_model else ""
    if priv_model and pub_model and priv_model != pub_model:
        errors.append(
            f"Embedding model mismatch: private={priv_model}, public={pub_model}. "
            "Rebuild both indexes with the same model."
        )
    if private.normalize_embeddings != public.normalize_embeddings:
        errors.append(
            "Embedding L2-normalization mismatch. Both indexes must be built with "
            "normalize_embeddings=True for FlatIP cosine distances to agree."
        )

    both_multi = (
        private.granularity == GRANULARITY_MULTI
        and public.granularity == GRANULARITY_MULTI
    )
    if both_multi:
        if (
            private.chunk_size
            and public.chunk_size
            and int(private.chunk_size) != int(public.chunk_size)
        ):
            errors.append(
                f"chunk_size mismatch: private={private.chunk_size}, "
                f"public={public.chunk_size}."
            )
        if (
            private.chunk_overlap is not None
            and public.chunk_overlap is not None
            and int(private.chunk_overlap) != int(public.chunk_overlap)
        ):
            errors.append(
                f"chunk_overlap mismatch: private={private.chunk_overlap}, "
                f"public={public.chunk_overlap}."
            )
        if (
            private.max_tokens
            and public.max_tokens
            and int(private.max_tokens) != int(public.max_tokens)
        ):
            errors.append(
                f"max_tokens mismatch: private={private.max_tokens}, "
                f"public={public.max_tokens}."
            )
    return errors


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
