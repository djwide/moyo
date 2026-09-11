"""GUI-style phrase indexes must load for barrier analysis and reject dim mismatch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shared_utils.faiss_index import FAISSIndex, resolve_index_directory
from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer, LLMFuzzerConfig
from moyo.publicside.barrierprobe.public_index_builder import load_public_index


def _write_gui_index(directory: Path, *, dim: int, model: str, n: int = 4) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1.0, norms)
    index = FAISSIndex(dimension=dim, index_type="flat")
    texts = [f"phrase {i}" for i in range(n)]
    index.add_vectors_with_texts(vectors.tolist(), texts)
    index.save(
        directory,
        name=directory.name,
        extra_info={"embedding_model": model, "normalize_embeddings": True, "granularity": "phrases"},
    )
    return directory


def test_resolve_index_directory_finds_nested_faiss(tmp_path: Path):
    nested = tmp_path / "indexes" / "public" / "acme"
    _write_gui_index(nested, dim=8, model="all-MiniLM-L6-v2")
    resolved = resolve_index_directory(tmp_path / "indexes" / "public")
    assert resolved == nested


def test_search_rejects_dimension_mismatch(tmp_path: Path):
    directory = _write_gui_index(tmp_path / "idx", dim=8, model="toy-8d")
    index = FAISSIndex.load(directory)
    assert index.embedding_model == "toy-8d"
    assert index.dimension == 8
    with pytest.raises(ValueError, match="8-d"):
        index.search([0.1] * 384, k=2)


def test_load_public_index_gui_phrase_layout(tmp_path: Path):
    directory = _write_gui_index(
        tmp_path / "indexes" / "public" / "acme",
        dim=8,
        model="BAAI/bge-base-en-v1.5",
    )
    builder = load_public_index(str(tmp_path / "indexes" / "public"))
    assert builder is not None
    assert builder.faiss_index is not None
    assert builder.faiss_index.get_vector_count() == 4
    assert builder.config.embedding_model == "BAAI/bge-base-en-v1.5"
    # GUI metadata.json is a list, not a builder config block
    raw = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert isinstance(raw, list)


def test_search_uses_string_store_when_metadata_is_corpus_dict(tmp_path: Path):
    directory = _write_gui_index(tmp_path / "idx", dim=8, model="toy-8d", n=4)
    index = FAISSIndex.load(directory)
    # PublicIndexBuilder writes corpus-level metadata.json, not a per-vector list.
    index.metadata = {
        "id": "corpus",
        "config": {"embedding_model": "toy-8d"},
        "chunk_count": 4,
    }
    query = index.index.reconstruct(0).tolist()
    _distances, _indices, rows = index.search(query, k=2)
    assert rows
    assert rows[0].get("text")
    assert "phrase" in rows[0]["text"]
    assert rows[0].get("error") != "metadata_not_found"


def test_fuzzer_prefers_index_embedding_model():
    index = FAISSIndex(dimension=768, index_type="flat")
    index.embedding_model = "BAAI/bge-base-en-v1.5"
    fuzzer = LLMFuzzer(LLMFuzzerConfig(embedding_model="all-MiniLM-L6-v2"))
    assert fuzzer._embedding_model_for_index(index) == "BAAI/bge-base-en-v1.5"
    assert fuzzer.config.embedding_model == "BAAI/bge-base-en-v1.5"
