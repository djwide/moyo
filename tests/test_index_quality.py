"""Tests for chunking defaults, L2-normalization, dedup floors, and calibration."""

from __future__ import annotations

import numpy as np

from shared_utils.chunking import (
    DEFAULT_MIN_SECTION_CHARS,
    default_chunk_overlap,
    keep_granular_chunk,
    resolve_chunk_max_tokens,
)
from shared_utils.embeddings import l2_normalize, resolve_normalize
from shared_utils.index_spec import (
    GRANULARITY_MULTI,
    GRANULARITY_PHRASES,
    IndexBuildSpec,
    compare_index_specs,
)
from shared_utils.model_config import get_max_seq_tokens
from moyo.publicside.barrierprobe.calibrate import calibrate_from_distances


def test_default_overlap_is_ten_percent():
    assert default_chunk_overlap(512) == 51
    assert default_chunk_overlap(1000) == 100
    assert default_chunk_overlap(0) == 0


def test_max_tokens_follows_embedding_model():
    assert get_max_seq_tokens("mini") == 256
    assert get_max_seq_tokens("all-MiniLM-L6-v2") == 256
    assert get_max_seq_tokens("mpnet") == 384
    assert get_max_seq_tokens("bge-base") == 512
    assert resolve_chunk_max_tokens("mini") == 256
    assert resolve_chunk_max_tokens("mpnet", override=200) == 200


def test_keep_granular_chunk_drops_public_boilerplate_keeps_secrets():
    assert not keep_granular_chunk(
        "section",
        "Copyright 2024",
        min_section_chars=DEFAULT_MIN_SECTION_CHARS,
        has_finer_children=False,
        keep_short_atomic=False,
    )
    assert keep_granular_chunk(
        "section",
        "vault path prod/db/root",
        min_section_chars=DEFAULT_MIN_SECTION_CHARS,
        has_finer_children=False,
        keep_short_atomic=True,
    )
    assert keep_granular_chunk("sentence", "vault path prod/db/root")
    assert keep_granular_chunk(
        "section",
        "x" * 80,
        min_section_chars=DEFAULT_MIN_SECTION_CHARS,
    )


def test_l2_normalize_unit_rows():
    rows = l2_normalize([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    assert abs((rows[0][0] ** 2 + rows[0][2] ** 2) - 1.0) < 1e-6
    assert rows[1] == [0.0, 0.0, 0.0]


def test_resolve_normalize_defaults_on(monkeypatch):
    monkeypatch.delenv("MOYO_EMBEDDING_NORMALIZE", raising=False)
    assert resolve_normalize(None) is True
    assert resolve_normalize(True) is True
    monkeypatch.setenv("MOYO_EMBEDDING_NORMALIZE", "false")
    assert resolve_normalize(None) is False


def test_compare_specs_requires_model_and_normalize():
    private = IndexBuildSpec(
        embedding_model="all-MiniLM-L6-v2",
        normalize_embeddings=True,
        chunk_size=512,
        chunk_overlap=50,
        max_tokens=256,
        granularity=GRANULARITY_MULTI,
    )
    public = IndexBuildSpec(
        embedding_model="all-mpnet-base-v2",
        normalize_embeddings=False,
        chunk_size=256,
        chunk_overlap=20,
        max_tokens=384,
        granularity=GRANULARITY_MULTI,
    )
    errors = compare_index_specs(private, public)
    assert any("Embedding model mismatch" in e for e in errors)
    assert any("L2-normalization" in e for e in errors)
    assert any("chunk_size" in e for e in errors)

    phrase = IndexBuildSpec(
        embedding_model="all-MiniLM-L6-v2",
        normalize_embeddings=True,
        granularity=GRANULARITY_PHRASES,
    )
    public_ok = IndexBuildSpec(
        embedding_model="all-MiniLM-L6-v2",
        normalize_embeddings=True,
        chunk_size=512,
        chunk_overlap=50,
        granularity=GRANULARITY_MULTI,
    )
    assert compare_index_specs(phrase, public_ok) == []


def test_calibrate_from_distances_balanced_quantile():
    # 100 distances, evenly 0.01 .. 1.00. Balanced = 10th percentile ≈ 0.10
    distances = [(i + 1) / 100.0 for i in range(100)]
    result = calibrate_from_distances(distances, profile="balanced")
    assert result.n_private == 100
    assert abs(result.recommended_distance - 0.10) < 0.02
    assert result.n_flagged_at_recommended == 10
    strict = calibrate_from_distances(distances, profile="strict")
    recall = calibrate_from_distances(distances, profile="recall")
    assert strict.recommended_distance < result.recommended_distance < recall.recommended_distance
    assert "not reuse this cutoff" in " ".join(result.notes)
