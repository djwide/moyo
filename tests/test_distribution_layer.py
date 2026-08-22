"""Distribution layer: margin, top-k entropy, JS occupancy, diagnostic KL."""

from __future__ import annotations

import numpy as np
import pytest

from moyo.publicside.barrierprobe.distribution import (
    CONCENTRATED_MAX_NORM_ENTROPY,
    CONCENTRATED_MIN_MARGIN,
    attach_neighborhood,
    build_distribution_layer,
    is_concentrated,
    js_distance,
    js_divergence,
    kl_divergence,
    neighborhood_from_distances,
    normalized_entropy,
    occupancy_from_embeddings,
    pairwise_exposure_label,
    softmax_from_distances,
    smooth_occupancy,
)


def test_softmax_is_confined_to_topk_not_corpus_size():
    peaked = [0.13, 0.31, 0.34, 0.36] + [0.40] * 16
    dense = [0.13, 0.14, 0.15, 0.15] + [0.16] * 16
    p_peaked = softmax_from_distances(peaked, temperature=0.10)
    p_dense = softmax_from_distances(dense, temperature=0.10)
    assert p_peaked[0] > 0.35
    assert p_dense[0] < 0.20
    # Padding extra far neighbors must not flatten a peaked row.
    padded = peaked + [0.80] * 80
    p_pad = softmax_from_distances(padded[:20], temperature=0.10)
    assert abs(normalized_entropy(p_peaked) - normalized_entropy(p_pad)) < 1e-9


def test_distinctive_vs_dense_neighborhood():
    distinctive = neighborhood_from_distances(
        [0.13, 0.31, 0.34, 0.36] + [0.40] * 16,
        k=20,
    )
    dense = neighborhood_from_distances(
        [0.13, 0.14, 0.15, 0.15] + [0.16] * 16,
        k=20,
    )
    assert distinctive.nn_distance == pytest.approx(0.13)
    assert distinctive.margin == pytest.approx(0.18)
    assert dense.margin == pytest.approx(0.01)
    assert distinctive.normalized_entropy < dense.normalized_entropy
    assert distinctive.concentrated is True
    assert dense.concentrated is False


def test_entropy_normalized_against_k():
    uniform20 = softmax_from_distances([0.2] * 20)
    uniform50 = softmax_from_distances([0.2] * 50)
    assert normalized_entropy(uniform20) == pytest.approx(1.0)
    assert normalized_entropy(uniform50) == pytest.approx(1.0)
    one_hot = np.array([1.0] + [0.0] * 19)
    assert normalized_entropy(one_hot) == pytest.approx(0.0)


def test_concentrated_requires_close_and_peaked():
    assert is_concentrated(0.13, 0.18, 0.40) is True
    assert is_concentrated(0.13, 0.01, 0.90) is False
    assert is_concentrated(0.55, 0.20, 0.20) is False
    assert is_concentrated(0.20, CONCENTRATED_MIN_MARGIN, 0.99) is True
    assert is_concentrated(0.20, 0.01, CONCENTRATED_MAX_NORM_ENTROPY) is True


def test_pairwise_exposure_uses_closest_pair_band():
    assert pairwise_exposure_label([0.42, 0.08, 0.55]) == "High"
    assert pairwise_exposure_label([0.22, 0.31]) == "Medium"
    assert pairwise_exposure_label([0.41]) == "Low"
    assert pairwise_exposure_label([0.71, 0.88]) == "None"
    assert pairwise_exposure_label([]) == "None"


def test_js_bounded_and_kl_asymmetric():
    p = np.array([0.80, 0.15, 0.05])
    q = np.array([0.20, 0.50, 0.30])
    assert 0.0 <= js_divergence(p, q) <= 1.0
    assert 0.0 <= js_distance(p, q) <= 1.0
    assert js_distance(p, p) == pytest.approx(0.0)
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))
    assert js_distance(p, q) == pytest.approx(js_distance(q, p))


def test_occupancy_js_rises_when_clouds_separate():
    rng = np.random.default_rng(0)
    shared = rng.normal(0.0, 0.05, size=(30, 8))
    overlap = occupancy_from_embeddings(shared[:15], shared[15:], n_clusters=4, random_state=0)
    priv = rng.normal(-2.0, 0.05, size=(15, 8))
    pub = rng.normal(2.0, 0.05, size=(15, 8))
    split = occupancy_from_embeddings(priv, pub, n_clusters=4, random_state=0)
    assert split.semantic_separation > overlap.semantic_separation
    assert split.kl_private_to_public > 0
    assert split.kl_public_to_private > 0


def test_headline_omits_kl_and_counts_concentrated():
    # Four private rows, 20 public columns. Row 0 is a distinctive leak;
    # the others sit in a dense public neighborhood.
    matrix = np.full((4, 20), 0.45)
    matrix[0] = np.array([0.12, 0.32, 0.34, 0.36] + [0.40] * 16)
    matrix[1] = np.array([0.13, 0.14, 0.15, 0.15] + [0.16] * 16)
    rng = np.random.default_rng(1)
    priv = rng.normal(0.0, 0.1, size=(4, 6))
    pub = rng.normal(0.0, 0.1, size=(20, 6))
    layer = build_distribution_layer(matrix, priv, pub, neighborhood_k=20, n_clusters=3)
    assert layer.pairwise_exposure == "Medium"
    assert layer.concentrated_matches >= 1
    assert layer.semantic_separation is not None
    headline = "\n".join(layer.headline_lines())
    assert "Semantic Separation:" in headline
    assert "Pairwise Exposure:" in headline
    assert "Concentrated Matches:" in headline
    assert "KL diagnostic" in headline
    assert "kl_private_to_public" in layer.diagnostics()
    # KL must not be a headline key.
    assert not headline.startswith("KL")


def test_attach_neighborhood_copies_fields():
    score = neighborhood_from_distances([0.13, 0.31, 0.34], k=3)
    row = attach_neighborhood({"distance": 0.13, "private_index": 0}, score)
    assert row["margin"] == pytest.approx(0.18)
    assert row["concentrated"] is True
    assert "normalized_entropy" in row


def test_smooth_occupancy_avoids_zero_bins():
    p = smooth_occupancy([3, 0, 1])
    assert p.min() > 0
    assert p.sum() == pytest.approx(1.0)
