"""Distribution layer on top of cosine nearest-neighbor distances.

Pair level remains cosine NN distance. This module adds:

* Neighborhood: top-1/top-2 margin and top-k normalized entropy
* Corpus: JS distance over joint cluster occupancy (Semantic Separation)
* Diagnostic: directional KL (private→public and public→private)

High Semantic Separation is not barrier integrity: one leaked fact can
barely move global occupancy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:
    from sklearn.cluster import KMeans

    _SKLEARN = True
except Exception:  # pragma: no cover - optional extra
    _SKLEARN = False

DEFAULT_NEIGHBORHOOD_K = 20
DEFAULT_SOFTMAX_TEMPERATURE = 0.10
CLUSTER_MIN = 2
CLUSTER_MAX = 12
DIRICHLET_PSEUDOCOUNT = 0.5

# Pair-level distance bands (same as BarrierAnalyzer risk labels).
PAIR_HIGH = 0.10
PAIR_MEDIUM = 0.30
PAIR_LOW = 0.50

# A match is "concentrated" only if it is close enough to matter *and*
# the public neighborhood is peaked (margin and/or entropy).
CONCENTRATED_MAX_NN_DISTANCE = PAIR_MEDIUM
CONCENTRATED_MIN_MARGIN = 0.08
CONCENTRATED_MAX_NORM_ENTROPY = 0.70


@dataclass
class NeighborhoodScore:
    """Per-private neighborhood metrics over the nearest public passages."""

    private_index: int
    nn_distance: float
    second_distance: Optional[float]
    margin: Optional[float]
    entropy: Optional[float]
    normalized_entropy: Optional[float]
    k_used: int
    concentrated: bool


@dataclass
class CorpusSeparation:
    """Corpus-level occupancy comparison. JS is the headline; KL is diagnostic."""

    semantic_separation: float
    js_divergence: float
    kl_private_to_public: float
    kl_public_to_private: float
    n_clusters: int
    private_occupancy: List[float]
    public_occupancy: List[float]
    cluster_method: str


@dataclass
class DistributionLayer:
    """Headline trio plus diagnostics for the distribution layer."""

    semantic_separation: Optional[float]
    pairwise_exposure: str
    concentrated_matches: int
    neighborhoods: List[NeighborhoodScore] = field(default_factory=list)
    corpus: Optional[CorpusSeparation] = None
    neighborhood_k: int = DEFAULT_NEIGHBORHOOD_K
    softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE

    def by_private_index(self) -> Dict[int, NeighborhoodScore]:
        return {row.private_index: row for row in self.neighborhoods}

    def diagnostics(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "neighborhood_k": self.neighborhood_k,
            "softmax_temperature": self.softmax_temperature,
            "concentrated_rule": {
                "max_nn_distance": CONCENTRATED_MAX_NN_DISTANCE,
                "min_margin": CONCENTRATED_MIN_MARGIN,
                "max_normalized_entropy": CONCENTRATED_MAX_NORM_ENTROPY,
            },
            "note": (
                "Directional KL is a diagnostic only and is not part of the "
                "headline score. High semantic separation is not barrier "
                "integrity — a single leaked fact may barely move occupancy."
            ),
        }
        if self.corpus is not None:
            payload.update(
                {
                    "js_divergence": self.corpus.js_divergence,
                    "js_distance": self.corpus.semantic_separation,
                    "kl_private_to_public": self.corpus.kl_private_to_public,
                    "kl_public_to_private": self.corpus.kl_public_to_private,
                    "n_clusters": self.corpus.n_clusters,
                    "private_occupancy": self.corpus.private_occupancy,
                    "public_occupancy": self.corpus.public_occupancy,
                    "cluster_method": self.corpus.cluster_method,
                }
            )
        return payload

    def headline_lines(self) -> List[str]:
        sep = (
            f"{self.semantic_separation:.2f}"
            if self.semantic_separation is not None
            else "n/a"
        )
        lines = [
            f"Semantic Separation: {sep}",
            f"Pairwise Exposure: {self.pairwise_exposure}",
            f"Concentrated Matches: {self.concentrated_matches}",
        ]
        if self.corpus is not None:
            lines.append(
                f"KL diagnostic (not in headline): "
                f"private→public={self.corpus.kl_private_to_public:.3f}, "
                f"public→private={self.corpus.kl_public_to_private:.3f}"
            )
            lines.append(
                "High Semantic Separation does not mean the information "
                "barrier is secure — occupancy can miss a single leaked fact."
            )
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_separation": self.semantic_separation,
            "pairwise_exposure": self.pairwise_exposure,
            "concentrated_matches": self.concentrated_matches,
            "neighborhoods": [asdict(row) for row in self.neighborhoods],
            "corpus": asdict(self.corpus) if self.corpus is not None else None,
            "diagnostics": self.diagnostics(),
        }


def softmax_from_distances(
    distances: Sequence[float],
    temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
) -> np.ndarray:
    """Softmax over nearest neighbors: p ∝ exp(−d / τ).

    Temperature is in cosine-distance units so a 0.10 gap is a factor of e
    at the default τ. Corpus size does not enter — call this on top-k only.
    """
    arr = np.asarray(distances, dtype=np.float64)
    if arr.size == 0:
        return arr
    tau = float(temperature) if temperature and temperature > 0 else 1.0
    logits = -arr / tau
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(arr.shape, 1.0 / arr.size)
    return weights / total


def shannon_entropy(probs: Sequence[float]) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def normalized_entropy(probs: Sequence[float]) -> float:
    """Entropy divided by log(k) so the value stays in [0, 1] as k changes."""
    arr = np.asarray(probs, dtype=np.float64)
    k = int(arr.size)
    if k <= 1:
        return 0.0
    return float(shannon_entropy(arr) / np.log(k))


def nn_margin(d1: float, d2: Optional[float]) -> Optional[float]:
    if d2 is None:
        return None
    return float(d2 - d1)


def is_concentrated(
    nn_distance: float,
    margin: Optional[float],
    norm_entropy: Optional[float],
    *,
    max_nn_distance: float = CONCENTRATED_MAX_NN_DISTANCE,
    min_margin: float = CONCENTRATED_MIN_MARGIN,
    max_norm_entropy: float = CONCENTRATED_MAX_NORM_ENTROPY,
) -> bool:
    if nn_distance > max_nn_distance:
        return False
    peaked_margin = margin is not None and margin >= min_margin
    peaked_entropy = norm_entropy is not None and norm_entropy <= max_norm_entropy
    return bool(peaked_margin or peaked_entropy)


def pairwise_exposure_label(nn_distances: Sequence[float]) -> str:
    """Worst pair-level band across all private nearest neighbors."""
    if not nn_distances:
        return "None"
    closest = float(np.min(np.asarray(nn_distances, dtype=np.float64)))
    if closest <= PAIR_HIGH:
        return "High"
    if closest <= PAIR_MEDIUM:
        return "Medium"
    if closest <= PAIR_LOW:
        return "Low"
    return "None"


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """D_KL(P‖Q) in bits. Inputs must be strictly positive and sum to 1."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    return float(np.sum(p_arr * np.log2(p_arr / q_arr)))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen–Shannon divergence in bits, range [0, 1]."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    mid = 0.5 * (p_arr + q_arr)
    return 0.5 * kl_divergence(p_arr, mid) + 0.5 * kl_divergence(q_arr, mid)


def js_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """sqrt(JS divergence). With log2, this lives in [0, 1]."""
    return float(np.sqrt(js_divergence(p, q)))


def smooth_occupancy(counts: Sequence[float], *, pseudocount: float = DIRICHLET_PSEUDOCOUNT) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float64)
    k = max(int(arr.size), 1)
    smoothed = arr + float(pseudocount)
    return smoothed / smoothed.sum() if smoothed.sum() > 0 else np.full(k, 1.0 / k)


def choose_cluster_count(n_points: int, requested: Optional[int] = None) -> int:
    if n_points <= 1:
        return 1
    if requested is not None:
        return int(np.clip(requested, 1, n_points))
    auto = int(round(np.sqrt(n_points / 2.0)))
    return int(np.clip(auto, CLUSTER_MIN, min(CLUSTER_MAX, n_points)))


def _numpy_kmeans(matrix: np.ndarray, k: int, random_state: int = 0, n_iter: int = 25) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = int(matrix.shape[0])
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    k = min(k, n)
    centers = matrix[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        delta = matrix[:, None, :] - centers[None, :, :]
        labels = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        for j in range(k):
            members = matrix[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels


def cluster_labels(matrix: np.ndarray, k: int, random_state: int = 0) -> tuple[np.ndarray, str]:
    n = int(matrix.shape[0])
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=int), "single"
    if _SKLEARN:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        return model.fit_predict(matrix), "sklearn"
    return _numpy_kmeans(matrix, k, random_state=random_state), "numpy"


def occupancy_from_embeddings(
    private_embeddings: np.ndarray,
    public_embeddings: np.ndarray,
    *,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
) -> CorpusSeparation:
    """Joint k-means occupancy, then JS (headline) and both KL directions."""
    private = np.asarray(private_embeddings, dtype=np.float64)
    public = np.asarray(public_embeddings, dtype=np.float64)
    if private.ndim != 2 or public.ndim != 2 or private.shape[0] == 0 or public.shape[0] == 0:
        raise ValueError("Both embedding matrices must be non-empty 2-D arrays")

    stacked = np.vstack([private, public])
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    stacked = stacked / np.where(norms == 0, 1.0, norms)

    k = choose_cluster_count(int(stacked.shape[0]), n_clusters)
    labels, method = cluster_labels(stacked, k, random_state=random_state)
    priv_labels = labels[: private.shape[0]]
    pub_labels = labels[private.shape[0] :]
    priv_counts = np.bincount(priv_labels, minlength=k).astype(np.float64)
    pub_counts = np.bincount(pub_labels, minlength=k).astype(np.float64)
    p = smooth_occupancy(priv_counts)
    q = smooth_occupancy(pub_counts)
    return CorpusSeparation(
        semantic_separation=js_distance(p, q),
        js_divergence=js_divergence(p, q),
        kl_private_to_public=kl_divergence(p, q),
        kl_public_to_private=kl_divergence(q, p),
        n_clusters=k,
        private_occupancy=[float(x) for x in p],
        public_occupancy=[float(x) for x in q],
        cluster_method=method,
    )


def neighborhood_from_distances(
    distances: Sequence[float],
    *,
    private_index: int = 0,
    k: int = DEFAULT_NEIGHBORHOOD_K,
    temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
) -> NeighborhoodScore:
    """Margin + top-k entropy for one private chunk's public distances."""
    arr = np.asarray(distances, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return NeighborhoodScore(
            private_index=private_index,
            nn_distance=1.0,
            second_distance=None,
            margin=None,
            entropy=None,
            normalized_entropy=None,
            k_used=0,
            concentrated=False,
        )
    order = np.argsort(arr)
    ranked = arr[order]
    k_used = int(min(max(k, 1), ranked.size))
    top = ranked[:k_used]
    d1 = float(top[0])
    d2 = float(top[1]) if k_used >= 2 else None
    margin = nn_margin(d1, d2)
    if k_used >= 2:
        probs = softmax_from_distances(top, temperature=temperature)
        ent = shannon_entropy(probs)
        ent_norm = normalized_entropy(probs)
    else:
        ent = 0.0
        ent_norm = 0.0
    return NeighborhoodScore(
        private_index=private_index,
        nn_distance=d1,
        second_distance=d2,
        margin=margin,
        entropy=ent,
        normalized_entropy=ent_norm,
        k_used=k_used,
        concentrated=is_concentrated(d1, margin, ent_norm if k_used >= 2 else None),
    )


def build_distribution_layer(
    distance_matrix: np.ndarray,
    private_embeddings: np.ndarray,
    public_embeddings: np.ndarray,
    *,
    neighborhood_k: int = DEFAULT_NEIGHBORHOOD_K,
    temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
) -> DistributionLayer:
    """Score pair / neighborhood / corpus layers from one distance matrix."""
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    neighborhoods = [
        neighborhood_from_distances(
            matrix[i],
            private_index=i,
            k=neighborhood_k,
            temperature=temperature,
        )
        for i in range(matrix.shape[0])
    ]
    corpus: Optional[CorpusSeparation] = None
    try:
        corpus = occupancy_from_embeddings(
            private_embeddings,
            public_embeddings,
            n_clusters=n_clusters,
            random_state=random_state,
        )
    except ValueError:
        corpus = None

    nn_distances = [row.nn_distance for row in neighborhoods]
    return DistributionLayer(
        semantic_separation=corpus.semantic_separation if corpus else None,
        pairwise_exposure=pairwise_exposure_label(nn_distances),
        concentrated_matches=sum(1 for row in neighborhoods if row.concentrated),
        neighborhoods=neighborhoods,
        corpus=corpus,
        neighborhood_k=neighborhood_k,
        softmax_temperature=temperature,
    )


def attach_neighborhood(
    match: Mapping[str, Any],
    score: Optional[NeighborhoodScore],
) -> Dict[str, Any]:
    """Copy neighborhood fields onto a pair-level match / breach dict."""
    out = dict(match)
    if score is None:
        out.setdefault("margin", None)
        out.setdefault("second_distance", None)
        out.setdefault("neighborhood_entropy", None)
        out.setdefault("normalized_entropy", None)
        out.setdefault("concentrated", False)
        return out
    out["margin"] = score.margin
    out["second_distance"] = score.second_distance
    out["neighborhood_entropy"] = score.entropy
    out["normalized_entropy"] = score.normalized_entropy
    out["concentrated"] = score.concentrated
    return out
