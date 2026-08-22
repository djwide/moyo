"""Calibrate barrier cosine-distance thresholds from unlabeled NN distances.

Absolute distances are embedding-space specific. After any model change,
re-run calibration rather than reusing a MiniLM cutoff on MPNet/BGE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

# Quantile of the per-private nearest-public distance distribution.
# Lower quantile = fewer flags (higher precision).
PROFILE_QUANTILES: Mapping[str, float] = {
    "strict": 0.05,
    "balanced": 0.10,
    "recall": 0.25,
}

REFERENCE_THRESHOLDS = (0.10, 0.20, 0.30, 0.50, 0.80)


@dataclass
class ThresholdCalibration:
    """Suggested cosine-distance cutoff and supporting distribution stats."""

    profile: str
    recommended_distance: float
    n_private: int
    n_flagged_at_recommended: int
    profiles: Dict[str, float]
    fire_counts: Dict[str, int]
    min_distance: float
    max_distance: float
    mean_distance: float
    median_distance: float
    std_distance: float
    percentile_05: float
    percentile_10: float
    percentile_25: float
    notes: List[str] = field(default_factory=list)
    embedding_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calibrate_from_distances(
    distances: Sequence[float],
    *,
    profile: str = "balanced",
    embedding_model: Optional[str] = None,
) -> ThresholdCalibration:
    """Recommend a cosine-distance threshold from nearest-neighbor distances.

    ``profile`` selects which left-tail quantile becomes the cutoff:
    closest 5% (strict), 10% (balanced), or 25% (recall).
    """
    if profile not in PROFILE_QUANTILES:
        raise ValueError(
            f"Unknown profile {profile!r}. Choose one of {sorted(PROFILE_QUANTILES)}"
        )

    arr = np.asarray(list(distances), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite nearest-neighbor distances to calibrate from")

    profiles = {
        name: float(np.quantile(arr, q)) for name, q in PROFILE_QUANTILES.items()
    }
    recommended = profiles[profile]
    notes = [
        f"Recommended cosine-distance threshold ({profile}) is "
        f"{recommended:.4f}: flag the closest {int(PROFILE_QUANTILES[profile] * 100)}% "
        "of private phrases.",
        "Do not reuse this cutoff after changing the embedding model; "
        "absolute distances are not comparable across embedding spaces.",
        "similarity_threshold in BarrierProbeConfig is compared as cosine "
        "distance (smaller = closer).",
    ]
    median = float(np.median(arr))
    if median < 0.30:
        notes.append(
            f"Median nearest-neighbor distance is already low ({median:.4f}); "
            "overlap may be structural. Review flagged pairs rather than "
            "tightening the cutoff further."
        )

    fire_counts: Dict[str, int] = {}
    for value in list(REFERENCE_THRESHOLDS) + [recommended]:
        key = f"{value:.4f}"
        fire_counts[key] = int(np.sum(arr <= value))

    return ThresholdCalibration(
        profile=profile,
        recommended_distance=float(recommended),
        n_private=int(arr.size),
        n_flagged_at_recommended=int(np.sum(arr <= recommended)),
        profiles=profiles,
        fire_counts=fire_counts,
        min_distance=float(np.min(arr)),
        max_distance=float(np.max(arr)),
        mean_distance=float(np.mean(arr)),
        median_distance=median,
        std_distance=float(np.std(arr)),
        percentile_05=float(np.quantile(arr, 0.05)),
        percentile_10=float(np.quantile(arr, 0.10)),
        percentile_25=float(np.quantile(arr, 0.25)),
        notes=notes,
        embedding_model=embedding_model,
    )
