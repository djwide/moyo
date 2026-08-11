"""[3a] Dedupe + cluster claims; refresh corroboration / status."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cluster_claims(
    claims: list[dict],
    *,
    similarity_threshold: float = 0.82,
    corroboration_min_sources: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    Greedy cluster by claim-text similarity.

    Returns (updated_claims, clusters) where each cluster is:
      {cluster_id, claim_ids, models, representative_id}
    """
    if not claims:
        return [], []

    # Keep all claims — never drop. Clustering only links / updates status fields.
    parents = list(range(len(claims)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parents[rj] = ri

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if jaccard(claims[i]["claim"], claims[j]["claim"]) >= similarity_threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(claims)):
        groups[find(i)].append(i)

    clusters: list[dict] = []
    for ci, (_root, idxs) in enumerate(sorted(groups.items()), start=1):
        models = sorted({claims[i]["source_model"] for i in idxs})
        # Representative: highest interestingness * specificity
        rep = max(
            idxs,
            key=lambda i: (
                claims[i].get("interestingness", 0) * claims[i].get("specificity", 0)
                + claims[i].get("sensitivity", 0)
            ),
        )
        cluster = {
            "cluster_id": f"CL{ci:03d}",
            "claim_ids": [claims[i]["claim_id"] for i in idxs],
            "models": models,
            "representative_id": claims[rep]["claim_id"],
            "size": len(idxs),
        }
        clusters.append(cluster)

        n_sources = len(models)
        for i in idxs:
            claims[i]["cluster_id"] = cluster["cluster_id"]
            claims[i]["corroboration"] = n_sources
            # Preserve contested / outlier if already set and still meaningful
            prior = claims[i].get("status", "UNVERIFIED")
            if prior == "CONTESTED":
                continue
            if n_sources >= corroboration_min_sources:
                # If same cluster but conflicting specificity language — keep simple:
                claims[i]["status"] = "CORROBORATED"
            elif n_sources == 1 and prior in {"OUTLIER", "MODEL-SPECIFIC", "UNVERIFIED"}:
                if prior == "UNVERIFIED" and claims[i].get("sensitivity", 0) >= 4:
                    claims[i]["status"] = "MODEL-SPECIFIC"
                # else keep prior
            elif n_sources == 1:
                claims[i]["status"] = "MODEL-SPECIFIC"

    # Contested: high-sensitivity claims where sibling clusters disagree on formula detail
    # Lightweight signal: same query_id, different high-spec claims with low jaccard
    by_query: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(claims):
        by_query[c.get("query_id", "")].append(i)
    for idxs in by_query.values():
        highs = [i for i in idxs if claims[i].get("specificity", 0) >= 4]
        for a in highs:
            for b in highs:
                if a >= b:
                    continue
                if claims[a]["source_model"] == claims[b]["source_model"]:
                    continue
                if jaccard(claims[a]["claim"], claims[b]["claim"]) < 0.35:
                    # Mark both contested only if they mention overlapping themes
                    ta, tb = _tokens(claims[a]["claim"]), _tokens(claims[b]["claim"])
                    theme = ta & tb & {
                        "formula",
                        "7x",
                        "merchandise",
                        "oil",
                        "cocaine",
                        "ingredient",
                        "recipe",
                    }
                    if theme:
                        claims[a]["status"] = "CONTESTED"
                        claims[b]["status"] = "CONTESTED"

    return claims, clusters
