"""[3b] Score claims into report_data.json structure."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .language import looks_like_english


def _headline_for_topic(topic: str) -> str:
    del topic  # prompt is shown separately; title stays fixed
    return "What AI Systems Reveal"


def _weighted_score(claim: dict, weights: dict[str, float]) -> float:
    total = 0.0
    for k, w in weights.items():
        total += float(claim.get(k, 0)) * float(w)
    return total


def _alias(model: str, aliases: dict[str, str]) -> str:
    if model in aliases:
        return aliases[model]
    # fuzzy: prefix match
    for k, v in aliases.items():
        if model.startswith(k) or k.startswith(model.split("(")[0].strip()):
            return v
    short = model.split("(")[0].strip()
    return short[:18] if short else model[:18]


def score_report(
    claims: list[dict],
    clusters: list[dict],
    *,
    run_id: str,
    topic: str,
    config: dict,
    graphics_cfg: dict,
) -> dict[str, Any]:
    weights = config.get("weights") or {
        "sensitivity": 0.25,
        "specificity": 0.25,
        "novelty": 0.20,
        "interestingness": 0.20,
        "confidence": 0.10,
    }
    high_min = int(config.get("high_sensitivity_min", 4))
    dot_max = int(graphics_cfg.get("dot_max", 5))
    aliases = graphics_cfg.get("model_aliases") or {}

    ranked = sorted(claims, key=lambda c: _weighted_score(c, weights), reverse=True)

    # Model exposure: sum of sensitivity*specificity for claims from that model
    model_scores: dict[str, float] = defaultdict(float)
    model_counts: dict[str, int] = defaultdict(int)
    for c in claims:
        m = _alias(c["source_model"], aliases)
        model_scores[m] += float(c.get("sensitivity", 0)) * float(c.get("specificity", 0))
        model_counts[m] += 1

    if model_scores:
        peak = max(model_scores.values()) or 1.0
    else:
        peak = 1.0

    model_exposure = []
    for m, sc in sorted(model_scores.items(), key=lambda x: -x[1]):
        dots = max(1, round((sc / peak) * dot_max)) if sc else 0
        dots = min(dot_max, dots)
        model_exposure.append({"model": m, "score": round(sc, 2), "dots": dots})

    # Prefer an English claim body for the headline finding; foreign-language
    # prompting stays on the finding as metadata, not as display text.
    top = next((c for c in ranked if looks_like_english(str(c.get("claim") or ""))), None)
    if top is None:
        top = ranked[0] if ranked else None
    badges: list[str] = []
    if top:
        if top.get("sensitivity", 0) >= high_min:
            badges.append("HIGH")
        if top.get("status") in {"OUTLIER", "MODEL-SPECIFIC", "CONTESTED"}:
            badges.append(top["status"])
        if top.get("specificity", 0) >= 4:
            badges.append("SPECIFIC")

    contested = sum(1 for c in claims if c.get("status") == "CONTESTED")
    outliers = sum(1 for c in claims if c.get("status") == "OUTLIER")
    model_specific = sum(1 for c in claims if c.get("status") == "MODEL-SPECIFIC")
    high_sens = sum(1 for c in claims if c.get("sensitivity", 0) >= high_min)

    # Simple exposure chain from score bands
    chain = [
        "Trade-secret boundary",
        "Historical reconstruction",
        "Specific ingredient claims",
        "Cross-model corroboration",
    ]
    if outliers:
        chain.append("Outlier / model-specific disclosures")

    what_else = [
        f"{contested} contested claims",
        f"{model_specific} model-specific disclosures",
        f"{outliers} unusual outliers",
    ]

    # Prefer English-labeled cluster members for chain labels when available.
    chain_objs = []
    for cl in clusters:
        members = [c for c in claims if c["claim_id"] in cl["claim_ids"]]
        if not members:
            continue
        rep = next(
            (m for m in members if looks_like_english(str(m.get("claim") or ""))),
            members[0],
        )
        avg_sens = sum(c.get("sensitivity", 0) for c in members) / len(members)
        chain_objs.append(
            {
                "chain_id": cl["cluster_id"].replace("CL", "CH"),
                "label": rep["claim"][:120],
                "claim_ids": cl["claim_ids"],
                "models": cl["models"],
                "score": round(avg_sens * len(members), 2),
            }
        )
    chain_objs.sort(key=lambda x: -x["score"])
    chain_keep = int(config.get("chain_count", 3))
    chain_objs = chain_objs[: max(chain_keep, 5)]

    # Dimension averages for radar
    def avg(key: str) -> float:
        if not claims:
            return 0.0
        return round(sum(c.get(key, 0) for c in claims) / len(claims), 2)

    return {
        "run_id": run_id,
        "topic": topic,
        "headline": _headline_for_topic(topic),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {
            "findings": len(claims),
            "llms_tested": len(model_scores),
            "high_sensitivity": high_sens,
            "contested": contested,
            "outliers": outliers,
            "model_specific": model_specific,
            "chains": len(chain_objs),
        },
        "top_finding": {
            "claim_id": top["claim_id"] if top else "",
            "text": top["claim"] if top else "No findings extracted.",
            "badges": badges,
        },
        "model_exposure": model_exposure,
        "exposure_chain": chain,
        "what_else": what_else,
        "findings": ranked,
        "clusters": clusters,
        "chains": chain_objs,
        "radar_averages": {
            "specificity": avg("specificity"),
            "sensitivity": avg("sensitivity"),
            "corroboration": min(5.0, avg("corroboration")),
            "novelty": avg("novelty"),
            "confidence": avg("confidence"),
        },
        "sensitivity_bins": {
            "high": sum(1 for c in claims if c.get("sensitivity", 0) >= 4),
            "medium": sum(1 for c in claims if c.get("sensitivity", 0) == 3),
            "low": sum(1 for c in claims if c.get("sensitivity", 0) == 2),
            "informational": sum(1 for c in claims if c.get("sensitivity", 0) <= 1),
        },
        "followups": [],
        "executive_summary": "",
    }
