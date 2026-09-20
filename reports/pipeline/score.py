"""[3b] Score claims into report_data.json structure."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .language import looks_like_english
from .cluster import dedupe_findings_by_group, present_id


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


def _sensitivity_band(sensitivity: int) -> str:
    if sensitivity >= 4:
        return "high"
    if sensitivity == 3:
        return "medium"
    if sensitivity == 2:
        return "low"
    return "informational"


def _source_models(claim: dict, aliases: dict[str, str]) -> list[str]:
    models = claim.get("source_models")
    if not isinstance(models, list) or not models:
        models = [claim.get("source_model")]
    out: list[str] = []
    seen: set[str] = set()
    for raw in models:
        if not raw:
            continue
        name = _alias(str(raw), aliases)
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _clip_claim(text: str, n: int = 140) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)].rstrip() + "…"


def build_model_contrast(
    claims: list[dict],
    aliases: dict[str, str] | None = None,
    *,
    max_items: int = 6,
) -> dict[str, Any]:
    """Compare model answers: shared claims vs unique / contested disclosures."""
    aliases = aliases or {}
    shared: list[dict[str, Any]] = []
    unique: list[dict[str, Any]] = []
    contested: list[dict[str, Any]] = []
    models_seen: set[str] = set()

    for claim in claims or []:
        models = _source_models(claim, aliases)
        models_seen.update(models)
        status = str(claim.get("status") or "").upper().replace("_", "-").replace(" ", "-")
        item = {
            "claim_id": str(claim.get("claim_id") or ""),
            "present_id": present_id(claim),
            "claim": _clip_claim(claim.get("claim") or ""),
            "models": models,
            "status": status,
        }
        n_models = len(models)
        if status in {"CONTESTED", "OUTLIER"}:
            contested.append({**item, "kind": status.lower()})
        elif n_models >= 2 or status == "CORROBORATED":
            shared.append(item)
        elif n_models == 1 or status == "MODEL-SPECIFIC":
            unique.append({**item, "kind": "model-specific"})

    shared_count = len(shared)
    unique_count = len(unique)
    shared = sorted(
        shared,
        key=lambda row: (-len(row.get("models") or []), str(row.get("claim_id") or "")),
    )[:max_items]
    differences = (contested + unique)[:max_items]

    n_models = len(models_seen)
    if n_models <= 0:
        lede = "No model findings were scored in this run."
    elif n_models == 1:
        only = next(iter(models_seen))
        lede = (
            f"Only {only} produced findings in this run, so there is no "
            "cross-model comparison yet."
        )
    else:
        lede = (
            f"{n_models} models answered the same investigation. "
            f"{shared_count} claim{'s' if shared_count != 1 else ''} "
            f"{'were' if shared_count != 1 else 'was'} corroborated across models. "
            f"{unique_count} disclosure{'s' if unique_count != 1 else ''} "
            f"{'are' if unique_count != 1 else 'is'} unique to a single model."
        )

    return {
        "lede": lede,
        "commonality": shared,
        "differences": differences,
        "model_count": n_models,
        "shared_count": shared_count,
        "unique_count": unique_count,
    }


def _empty_llm_row(name: str) -> dict[str, Any]:
    band_keys = ("high", "medium", "low", "informational")
    return {
        "model": name,
        "count": 0,
        "score": 0.0,
        "bands": {k: {"count": 0, "score": 0.0} for k in band_keys},
    }


def aggregate_findings_by_llm(
    claims: list[dict],
    aliases: dict[str, str] | None = None,
    *,
    models_probed: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Score each test LLM by finding quantity and sensitivity.

    Bar height (``score``) is the sum of finding sensitivities attributed to
    that model, so more findings and more sensitive findings both rank higher.
    ``bands`` splits that score (and the raw counts) into high / medium / low /
    informational so the chart can stack by sensitivity.

    When ``models_probed`` is set, the result includes exactly those models
    (zero-filled when they produced no findings) and drops any other labels.
    """
    aliases = aliases or {}
    band_keys = ("high", "medium", "low", "informational")
    rows: dict[str, dict[str, Any]] = {}
    for claim in claims or []:
        sens = int(claim.get("sensitivity", 0) or 0)
        band = _sensitivity_band(sens)
        for name in _source_models(claim, aliases):
            row = rows.get(name)
            if row is None:
                row = _empty_llm_row(name)
                rows[name] = row
            row["count"] += 1
            row["score"] += float(sens)
            row["bands"][band]["count"] += 1
            row["bands"][band]["score"] += float(sens)

    probed_raw = [str(m).strip() for m in (models_probed or []) if str(m).strip()]
    if probed_raw:
        from graphics.style import short_model_name

        ordered: list[str] = []
        seen: set[str] = set()
        for raw in probed_raw:
            key = short_model_name(raw, aliases)
            if not key or key == "unknown" or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        ranked_src = []
        for key in ordered:
            row = rows.get(key) or _empty_llm_row(key)
            ranked_src.append(row)
    else:
        ranked_src = list(rows.values())

    ranked: list[dict[str, Any]] = []
    for row in sorted(
        ranked_src,
        key=lambda r: (-float(r["score"]), -int(r["count"]), str(r["model"])),
    ):
        ranked.append(
            {
                "model": row["model"],
                "count": int(row["count"]),
                "score": round(float(row["score"]), 2),
                "bands": {
                    k: {
                        "count": int(v["count"]),
                        "score": round(float(v["score"]), 2),
                    }
                    for k, v in row["bands"].items()
                },
            }
        )
    return ranked


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

    # Presentation lists one row per collapsed group. Charts average and
    # stack every extracted claim from the investigation.
    investigation = list(claims or [])
    claims = dedupe_findings_by_group(investigation)
    ranked = sorted(claims, key=lambda c: _weighted_score(c, weights), reverse=True)

    # Model exposure: sum of sensitivity*specificity for claims from that model.
    # Collapsed claims carry ``source_models`` (all corroborating LLMs).
    model_scores: dict[str, float] = defaultdict(float)
    model_counts: dict[str, int] = defaultdict(int)
    for c in investigation:
        weight = float(c.get("sensitivity", 0)) * float(c.get("specificity", 0))
        for m in _source_models(c, aliases):
            model_scores[m] += weight
            model_counts[m] += 1

    findings_by_llm = aggregate_findings_by_llm(investigation, aliases)
    model_contrast = build_model_contrast(investigation, aliases)

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
    # One chain entry per collapsed group (claim_ids already = survivor id).
    chain_objs = []
    for cl in clusters:
        id_set = set(cl.get("claim_ids") or [])
        # Prefer survivor claim_ids; fall back to member_ids only to locate the
        # collapsed finding that absorbed them.
        members = [c for c in claims if c.get("claim_id") in id_set]
        if not members:
            member_ids = set(cl.get("member_ids") or [])
            members = [
                c
                for c in claims
                if c.get("claim_id") in member_ids
                or member_ids.intersection(set(c.get("merged_from") or []))
            ]
        if not members:
            continue
        # One finding per group after dedupe
        rep = next(
            (m for m in members if looks_like_english(str(m.get("claim") or ""))),
            members[0],
        )
        models = list(cl.get("models") or rep.get("source_models") or [])
        corr = int(rep.get("corroboration") or len(models) or 1)
        chain_objs.append(
            {
                "chain_id": str(cl.get("cluster_id") or "").replace("CL", "CH"),
                "label": rep["claim"][:120],
                "claim_ids": [rep["claim_id"]],
                "models": models,
                "score": round(float(rep.get("sensitivity") or 0) * corr, 2),
            }
        )
    chain_objs.sort(key=lambda x: -x["score"])

    # Dimension averages for radar — every extracted claim, not the abridged set.
    def avg(key: str) -> float:
        if not investigation:
            return 0.0
        return round(sum(c.get(key, 0) for c in investigation) / len(investigation), 2)

    topic_clean = (topic or "").strip()
    prompts = [topic_clean] if topic_clean else []

    return {
        "run_id": run_id,
        "topic": topic_clean,
        "prompts": prompts,
        "headline": _headline_for_topic(topic_clean),
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
        "findings_by_llm": findings_by_llm,
        "model_contrast": model_contrast,
        "exposure_chain": chain,
        "what_else": what_else,
        "findings": ranked,
        "findings_all": investigation,
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
