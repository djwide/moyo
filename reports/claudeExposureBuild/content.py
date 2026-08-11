"""Content assembly for the claudeExposureBuild product.

Consumes the shared scored artifact (``report_data.json``) and produces a
self-contained content document (including inline chart SVGs) for this
product's own templates. Only pure data helpers are reused from the pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from graphics.style import format_source_cite, short_model_name
from pipeline.glossary import glossary_groups
from pipeline.sources import build_source_registry, top_source_labels
from pipeline.synthesize import parse_executive_payload
from pipeline.textclean import plain_text, strip_markdown

from . import charts


def format_report_date(raw: str | None) -> str:
    if raw:
        try:
            d = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return str(raw).upper()
    else:
        d = datetime.now()
    return d.strftime("%-d %b %Y").upper()


def _severity_label(sensitivity: int) -> str:
    if sensitivity >= 4:
        return "high"
    if sensitivity == 3:
        return "medium"
    if sensitivity == 2:
        return "low"
    return "info"


def _enrich_findings(
    findings: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    peers_by_cluster = {
        c.get("cluster_id"): list(c.get("models") or [])
        for c in clusters
        if c.get("cluster_id")
    }
    out = []
    for f in findings:
        row = dict(f)
        peers = peers_by_cluster.get(row.get("cluster_id"))
        row["claim"] = plain_text(row.get("claim"))
        row["category"] = plain_text(row.get("category")) or "unclassified"
        row["source_short"] = short_model_name(row.get("source_model") or "", aliases)
        row["source_cite"] = format_source_cite(
            row.get("source_model") or "",
            corroboration=row.get("corroboration"),
            peer_models=peers,
            aliases=aliases,
        )
        row["severity"] = _severity_label(int(row.get("sensitivity") or 0))
        out.append(row)
    return out


_PUBLIC_SOURCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bFEC\b|Federal Election Commission", re.I), "Federal Election Commission (FEC) filings"),
    (re.compile(r"OpenSecrets", re.I), "OpenSecrets.org"),
    (re.compile(r"House (Clerk|Ethics)|Clerk of the House|financial disclosure", re.I),
     "U.S. House Clerk / Ethics financial disclosures"),
    (re.compile(r"Business Insider", re.I), "Business Insider"),
    (re.compile(r"Texas Tribune", re.I), "The Texas Tribune"),
    (re.compile(r"The Intercept", re.I), "The Intercept"),
    (re.compile(r"Politico", re.I), "Politico"),
    (re.compile(r"Washington Post", re.I), "The Washington Post"),
    (re.compile(r"PACER|federal court|district court", re.I), "Federal court dockets (PACER)"),
    (re.compile(r"OpenSecrets|Center for Responsive", re.I), "OpenSecrets.org"),
]


def _infer_public_sources(findings: list[dict[str, Any]], limit: int = 3) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for f in findings:
        blob = " ".join(str(f.get(k) or "") for k in ("claim", "raw_excerpt", "category"))
        for pat, label in _PUBLIC_SOURCE_PATTERNS:
            if label in seen:
                continue
            if pat.search(blob):
                found.append(label)
                seen.add(label)
                if len(found) >= limit:
                    return found
    return found


def _confidence_label(score: int) -> str:
    if score >= 4:
        return "High"
    if score == 3:
        return "Medium"
    return "Low"


def _exec_fields(
    report_data: dict,
    findings: list[dict],
    pull: str,
    sources: list[dict] | None = None,
) -> dict:
    raw = report_data.get("executive_fields") or report_data.get("executive_summary") or ""
    fields = parse_executive_payload(raw)

    body = (fields.get("summary") or fields.get("top_finding_blurb") or pull or "").strip()

    top_id = (report_data.get("top_finding") or {}).get("claim_id") or ""
    top_claim = next((f for f in findings if f.get("claim_id") == top_id), None)
    conf_score = int((top_claim or (findings[0] if findings else {})).get("confidence") or 0)

    # Real citations from the model answers first; guessed labels only as fallback.
    public_sources = top_source_labels(list(sources or []), limit=3)
    if len(public_sources) < 3:
        candidates = [
            plain_text(s) for s in (fields.get("public_sources") or [])
        ] + _infer_public_sources(findings, 3)
        for s in candidates:
            if s and s not in public_sources:
                public_sources.append(s)
            if len(public_sources) >= 3:
                break

    inference = list(fields.get("inference_chain") or [])[:5]
    if not inference:
        chains = report_data.get("chains") or []
        if chains:
            c = chains[0]
            models = ", ".join(c.get("models") or [])
            ids = ", ".join(c.get("claim_ids") or [])
            inference = [
                (c.get("label") or "Lead cluster of related disclosures"),
                f"Corroborating models: {models}" if models else "Single-model disclosure",
                f"Evidence claims: {ids}" if ids else "See finding index for claim IDs",
            ]

    exposure_steps = list(fields.get("exposure_chain") or report_data.get("exposure_chain") or [])
    teaser = (fields.get("exposure_teaser") or "").strip()
    if not teaser and exposure_steps:
        teaser = "Exposure chain teaser: " + " \u2192 ".join(str(s) for s in exposure_steps[:3])

    defensive = (fields.get("defensive_action") or "").strip()
    if not defensive:
        fups = report_data.get("followups") or []
        if fups:
            defensive = str(fups[0].get("action") or "").strip()

    why = (fields.get("why_it_matters") or "").strip()
    if not why and pull:
        why = (
            "This disclosure is concrete enough for adversaries or reporters to act "
            "on without further invention, raising reputational and compliance risk."
        )

    conf_label = (fields.get("confidence_label") or "").strip() or _confidence_label(conf_score)
    conf_rationale = (fields.get("confidence_rationale") or "").strip()
    if not conf_rationale and top_claim:
        conf_rationale = (
            f"Lead finding confidence {conf_score}/5 "
            f"({top_claim.get('status') or 'UNVERIFIED'}; "
            f"corroboration {top_claim.get('corroboration') or 1})."
        )

    return {
        "body": strip_markdown(body),
        "pull_quote": plain_text(pull),
        "public_sources": public_sources[:3],
        "inference_chain": [plain_text(s) for s in inference if str(s).strip()],
        "confidence_label": plain_text(conf_label),
        "confidence_score": conf_score,
        "confidence_rationale": plain_text(conf_rationale),
        "why_it_matters": strip_markdown(why),
        "defensive_action": strip_markdown(defensive),
        "exposure_teaser": plain_text(teaser),
    }


def _model_rows(findings: list[dict], aliases: dict[str, str]) -> list[dict]:
    counts: dict[str, int] = {}
    for f in findings:
        m = short_model_name(f.get("source_model") or "", aliases)
        counts[m] = counts.get(m, 0) + 1
    return [{"model": m, "count": n} for m, n in counts.items()]


def build_content(
    report_data: dict[str, Any],
    *,
    report_date_cfg: str | None,
    aliases: dict[str, str] | None = None,
    prompts: list[str] | None = None,
) -> dict[str, Any]:
    aliases = aliases or {}
    counts = report_data.get("counts") or {}
    top = report_data.get("top_finding") or {}
    findings = _enrich_findings(
        list(report_data.get("findings") or []),
        list(report_data.get("clusters") or []),
        aliases,
    )
    sources, findings = build_source_registry(findings)
    pull = plain_text(top.get("text") or "")[:280]
    exec_fields = _exec_fields(report_data, findings, pull, sources)

    # Abridged sets (honor prior stipulation: top 5 findings/claims, 2 specific)
    abridged = findings[:5]
    specific = [
        f for f in findings
        if int(f.get("specificity") or 0) >= 4 and f.get("claim_id") != top.get("claim_id")
    ][:2]
    if not specific:
        specific = [f for f in findings if f.get("claim_id") != top.get("claim_id")][:2]

    if not prompts:
        raw_prompts = report_data.get("prompts") or report_data.get("prompt")
        if isinstance(raw_prompts, list):
            prompts = [str(p).strip() for p in raw_prompts if str(p).strip()]
        elif isinstance(raw_prompts, str) and raw_prompts.strip():
            prompts = [raw_prompts.strip()]
        else:
            prompts = [report_data.get("topic") or ""]

    explore_meta = report_data.get("explore_meta") or {}
    models_tested = list(explore_meta.get("models_tested") or [])
    if not models_tested:
        models_tested = [
            (m.get("model") or "").strip()
            for m in (report_data.get("model_exposure") or [])
            if (m.get("model") or "").strip()
        ]
    strategies = list(explore_meta.get("strategies") or []) or [
        "paraphrase", "translate", "summarize"
    ]

    sensitivity_bins = report_data.get("sensitivity_bins") or {}
    radar = report_data.get("radar_averages") or {}
    model_rows = _model_rows(findings, aliases)

    return {
        "meta": {
            "run_id": report_data.get("run_id"),
            "topic": report_data.get("topic"),
            "prompts": prompts,
            "headline": "What AI Systems Reveal",
            "report_date": format_report_date(report_date_cfg or report_data.get("report_date")),
            "fuzz_mode": explore_meta.get("fuzz_mode") or "basic",
            "strategies": strategies,
            "models_tested": models_tested,
            "counts": {
                "findings": counts.get("findings", len(findings)),
                "llms_tested": counts.get("llms_tested", 0),
                "high_sensitivity": counts.get("high_sensitivity", 0),
                "contested": counts.get("contested", 0),
                "outliers": counts.get("outliers", 0),
                "model_specific": counts.get("model_specific", 0),
                "chains": counts.get("chains", 0),
            },
        },
        "exec": exec_fields,
        "top_finding": {**top, "text": plain_text(top.get("text"))},
        "findings": findings,
        "abridged_findings": abridged,
        "specific_findings": specific,
        "model_exposure": report_data.get("model_exposure") or [],
        "sources": sources,
        "glossary": glossary_groups(),
        "followups": [
            {
                **item,
                "method": plain_text(item.get("method")),
                "action": strip_markdown(item.get("action")),
            }
            for item in (report_data.get("followups") or [])
        ],
        "charts": {
            "severity_donut": charts.severity_donut_svg(sensitivity_bins),
            "severity_donut_compact": charts.severity_donut_svg(sensitivity_bins, compact=True),
            "dimension_bars": charts.dimension_bars_svg(radar),
            "model_bars": charts.model_bars_svg(model_rows),
        },
    }
