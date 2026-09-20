"""Write per-run content package: report.md + report.yaml (LLM-editable).

Presentation stays in ``reports/design-system/``. Charts land in ``assets/`` as SVG.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from graphics.graph import snapshot_graph_findings
from graphics.style import format_source_cite, short_model_name
from .cluster import dedupe_findings_by_group, present_id
from pipeline.graphics import ASSET_NAMES
from pipeline.synthesize import parse_executive_payload
from pipeline.score import build_model_contrast
from pipeline.basis import build_basis_section
from pipeline.glossary import glossary_groups
from pipeline.isvf import load_isvf_controls, select_remediation
from pipeline.language import (
    default_translate_fn,
    englishize_findings,
    is_foreign_language,
    languages_from_findings,
    looks_like_english,
)
from pipeline.sources import build_source_registry, top_source_labels
from pipeline.textclean import plain_text, strip_markdown


def _severity_label(sensitivity: int) -> str:
    if sensitivity >= 4:
        return "high"
    if sensitivity == 3:
        return "medium"
    if sensitivity == 2:
        return "low"
    return "info"


def build_next_steps(*, include_remediation: bool = False) -> dict[str, Any]:
    """Next-steps blocks for Exposure Snapshot and Basis Report.

    Snapshot points readers toward Basis Report depth this abridged product
    omits. Basis recommends red-teaming. Both push denser, better-strategized
    prompting and fuller use of MOYO.
    """
    snapshot_items: list[dict[str, str]] = [
        {
            "title": "Request the Basis Report",
            "body": (
                "This snapshot keeps the top findings. The Basis Report has "
                "the ranked inventory, full evidence excerpts with line "
                "offsets, corroborating outputs, derivation steps, cited "
                "sources, and exploitation implications."
            ),
        },
        {
            "title": "Review the complete exposure inventory",
            "body": (
                "Move beyond the top findings shown here. The Basis Report "
                "retains every extracted claim — including outliers, contested "
                "items, and single-model disclosures — ranked by weighted "
                "exposure so nothing material is dropped."
            ),
        },
        {
            "title": "Trace derivation and corroboration",
            "body": (
                "Use the Basis Report to see exactly how each conclusion was "
                "reached: query seeds, model fan-out, line-offset evidence, "
                "and multi-model paraphrases that corroborate the same "
                "exposure."
            ),
        },
        {
            "title": "Map exploitation implications",
            "body": (
                "The Basis Report pairs each exposure with a concrete "
                "implication scenario — how an adversary or opposition "
                "researcher could weaponize the recovered synthesis."
            ),
        },
    ]
    if include_remediation:
        snapshot_items.append(
            {
                "title": "Apply ISVF-aligned remediation",
                "body": (
                    "When remediation is enabled, the Basis Report maps "
                    "findings to Idea Security Verification Framework "
                    "controls and a run-specific follow-up playbook."
                ),
            }
        )
    snapshot_items.append(
        {
            "title": "Prompt harder and use MOYO more deliberately",
            "body": (
                "Re-run with denser, better-strategized prompts: paraphrase "
                "and translate angles, multi-step retrieval seeds, targeted "
                "follow-ups on high-sensitivity claims, and broader model "
                "coverage. Treat this snapshot as a scout pass, not the "
                "ceiling of what MOYO can surface."
            ),
        }
    )

    basis_items: list[dict[str, str]] = [
        {
            "title": "Red-team the reachable conclusions",
            "body": (
                "Escalate from passive exposure assessment to adversarial "
                "testing. Use MOYO red-teaming (white-box when secrets are "
                "known; black-box when probing blindly) to pressure-test "
                "whether high-sensitivity conclusions remain reachable under "
                "hostile prompting, paraphrase, and multi-step chaining."
            ),
        },
        {
            "title": "Prompt harder and use MOYO more deliberately",
            "body": (
                "This report reflects the prompts and strategies used in this "
                "run. Deepen coverage with better-strategized prompting: "
                "reworded seeds, translation and summarization paths, "
                "targeted probes on contested or single-model claims, and "
                "additional models. Iterate until high-value exposures are "
                "either corroborated or ruled out."
            ),
        },
        {
            "title": "Close the loop on high-severity chains",
            "body": (
                "Prioritize the highest-scoring exposure chains for "
                "follow-up runs, domain-boundary review, and — where "
                "applicable — remediation against Unreachable Statement "
                "Classes and permitted-join policy."
            ),
        },
    ]

    return {
        "snapshot": {
            "title": "This snapshot is scouting  for AI assertions, not a record of truths",
            "lede": (
                "The abridged product stops at the top findings. Use the "
                "Basis Report for the rest of the inventory, then re-prompt "
                "on the high-sensitivity claims."
            ),
            "items": snapshot_items,
        },
        "basis": {
            "title": "Turn the inventory into a test plan",
            "lede": (
                "The exposure basis is complete for this run. Red-team the "
                "reachable conclusions and tighten the prompts until "
                "high-value claims are corroborated or ruled out."
            ),
            "items": basis_items,
        },
    }


def _enrich_findings(
    findings: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    *,
    aliases: dict[str, str] | None = None,
    translate: bool = True,
    llm_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach source cites and present every finding in English.

    Foreign-language prompt provenance is kept as ``language_annotation``;
    claim / excerpt bodies shown in reports are English only.
    """
    aliases = aliases or {}
    peers_by_cluster = {
        c.get("cluster_id"): list(c.get("models") or [])
        for c in clusters
        if c.get("cluster_id")
    }
    translate_fn = default_translate_fn(llm_config) if translate else None
    normalized = englishize_findings(findings, translate=translate_fn)
    out: list[dict[str, Any]] = []
    for f in normalized:
        row = dict(f)
        peers = peers_by_cluster.get(row.get("cluster_id"))
        row["claim"] = plain_text(row.get("claim"))
        row["category"] = plain_text(row.get("category")) or "unclassified"
        row["present_id"] = present_id(row)
        row["source_short"] = short_model_name(row.get("source_model") or "", aliases)
        source_models = row.get("source_models")
        if isinstance(source_models, list) and len(source_models) > 1:
            peers = [str(m) for m in source_models]
        cite = format_source_cite(
            row.get("source_model") or "",
            corroboration=row.get("corroboration"),
            peer_models=peers,
            aliases=aliases,
        )
        note = (row.get("language_annotation") or "").strip()
        if note:
            row["source_cite"] = f"{cite} · {note}" if cite else note
        else:
            row["source_cite"] = cite
        out.append(row)
    return out


def _prompt_languages(
    report_data: dict[str, Any],
    findings: list[dict[str, Any]],
) -> list[str]:
    explore_meta = report_data.get("explore_meta") or {}
    langs = list(explore_meta.get("languages") or [])
    if not langs:
        langs = languages_from_findings(findings)
    # Dedupe, English first
    out: list[str] = []
    seen: set[str] = set()
    for name in langs:
        key = (name or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name.strip())
    eng = [x for x in out if not is_foreign_language(x)]
    foreign = [x for x in out if is_foreign_language(x)]
    return eng + foreign


def _sync_top_finding_english(
    top: dict[str, Any],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ensure top_finding.text is English and carries prompt-language metadata."""
    top = dict(top or {})

    def _usable(f: dict[str, Any]) -> bool:
        return looks_like_english(str(f.get("claim") or "")) and not f.get(
            "english_pending"
        )

    top_id = top.get("claim_id") or ""
    match = next((f for f in findings if f.get("claim_id") == top_id), None)
    if match and _usable(match):
        top["text"] = plain_text(match.get("claim") or top.get("text"))
        note = (match.get("language_annotation") or "").strip()
        badges = [
            b
            for b in list(top.get("badges") or [])
            if not str(b).upper().startswith("VIA ")
        ]
        if note:
            top["language_annotation"] = note
            top["prompt_language"] = match.get("prompt_language") or match.get(
                "language"
            )
        top["badges"] = badges
        return top

    # Fall back to the highest-ranked English finding for display.
    english = next((f for f in findings if _usable(f)), None)
    if english:
        badges: list[str] = []
        if int(english.get("sensitivity") or 0) >= 4:
            badges.append("HIGH")
        status = (english.get("status") or "").upper()
        if status in {"OUTLIER", "MODEL-SPECIFIC", "CONTESTED"}:
            badges.append(status)
        if int(english.get("specificity") or 0) >= 4:
            badges.append("SPECIFIC")
        top = {
            "claim_id": english.get("claim_id"),
            "text": plain_text(english.get("claim")),
            "badges": badges,
            "language_annotation": english.get("language_annotation") or "",
            "prompt_language": english.get("prompt_language")
            or english.get("language"),
        }
        return top

    top["text"] = plain_text(top.get("text"))
    return top


_PUBLIC_SOURCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bFEC\b|Federal Election Commission", re.I), "Federal Election Commission (FEC) filings"),
    (re.compile(r"OpenSecrets", re.I), "OpenSecrets.org"),
    (
        re.compile(r"House (Clerk|Ethics)|Clerk of the House|financial disclosure", re.I),
        "U.S. House Clerk / Ethics financial disclosures",
    ),
    (re.compile(r"Business Insider", re.I), "Business Insider"),
    (re.compile(r"Texas Tribune", re.I), "The Texas Tribune"),
    (re.compile(r"The Intercept", re.I), "The Intercept"),
    (re.compile(r"Politico", re.I), "Politico"),
    (re.compile(r"Washington Post", re.I), "The Washington Post"),
    (re.compile(r"PACER|federal court|district court", re.I), "Federal court dockets (PACER)"),
    (re.compile(r"Office of Congressional Ethics|\bOCE\b", re.I), "Office of Congressional Ethics (OCE)"),
]


def _infer_public_sources(findings: list[dict[str, Any]], limit: int = 3) -> list[str]:
    """Fallback source labels for runs whose answers carried no citations."""
    found: list[str] = []
    seen: set[str] = set()
    for f in findings:
        cites = f.get("citations") or []
        cite_blob = " ".join(str(c) for c in cites) if isinstance(cites, list) else ""
        blob = " ".join(
            [
                str(f.get(k) or "")
                for k in ("claim", "raw_excerpt", "category")
            ]
            + [cite_blob]
        )
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


def _prompts_list(report_data: dict[str, Any]) -> list[str]:
    raw = report_data.get("prompts") or report_data.get("prompt")
    if isinstance(raw, list):
        out = [str(p).strip() for p in raw if str(p).strip()]
        if out:
            return out
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    topic = (report_data.get("topic") or "").strip()
    run_id = str(report_data.get("run_id") or "").strip()
    # Avoid printing a bare run-id as the cover "Prompt" when topic was never
    # recovered from exploration.md (e.g. mid-pipeline rebuild without -e).
    if topic and topic != run_id and topic.replace(" ", "_") != run_id:
        return [topic]
    return [topic] if topic and topic != run_id else []


def _build_executive_page(
    report_data: dict[str, Any],
    *,
    findings: list[dict[str, Any]],
    top: dict[str, Any],
    pull: str,
    sources: list[dict[str, Any]] | None = None,
    include_remediation: bool = False,
) -> dict[str, Any]:
    fields = report_data.get("executive_fields")
    if not isinstance(fields, dict) or not fields:
        fields = parse_executive_payload(
            report_data.get("executive_summary")
            or report_data.get("executive_fields")
            or ""
        )

    body = (
        fields.get("summary")
        or fields.get("top_finding_blurb")
        or pull
        or ""
    ).strip()

    top_id = top.get("claim_id") or ""
    top_claim = next((f for f in findings if f.get("claim_id") == top_id), None)
    conf_score = int((top_claim or {}).get("confidence") or 0)
    if not conf_score and findings:
        conf_score = int(findings[0].get("confidence") or 0)

    # Real citations carried by the model answers outrank anything the LLM or
    # the fallback patterns guessed.
    public_sources = top_source_labels(list(sources or []), limit=3)
    if len(public_sources) < 3:
        candidates = [
            plain_text(s) for s in (fields.get("public_sources") or [])
        ] + _infer_public_sources(findings, limit=3)
        for src in candidates:
            if src and src not in public_sources:
                public_sources.append(src)
            if len(public_sources) >= 3:
                break

    inference = list(fields.get("inference_chain") or [])[:5]
    if not inference:
        chains = report_data.get("chains") or []
        if chains:
            label = (chains[0].get("label") or "").strip()
            models = ", ".join(chains[0].get("models") or [])
            ids = ", ".join(chains[0].get("claim_ids") or [])
            inference = [
                label or "Lead cluster of related disclosures",
                f"Corroborating models: {models}" if models else "Single-model disclosure",
                f"Evidence claims: {ids}" if ids else "See finding index for claim IDs",
            ]

    exposure_steps = list(
        fields.get("exposure_chain") or report_data.get("exposure_chain") or []
    )
    teaser = (fields.get("exposure_teaser") or "").strip()
    if not teaser and exposure_steps:
        teaser = " → ".join(str(s) for s in exposure_steps[:3])
    elif not teaser and inference:
        teaser = " → ".join(inference[:3])

    defensive = ""
    if include_remediation:
        defensive = (fields.get("defensive_action") or "").strip()
        if not defensive:
            followups = report_data.get("followups") or []
            if followups:
                defensive = str(followups[0].get("action") or "").strip()

    why = (fields.get("why_it_matters") or "").strip()

    conf_label = (fields.get("confidence_label") or "").strip() or _confidence_label(
        conf_score
    )
    conf_rationale = (fields.get("confidence_rationale") or "").strip()
    if not conf_rationale and top_claim:
        conf_rationale = (
            f"Lead finding confidence score {conf_score}/5 "
            f"({top_claim.get('status') or 'UNVERIFIED'}; "
            f"corroboration {top_claim.get('corroboration') or 1})."
        )

    return {
        "body": strip_markdown(body),
        "pull_quote": plain_text(pull),
        "public_sources": public_sources[:3],
        "inference_chain": [plain_text(step) for step in inference if str(step).strip()],
        "confidence_label": plain_text(conf_label),
        "confidence_score": conf_score,
        "confidence_rationale": plain_text(conf_rationale),
        "why_it_matters": strip_markdown(why),
        "defensive_action": strip_markdown(defensive),
        "exposure_teaser": plain_text(teaser),
        "model_commonality": [
            plain_text(x) for x in (fields.get("model_commonality") or []) if str(x).strip()
        ][:5],
        "model_differences": [
            plain_text(x) for x in (fields.get("model_differences") or []) if str(x).strip()
        ][:5],
    }


def build_content_doc(
    report_data: dict[str, Any],
    *,
    report_date: str,
    aliases: dict[str, str] | None = None,
    isvf_path: Path | None = None,
    include_remediation: bool = False,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured content document consumed by design-system templates."""
    counts = report_data.get("counts") or {}
    top = report_data.get("top_finding") or {}
    findings_enriched = _enrich_findings(
        list(report_data.get("findings_all") or report_data.get("findings") or []),
        list(report_data.get("clusters") or []),
        aliases=aliases,
        llm_config=llm_config,
    )
    sources, findings_enriched = build_source_registry(findings_enriched)
    # Snapshot lists collapsed groups; basis narrative can use every claim.
    findings = dedupe_findings_by_group(findings_enriched)
    top = _sync_top_finding_english(top, findings)
    pull = plain_text(top.get("text") or "")[:280]
    prompts = [plain_text(p) for p in _prompts_list(report_data)]
    exec_page = _build_executive_page(
        report_data,
        findings=findings,
        top=top,
        pull=pull,
        sources=sources,
        include_remediation=include_remediation,
    )

    contrast = dict(report_data.get("model_contrast") or {})
    if not contrast.get("lede") and not (
        contrast.get("commonality") or contrast.get("differences")
    ):
        contrast = build_model_contrast(findings_enriched or findings, aliases or {})
    contrast["commonality_prose"] = list(exec_page.get("model_commonality") or [])
    contrast["differences_prose"] = list(exec_page.get("model_differences") or [])

    # One-pager budget: fill a single A4 landscape sheet, nothing more.
    # Lead + ~4 rail claims + ~6 compact further rows is the ceiling.
    specific_min = 4
    specific_cap = 4
    snapshot_cap = 12
    onepage_more_cap = 6
    top_id = top.get("claim_id") or ""

    def _specific_rank(f: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(f.get("specificity") or 0),
            int(f.get("sensitivity") or 0),
            int(f.get("confidence") or 0),
        )

    english_findings = [
        f
        for f in findings
        if looks_like_english(str(f.get("claim") or "")) and not f.get("english_pending")
    ] or [
        f for f in findings if looks_like_english(str(f.get("claim") or ""))
    ] or findings
    specific_findings = sorted(
        [
            f
            for f in english_findings
            if int(f.get("specificity") or 0) >= specific_min
            and f.get("claim_id") != top_id
        ],
        key=_specific_rank,
        reverse=True,
    )[:specific_cap]
    if not specific_findings:
        specific_findings = sorted(
            [f for f in english_findings if f.get("claim_id") != top_id],
            key=_specific_rank,
            reverse=True,
        )[:specific_cap]

    abridged = english_findings[:snapshot_cap]
    evidence_findings = snapshot_graph_findings(
        english_findings,
        cap=snapshot_cap,
        limit=10,
    )
    shown_ids = {top_id} | {f.get("claim_id") for f in specific_findings}
    onepage_more = [
        f
        for f in english_findings
        if f.get("claim_id") not in shown_ids
    ][:onepage_more_cap]

    explore_meta = report_data.get("explore_meta") or {}
    alias_map = aliases or {}
    models_tested: list[str] = []
    seen_models: set[str] = set()
    for raw in list(explore_meta.get("models_tested") or []):
        name = short_model_name(str(raw or ""), alias_map)
        if not name or name == "unknown" or name in seen_models:
            continue
        seen_models.add(name)
        models_tested.append(name)
    if not models_tested:
        for m in report_data.get("model_exposure") or []:
            name = (m.get("model") or "").strip()
            if not name or name in seen_models:
                continue
            seen_models.add(name)
            models_tested.append(name)
    strategies = list(explore_meta.get("strategies") or [])
    if not strategies:
        strategies = ["original", "paraphrase"]

    collection_issues = list(report_data.get("collection_issues") or [])
    if not collection_issues:
        collection_issues = list(explore_meta.get("collection_issues") or [])

    prompt_languages = _prompt_languages(report_data, findings)
    foreign_languages = [x for x in prompt_languages if is_foreign_language(x)]
    languages_count = len(prompt_languages) if foreign_languages else 0

    headline = (report_data.get("headline") or "").strip()
    if not headline or headline.lower().startswith("what ai systems reveal about"):
        headline = "What AI Systems Reveal"
    if headline.lower() == "what ai systems reveal":
        headline = "What AI Systems Reveal"

    # Basis Report content; ISVF remediation only when explicitly enabled.
    remediation: list[dict[str, Any]] = []
    followups: list[dict[str, Any]] = []
    if include_remediation:
        followups = list(report_data.get("followups") or [])
        resolved_isvf = isvf_path or (
            report_data.get("isvf_path") and Path(report_data["isvf_path"])
        )
        if resolved_isvf:
            catalog = Path(resolved_isvf) / "controls" / "control-catalog.md"
            remediation = select_remediation(load_isvf_controls(catalog))
    basis_section = build_basis_section(
        report_data,
        findings,
        remediation=remediation,
        all_findings=findings_enriched,
    )
    followups = [
        {
            **item,
            "method": plain_text(item.get("method")),
            "action": strip_markdown(item.get("action")),
        }
        for item in followups
    ]

    n_findings = int(counts.get("findings", len(findings)) or 0)
    n_high = int(counts.get("high_sensitivity", 0) or 0)
    n_models = int(counts.get("llms_tested") or len(models_tested) or 0)
    coverage = dict(explore_meta.get("coverage") or {})
    n_attempted = int(coverage.get("attempted") or counts.get("llms_attempted") or n_models or 0)
    n_received = int(coverage.get("response_received") or counts.get("llms_response_received") or 0)
    n_substantive = int(
        coverage.get("substantive_response") or counts.get("llms_substantive") or 0
    )
    n_contributed = int(
        coverage.get("claims_contributed") or counts.get("llms_claims_contributed") or 0
    )
    if n_substantive:
        n_models = n_substantive
    n_abridged = len(abridged)
    if n_high:
        exec_page["title"] = (
            f"{n_high} high-sensitivity disclosures across {n_models} models"
        )
    else:
        exec_page["title"] = (
            f"{n_findings} disclosures across {n_models} models"
        )

    return {
        "meta": {
            "run_id": report_data.get("run_id"),
            "topic": report_data.get("topic"),
            "prompts": prompts,
            "headline": headline,
            "generated_at": report_data.get("generated_at")
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "report_date": report_date,
            "fuzz_mode": explore_meta.get("fuzz_mode") or "basic",
            "strategies": strategies,
            "models_tested": models_tested,
            "languages": prompt_languages,
            "include_remediation": bool(include_remediation),
            "collection_issues": collection_issues,
            "coverage": {
                "attempted": n_attempted,
                "response_received": n_received,
                "substantive_response": n_substantive,
                "claims_contributed": n_contributed,
            },
            "counts": {
                "findings": counts.get("findings", len(findings)),
                "llms_tested": n_models,
                "llms_attempted": n_attempted,
                "llms_response_received": n_received,
                "llms_substantive": n_substantive,
                "llms_claims_contributed": n_contributed,
                "high_sensitivity": counts.get("high_sensitivity", 0),
                "contested": counts.get("contested", 0),
                "outliers": counts.get("outliers", 0),
                "model_specific": counts.get("model_specific", 0),
                "chains": counts.get("chains", 0),
                "languages": languages_count,
            },
        },
        "pages": {
            "executive_summary": exec_page,
            "risk_overview": {
                "title": "Which models disclosed the most",
                "body": (
                    f"{n_findings} findings from {n_substantive or n_models} models "
                    f"with a substantive answer "
                    f"({n_attempted} attempted, {n_received} responded, "
                    f"{n_contributed} contributed claims). "
                    "Bar height is the sum of finding sensitivities across the "
                    "full response corpus."
                ),
                "chart_captions": {
                    "findings_by_llm": (
                        "Each bar is the sum of finding sensitivities per model."
                    ),
                    "exposure_radar": (
                        "Mean metrics across every extracted claim"
                    ),
                },
            },
            "findings": {
                "title": (
                    "1 finding that carries this exposure"
                    if n_abridged == 1
                    else f"{n_abridged} findings that carry this exposure"
                    if n_abridged
                    else "Findings that carry this exposure"
                ),
                "body": "",
            },
            "evidence": {
                "title": "Verbatim excerpts with line numbers",
                "body": "",
            },
            "model_comparison": {
                "title": "Where the models validate and surface the same information, and where they don't",
                "body": contrast.get("lede") or "",
                "heatmap_title": "Sensitivity by model and claim",
            },
            "appendix": {
                "claims_title": "Cluster index",
                "claims_body": "",
                "corpus_title": "Normalized responses (every model × probe)",
                "corpus_lede": (
                    "Verbatim normalized answers after localization and after "
                    "stripping hidden reasoning. Provider payloads are in "
                    "provider_responses.jsonl (auth headers removed)."
                ),
            },
            "inventory": {
                "title": "Every cluster, ranked",
            },
            "sources": {
                "title": "What the models cited",
                "lede": (
                    "Sources the model answers named, numbered once for the "
                    "run. Findings point here as S1, S2, and so on. A citation "
                    "records what a model named; it is not an endorsement."
                ),
                "empty": (
                    "No model answer in this run cited an external source, so "
                    "findings are attributed to the model that produced them."
                ),
            },
            "glossary": {
                "title": "How to read the scores",
                "lede": (
                    "Identifiers and score dimensions used in this report."
                ),
            },
            "next_steps": {
                "title": "This snapshot is scouting  for AI assertions, not a record of truths",
                "body": "",
            },
        },
        "top_finding": top,
        "findings": findings,
        "abridged_findings": abridged,
        "evidence_findings": evidence_findings,
        "basis": basis_section,
        "next_steps": build_next_steps(include_remediation=include_remediation),
        "specific_findings": specific_findings,
        "onepage_more": onepage_more,
        "sources": sources,
        "glossary": glossary_groups(),
        "what_else": [plain_text(w) for w in (report_data.get("what_else") or [])],
        "model_exposure": report_data.get("model_exposure") or [],
        "findings_by_llm": report_data.get("findings_by_llm") or [],
        "response_corpus": report_data.get("response_corpus") or [],
        "model_contrast": contrast,
        "chains": report_data.get("chains") or [],
        "followups": followups,
        "radar_averages": report_data.get("radar_averages") or {},
        "sensitivity_bins": report_data.get("sensitivity_bins") or {},
        "assets": {
            "company_logo": "assets/company-logo.svg",
            **{k: f"assets/{v}" for k, v in ASSET_NAMES.items()},
        },
    }


def render_report_md(content: dict[str, Any]) -> str:
    """Human-editable markdown mirror of report.yaml (content only)."""
    meta = content["meta"]
    pages = content["pages"]
    lines = [
        f"# {meta.get('headline') or meta.get('topic')}",
        "",
        f"_Run `{meta.get('run_id')}` · {meta.get('report_date')}_",
        "",
        "## What the models disclosed",
        "",
        pages["executive_summary"]["body"],
        "",
    ]
    exec_page = pages["executive_summary"]
    if exec_page.get("pull_quote"):
        lines += [f"> {exec_page['pull_quote']}", ""]
    if exec_page.get("public_sources"):
        lines += ["### Public sources", ""]
        for src in exec_page["public_sources"]:
            lines.append(f"- {src}")
        lines.append("")
    if exec_page.get("inference_chain"):
        lines += ["### Inference chain", ""]
        for i, step in enumerate(exec_page["inference_chain"], 1):
            lines.append(f"{i}. {step}")
        lines.append("")
    if exec_page.get("confidence_label"):
        lines += [
            f"**Confidence:** {exec_page['confidence_label']}"
            + (
                f" — {exec_page['confidence_rationale']}"
                if exec_page.get("confidence_rationale")
                else ""
            ),
            "",
        ]
    if exec_page.get("why_it_matters"):
        lines += ["### Why it matters", "", exec_page["why_it_matters"], ""]
    if exec_page.get("defensive_action"):
        lines += ["### Recommended defensive action", "", exec_page["defensive_action"], ""]
    if exec_page.get("exposure_teaser"):
        lines += ["### Exposure chain teaser", "", exec_page["exposure_teaser"], ""]

    lines += [
        "## Which models disclosed the most",
        "",
        pages["risk_overview"]["body"],
        "",
        f"- High: {meta['counts'].get('high_sensitivity', 0)}",
        f"- Contested: {meta['counts'].get('contested', 0)}",
        f"- Outliers: {meta['counts'].get('outliers', 0)}",
        "",
        "## Where the models validate and surface the same information, and where they don't",
        "",
        (pages.get("model_comparison") or {}).get("body")
        or (content.get("model_contrast") or {}).get("lede")
        or "",
        "",
    ]
    contrast = content.get("model_contrast") or {}
    common_lines = contrast.get("commonality_prose") or []
    if not common_lines:
        common_lines = [
            f"{row.get('claim_id')} {row.get('claim')} "
            f"({', '.join(row.get('models') or [])})".strip()
            for row in (contrast.get("commonality") or [])
        ]
    diff_lines = contrast.get("differences_prose") or []
    if not diff_lines:
        diff_lines = [
            f"{row.get('claim_id')} {row.get('claim')} "
            f"({', '.join(row.get('models') or [])})".strip()
            for row in (contrast.get("differences") or [])
        ]
    if common_lines:
        lines += ["### Commonality", ""]
        for line in common_lines:
            lines.append(f"- {line}")
        lines.append("")
    if diff_lines:
        lines += ["### Differences", ""]
        for line in diff_lines:
            lines.append(f"- {line}")
        lines.append("")
    lines += [
        "## Findings that carry this exposure",
        "",
        pages["findings"]["body"],
        "",
    ]
    for f in content.get("abridged_findings") or content.get("findings") or []:
        sev = _severity_label(int(f.get("sensitivity") or 0))
        source = f.get("source_cite") or f.get("source_model")
        refs = ", ".join(f.get("source_refs") or [])
        lines.append(
            f"- **{f.get('claim_id')}** [{f.get('status')}/{sev}] "
            f"{f.get('claim')} — _{source}_"
            + (f" ({refs})" if refs else "")
        )
    if content.get("sources"):
        lines += ["", "## Sources and citations", ""]
        for src in content["sources"]:
            url = f" — {src['url']}" if src.get("url") else ""
            lines.append(f"- **{src.get('ref')}** {src.get('label')}{url}")
    if content.get("meta", {}).get("include_remediation") and content.get("followups"):
        lines += ["", "## Remediation", ""]
        for item in content.get("followups") or []:
            ids = ", ".join(item.get("claim_ids") or [])
            lines.append(f"- **{item.get('method')}** — {item.get('action')} ({ids})")
        lines.append("")
    else:
        lines.append("")

    snap_ns = (content.get("next_steps") or {}).get("snapshot") or {}
    if snap_ns.get("items"):
        lines += [
            f"## {snap_ns.get('title') or 'Where this assessment should go next'}",
            "",
            snap_ns.get("lede") or "",
            "",
        ]
        for item in snap_ns["items"]:
            lines.append(f"- **{item.get('title')}** — {item.get('body')}")
        lines.append("")

    corpus = content.get("response_corpus") or []
    if corpus:
        pages_app = (content.get("pages") or {}).get("appendix") or {}
        lines += [
            f"## {pages_app.get('corpus_title') or 'Normalized responses (every model × probe)'}",
            "",
            pages_app.get("corpus_lede") or "",
            "",
        ]
        for row in corpus:
            qid = row.get("query_id") or ""
            strat = f" [{row.get('strategy')}]" if row.get("strategy") else ""
            lang = f" ({row.get('language')})" if row.get("language") else ""
            lines.append(f"### {row.get('model')}{lang} — {qid}{strat}")
            lines.append("")
            lines.append(f"_{row.get('query') or ''}_")
            lines.append("")
            if row.get("failed"):
                lines.append("> Retrieval failed or returned no usable content.")
            else:
                body = (row.get("text") or "").strip() or "> (no content returned)"
                lines.append(body)
            lines.append("")

    return "\n".join(lines)


def write_content_package(
    report_data: dict[str, Any],
    run_dir: Path,
    *,
    report_date: str,
    logo_src: Path | None = None,
    partner_logo_src: Path | None = None,
    graphics_svgs: dict[str, str] | None = None,
    aliases: dict[str, str] | None = None,
    overwrite_graphics: bool = True,
    keep_content: bool = False,
    isvf_path: Path | None = None,
    include_remediation: bool = False,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write report.md, report.yaml, and SVG assets under run_dir.

    Layout::

        <run_dir>/
          report.md
          report.yaml
          assets/*.svg

    When ``overwrite_graphics`` is false, existing ``assets/*.svg`` are left
    untouched (use after hand-editing charts before PDF rebuild).
    When ``keep_content`` is true and ``report.yaml`` already exists, reuse it
    (and ``report.md``) instead of regenerating from ``report_data.json``.
    """
    yaml_path = run_dir / "report.yaml"
    md_path = run_dir / "report.md"
    if keep_content and yaml_path.is_file():
        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        content = loaded if isinstance(loaded, dict) else {}
    else:
        content = build_content_doc(
            report_data,
            report_date=report_date,
            aliases=aliases,
            isvf_path=isvf_path,
            include_remediation=include_remediation,
            llm_config=llm_config,
        )
    assets_dir = run_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "screenshots").mkdir(exist_ok=True)

    if logo_src and logo_src.exists():
        suffix = logo_src.suffix.lower() or ".png"
        if suffix == ".svg":
            (assets_dir / "company-logo.svg").write_bytes(logo_src.read_bytes())
        else:
            (assets_dir / f"company-logo{suffix}").write_bytes(logo_src.read_bytes())
            # Stable name used by templates
            (assets_dir / "company-logo.png").write_bytes(logo_src.read_bytes())

    if partner_logo_src and partner_logo_src.exists():
        partner_dest = assets_dir / f"partner-logo{partner_logo_src.suffix.lower()}"
        partner_dest.write_bytes(partner_logo_src.read_bytes())
        # Prefer stable PNG name for templates
        if partner_dest.name != "partner-logo.png":
            pngish = assets_dir / "partner-logo.png"
            pngish.write_bytes(partner_logo_src.read_bytes())

    if overwrite_graphics:
        graphics_svgs = graphics_svgs or {}
        for key, filename in ASSET_NAMES.items():
            svg = graphics_svgs.get(key)
            if svg:
                (assets_dir / filename).write_text(svg, encoding="utf-8")

    if not (keep_content and yaml_path.is_file()):
        yaml_path.write_text(
            yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        md_path.write_text(render_report_md(content), encoding="utf-8")
    return content
