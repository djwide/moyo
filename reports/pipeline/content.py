"""Write per-run content package: report.md + report.yaml (LLM-editable).

Presentation stays in ``reports/design-system/``. Charts land in ``assets/`` as SVG.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from graphics.style import format_source_cite, short_model_name
from pipeline.graphics import ASSET_NAMES
from pipeline.synthesize import parse_executive_payload
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
                "This snapshot is abridged. The Basis Report delivers the "
                "complete prioritized exposure inventory, full findings with "
                "severity rationale and confidence, verbatim evidence excerpts, "
                "corroborating model outputs, derivation of how MOYO reached "
                "each conclusion, cited real-world sources, and exploitation "
                "implications."
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
            "title": "Next Steps",
            "lede": (
                "This Exposure Snapshot is an abridged scout. The steps below "
                "point to the Basis Report depth omitted here, and to denser "
                "use of MOYO."
            ),
            "items": snapshot_items,
        },
        "basis": {
            "title": "Next Steps",
            "lede": (
                "You now have the complete exposure basis. Escalate to "
                "adversarial testing and denser prompting so remaining "
                "reachability risk is measured, not assumed."
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
    translate_fn = default_translate_fn() if translate else None
    normalized = englishize_findings(findings, translate=translate_fn)
    out: list[dict[str, Any]] = []
    for f in normalized:
        row = dict(f)
        peers = peers_by_cluster.get(row.get("cluster_id"))
        row["claim"] = plain_text(row.get("claim"))
        row["category"] = plain_text(row.get("category")) or "unclassified"
        row["source_short"] = short_model_name(row.get("source_model") or "", aliases)
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
        return [str(p).strip() for p in raw if str(p).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    topic = (report_data.get("topic") or "").strip()
    return [topic] if topic else []


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
        teaser = (
            "Exposure chain teaser: "
            + " → ".join(str(s) for s in exposure_steps[:3])
        )
    elif not teaser and inference:
        teaser = "Exposure chain teaser: " + " → ".join(inference[:3])

    defensive = ""
    if include_remediation:
        defensive = (fields.get("defensive_action") or "").strip()
        if not defensive:
            followups = report_data.get("followups") or []
            if followups:
                defensive = str(followups[0].get("action") or "").strip()

    why = (fields.get("why_it_matters") or "").strip()
    if not why and pull:
        why = (
            "This disclosure is concrete enough for adversaries or reporters to "
            "act on without further invention, raising reputational and compliance risk."
        )

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
    }


def build_content_doc(
    report_data: dict[str, Any],
    *,
    report_date: str,
    aliases: dict[str, str] | None = None,
    isvf_path: Path | None = None,
    include_remediation: bool = False,
) -> dict[str, Any]:
    """Structured content document consumed by design-system templates."""
    counts = report_data.get("counts") or {}
    top = report_data.get("top_finding") or {}
    findings = _enrich_findings(
        list(report_data.get("findings") or []),
        list(report_data.get("clusters") or []),
        aliases=aliases,
    )
    # Real-world citations extracted from exploration.md, numbered once per run
    # (S1, S2, …) so every product cites the same registry.
    sources, findings = build_source_registry(findings)
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

    # High-specificity findings for the one-pager "Specific" panel (bottom left).
    # Cap at 2 so the snapshot stays a single page. Prefer English claim bodies.
    specific_min = 4
    english_findings = [
        f
        for f in findings
        if looks_like_english(str(f.get("claim") or "")) and not f.get("english_pending")
    ] or [
        f for f in findings if looks_like_english(str(f.get("claim") or ""))
    ] or findings
    specific_findings = [
        f
        for f in english_findings
        if int(f.get("specificity") or 0) >= specific_min
        and f.get("claim_id") != (top.get("claim_id") or "")
    ][:2]
    if not specific_findings:
        specific_findings = [
            f
            for f in english_findings
            if f.get("claim_id") != (top.get("claim_id") or "")
        ][:2]

    abridged = english_findings[:5]

    explore_meta = report_data.get("explore_meta") or {}
    models_tested = list(explore_meta.get("models_tested") or [])
    if not models_tested:
        models_tested = [
            (m.get("model") or "").strip()
            for m in (report_data.get("model_exposure") or [])
            if (m.get("model") or "").strip()
        ]
    strategies = list(explore_meta.get("strategies") or [])
    if not strategies:
        strategies = ["paraphrase", "translate", "summarize"]

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
        report_data, findings, remediation=remediation
    )
    followups = [
        {
            **item,
            "method": plain_text(item.get("method")),
            "action": strip_markdown(item.get("action")),
        }
        for item in followups
    ]

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
            "counts": {
                "findings": counts.get("findings", len(findings)),
                "llms_tested": counts.get("llms_tested", 0),
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
                "body": (
                    "Severity distribution and multi-axis exposure scores "
                    "summarize how models disclosed sensitive or specific material."
                ),
                "chart_captions": {
                    "sensitivity_distribution": (
                        "Sensitivity distribution: how many findings fall into "
                        "high, medium, low, and informational sensitivity bands."
                    ),
                    "exposure_radar": (
                        "Exposure radar: average specificity, sensitivity, "
                        "corroboration, novelty, and confidence across extracted claims."
                    ),
                },
            },
            "findings": {
                "title": "Abridged Findings",
                "body": "",
            },
            "evidence": {
                "title": "Abridged Evidence",
                "body": "",
            },
            "model_comparison": {
                "title": "Model Comparison",
                "body": "",
            },
            "appendix": {
                "claims_title": "Abridged Claims",
                "claims_body": "",
            },
            "sources": {
                "title": "Sources and Citations",
                "lede": (
                    "Real-world sources the model answers cited, numbered once "
                    "for the whole run. Findings reference them as S1, S2, and "
                    "so on. Presence here records what a model cited; it is not "
                    "an endorsement of the source."
                ),
                "empty": (
                    "No model answer in this run cited an external source, so "
                    "findings are attributed to the model that produced them."
                ),
            },
            "glossary": {
                "title": "Glossary",
                "lede": (
                    "Terms, identifiers, and score dimensions used throughout "
                    "this report."
                ),
            },
            "next_steps": {
                "title": "Next Steps",
                "body": "",
            },
        },
        "top_finding": top,
        "findings": findings,
        "abridged_findings": abridged,
        "basis": basis_section,
        "next_steps": build_next_steps(include_remediation=include_remediation),
        "specific_findings": specific_findings,
        "sources": sources,
        "glossary": glossary_groups(),
        "what_else": [plain_text(w) for w in (report_data.get("what_else") or [])],
        "model_exposure": report_data.get("model_exposure") or [],
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
        "## Executive summary",
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
        "## Risk overview",
        "",
        pages["risk_overview"]["body"],
        "",
        f"- High: {meta['counts'].get('high_sensitivity', 0)}",
        f"- Contested: {meta['counts'].get('contested', 0)}",
        f"- Outliers: {meta['counts'].get('outliers', 0)}",
        "",
        "## Abridged Findings",
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
        lines += ["## Next Steps", "", snap_ns.get("lede") or "", ""]
        for item in snap_ns["items"]:
            lines.append(f"- **{item.get('title')}** — {item.get('body')}")
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
    isvf_path: Path | None = None,
    include_remediation: bool = False,
) -> dict[str, Any]:
    """Write report.md, report.yaml, and SVG assets under run_dir.

    Layout::

        <run_dir>/
          report.md
          report.yaml
          assets/*.svg

    When ``overwrite_graphics`` is false, existing ``assets/*.svg`` are left
    untouched (use after hand-editing charts before PDF rebuild).
    """
    content = build_content_doc(
        report_data,
        report_date=report_date,
        aliases=aliases,
        isvf_path=isvf_path,
        include_remediation=include_remediation,
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

    (run_dir / "report.yaml").write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(render_report_md(content), encoding="utf-8")
    return content
