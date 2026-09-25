"""Write per-run content package: report.md + report.yaml (LLM-editable).

Presentation stays in ``reports/design-system/``. Charts land in ``assets/`` as SVG.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from graphics.style import format_source_cite, full_model_name, short_model_name
from .cluster import dedupe_findings_by_group, present_id
from pipeline.graphics import ASSET_NAMES
from pipeline.model_dossiers import build_model_dossiers
from pipeline.synthesize import parse_executive_payload
from pipeline.audience import (
    OPPOSITION,
    disclosure_class as audience_disclosure_class,
    normalize_audience,
    profile as audience_profile,
    retain_finding,
    sort_key,
)
from .provenance import (
    claim_tokens,
    evidence_status,
    has_source_url,
    presentation_rows,
    provenance_label,
    research_significance,
)
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
from pipeline.overview import (
    EVIDENCE_ORDER,
    evidence_quality,
    exposure_overview,
    finding_models,
    model_count,
    number_word,
    reproduction,
    verification_plan,
)
from pipeline.sources import build_source_registry
from pipeline.textclean import plain_text, strip_markdown


def group_response_corpus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve first-seen model order while grouping probes under each model."""
    groups: list[dict[str, Any]] = []
    index: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model = str(row.get("model") or "Unknown")
        items = index.get(model)
        if items is None:
            items = []
            index[model] = items
            groups.append({"model": model, "items": items})
        items.append(row)
    return groups


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
                "follow-ups on high-significance claims, and broader model "
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
                "whether high-significance conclusions remain reachable under "
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
        {
            "title": "Verify organizational policy",
            "body": (
                "Use the Idea Security Verification Framework as a "
                "follow-up to verify proper organizational information "
                "security policy: whether permitted joins, Unreachable "
                "Statement Classes, and domain-boundary controls actually "
                "block the conclusions recovered in this run."
            ),
        },
    ]

    return {
        "snapshot": {
            "title": "Recommended Next",
            "lede": (
                "The abridged product stops at the top findings. Use the "
                "Basis Report for the rest of the inventory, then re-prompt "
                "on the high-significance claims."
            ),
            "items": snapshot_items,
        },
        "basis": {
            "title": "Follow-Up Plan",
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


def _language_phrase(languages: list[str]) -> str:
    names = [str(name).strip() for name in languages if str(name).strip()]
    if not names:
        names = ["English"]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} + {names[1]}"
    return ", ".join(names[:-1]) + " + " + names[-1]


def _extracted_lead_count(findings: list[dict[str, Any]]) -> int:
    total = 0
    for finding in findings:
        members = finding.get("merged_from")
        if isinstance(members, list) and members:
            total += len(members)
        else:
            total += 1
    return total or len(findings)


def _model_count(finding: dict[str, Any]) -> int:
    try:
        return max(1, int(finding.get("corroboration") or 1))
    except (TypeError, ValueError):
        return 1


def _why_it_matters(finding: dict[str, Any]) -> str:
    category = plain_text(finding.get("category") or "this topic").replace("_", " ")
    return f"{research_significance(finding)} significance. Category: {category}."


def _next_verification(finding: dict[str, Any]) -> str:
    sourced = has_source_url(finding)
    if not sourced:
        return "Locate a primary source before treating this as a fact."
    if _model_count(finding) < 2:
        return "Check the cited source and look for a second independent record."
    return "Confirm the citation states the claim, not only the topic."


def _model_labels(finding: dict[str, Any], aliases: dict[str, str]) -> str:
    raw = finding.get("source_models")
    names = [str(name).strip() for name in raw if str(name).strip()] if isinstance(raw, list) else []
    if not names:
        one = str(finding.get("source_model") or "").strip()
        names = [one] if one else []
    labels: list[str] = []
    seen: set[str] = set()
    for name in names:
        label = short_model_name(name, aliases) or name.split("(")[0].strip()
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return ", ".join(labels)


_QUOTE_RE = re.compile(r"(?:^|\s)['\"“‘]([^'\"”’]{6,60})['\"”’]")


def _quoted_phrases(text: str) -> set[str]:
    """Quoted multi-word phrases: two wordings of one quote are one lead."""
    out: set[str] = set()
    for match in _QUOTE_RE.findall(text or ""):
        phrase = " ".join(re.findall(r"\w+", match.lower()))
        if len(phrase.split()) >= 2:
            out.add(phrase)
    return out


def _brief_rows(
    findings: list[dict[str, Any]],
    aliases: dict[str, str],
    *,
    limit: int = 5,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: list[tuple[set[str], set[str]]] = []
    for finding in findings:
        if len(rows) >= limit:
            break
        claim = plain_text(finding.get("claim") or "").strip()
        if not claim:
            continue
        tokens = claim_tokens(claim)
        quotes = _quoted_phrases(claim)
        if any(
            (quotes & prior_quotes)
            or (
                len(tokens & prior) >= 4
                and len(tokens & prior) / max(1, min(len(tokens), len(prior))) >= 0.4
            )
            for prior, prior_quotes in seen
        ):
            continue
        seen.append((tokens, quotes))
        rows.append(
            {
                "finding": claim,
                "why": _why_it_matters(finding),
                "evidence": evidence_status(finding),
                "models": _model_labels(finding, aliases),
                "next_step": _next_verification(finding),
            }
        )
    return rows


def _brief_package(
    findings: list[dict[str, Any]],
    *,
    models: int,
    languages: list[str],
    lead_text: str,
    aliases: dict[str, str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    leads = _extracted_lead_count(findings)
    corroborated = sum(1 for finding in findings if _model_count(finding) >= 2)
    high = sum(1 for finding in findings if int(finding.get("sensitivity") or 0) >= 4)
    verified = sum(1 for finding in findings if has_source_url(finding))
    pending = max(0, len(findings) - verified)
    lang_names = [str(name).strip() for name in languages if str(name).strip()] or ["English"]
    phrase = _language_phrase(lang_names)
    scan_line = (
        f"{models} models | {phrase} prompting | {leads} extracted leads"
    )
    material = (lead_text or "").strip()
    if material:
        material = (
            "The lead the models ranked highest, still a model output: "
            + material
        )
    else:
        material = "No lead finding was ranked for this run."
    brief_rows = _brief_rows(rows, aliases)
    n_rows = len(brief_rows)
    rows_verified = sum(1 for row in brief_rows if row["evidence"] == "Externally verified")
    rows_high = sum(1 for row in brief_rows if row["why"].startswith("High"))
    if not n_rows:
        leads_takeaway = "No lead in this run cites a source yet."
    elif rows_verified == n_rows:
        leads_takeaway = (
            f"All {number_word(n_rows)} top leads are externally verified"
            + (" and high significance." if rows_high == n_rows else ".")
        )
    else:
        leads_takeaway = (
            f"{number_word(rows_verified).capitalize()} of the {number_word(n_rows)} "
            "top leads are externally verified."
        )
    return {
        "scan_line": scan_line,
        "takeaway": (
            f"{number_word(models).capitalize()} models surfaced {len(findings)} "
            f"distinct findings. {high} are high significance; {verified} cite "
            "an external source."
        ),
        "leads_takeaway": leads_takeaway,
        "kpis": [
            {"value": leads, "label": "Extracted leads"},
            {"value": corroborated, "label": "Cross-model corroborated"},
            {"value": models, "label": "Models tested"},
            {"value": len(lang_names), "label": "Prompting languages"},
        ],
        "secondary": [
            {"value": high, "label": "High significance"},
            {"value": verified, "label": "Externally verified"},
            {"value": pending, "label": "Verification pending"},
        ],
        "blocks": [
            {
                "title": "What the models surfaced",
                "body": (
                    f"{models} models were prompted in {phrase} and returned "
                    f"{leads} extracted leads. {corroborated} findings are "
                    "stated by more than one model."
                ),
            },
            {
                "title": "What appears most material",
                "body": material,
            },
            {
                "title": "What requires verification",
                "body": (
                    f"{pending} findings are not externally verified. "
                    f"{high} are high significance; that label is not an "
                    "evidence status. Start with externally verified, "
                    "cross-model leads, then check the rest against the record."
                ),
            },
        ],
        "rows": brief_rows,
    }


_SIGNIFICANCE_RANK = {"High": 0, "Medium": 1, "Low": 2}


def _claim_inventory(
    findings: list[dict[str, Any]],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    """Every distinct finding, ordered by significance, then evidence status."""
    rows = [
        {
            "id": present_id(f) or f.get("cluster_id") or f.get("claim_id"),
            "claim": plain_text(f.get("claim") or ""),
            "evidence_status": f.get("evidence_status") or evidence_status(f),
            "significance": f.get("significance") or research_significance(f),
            "models": ", ".join(finding_models(f, aliases)),
            "n_models": model_count(f, aliases),
            "source_refs": list(f.get("source_refs") or []),
        }
        for f in findings
    ]
    rows.sort(
        key=lambda row: (
            _SIGNIFICANCE_RANK.get(row["significance"], 3),
            EVIDENCE_ORDER.index(row["evidence_status"])
            if row["evidence_status"] in EVIDENCE_ORDER
            else len(EVIDENCE_ORDER),
            -row["n_models"],
        )
    )
    return rows


def _methodology_page(
    *,
    models_tested: list[str],
    languages: list[str],
    strategies: list[str],
    n_attempted: int,
    n_substantive: int,
    leads: int,
    distinct: int,
) -> dict[str, Any]:
    missing = max(0, n_attempted - n_substantive)
    limitations = [
        "Models repeat each other. Cross-model agreement raises confidence "
        "that a claim is widely stated, not that it is true.",
        "A source URL means the model named a source. It does not mean the "
        "source says what the model says. Verification is still required.",
        "Research significance is a rubric score assigned during extraction. "
        "It describes stakes if true, not likelihood.",
        "Results reflect these prompts, these models, and this date. Other "
        "prompts or later model versions can surface different material.",
    ]
    if missing:
        limitations.append(
            f"{missing} of {n_attempted} models gave no substantive answer. "
            "The findings come from the rest."
        )
    return {
        "title": "Methodology and Limitations",
        "takeaway": (
            "This is a map of what AI models say about the subject, "
            "not a fact-check."
        ),
        "facts": [
            {"label": "Models queried", "value": ", ".join(models_tested) or "—"},
            {"label": "Prompting languages", "value": _language_phrase(languages)},
            {
                "label": "Prompt techniques",
                "value": ", ".join(s.replace("_", " ").capitalize() for s in strategies) or "—",
            },
            {
                "label": "Coverage",
                "value": f"{n_substantive} of {n_attempted} models gave a substantive answer",
            },
            {
                "label": "Findings",
                "value": f"{leads} extracted leads, clustered into {distinct} distinct findings",
            },
        ],
        "steps": [
            "Each model received the same investigation prompts, in every "
            "prompting language, in original and reworded form.",
            "Every answer was split into individual claims. Claims that say "
            "the same thing were clustered into one finding.",
            "Each finding got one evidence status (externally verified, "
            "cross-model corroborated, single-model lead, or contested) and "
            "one research significance (high, medium, or low).",
            "Findings were ranked by significance and evidence. The core "
            "report shows the leads; the appendix lists every finding.",
        ],
        "limitations": limitations,
    }


def _confidence_label(score: int) -> str:
    if score >= 4:
        return "High"
    if score == 3:
        return "Medium"
    return "Low"


def _tested_model_labels(
    raw_names: list[str],
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """Exact model labels for covers: vendor/id kept, language suffix dropped.

    Dedupes by short alias so ``ChatGPT (OpenAI gpt-4o) (French)`` does not
    appear twice, but the printed name stays ``ChatGPT (OpenAI gpt-4o)``.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for raw in raw_names:
        text = str(raw or "").strip()
        if not text:
            continue
        key = short_model_name(text, aliases)
        if not key or key == "unknown" or key in seen:
            continue
        seen.add(key)
        labels.append(full_model_name(text))
    return labels


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


def _looks_like_scan_wrap(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered.startswith("what do ai systems already know") or lowered.startswith(
        "what personal, biographical"
    )


def _display_topic(report_data: dict[str, Any]) -> str:
    """Customer-facing topic: never the reconstructed retrieval wrap."""
    display = plain_text(report_data.get("display_topic") or "").strip()
    if display:
        return display
    topic = plain_text(report_data.get("topic") or "").strip()
    if topic and not _looks_like_scan_wrap(topic):
        return topic
    for candidate in _prompts_list(report_data):
        text = plain_text(candidate).strip()
        if text and not _looks_like_scan_wrap(text):
            return text
    return ""


def _customer_prompts(report_data: dict[str, Any]) -> list[str]:
    display = _display_topic(report_data)
    if display:
        return [display]
    out: list[str] = []
    for candidate in _prompts_list(report_data):
        text = plain_text(candidate).strip()
        if text and not _looks_like_scan_wrap(text):
            out.append(text)
    return out


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

    # Only sources that include a URL. A guessed outlet name is not a citation.
    public_sources = []
    for source in list(sources or []):
        url = str(source.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        label = plain_text(source.get("label") or "").strip()
        public_sources.append(f"{label} — {url}" if label else url)
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
        "title": "Disclosure Summary",
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
    voice = normalize_audience(report_data.get("audience"))
    voice_profile = audience_profile(voice)
    top = report_data.get("top_finding") or {}
    findings_enriched = _enrich_findings(
        [
            f
            for f in (report_data.get("findings_all") or report_data.get("findings") or [])
            if retain_finding(f, voice)
        ],
        list(report_data.get("clusters") or []),
        aliases=aliases,
        llm_config=llm_config,
    )
    sources, findings_enriched = build_source_registry(findings_enriched)
    # Snapshot lists collapsed groups; basis narrative can use every claim.
    findings = dedupe_findings_by_group(findings_enriched)
    findings.sort(key=lambda f: sort_key(f, voice))
    for row in findings:
        row["provenance"] = provenance_label(row, voice)
        row["evidence_status"] = evidence_status(row)
        row["significance"] = research_significance(row)
    shown, parked = presentation_rows(findings, voice)
    top = _sync_top_finding_english(top, shown or findings)
    shown_ids = {row.get("claim_id") for row in shown}
    if top.get("claim_id") not in shown_ids:
        if shown:
            lead = shown[0]
            top = {
                "claim_id": lead.get("claim_id"),
                "text": plain_text(lead.get("claim")),
                "badges": [],
                "language_annotation": lead.get("language_annotation") or "",
                "prompt_language": lead.get("prompt_language") or lead.get("language"),
                "provenance": lead.get("provenance") or "",
                "evidence_status": lead.get("evidence_status") or "",
                "significance": lead.get("significance") or "",
            }
        else:
            top = {
                "claim_id": "",
                "text": (
                    "No sourced finding in this run is corroborated enough to lead."
                ),
                "badges": [],
                "provenance": "",
            }
    elif shown:
        match = next(
            (row for row in shown if row.get("claim_id") == top.get("claim_id")),
            None,
        )
        if match:
            top["provenance"] = match.get("provenance") or ""
            top["evidence_status"] = match.get("evidence_status") or ""
            top["significance"] = match.get("significance") or ""
            top["prompt_language"] = match.get("prompt_language") or top.get("prompt_language")
    pull = plain_text(top.get("text") or "")
    display_topic = _display_topic(report_data)
    # Customer-facing templates never show reconstructed retrieval wraps.
    prompts = _customer_prompts(report_data)
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

    # One row per episode. Unsourced and single-model damaging claims stay
    # off the priority list. Opposition ranks them in a separate one-pager section.
    specific_cap = 4
    snapshot_cap = 12
    onepage_more_cap = 6
    top_id = top.get("claim_id") or ""

    def _english_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        english = [
            row
            for row in rows
            if looks_like_english(str(row.get("claim") or ""))
            and not row.get("english_pending")
        ]
        return english or list(rows)

    page_rows = _english_rows(shown)
    top["badges"] = []
    # Opposition one-pager ranks verified and needs-verification separately.
    onepage_verified = page_rows[:4] if voice == OPPOSITION else []
    onepage_needs_verification = (
        _english_rows(parked)[:4] if voice == OPPOSITION else []
    )
    specific_findings = [row for row in page_rows if row.get("claim_id") != top_id][
        :specific_cap
    ]
    abridged = page_rows[:snapshot_cap]
    evidence_findings = [row for row in page_rows if row.get("raw_excerpt")][:3]
    used_ids = {top_id} | {row.get("claim_id") for row in specific_findings}
    onepage_more = [
        row for row in page_rows if row.get("claim_id") not in used_ids
    ][:onepage_more_cap]
    reserve_label = {
        "competitive": "Commercially sensitive",
        "security": "Security relevant",
    }.get(voice)
    if reserve_label:
        placed = used_ids | {row.get("claim_id") for row in onepage_more} | {
            row.get("claim_id") for row in abridged
        }
        missing = [
            row
            for row in page_rows
            if audience_disclosure_class(row, voice) == reserve_label
            and row.get("claim_id") not in placed
        ]
        if missing:
            extra = missing[:2]
            onepage_more = list(onepage_more) + extra
            abridged = list(abridged) + [row for row in extra if row not in abridged]

    explore_meta = report_data.get("explore_meta") or {}
    alias_map = aliases or {}
    models_tested = _tested_model_labels(
        list(explore_meta.get("models_tested") or []),
        alias_map,
    )
    if not models_tested:
        models_tested = _tested_model_labels(
            [
                str(m.get("model") or "")
                for m in (report_data.get("model_exposure") or [])
            ],
            alias_map,
        )
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
    if voice != "organization" and (
        not headline or headline.lower().startswith("what ai systems reveal")
    ):
        headline = voice_profile["headline"]
    elif not headline or headline.lower().startswith("what ai systems reveal about"):
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
        audience=voice,
    )
    followups = [
        {
            **item,
            "method": plain_text(item.get("method")),
            "action": strip_markdown(item.get("action")),
        }
        for item in followups
    ]

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
    exec_page["title"] = "Executive Summary"
    brief = _brief_package(
        findings,
        models=n_models,
        languages=prompt_languages,
        lead_text=pull,
        aliases=alias_map if isinstance(alias_map, dict) else {},
        rows=page_rows,
    )
    exec_page["blocks"] = brief["blocks"]
    exec_page["takeaway"] = brief["takeaway"]

    alias_dict = alias_map if isinstance(alias_map, dict) else {}
    roster = list(explore_meta.get("models_tested") or []) or None
    overview = exposure_overview(findings, alias_dict, roster)
    comparison = reproduction(findings, alias_dict, roster)
    evidence_summary = evidence_quality(findings, sources)
    plan = verification_plan(findings, alias_dict)
    basis_extra: list[dict[str, Any]] = []
    if voice == "organization":
        basis_extra = build_next_steps(include_remediation=include_remediation)["basis"]["items"]
    elif voice == "security":
        basis_extra = [
            {
                "title": "Red-team the reachable conclusions",
                "body": (
                    "Pressure-test whether the security-relevant conclusions "
                    "remain reachable under hostile prompting, paraphrase, "
                    "and multi-step chaining."
                ),
            }
        ]
    next_steps = {
        "snapshot": plan,
        "basis": {**plan, "items": list(plan["items"]) + basis_extra},
    }
    inventory = _claim_inventory(findings, alias_dict)
    methodology = _methodology_page(
        models_tested=models_tested,
        languages=prompt_languages,
        strategies=strategies,
        n_attempted=n_attempted,
        n_substantive=n_substantive or n_models,
        leads=_extracted_lead_count(findings),
        distinct=len(findings),
    )

    return {
        "meta": {
            "run_id": report_data.get("run_id"),
            "topic": display_topic or report_data.get("topic"),
            "display_topic": display_topic or None,
            "subject_detail": (report_data.get("subject_detail") or "").strip() or None,
            "prompts": prompts,
            "headline": headline,
            "audience": voice,
            "kicker": voice_profile["kicker"],
            "basis_kicker": voice_profile["basis_kicker"],
            "eyebrow": voice_profile["eyebrow"],
            "priority_label": voice_profile["priority_label"],
            "generated_at": report_data.get("generated_at")
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "report_date": report_date,
            "fuzz_mode": explore_meta.get("fuzz_mode") or "basic",
            "strategies": strategies,
            "models_tested": models_tested,
            "languages": prompt_languages,
            "brief": brief,
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
                "title": "Exposure Overview",
                "takeaway": overview["takeaway"],
                "body": (
                    f"{n_substantive or n_models} of {n_attempted} models gave a "
                    f"substantive answer. {overview['detail']} Each model has "
                    "two bars: every finding it stated, and the "
                    "high-significance subset. Volume is not verification."
                ).replace("  ", " "),
                "chart_title": "Findings and high-significance findings by model",
            },
            "brief_findings": {
                "title": "Priority Leads",
                "takeaway": brief["leads_takeaway"],
                "body": (
                    "The highest-ranked leads. Each has one evidence status "
                    "and one research significance."
                ),
            },
            "evidence": {
                "title": "Evidence and Source Quality",
                "takeaway": evidence_summary["takeaway"],
                "body": (
                    "Evidence status is the conclusion; the excerpt is the "
                    "exhibit. These are the models' own words for the top leads."
                ),
            },
            "model_comparison": {
                "title": "Cross-Model Comparison",
                "takeaway": comparison["title"],
                "body": comparison["lede"],
                "chart_caption": (
                    "Each column is one "
                    + (
                        "high-significance finding"
                        if comparison["scope"] != "findings"
                        else "finding"
                    )
                    + ". A filled cell means that model stated it. "
                    "Numbered columns are described below the chart."
                ),
            },
            "methodology": methodology,
            "appendix": {
                "title": "Technical Appendix",
                "lede": (
                    "The database behind the core report: every finding, "
                    "each model's profile, every cited source, and the terms used."
                ),
                "claims_title": "Complete Claim Inventory",
                "claims_body": (
                    "Every distinct finding, ordered by research significance, "
                    "then evidence status. S-numbers point to cited sources."
                ),
                "method_title": "Collection Method",
                "method_body": (
                    "Multi-LLM fan-out against the naive prompt. Responses "
                    "are compiled from exploration.md with lossless chunking, "
                    "per-chunk extraction, clustering, and scored reporting. "
                    "Contested, outlier, sensitive, and single-source findings "
                    "are retained and classified — not dropped. Hidden "
                    "reasoning is omitted from the normalized text."
                ),
                "corpus_title": "Normalized Responses",
                "corpus_lede": (
                    "Every model × probe after localization. Provider payloads "
                    "are in provider_responses.jsonl (auth headers removed)."
                ),
            },
            "inventory": {
                "title": "Cluster Inventory",
            },
            "basis_findings": {
                "title": "Cluster Transcripts",
            },
            "exposure": {
                "title": "Exposure Chains",
            },
            "remediation": {
                "title": "Matching Controls",
            },
            "sources": {
                "title": "Cited Sources",
                "lede": (
                    "Sources named in a claim's own excerpt that include a "
                    "link. A headline without a URL is not listed. Findings "
                    "point here as S1, S2, and so on."
                ),
                "empty": (
                    "No claim in this run cited a source URL. Unsourced "
                    "claims are listed as unverified and attributed to the "
                    "model that produced them."
                ),
            },
            "glossary": {
                "title": "Glossary",
                "lede": (
                    "Identifiers, labels, and scores used in this report."
                ),
            },
            "model_dossiers": {
                "title": "Model Dossiers",
            },
        },
        "top_finding": top,
        "findings": findings,
        "abridged_findings": abridged,
        "unverified_findings": parked,
        "evidence_findings": evidence_findings,
        "overview": overview,
        "comparison": comparison,
        "evidence_summary": evidence_summary,
        "inventory": inventory,
        "basis": basis_section,
        "next_steps": next_steps,
        "specific_findings": specific_findings,
        "onepage_more": onepage_more,
        "onepage_verified": onepage_verified,
        "onepage_needs_verification": onepage_needs_verification,
        "sources": sources,
        "glossary": glossary_groups(voice),
        "what_else": [plain_text(w) for w in (report_data.get("what_else") or [])],
        "model_exposure": report_data.get("model_exposure") or [],
        "findings_by_llm": report_data.get("findings_by_llm") or [],
        "response_corpus": report_data.get("response_corpus") or [],
        "corpus_groups": group_response_corpus(
            list(report_data.get("response_corpus") or [])
        ),
        "model_dossiers": build_model_dossiers(
            findings,
            corpus=list(report_data.get("response_corpus") or []),
            sources=sources,
            radar_averages=report_data.get("radar_averages") or {},
            models_probed=list(
                explore_meta.get("models_tested") or models_tested or []
            ),
            aliases=alias_map,
            audience=voice,
        ),
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
        "# AI Exposure Assessment",
        "",
        str(meta.get("display_topic") or meta.get("topic") or ""),
        "",
        f"_Run `{meta.get('run_id')}` · {meta.get('report_date')}_",
        "",
        str((meta.get("brief") or {}).get("scan_line") or ""),
        "",
        f"## {pages['executive_summary'].get('title') or 'Executive Summary'}",
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
    if exec_page.get("blocks"):
        for block in exec_page["blocks"]:
            lines += [f"### {block.get('title') or ''}", "", block.get("body") or "", ""]
    if exec_page.get("why_it_matters"):
        lines += ["### Why it matters", "", exec_page["why_it_matters"], ""]
    if exec_page.get("defensive_action"):
        lines += ["### Recommended defensive action", "", exec_page["defensive_action"], ""]
    if exec_page.get("exposure_teaser"):
        lines += ["### Exposure chain teaser", "", exec_page["exposure_teaser"], ""]

    def _section(key: str, fallback: str) -> list[str]:
        page = pages.get(key) or {}
        out = [f"## {page.get('title') or fallback}", ""]
        if page.get("takeaway"):
            out += [f"**{page['takeaway']}**", ""]
        if page.get("body"):
            out += [page["body"], ""]
        return out

    lines += _section("risk_overview", "Exposure Overview")
    for row in (content.get("overview") or {}).get("rows") or []:
        lines.append(
            f"- {row['model']}: {row['findings']} findings, {row['high']} high "
            f"significance, {row['corroborated_pct']}% corroborated, "
            f"{row['verified']} externally verified"
        )
    lines.append("")
    lines += _section("brief_findings", "Priority Leads")
    for row in (meta.get("brief") or {}).get("rows") or []:
        lines.append(
            f"- {row['finding']} — {row['evidence']}; {row['why']} "
            f"Models: {row['models']}. Next: {row['next_step']}"
        )
    lines.append("")
    lines += _section("evidence", "Evidence and Source Quality")
    for step in (content.get("evidence_summary") or {}).get("ladder") or []:
        lines.append(f"- {step['label']}: {step['count']} ({step['pct']}%)")
    lines.append("")
    lines += _section("model_comparison", "Cross-Model Comparison")
    for note in (content.get("comparison") or {}).get("notes") or []:
        lines.append(f"{note['marker']}. {note['why']}. {note['label']}")
    lines.append("")

    plan = (content.get("next_steps") or {}).get("snapshot") or {}
    if plan.get("items"):
        lines += [f"## {plan.get('title') or 'Verification Plan'}", ""]
        if plan.get("takeaway"):
            lines += [f"**{plan['takeaway']}**", ""]
        for item in plan["items"]:
            lines.append(f"- **{item.get('title')}** — {item.get('body')}")
        lines.append("")

    method = pages.get("methodology") or {}
    if method:
        lines += _section("methodology", "Methodology and Limitations")
        for fact in method.get("facts") or []:
            lines.append(f"- {fact['label']}: {fact['value']}")
        lines.append("")
        for item in method.get("limitations") or []:
            lines.append(f"- {item}")
        lines.append("")

    inventory = content.get("inventory") or []
    if inventory:
        title = (pages.get("appendix") or {}).get("claims_title") or "Complete Claim Inventory"
        lines += [f"## {title}", ""]
        for row in inventory:
            refs = ", ".join(row.get("source_refs") or [])
            lines.append(
                f"- **{row['id']}** [{row['evidence_status']} / {row['significance']}] "
                f"{row['claim']} — _{row['models']}_" + (f" ({refs})" if refs else "")
            )
        lines.append("")
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
