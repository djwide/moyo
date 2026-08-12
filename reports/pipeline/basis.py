"""Assemble Basis Report content from scored run data.

The Basis Report is the comprehensive counterpart to the Exposure Snapshot. It
reuses the same pipeline artifacts (``report_data.json`` findings, clusters, and
exposure chains) and adds full, non-abridged detail plus ISVF-sourced
remediation. No re-extraction or extra LLM calls are required.
"""

from __future__ import annotations

from typing import Any

from .textclean import plain_text
from .cluster import dedupe_findings_by_group


def _severity_label(sensitivity: int) -> str:
    if sensitivity >= 4:
        return "high"
    if sensitivity == 3:
        return "medium"
    if sensitivity == 2:
        return "low"
    return "info"


def _rationale(f: dict[str, Any]) -> str:
    """Plain-English severity rationale from the scored dimensions."""
    sens = int(f.get("sensitivity") or 0)
    spec = int(f.get("specificity") or 0)
    nov = int(f.get("novelty") or 0)
    corr = int(f.get("corroboration") or 1)
    status = (f.get("status") or "UNVERIFIED").upper()
    sev = _severity_label(sens).capitalize()

    detail_bits = []
    if spec >= 4:
        detail_bits.append("concrete, specific detail")
    elif spec <= 2:
        detail_bits.append("low specificity")
    if nov >= 4:
        detail_bits.append("novel relative to consensus")
    detail = "; ".join(detail_bits) if detail_bits else "moderate specificity"

    if corr >= 2:
        corr_note = f"corroborated across {corr} model outputs"
    else:
        corr_note = "single-model disclosure"

    return (
        f"{sev} severity (sensitivity {sens}/5, specificity {spec}/5, "
        f"novelty {nov}/5). {detail.capitalize()}; status {status}; {corr_note}."
    )


_CATEGORY_SCENARIOS: dict[str, str] = {
    "campaign_finance": (
        "An adversary or opposition researcher assembles a financial-misconduct "
        "narrative from individually public filings, arriving at a prejudicial "
        "conclusion no single record states outright."
    ),
    "proprietary_adjacent": (
        "A competitor reconstructs proprietary-adjacent detail by paraphrase and "
        "synthesis, obtaining protected meaning without touching the original "
        "source material."
    ),
    "public_fact": (
        "Individually harmless public facts are combined into a targeting or "
        "profiling dossier at machine speed, collapsing the friction that once "
        "limited such synthesis."
    ),
}

_GENERIC_SCENARIO = (
    "Because several independent models reached the same conclusion from public "
    "fragments, the conclusion is reliably reachable on demand — an adversary "
    "needs only routine prompting, not privileged access, to reproduce it."
)


def _dominant_category(members: list[dict]) -> str:
    counts: dict[str, int] = {}
    for m in members:
        cat = (m.get("category") or "unclassified").strip()
        counts[cat] = counts.get(cat, 0) + 1
    if not counts:
        return "unclassified"
    return max(counts, key=counts.get)


def _corroborating_outputs(finding: dict[str, Any]) -> list[dict[str, Any]]:
    """Model-level rows for a collapsed group (from member_scores when present)."""
    scores = finding.get("member_scores")
    if isinstance(scores, list) and scores:
        rows = []
        for m in scores:
            rows.append(
                {
                    "claim_id": m.get("claim_id"),
                    "source": m.get("source_model"),
                    "source_model": m.get("source_model"),
                    "claim": plain_text(m.get("claim")),
                    "excerpt": m.get("raw_excerpt") or finding.get("raw_excerpt"),
                    "start_line": m.get("raw_start_line") or finding.get("raw_start_line"),
                    "end_line": m.get("raw_end_line") or finding.get("raw_end_line"),
                    "citations": list(m.get("citations") or []),
                }
            )
        return rows
    return [
        {
            "claim_id": finding.get("claim_id"),
            "source": finding.get("source_cite") or finding.get("source_model"),
            "source_model": finding.get("source_model"),
            "claim": plain_text(finding.get("claim")),
            "excerpt": finding.get("raw_excerpt"),
            "start_line": finding.get("raw_start_line"),
            "end_line": finding.get("raw_end_line"),
            "citations": list(finding.get("citations_display") or []),
        }
    ]


def build_basis_section(
    report_data: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    remediation: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``basis`` content block consumed by the Basis Report template."""
    # One inventory / findings row per collapsed exposure group.
    findings = dedupe_findings_by_group(list(findings or []))
    by_id = {f.get("claim_id"): f for f in findings if f.get("claim_id")}
    chains = list(report_data.get("chains") or [])

    # 1) Complete prioritized exposure inventory (collapsed groups only).
    inventory = [
        {
            "claim_id": f.get("claim_id"),
            "claim": plain_text(f.get("claim")),
            "category": plain_text(f.get("category")).replace("_", " ") or "unclassified",
            "corroboration": f.get("corroboration") or 1,
            "source_cite": f.get("source_cite") or f.get("source_model"),
            "source_refs": list(f.get("source_refs") or []),
        }
        for f in findings
    ]

    # 2) Full findings with severity + rationale + confidence + evidence.
    findings_full = []
    for f in findings:
        row = dict(f)
        row["claim"] = plain_text(f.get("claim"))
        row["severity"] = _severity_label(int(f.get("sensitivity") or 0))
        row["rationale"] = _rationale(f)
        findings_full.append(row)

    # 3) Exposure chains: one collapsed finding per chain; corroborating model
    #    paraphrases come from member_scores, not duplicate inventory rows.
    chain_details = []
    for ch in chains:
        member_ids = list(ch.get("claim_ids") or [])
        members = [by_id[c] for c in member_ids if c in by_id]
        if not members:
            # Locate survivor that absorbed these member ids
            wanted = set(member_ids)
            members = [
                f
                for f in findings
                if wanted.intersection(set(f.get("merged_from") or []))
                or f.get("claim_id") in wanted
            ]
        if not members:
            continue
        rep = members[0]
        models = list(
            ch.get("models")
            or rep.get("source_models")
            or ([rep.get("source_model")] if rep.get("source_model") else [])
        )
        models = [m for m in models if m]
        cited: list[dict[str, str]] = []
        for entry in rep.get("citations_display") or []:
            if all(c.get("ref") != entry.get("ref") for c in cited):
                cited.append(entry)
        query_ids = sorted(
            {
                *(rep.get("query_ids") or []),
                *([rep.get("query_id")] if rep.get("query_id") else []),
            }
        )
        line_spans = []
        if rep.get("raw_start_line"):
            line_spans.append(
                f"lines {rep.get('raw_start_line')}\u2013{rep.get('raw_end_line')}"
            )
        group_size = int(rep.get("merged_count") or len(rep.get("member_scores") or []) or 1)
        chain_details.append(
            {
                "chain_id": ch.get("chain_id"),
                "label": plain_text(ch.get("label") or rep.get("claim"))[:120],
                "score": ch.get("score"),
                "recovered": plain_text(rep.get("claim")),
                "model_count": len(models) or int(rep.get("corroboration") or 1),
                "models": models,
                "member_ids": list(rep.get("merged_from") or member_ids),
                "citations": cited[:8],
                "corroborating_outputs": _corroborating_outputs(rep),
                "derivation": {
                    "query_ids": query_ids,
                    "models": models,
                    "line_spans": line_spans[:6],
                    "cluster_size": group_size,
                    "score": ch.get("score"),
                    "steps": [
                        (
                            f"Reworded the operator prompt into retrieval seeds "
                            f"({', '.join(str(q) for q in query_ids) or 'multiple queries'})."
                        ),
                        (
                            f"Fanned out to {len(models) or int(rep.get('corroboration') or 1)} "
                            f"model(s): {', '.join(str(m) for m in models) or 'multiple models'}."
                        ),
                        (
                            "Extracted atomic claims with source line offsets "
                            + (f"({'; '.join(line_spans[:4])})." if line_spans else ".")
                        ),
                        (
                            f"Collapsed {group_size} paraphrase(s) into one "
                            f"exposure and scored it ({ch.get('score')})."
                        ),
                    ],
                },
                "implication": _CATEGORY_SCENARIOS.get(
                    _dominant_category([rep]), _GENERIC_SCENARIO
                ),
            }
        )

    # A cross-chain implication when several exposures corroborate each other.
    cross_chain_implication = (
        _GENERIC_SCENARIO
        if len([c for c in chain_details if c["model_count"] >= 2]) >= 2
        else ""
    )

    return {
        "title": "Basis Report",
        "subtitle": (
            "Complete exposure basis, derivation, and remediation"
            if remediation
            else "Complete exposure basis and derivation"
        ),
        "inventory": inventory,
        "findings_full": findings_full,
        "chain_details": chain_details,
        "cross_chain_implication": cross_chain_implication,
        "remediation": list(remediation or []),
        "counts": report_data.get("counts") or {},
    }
