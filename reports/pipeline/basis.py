"""Assemble Basis Report content from scored run data.

The Basis Report is the comprehensive counterpart to the Exposure Snapshot. It
reuses the same pipeline artifacts (``report_data.json`` findings, clusters, and
exposure chains) and adds full, non-abridged detail plus ISVF-sourced
remediation. No re-extraction or extra LLM calls are required.
"""

from __future__ import annotations

from typing import Any

from .textclean import plain_text


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


def build_basis_section(
    report_data: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    remediation: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``basis`` content block consumed by the Basis Report template."""
    findings = list(findings or [])
    by_id = {f.get("claim_id"): f for f in findings if f.get("claim_id")}
    chains = list(report_data.get("chains") or [])

    # 1) Complete prioritized exposure inventory (all findings, ranked).
    #    Kept to five columns so the table fits the page width.
    inventory = [
        {
            "claim_id": f.get("claim_id"),
            "claim": plain_text(f.get("claim")),
            # Underscored slugs break mid-word in a narrow column.
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

    # 3) What was inferred/recovered + corroborating outputs + derivation,
    #    organized per exposure chain (clusters of related claims).
    chain_details = []
    for ch in chains:
        member_ids = list(ch.get("claim_ids") or [])
        members = [by_id[c] for c in member_ids if c in by_id]
        if not members:
            continue
        models = sorted({m.get("source_model") for m in members if m.get("source_model")})
        cited: list[dict[str, str]] = []
        for m in members:
            for entry in m.get("citations_display") or []:
                if all(c.get("ref") != entry.get("ref") for c in cited):
                    cited.append(entry)
        query_ids = sorted({m.get("query_id") for m in members if m.get("query_id")})
        line_spans = [
            f"lines {m.get('raw_start_line')}\u2013{m.get('raw_end_line')}"
            for m in members
            if m.get("raw_start_line")
        ]
        rep = members[0]
        chain_details.append(
            {
                "chain_id": ch.get("chain_id"),
                "label": plain_text(ch.get("label") or rep.get("claim"))[:120],
                "score": ch.get("score"),
                "recovered": plain_text(rep.get("claim")),
                "model_count": len(models),
                "models": models,
                "member_ids": member_ids,
                "citations": cited[:8],
                # Multiple corroborating model outputs (verbatim per model).
                "corroborating_outputs": [
                    {
                        "claim_id": m.get("claim_id"),
                        "source": m.get("source_cite") or m.get("source_model"),
                        "source_model": m.get("source_model"),
                        "claim": plain_text(m.get("claim")),
                        "excerpt": m.get("raw_excerpt"),
                        "start_line": m.get("raw_start_line"),
                        "end_line": m.get("raw_end_line"),
                        "citations": list(m.get("citations_display") or []),
                    }
                    for m in members
                ],
                # Exactly how MOYO reached this conclusion (provenance trail).
                "derivation": {
                    "query_ids": query_ids,
                    "models": models,
                    "line_spans": line_spans[:6],
                    "cluster_size": len(members),
                    "score": ch.get("score"),
                    "steps": [
                        (
                            f"Reworded the operator prompt into retrieval seeds "
                            f"({', '.join(query_ids) or 'multiple queries'})."
                        ),
                        (
                            f"Fanned out to {len(models)} model(s): "
                            f"{', '.join(models) or 'multiple models'}."
                        ),
                        (
                            "Extracted atomic claims with source line offsets "
                            + (f"({'; '.join(line_spans[:4])})." if line_spans else ".")
                        ),
                        (
                            f"Clustered {len(members)} paraphrases into one "
                            f"exposure and scored it ({ch.get('score')})."
                        ),
                    ],
                },
                # Specific implication / exploitation scenario.
                "implication": _CATEGORY_SCENARIOS.get(
                    _dominant_category(members), _GENERIC_SCENARIO
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
