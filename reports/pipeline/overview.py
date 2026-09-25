"""Decision-oriented summaries: per-model counts, reproduction, evidence, verification."""

from __future__ import annotations

from collections import Counter
from typing import Any
from urllib.parse import urlparse

from .provenance import evidence_status, has_source_url, research_significance

EVIDENCE_ORDER = (
    "Externally verified",
    "Cross-model corroborated",
    "Single-model lead",
    "Contested",
)

_SMALL = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def number_word(n: int) -> str:
    return _SMALL.get(int(n), str(int(n)))


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or f"{one}s")


def _pct(part: int, whole: int) -> int:
    return round(100 * part / whole) if whole else 0


def finding_models(finding: dict[str, Any], aliases: dict[str, str] | None = None) -> list[str]:
    from pipeline.score import _source_models

    return _source_models(finding, aliases or {})


def model_count(finding: dict[str, Any], aliases: dict[str, str] | None = None) -> int:
    try:
        stated = int(finding.get("corroboration") or 1)
    except (TypeError, ValueError):
        stated = 1
    return max(1, stated, len(finding_models(finding, aliases)))


def clip_label(text: Any, n: int = 72) -> str:
    words = " ".join(str(text or "").split())
    if len(words) <= n:
        return words
    cut = words[: n - 1].rsplit(" ", 1)[0].rstrip(",;:–—-")
    return f"{cut}…"


def model_rows(
    findings: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
    roster: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per model: findings, high significance, corroborated, verified.

    With a roster, every probed model gets a row (zero if it returned nothing)
    and labels outside the roster are ignored.
    """
    from graphics.style import short_model_name

    def _empty(name: str) -> dict[str, Any]:
        return {"model": name, "findings": 0, "high": 0, "corroborated": 0, "verified": 0}

    rows: dict[str, dict[str, Any]] = {}
    allowed: set[str] | None = None
    if roster:
        allowed = set()
        for raw in roster:
            name = short_model_name(str(raw), aliases or {})
            if name and name != "unknown":
                allowed.add(name)
                rows.setdefault(name, _empty(name))
    for finding in findings:
        high = research_significance(finding) == "High"
        shared = model_count(finding, aliases) >= 2
        sourced = has_source_url(finding)
        for name in finding_models(finding, aliases):
            if allowed is not None and name not in allowed:
                continue
            row = rows.setdefault(name, _empty(name))
            row["findings"] += 1
            row["high"] += int(high)
            row["corroborated"] += int(shared)
            row["verified"] += int(sourced)
    out = list(rows.values()) if allowed is not None else [r for r in rows.values() if r["findings"]]
    for row in out:
        row["corroborated_pct"] = _pct(row["corroborated"], row["findings"])
        row["verified_pct"] = _pct(row["verified"], row["findings"])
    out.sort(key=lambda row: (-row["findings"], -row["high"], row["model"]))
    return out


def exposure_overview(
    findings: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
    roster: list[str] | None = None,
) -> dict[str, Any]:
    rows = model_rows(findings, aliases, roster)
    total = len(findings)
    high = sum(1 for f in findings if research_significance(f) == "High")
    corroborated = sum(1 for f in findings if model_count(f, aliases) >= 2)
    verified = sum(1 for f in findings if has_source_url(f))
    rate = _pct(corroborated, total)
    detail = ""
    if rows and rows[0]["findings"]:
        top = rows[0]
        takeaway = (
            f"{top['model']} surfaced the most findings, "
            f"{'but only' if rate < 50 else 'and'} {rate}% of all findings "
            "were stated by more than one model."
        )
        top_high = max(rows, key=lambda row: (row["high"], row["findings"]))
        if top_high["model"] != top["model"] and top_high["high"]:
            detail = (
                f"{top_high['model']} surfaced the most high-significance "
                f"findings ({top_high['high']})."
            )
    else:
        takeaway = "No model produced a finding in this run."
    return {
        "rows": rows,
        "totals": {
            "findings": total,
            "high": high,
            "corroborated": corroborated,
            "corroborated_pct": rate,
            "verified": verified,
            "verified_pct": _pct(verified, total),
        },
        "stats": [
            {"value": f"{rate}%", "label": "Cross-model corroboration rate",
             "note": f"{corroborated} of {total} findings stated by 2+ models"},
            {"value": str(verified), "label": "Externally verified findings",
             "note": f"{_pct(verified, total)}% cite a source URL"},
            {"value": str(high), "label": "High-significance findings",
             "note": f"of {total} distinct findings"},
        ],
        "takeaway": takeaway,
        "detail": detail,
    }


def reproduction(
    findings: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
    roster: list[str] | None = None,
    *,
    max_columns: int = 48,
) -> dict[str, Any]:
    """High-significance findings by model, sorted by how many models stated them."""
    rows = [row["model"] for row in model_rows(findings, aliases, roster)]
    pool = [f for f in findings if research_significance(f) == "High"]
    scope = "high-significance disclosures"
    if not pool:
        pool = list(findings)
        scope = "findings"
    columns: list[dict[str, Any]] = []
    for finding in pool:
        models = finding_models(finding, aliases)
        if not models:
            continue
        columns.append(
            {
                "id": str(finding.get("present_id") or finding.get("cluster_id") or finding.get("claim_id") or ""),
                "label": clip_label(finding.get("claim"), 140),
                "models": models,
                "n": model_count(finding, aliases),
                "sensitivity": int(finding.get("sensitivity") or 0),
                "evidence_status": evidence_status(finding),
            }
        )
    columns.sort(key=lambda col: (-col["n"], -col["sensitivity"], col["id"]))
    total = len(columns)
    shown = columns[:max_columns]
    single = sum(1 for col in columns if col["n"] < 2)
    multi = total - single
    everyone = sum(1 for col in columns if len(rows) > 1 and col["n"] >= len(rows))

    if not total:
        title = "No model produced a finding to compare."
    elif len(rows) < 2:
        title = "Only one model produced findings, so nothing can be reproduced across models."
    elif single > 0.6 * total:
        title = (
            f"Most {scope} are model-specific rather than consistently "
            "reproduced across models."
        )
    elif single < 0.4 * total:
        title = f"Most {scope} are reproduced by more than one model."
    else:
        title = (
            f"Only about half of the {scope} are reproduced by a second model"
            + (
                f"; {number_word(everyone)} {'was' if everyone == 1 else 'were'} "
                f"stated by all {number_word(len(rows))}."
                if everyone
                else "; none was stated by every model."
            )
        )
    lede = (
        f"{single} of {total} {scope} came from one model only. "
        f"{multi} {'was' if multi == 1 else 'were'} stated by two or more"
        + (f"; {everyone} by all {len(rows)} models." if everyone else ".")
    ) if total else ""

    notes: list[dict[str, Any]] = []
    if shown:
        notes.append({**shown[0], "why": f"Most reproduced: {shown[0]['n']} models"})
        singles = [col for col in shown if col["n"] < 2]
        if singles and singles[0] is not shown[0]:
            only = singles[0]
            notes.append({**only, "why": f"Only {only['models'][0]} stated this"})
        used = {note["id"] for note in notes}
        mids = [col for col in shown if 2 <= col["n"] < shown[0]["n"] and col["id"] not in used]
        if mids:
            notes.append({**mids[0], "why": f"Split: {mids[0]['n']} of {len(rows)} models"})
        elif len(singles) > 1:
            other = next(
                (col for col in singles[1:] if col["models"][0] != singles[0]["models"][0]),
                None,
            )
            if other:
                notes.append({**other, "why": f"Only {other['models'][0]} stated this"})
    for index, note in enumerate(notes, 1):
        note["marker"] = index
    marks = {note["id"]: note["marker"] for note in notes}
    for col in shown:
        col["marker"] = marks.get(col["id"])

    return {
        "rows": rows,
        "columns": shown,
        "total": total,
        "single": single,
        "multi": multi,
        "notes": notes,
        "scope": scope,
        "title": title,
        "lede": lede,
    }


def _domain(url: str) -> str:
    host = urlparse(str(url or "")).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def evidence_quality(
    findings: list[dict[str, Any]],
    sources: list[dict[str, Any]] | None = None,
    *,
    top_sources: int = 6,
) -> dict[str, Any]:
    counts = Counter(evidence_status(f) for f in findings)
    total = len(findings)
    ladder = [
        {"label": label, "count": counts.get(label, 0), "pct": _pct(counts.get(label, 0), total)}
        for label in EVIDENCE_ORDER
    ]
    ranked = sorted(
        (s for s in sources or [] if s.get("url")),
        key=lambda s: (-int(s.get("cited_by") or 0), str(s.get("ref") or "")),
    )
    domains = Counter(_domain(s.get("url")) for s in sources or [] if s.get("url"))
    verified = counts.get("Externally verified", 0)
    single = counts.get("Single-model lead", 0)
    takeaway = (
        f"{verified} of {total} findings cite an external source. "
        + (
            f"{single} rest on a single model with no source."
            if single
            else "Every other finding was stated by more than one model."
        )
    ) if total else "No findings were scored in this run."
    return {
        "ladder": ladder,
        "sources": [
            {
                "ref": s.get("ref"),
                "label": clip_label(s.get("label"), 90),
                "domain": _domain(s.get("url")),
                "cited_by": int(s.get("cited_by") or 0),
            }
            for s in ranked[:top_sources]
        ],
        "source_count": len(sources or []),
        "domain_count": len(domains),
        "takeaway": takeaway,
    }


def verification_plan(
    findings: list[dict[str, Any]],
    aliases: dict[str, str] | None = None,
    *,
    targets: int = 4,
) -> dict[str, Any]:
    """Verification order built from the evidence ladder. Numbers only, no voice copy."""
    status = {id(f): evidence_status(f) for f in findings}
    high = [f for f in findings if research_significance(f) == "High"]
    high_sourced = [f for f in high if status[id(f)] == "Externally verified"]
    other_sourced = sum(
        1 for f in findings
        if status[id(f)] == "Externally verified" and research_significance(f) != "High"
    )
    cross = sum(1 for f in findings if status[id(f)] == "Cross-model corroborated")
    single = sum(1 for f in findings if status[id(f)] == "Single-model lead")
    contested = sum(1 for f in findings if status[id(f)] == "Contested")

    items = [
        {
            "title": "Confirm the sourced high-significance findings",
            "count": len(high_sourced),
            "body": (
                f"{len(high_sourced)} high-significance {_plural(len(high_sourced), 'finding')} "
                "cite an external source. Open each citation and confirm it "
                "states the claim, not only the topic."
            ),
        },
        {
            "title": "Check the other sourced findings",
            "count": other_sourced,
            "body": (
                f"{other_sourced} more {_plural(other_sourced, 'finding')} cite a source. "
                "Check them in significance order."
            ),
        },
        {
            "title": "Find a record for cross-model findings",
            "count": cross,
            "body": (
                f"{cross} {_plural(cross, 'finding')} were stated by more than one "
                "model but cite no source. Agreement between models is not a "
                "record; locate the primary document."
            ),
        },
        {
            "title": "Hold single-model leads",
            "count": single,
            "body": (
                f"{single} {_plural(single, 'finding')} rest on one model with no "
                "source. Do not repeat them without independent confirmation."
            ),
        },
    ]
    if contested:
        items.append(
            {
                "title": "Resolve contested findings",
                "count": contested,
                "body": (
                    f"{contested} {_plural(contested, 'finding')} are contested: "
                    "models disagree. Establish which account the record supports."
                ),
                "exceptional": True,
            }
        )
    items = [item for item in items if item["count"] or item is items[0]]

    unsourced_high = [f for f in high if status[id(f)] != "Externally verified"]
    unsourced_high.sort(key=lambda f: (-model_count(f, aliases), -int(f.get("sensitivity") or 0)))
    rows = [
        {
            "id": str(f.get("present_id") or f.get("cluster_id") or f.get("claim_id") or ""),
            "claim": clip_label(f.get("claim"), 150),
            "models": ", ".join(finding_models(f, aliases)),
            "evidence_status": status[id(f)],
        }
        for f in unsourced_high[:targets]
    ]
    if high_sourced:
        takeaway = (
            f"Start with the {len(high_sourced)} high-significance "
            f"{_plural(len(high_sourced), 'finding')} that already cite a source."
        )
    elif high:
        takeaway = (
            "No high-significance finding cites a source yet. "
            "Start by locating primary records."
        )
    else:
        takeaway = "Start with the findings that already cite a source."
    return {
        "title": "Verification Plan",
        "takeaway": takeaway,
        "lede": (
            "Work down the evidence ladder. Each step says how many findings "
            "it covers and what would move them up a rung."
        ),
        "items": items,
        "targets": rows,
        "targets_title": "High-significance findings with no source yet",
    }
