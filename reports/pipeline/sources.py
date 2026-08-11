"""Real-world source registry for the reports.

Claims carry citations extracted from ``exploration.md`` (URLs, named sources,
and numbered references). This module folds them into one numbered registry so
tables can cite compactly (``S3``) while the report's Sources section prints the
full label and URL exactly once.
"""

from __future__ import annotations

import re
from typing import Any

from .citations import citation_entry
from .textclean import plain_text

_TRACKING_RE = re.compile(r"[?#].*$")
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)")


_LEAD_SPLIT_RE = re.compile(r"\s+[–—-]\s+|:\s+|,\s+")


def short_label(label: str, *, max_len: int = 44) -> str:
    """Compact form of a source label for one-line contexts."""
    text = _PARENTHETICAL_RE.sub("", str(label or "")).strip(" :;,-—")
    if len(text) <= max_len:
        return text
    # Prefer the publication name that leads most citation labels.
    head = _LEAD_SPLIT_RE.split(text, maxsplit=1)[0].strip(" :;,-—")
    if 3 < len(head) <= max_len:
        return head
    clipped = text[:max_len].rsplit(" ", 1)[0].rstrip(" :;,-—")
    return (clipped or text[:max_len]) + "…"


def _dedupe_key(entry: dict[str, str]) -> str:
    url = (entry.get("url") or "").strip().lower()
    if url:
        url = _TRACKING_RE.sub("", url).rstrip("/")
        return f"url:{url}"
    return f"label:{(entry.get('label') or entry.get('text') or '').strip().lower()}"


def build_source_registry(
    findings: list[dict[str, Any]],
    *,
    per_finding: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(sources, findings)`` with citations resolved to ``S#`` refs.

    ``sources`` is ordered by first appearance across ``findings``. Each
    returned finding gains ``citations_display`` (label/url/ref dicts) and
    ``source_refs`` (e.g. ``["S1", "S4"]``).
    """
    sources: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []

    for finding in findings:
        row = dict(finding)
        refs: list[str] = []
        for raw in list(row.get("citations") or [])[:per_finding]:
            entry = citation_entry(str(raw))
            label = plain_text(entry.get("label") or entry.get("text") or "")
            url = (entry.get("url") or "").strip()
            if not label and not url:
                continue
            key = _dedupe_key({"label": label, "url": url, "text": entry.get("text", "")})
            known = index.get(key)
            if known is None:
                known = {
                    "ref": f"S{len(sources) + 1}",
                    "number": len(sources) + 1,
                    "label": label or url,
                    "url": url,
                    "claim_ids": [],
                }
                index[key] = known
                sources.append(known)
            elif label and len(label) > len(known["label"]):
                known["label"] = label
            claim_id = row.get("claim_id")
            if claim_id and claim_id not in known["claim_ids"]:
                known["claim_ids"].append(claim_id)
            if known["ref"] not in refs:
                refs.append(known["ref"])
        row["source_refs"] = refs
        out.append(row)

    for source in sources:
        source["cited_by"] = len(source["claim_ids"])
        source["short"] = short_label(source["label"])

    # Labels can grow as later findings cite the same source, so resolve the
    # per-finding display list only once the registry is final.
    by_ref = {s["ref"]: s for s in sources}
    for row in out:
        row["citations_display"] = [
            {
                "ref": ref,
                "label": by_ref[ref]["label"],
                "short": by_ref[ref]["short"],
                "url": by_ref[ref]["url"],
            }
            for ref in row["source_refs"]
        ]
    return sources, out


def top_source_labels(sources: list[dict[str, Any]], limit: int = 3) -> list[str]:
    """Most-cited real-world sources, for the executive summary."""
    ranked = sorted(
        sources, key=lambda s: (-int(s.get("cited_by") or 0), int(s.get("number") or 0))
    )
    labels: list[str] = []
    for source in ranked:
        label = (source.get("short") or source.get("label") or "").strip()
        if not label or label in labels:
            continue
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels
