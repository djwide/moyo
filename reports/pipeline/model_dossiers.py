"""Per-model dossiers: analysis that the cross-model sections do not already show.

Existing report pages compare models side by side (bars, heatmap, contrast).
These records isolate one scanned model: exclusive disclosures, citation
originals, probe outcomes, and that model's score fingerprint versus the corpus.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from graphics.style import short_model_name
from pipeline.cluster import present_id
from pipeline.audience import chart_order, disclosure_bin, normalize_audience
from pipeline.score import _source_models

_RADAR_KEYS = (
    "specificity",
    "sensitivity",
    "corroboration",
    "novelty",
    "confidence",
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def model_slug(name: str, used: set[str] | None = None) -> str:
    base = _SLUG_RE.sub("-", str(name or "model").lower()).strip("-")[:36] or "model"
    slug = base
    n = 2
    used = used if used is not None else set()
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def dossier_chart_key(slug: str, kind: str) -> str:
    return f"d_{slug}_{kind}"


def _clip(text: str, n: int = 160) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= n:
        return raw
    return raw[: max(0, n - 1)].rstrip() + "…"


def _radar_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {k: 0.0 for k in _RADAR_KEYS}
    n = len(rows)
    return {
        k: round(sum(float(r.get(k) or 0) for r in rows) / n, 2) for k in _RADAR_KEYS
    }


def _roster(
    findings: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    *,
    models_probed: list[str] | None,
    aliases: dict[str, str],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        key = short_model_name(raw, aliases)
        if not key or key == "unknown" or key in seen:
            return
        seen.add(key)
        ordered.append(key)

    for raw in models_probed or []:
        _add(str(raw))
    if ordered:
        return ordered
    for row in corpus or []:
        _add(str(row.get("model") or ""))
    for finding in findings or []:
        for name in _source_models(finding, aliases):
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return ordered


def build_model_dossiers(
    findings: list[dict[str, Any]] | None,
    *,
    corpus: list[dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
    radar_averages: dict[str, Any] | None = None,
    models_probed: list[str] | None = None,
    aliases: dict[str, str] | None = None,
    audience: str | None = None,
) -> list[dict[str, Any]]:
    """One dossier per scanned model, roster order."""
    aliases = aliases or {}
    voice = normalize_audience(audience)
    band_keys = chart_order(voice)
    findings = list(findings or [])
    corpus = list(corpus or [])
    sources = list(sources or [])
    corpus_radar = {
        k: round(float((radar_averages or {}).get(k) or 0), 2) for k in _RADAR_KEYS
    }

    roster = _roster(
        findings, corpus, models_probed=models_probed, aliases=aliases
    )
    used_slugs: set[str] = set()
    by_model: dict[str, list[dict[str, Any]]] = {m: [] for m in roster}
    for finding in findings:
        for name in _source_models(finding, aliases):
            if name in by_model:
                by_model[name].append(finding)

    cited_by: dict[str, set[str]] = defaultdict(set)
    for finding in findings:
        models = set(_source_models(finding, aliases))
        for ref in finding.get("source_refs") or []:
            cited_by[str(ref)].update(models)

    out: list[dict[str, Any]] = []
    for model in roster:
        slug = model_slug(model, used_slugs)
        mine = by_model.get(model) or []
        unique_rows: list[dict[str, Any]] = []
        shared_rows: list[dict[str, Any]] = []
        bands = {key: 0 for key in band_keys}
        status_mix: dict[str, int] = defaultdict(int)
        languages: dict[str, int] = defaultdict(int)
        my_refs: dict[str, int] = defaultdict(int)

        for finding in mine:
            others = [n for n in _source_models(finding, aliases) if n != model]
            row = {
                "claim_id": finding.get("claim_id"),
                "present_id": present_id(finding),
                "claim": _clip(finding.get("claim") or "", 180),
                "status": finding.get("status") or "UNVERIFIED",
                "sensitivity": int(finding.get("sensitivity") or 0),
                "specificity": int(finding.get("specificity") or 0),
                "source_refs": list(finding.get("source_refs") or []),
            }
            if others:
                shared_rows.append(row)
            else:
                unique_rows.append(row)
            bands[disclosure_bin(finding, voice)] += 1
            status = str(finding.get("status") or "UNVERIFIED").upper()
            status_mix[status] += 1
            lang = str(finding.get("prompt_language") or "").strip()
            if lang:
                languages[lang] += 1
            for ref in finding.get("source_refs") or []:
                my_refs[str(ref)] += 1

        unique_rows.sort(key=lambda r: (-int(r["sensitivity"]), -int(r["specificity"])))
        shared_rows.sort(key=lambda r: (-int(r["sensitivity"]), -int(r["specificity"])))
        n_findings = len(mine)
        n_unique = len(unique_rows)
        overlap_pct = (
            round(100.0 * (n_findings - n_unique) / n_findings, 1) if n_findings else 0.0
        )

        probes = {"attempted": 0, "answered": 0, "failed": 0, "empty": 0}
        failed_detail: list[dict[str, str]] = []
        for row in corpus:
            if short_model_name(str(row.get("model") or ""), aliases) != model:
                continue
            probes["attempted"] += 1
            text = str(row.get("text") or "").strip()
            if row.get("failed"):
                probes["failed"] += 1
                failed_detail.append(
                    {
                        "query_id": str(row.get("query_id") or ""),
                        "query": _clip(row.get("query") or "", 120),
                        "strategy": str(row.get("strategy") or ""),
                    }
                )
            elif not text:
                probes["empty"] += 1
            else:
                probes["answered"] += 1

        exclusive_cites: list[dict[str, Any]] = []
        by_ref = {str(s.get("ref") or ""): s for s in sources}
        for ref, count in sorted(my_refs.items(), key=lambda kv: (-kv[1], kv[0])):
            owners = cited_by.get(ref) or set()
            src = by_ref.get(ref) or {}
            exclusive_cites.append(
                {
                    "ref": ref,
                    "label": src.get("short") or src.get("label") or ref,
                    "count": count,
                    "exclusive": owners == {model},
                }
            )
        exclusive_only = [c for c in exclusive_cites if c["exclusive"]]

        distinctive = unique_rows[:3]
        if len(distinctive) < 3:
            extra = [r for r in shared_rows if r not in distinctive]
            distinctive = (distinctive + extra)[:3]

        lede = _dossier_lede(model, n_findings, n_unique, overlap_pct, probes)

        out.append(
            {
                "slug": slug,
                "model": model,
                "id": f"model-{slug}",
                "detail_id": f"model-{slug}-detail",
                "lede": lede,
                "findings": n_findings,
                "unique": n_unique,
                "shared": len(shared_rows),
                "high": bands.get(band_keys[0], 0),
                "overlap_pct": overlap_pct,
                "radar": _radar_means(mine),
                "corpus_radar": corpus_radar,
                "bands": bands,
                "probes": probes,
                "failed_probes": failed_detail[:8],
                "distinctive": distinctive,
                "unique_findings": unique_rows[:16],
                "shared_findings": shared_rows[:8],
                "citations": exclusive_cites[:8],
                "exclusive_citations": exclusive_only[:8],
                "citation_count": len(my_refs),
                "exclusive_citation_count": len(exclusive_only),
                "languages": [
                    {"language": k, "count": v}
                    for k, v in sorted(languages.items(), key=lambda kv: (-kv[1], kv[0]))
                ],
                "status_mix": dict(status_mix),
                "charts": {
                    "fingerprint": dossier_chart_key(slug, "fingerprint"),
                    "mix": dossier_chart_key(slug, "mix"),
                    "probes": dossier_chart_key(slug, "probes"),
                    "overlap": dossier_chart_key(slug, "overlap"),
                },
            }
        )
    return out


def _dossier_lede(
    model: str,
    n_findings: int,
    n_unique: int,
    overlap_pct: float,
    probes: dict[str, int],
) -> str:
    if n_findings == 0:
        attempted = int(probes.get("attempted") or 0)
        failed = int(probes.get("failed") or 0)
        if failed and attempted:
            return (
                f"{model} was probed {attempted} time"
                f"{'s' if attempted != 1 else ''} and returned no usable "
                f"findings ({failed} failed retrieval"
                f"{'s' if failed != 1 else ''})."
            )
        return f"{model} produced no extracted findings in this run."
    unique_bit = (
        f"{n_unique} exclusive to this model"
        if n_unique
        else "none exclusive to this model"
    )
    return (
        f"{model} produced {n_findings} finding"
        f"{'s' if n_findings != 1 else ''} ({unique_bit}). "
        f"Cross-model overlap is {overlap_pct:.0f}%."
    )
