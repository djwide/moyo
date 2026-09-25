"""Evidence graph SVG: citations → model inference → claims → conclusions.

Visual language matches /methodology EvidenceGraph: teal bullseye nodes on every
column, teal curves only, ink/muted labels (no teal text).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

from .style import (
    CREAM,
    FONT,
    FONT_MONO,
    INK,
    MUTED,
    PRINT_MAX_HEIGHT,
    RULE,
    TEAL,
    WHITE,
    escape_xml,
    models_with_results,
    raw_finding_models,
    short_model_name,
    svg_root,
    truncate,
)

_URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+", re.I)
_TRACKING_RE = re.compile(r"[?#].*$")


def _curve(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> str:
    dx = max(36.0, abs(x2 - x1) * 0.4)
    c1x = x1 + dx
    c2x = x2 - dx * 0.65
    return (
        f'<path d="M {x1:.1f},{y1:.1f} C {c1x:.1f},{y1:.1f} {c2x:.1f},{y2:.1f} '
        f'{x2:.1f},{y2:.1f}" fill="none" stroke="{TEAL}" '
        f'stroke-opacity="0.28" stroke-width="0.75"/>'
    )


def _bullseye(x: float, y: float) -> str:
    """Same teal-ring / ink-center mark on every node."""
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{WHITE}" '
        f'stroke="{TEAL}" stroke-width="1.5"/>'
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{INK}"/>'
    )


def _ys(n: int, top: float, bot: float) -> list[float]:
    if n <= 1:
        return [(top + bot) / 2]
    return [top + i * (bot - top) / (n - 1) for i in range(n)]


def _citation_key(raw: str) -> tuple[str, str]:
    text = " ".join(str(raw or "").split()).strip()
    if not text:
        return ("", "")
    url_m = _URL_RE.search(text)
    url = ""
    if url_m:
        url = _TRACKING_RE.sub("", url_m.group(0)).rstrip("/").lower()
    label = text
    if url_m:
        label = (text[: url_m.start()] + text[url_m.end() :]).strip(" -—|:;")
    if not label and url:
        host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        label = host
    title = truncate(label or url or text, 28)
    key = f"url:{url}" if url else f"label:{title.lower()}"
    return key, title


def _collect_citations(findings: list[dict], *, limit: int = 8) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    index: dict[str, dict[str, str]] = {}
    for f in findings:
        display = f.get("citations_display")
        if isinstance(display, list) and display:
            for entry in display:
                if not isinstance(entry, dict):
                    continue
                ref = str(entry.get("ref") or "").strip()
                label = str(entry.get("short") or entry.get("label") or "").strip()
                url = str(entry.get("url") or "").strip()
                key = f"url:{url.lower()}" if url else f"label:{label.lower()}"
                if not key or key in ("url:", "label:") or key in index:
                    continue
                row = {
                    "ref": ref or f"S{len(out) + 1}",
                    "title": truncate(label or url, 28),
                    "key": key,
                }
                index[key] = row
                out.append(row)
                if len(out) >= limit:
                    return out
            continue
        for raw in list(f.get("citations") or [])[:4]:
            key, title = _citation_key(str(raw))
            if not key or not title or key in index:
                continue
            row = {"ref": f"S{len(out) + 1}", "title": title, "key": key}
            index[key] = row
            out.append(row)
            if len(out) >= limit:
                return out
    return out


def _finding_citation_keys(finding: dict) -> set[str]:
    keys: set[str] = set()
    display = finding.get("citations_display")
    if isinstance(display, list):
        for entry in display:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "").strip().lower()
            label = str(entry.get("short") or entry.get("label") or "").strip().lower()
            if url:
                keys.add(f"url:{url}")
            elif label:
                keys.add(f"label:{label}")
    for raw in list(finding.get("citations") or [])[:4]:
        key, _ = _citation_key(str(raw))
        if key:
            keys.add(key)
    return keys


def _finding_models(finding: dict, aliases: dict[str, str]) -> set[str]:
    out: set[str] = set()
    for raw in raw_finding_models(finding):
        m = short_model_name(raw, aliases)
        if m and m != "unknown":
            out.add(m)
    return out


def _present_id(finding: dict) -> str:
    return str(
        finding.get("present_id")
        or finding.get("cluster_id")
        or finding.get("claim_id")
        or ""
    ).strip()


def snapshot_graph_findings(
    findings: Sequence[dict],
    *,
    cap: int = 12,
    limit: int = 10,
) -> list[dict]:
    """Clusters the snapshot graph may show: abridged set, excerpted first.

    The Exposure Snapshot lists a capped English cluster set, then prints
    verbatim excerpts for a subset. Graph nodes must come from that same
    subset so every cluster on the graph appears in the following section.
    """
    from pipeline.cluster import dedupe_findings_by_group
    from pipeline.language import looks_like_english

    grouped = dedupe_findings_by_group(list(findings))
    english = [
        f
        for f in grouped
        if looks_like_english(str(f.get("claim") or ""))
        and not f.get("english_pending")
    ] or [
        f for f in grouped if looks_like_english(str(f.get("claim") or ""))
    ] or grouped
    abridged = english[: max(0, cap)]
    excerpted = [
        f for f in abridged if str(f.get("raw_excerpt") or "").strip()
    ][: max(0, limit)]
    return excerpted or abridged[: max(0, limit)]


def _clusters_by_agreement(
    findings: list[dict],
    *,
    model_keys: Sequence[str],
    aliases: dict[str, str],
    limit: int,
) -> list[str]:
    """Top cluster ids by cross-model agreement (same metric as the heatmap)."""
    key_set = {m for m in model_keys if m and m != "unknown"}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        kid = _present_id(f)
        if kid:
            grouped[kid].append(f)

    ranked: list[tuple[int, int, str]] = []
    for kid, members in grouped.items():
        models_in: set[str] = set()
        sens = 0
        for f in members:
            try:
                sens = max(sens, int(f.get("sensitivity") or 0))
            except (TypeError, ValueError):
                pass
            models_in |= _finding_models(f, aliases)
        agreement = len(models_in & key_set) if key_set else len(models_in)
        ranked.append((agreement, sens, kid))

    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [kid for _, _, kid in ranked[: max(0, limit)]]


def evidence_graph_svg(
    findings: Iterable[dict],
    chains: Iterable[dict],
    *,
    models_probed: Sequence[str] | None = None,
    max_citations: int = 8,
    max_models: int = 9,
    max_claims: int = 10,
    max_conclusions: int = 8,
    aliases: dict[str, str] | None = None,
    max_width: float = 640,
    max_height: float = PRINT_MAX_HEIGHT,
    include_conclusions: bool = True,
) -> str:
    """Print evidence graph matching /methodology EvidenceGraph columns."""
    aliases = aliases or {}
    all_findings = list(findings)
    by_id = {f.get("claim_id"): f for f in all_findings if f.get("claim_id")}
    chains = (
        [c for c in list(chains)[:max_conclusions] if c.get("chain_id")]
        if include_conclusions
        else []
    )

    width = min(max(max_width, 600), 660)
    if not include_conclusions:
        width = min(width, 490)
    height = min(max_height, 420)

    band_top = 8
    band_bottom = height - 10
    footer_h = 18
    node_top = band_top + 22
    node_bot = band_bottom - footer_h - 16

    citations = _collect_citations(all_findings, limit=max_citations)

    models: list[str] = []
    seen_m: set[str] = set()
    for raw in models_with_results(
        all_findings, models_probed=models_probed, aliases=aliases
    ):
        m = short_model_name(str(raw), aliases)
        if not m or m == "unknown" or m in seen_m:
            continue
        seen_m.add(m)
        models.append(m)
        if len(models) >= max_models:
            break

    # Same ranking as the deliverable model heatmap: most cross-model agreement.
    cluster_ids = _clusters_by_agreement(
        all_findings,
        model_keys=models,
        aliases=aliases,
        limit=max_claims,
    )

    def _chain_clusters(ch: dict) -> set[str]:
        wanted: set[str] = set()
        for cid in ch.get("claim_ids") or []:
            f = by_id.get(str(cid))
            wanted.add(_present_id(f) if f else str(cid))
        return wanted

    shown = set(cluster_ids)
    chains = [
        c
        for c in chains
        if _chain_clusters(c) & shown
    ] or chains

    cite_ys = _ys(len(citations) or 1, node_top, node_bot)
    model_ys = _ys(len(models) or 1, node_top, node_bot)
    claim_ys = _ys(len(cluster_ids) or 1, node_top, node_bot)
    chain_ys = _ys(len(chains) or 1, node_top, node_bot)

    # Column geometry: conclusions are id-only, so that band stays narrow.
    # Opposition graphs omit that column and the edges into it.
    x_cite, x_model, x_claim, x_chain = 28, 235, 400, 525
    col_boxes = [
        (8, 150, "Real-world citations"),
        (205, 125, "Model inference"),
        (370, 110, "Clusters"),
    ]
    if include_conclusions:
        col_boxes.append((500, 98, "Inferred conclusions"))

    bands: list[str] = []
    for x, w, label in col_boxes:
        bands.append(
            f'<rect x="{x}" y="{band_top}" width="{w}" height="{band_bottom - band_top}" '
            f'fill="{CREAM}" stroke="{RULE}"/>'
            f'<text x="{x + w / 2:.1f}" y="{band_bottom - 5:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="9" font-weight="600" fill="{MUTED}">'
            f"{escape_xml(label)}</text>"
        )

    cite_pos = {c["key"]: (x_cite, cite_ys[i]) for i, c in enumerate(citations)}
    model_pos = {m: (x_model, model_ys[i]) for i, m in enumerate(models)}
    claim_pos = {cid: (x_claim, claim_ys[i]) for i, cid in enumerate(cluster_ids)}
    chain_pos = {c["chain_id"]: (x_chain, chain_ys[i]) for i, c in enumerate(chains)}

    edges: list[str] = []
    seen_edges: set[tuple[float, float, float, float]] = set()

    def _add_edge(x1: float, y1: float, x2: float, y2: float) -> None:
        key = (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(_curve(x1, y1, x2, y2))

    # Citation → model inference
    for f in all_findings:
        f_keys = _finding_citation_keys(f)
        f_models = _finding_models(f, aliases)
        for key in f_keys:
            if key not in cite_pos:
                continue
            x1, y1 = cite_pos[key]
            for m in f_models:
                if m not in model_pos:
                    continue
                x2, y2 = model_pos[m]
                _add_edge(x1 + 28, y1, x2 - 10, y2)

    # Model inference → clusters
    for kid in cluster_ids:
        if kid not in claim_pos:
            continue
        x2, y2 = claim_pos[kid]
        linked: set[str] = set()
        for f in all_findings:
            if _present_id(f) != kid:
                continue
            linked |= _finding_models(f, aliases)
        for m in linked:
            if m not in model_pos:
                continue
            x1, y1 = model_pos[m]
            _add_edge(x1 + 14, y1, x2 - 10, y2)

    # Clusters → inferred conclusions
    if include_conclusions:
        for ch in chains:
            chid = ch["chain_id"]
            if chid not in chain_pos:
                continue
            x2, y2 = chain_pos[chid]
            wanted: set[str] = set()
            for cid in ch.get("claim_ids") or []:
                f = by_id.get(str(cid))
                wanted.add(_present_id(f) if f else str(cid))
            for kid in cluster_ids:
                if kid not in wanted or kid not in claim_pos:
                    continue
                x1, y1 = claim_pos[kid]
                _add_edge(x1 + 14, y1, x2 - 10, y2)

    nodes: list[str] = []
    for c in citations:
        x, y = cite_pos[c["key"]]
        title = c.get("title") or c["ref"]
        nodes.append(
            f"<g>"
            f"<title>{escape_xml(title)}</title>"
            f"{_bullseye(x, y)}"
            f'<text x="{x + 14:.1f}" y="{y + 4:.1f}" font-family="{FONT_MONO}" '
            f'font-size="10" fill="{INK}">'
            f"{escape_xml(c['ref'])}</text>"
            f"</g>"
        )

    for m, (x, y) in model_pos.items():
        nodes.append(
            f"<g>"
            f"{_bullseye(x, y)}"
            f'<text x="{x + 14:.1f}" y="{y + 4:.1f}" font-family="{FONT_MONO}" '
            f'font-size="10" fill="{INK}">'
            f"{escape_xml(truncate(m, 14))}</text>"
            f"</g>"
        )

    for cid in cluster_ids:
        x, y = claim_pos[cid]
        nodes.append(
            f"<g>"
            f"{_bullseye(x, y)}"
            f'<text x="{x + 14:.1f}" y="{y + 4:.1f}" font-family="{FONT_MONO}" '
            f'font-size="10" fill="{INK}">'
            f"{escape_xml(cid)}</text>"
            f"</g>"
        )

    if include_conclusions:
        for ch in chains:
            chid = ch["chain_id"]
            x, y = chain_pos[chid]
            tip = truncate(ch.get("label") or chid, 80)
            nodes.append(
                f"<g>"
                f"<title>{escape_xml(tip)}</title>"
                f"{_bullseye(x, y)}"
                f'<text x="{x + 14:.1f}" y="{y + 4:.1f}" font-family="{FONT_MONO}" '
                f'font-size="10" font-weight="600" fill="{INK}">'
                f"{escape_xml(chid)}</text>"
                f"</g>"
            )

    mid_y = (node_top + node_bot) / 2
    empties = [
        (not citations, x_cite, "No citations"),
        (not models, x_model, "No models"),
        (not cluster_ids, x_claim, "No clusters"),
    ]
    if include_conclusions:
        empties.append((not chains, x_chain, "No conclusions"))
    for empty, x, msg in empties:
        if not empty:
            continue
        nodes.append(
            f"{_bullseye(x, mid_y)}"
            f'<text x="{x + 14:.1f}" y="{mid_y + 4:.1f}" '
            f'font-family="{FONT}" font-size="9.5" fill="{MUTED}">'
            f"{escape_xml(msg)}</text>"
        )

    body = f"""  {"".join(bands)}
  {"".join(edges)}
  {"".join(nodes)}"""
    return svg_root(width, height, body)
