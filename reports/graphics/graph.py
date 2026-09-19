"""Evidence graph SVG: citations → model inference → claims → conclusions.

Visual language matches how-it-works EvidenceGraph: teal bullseye nodes on every
column, teal curves only, ink/muted labels (no teal text).
"""

from __future__ import annotations

import re
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
    *,
    opacity: float = 0.28,
) -> str:
    dx = max(36.0, abs(x2 - x1) * 0.4)
    c1x = x1 + dx
    c2x = x2 - dx * 0.65
    return (
        f'<path d="M {x1:.1f},{y1:.1f} C {c1x:.1f},{y1:.1f} {c2x:.1f},{y2:.1f} '
        f'{x2:.1f},{y2:.1f}" fill="none" stroke="{TEAL}" '
        f'stroke-opacity="{opacity:.2f}" stroke-width="1.2"/>'
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
    raw_models = finding.get("source_models")
    if not isinstance(raw_models, list) or not raw_models:
        raw_models = [finding.get("source_model") or ""]
    out: set[str] = set()
    for raw in raw_models:
        m = short_model_name(str(raw or ""), aliases)
        if m and m != "unknown":
            out.add(m)
    return out


def _claim_sort_key(finding: dict) -> tuple:
    try:
        sens = int(finding.get("sensitivity") or 0)
    except (TypeError, ValueError):
        sens = 0
    return (-sens, str(finding.get("claim_id") or ""))


def evidence_graph_svg(
    findings: Iterable[dict],
    chains: Iterable[dict],
    *,
    models_probed: Sequence[str] | None = None,
    max_citations: int = 8,
    max_models: int = 9,
    max_claims: int = 8,
    max_conclusions: int = 4,
    aliases: dict[str, str] | None = None,
    max_width: float = 700,
    max_height: float = PRINT_MAX_HEIGHT,
) -> str:
    """Print evidence graph matching /how-it-works EvidenceGraph columns."""
    aliases = aliases or {}
    all_findings = list(findings)
    by_id = {f.get("claim_id"): f for f in all_findings if f.get("claim_id")}
    chains = [c for c in list(chains)[:max_conclusions] if c.get("chain_id")]

    width = min(max(max_width, 680), 720)
    height = min(max_height, 420)

    band_top = 8
    band_bottom = height - 10
    footer_h = 18
    node_top = band_top + 22
    node_bot = band_bottom - footer_h - 16

    citations = _collect_citations(all_findings, limit=max_citations)

    models: list[str] = []
    seen_m: set[str] = set()
    probed = [str(m).strip() for m in (models_probed or []) if str(m).strip()]
    source_pool: list[str] = list(probed) if probed else []
    if not source_pool:
        for f in all_findings:
            source_pool.extend(_finding_models(f, aliases))
    for raw in source_pool:
        m = short_model_name(str(raw), aliases)
        if not m or m == "unknown" or m in seen_m:
            continue
        seen_m.add(m)
        models.append(m)
        if len(models) >= max_models:
            break

    # Prefer claims that participate in selected chains; fill from findings.
    claim_ids: list[str] = []
    seen_c: set[str] = set()
    for ch in chains:
        for cid in list(ch.get("claim_ids") or []):
            cid = str(cid)
            if not cid or cid in seen_c or cid not in by_id:
                continue
            seen_c.add(cid)
            claim_ids.append(cid)
            if len(claim_ids) >= max_claims:
                break
        if len(claim_ids) >= max_claims:
            break
    if len(claim_ids) < max_claims:
        ranked = sorted(
            (f for f in all_findings if f.get("claim_id")),
            key=_claim_sort_key,
        )
        for f in ranked:
            cid = str(f.get("claim_id"))
            if cid in seen_c:
                continue
            seen_c.add(cid)
            claim_ids.append(cid)
            if len(claim_ids) >= max_claims:
                break

    cite_ys = _ys(len(citations) or 1, node_top, node_bot)
    model_ys = _ys(len(models) or 1, node_top, node_bot)
    claim_ys = _ys(len(claim_ids) or 1, node_top, node_bot)
    chain_ys = _ys(len(chains) or 1, node_top, node_bot)

    # Column geometry scaled from the 960px how-it-works SVG.
    x_cite, x_model, x_claim, x_chain = 28, 250, 430, 560
    col_boxes = (
        (8, 155, "Real-world citations"),
        (220, 130, "Model inference"),
        (400, 115, "Claims"),
        (540, 150, "Inferred conclusions"),
    )

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
    claim_pos = {cid: (x_claim, claim_ys[i]) for i, cid in enumerate(claim_ids)}
    chain_pos = {c["chain_id"]: (x_chain, chain_ys[i]) for i, c in enumerate(chains)}

    edges: list[str] = []

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
                edges.append(_curve(x1 + 120, y1, x2 - 10, y2, opacity=0.22))

    # Model inference → claims
    for cid in claim_ids:
        f = by_id.get(cid)
        if not f or cid not in claim_pos:
            continue
        x2, y2 = claim_pos[cid]
        for m in _finding_models(f, aliases):
            if m not in model_pos:
                continue
            x1, y1 = model_pos[m]
            edges.append(_curve(x1 + 14, y1, x2 - 10, y2, opacity=0.28))

    # Claims → inferred conclusions
    for ch in chains:
        chid = ch["chain_id"]
        if chid not in chain_pos:
            continue
        x2, y2 = chain_pos[chid]
        wanted = {str(c) for c in (ch.get("claim_ids") or [])}
        for cid in claim_ids:
            if cid not in wanted or cid not in claim_pos:
                continue
            x1, y1 = claim_pos[cid]
            edges.append(_curve(x1 + 14, y1, x2 - 10, y2, opacity=0.28))

    nodes: list[str] = []
    for c in citations:
        x, y = cite_pos[c["key"]]
        nodes.append(
            f"<g>"
            f"{_bullseye(x, y)}"
            f'<text x="{x + 14:.1f}" y="{y - 3:.1f}" font-family="{FONT_MONO}" '
            f'font-size="9.5" font-weight="600" fill="{INK}">'
            f"{escape_xml(c['ref'])}</text>"
            f'<text x="{x + 14:.1f}" y="{y + 11:.1f}" font-family="{FONT}" '
            f'font-size="9.5" font-style="italic" fill="{MUTED}">'
            f"{escape_xml(c['title'])}</text>"
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

    for cid in claim_ids:
        x, y = claim_pos[cid]
        nodes.append(
            f"<g>"
            f"{_bullseye(x, y)}"
            f'<text x="{x + 14:.1f}" y="{y + 4:.1f}" font-family="{FONT_MONO}" '
            f'font-size="10" fill="{INK}">'
            f"{escape_xml(cid)}</text>"
            f"</g>"
        )

    for ch in chains:
        chid = ch["chain_id"]
        x, y = chain_pos[chid]
        title = truncate(ch.get("label") or chid, 22)
        nodes.append(
            f"<g>"
            f"<title>{escape_xml(title)}</title>"
            f"{_bullseye(x, y)}"
            f'<text x="{x + 14:.1f}" y="{y - 3:.1f}" font-family="{FONT_MONO}" '
            f'font-size="9.5" font-weight="600" fill="{INK}">'
            f"{escape_xml(chid)}</text>"
            f'<text x="{x + 14:.1f}" y="{y + 11:.1f}" font-family="{FONT}" '
            f'font-size="9.5" font-style="italic" fill="{MUTED}">'
            f"{escape_xml(title)}</text>"
            f"</g>"
        )

    mid_y = (node_top + node_bot) / 2
    empties = (
        (not citations, x_cite, "No citations"),
        (not models, x_model, "No models"),
        (not claim_ids, x_claim, "No claims"),
        (not chains, x_chain, "No conclusions"),
    )
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
