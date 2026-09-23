"""Compact per-model charts: fingerprint, sensitivity mix, probes, overlap.

These are not the cross-model bars / corpus radar / heatmap already in the
risk and comparison pages.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .style import (
    BAR_COLORS,
    BAR_LABELS,
    DISCLOSURE_CHART_ORDER,
    FONT,
    FONT_MONO,
    INK,
    MUTED,
    RULE,
    TEAL,
    TEAL_DEEP,
    WHITE,
    escape_xml,
    svg_root,
)

_AXES = (
    ("specificity", "Spec"),
    ("sensitivity", "Sens"),
    ("corroboration", "Corr"),
    ("novelty", "Nov"),
    ("confidence", "Conf"),
)


def model_fingerprint_svg(
    model_avgs: Mapping[str, float] | None,
    corpus_avgs: Mapping[str, float] | None,
    *,
    width: int = 280,
    height: int = 248,
) -> str:
    """Overlay this model's mean scores on the corpus mean (hairline)."""
    cx, cy = width / 2.0, 128.0
    r_max = 74.0
    n = len(_AXES)

    def _pt(i: int, level: float) -> tuple[float, float]:
        ang = -math.pi / 2 + (2 * math.pi * i / n)
        rr = r_max * (max(0.0, min(5.0, level)) / 5.0)
        return cx + rr * math.cos(ang), cy + rr * math.sin(ang)

    rings = []
    for level in (1, 3, 5):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_pt(i, level) for i in range(n)))
        rings.append(
            f'<polygon points="{pts}" fill="none" stroke="{RULE}" stroke-width="0.8"/>'
        )

    def _poly(avgs: Mapping[str, float] | None, fill: str, stroke: str, opacity: str) -> str:
        values = [float((avgs or {}).get(k, 0) or 0) for k, _ in _AXES]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_pt(i, values[i]) for i in range(n)))
        return (
            f'<polygon points="{pts}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="1.6"/>'
        )

    labels = []
    for i, (_, label) in enumerate(_AXES):
        ang = -math.pi / 2 + (2 * math.pi * i / n)
        lx = cx + (r_max + 16) * math.cos(ang)
        ly = cy + (r_max + 16) * math.sin(ang)
        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="9" fill="{INK}">'
            f"{escape_xml(label)}</text>"
        )

    legend = (
        f'<rect x="8" y="6" width="10" height="6" fill="{TEAL}" fill-opacity="0.45" '
        f'stroke="{TEAL_DEEP}" stroke-width="1"/>'
        f'<text x="22" y="12" font-family="{FONT}" font-size="9" fill="{INK}">This model</text>'
        f'<rect x="92" y="6" width="10" height="6" fill="none" stroke="{MUTED}" stroke-width="1.4"/>'
        f'<text x="106" y="12" font-family="{FONT}" font-size="9" fill="{INK}">Corpus mean</text>'
    )

    body = f"""  {"".join(rings)}
  {_poly(corpus_avgs, "none", MUTED, "0")}
  {_poly(model_avgs, TEAL, TEAL_DEEP, "0.38")}
  {"".join(labels)}
  {legend}"""
    return svg_root(width, height, body)


def model_mix_svg(
    bands: Mapping[str, Any] | None,
    *,
    width: int = 280,
    height: int = 112,
) -> str:
    """100% stacked bar of this model's disclosure mix."""
    keys = set((bands or {}).keys())
    if "damaging" in keys or "potentially_damaging" in keys:
        order = ("damaging", "potentially_damaging", "unexpected", "interesting")
    elif "commercially_sensitive" in keys or "potentially_strategic" in keys:
        order = (
            "commercially_sensitive",
            "potentially_strategic",
            "unexpected",
            "interesting",
        )
    elif "material" in keys:
        order = ("security_relevant", "material", "unexpected", "interesting")
    elif "sensitive" in keys and "security_relevant" not in keys:
        order = ("sensitive", "unexpected", "interesting", "expected")
    else:
        order = DISCLOSURE_CHART_ORDER
    counts = [max(0, int((bands or {}).get(k) or 0)) for k in order]
    total = sum(counts) or 1
    x, y, bar_h = 8, 28, 22
    usable = width - 16
    chunks = []
    cursor = x
    for key, count in zip(order, counts):
        w = usable * (count / total)
        if w <= 0:
            continue
        chunks.append(
            f'<rect x="{cursor:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" '
            f'fill="{BAR_COLORS[key]}"/>'
        )
        if w >= 28:
            chunks.append(
                f'<text x="{cursor + w / 2:.1f}" y="{y + 15:.1f}" text-anchor="middle" '
                f'font-family="{FONT_MONO}" font-size="9" fill="{WHITE}">'
                f"{count}</text>"
            )
        cursor += w
    legend = []
    for row_index, keys in enumerate((order[:2], order[2:])):
        lx = 8
        y = height - 34 + row_index * 16
        for key in keys:
            legend.append(
                f'<rect x="{lx}" y="{y}" width="8" height="8" fill="{BAR_COLORS[key]}"/>'
                f'<text x="{lx + 11}" y="{y + 8}" font-family="{FONT}" font-size="8" '
                f'fill="{INK}">{escape_xml(BAR_LABELS[key])}</text>'
            )
            lx += 136
    body = f"""  <text x="8" y="16" font-family="{FONT}" font-size="9" font-weight="600" fill="{MUTED}">Disclosure mix</text>
  {"".join(chunks)}
  {"".join(legend)}"""
    return svg_root(width, height, body)


def model_probes_svg(
    probes: Mapping[str, Any] | None,
    *,
    width: int = 280,
    height: int = 88,
) -> str:
    """Probe outcomes for this model only (answered / empty / failed)."""
    keys = ("answered", "empty", "failed")
    colors = {"answered": TEAL_DEEP, "empty": RULE, "failed": INK}
    labels = {"answered": "Answered", "empty": "Empty", "failed": "Failed"}
    counts = [max(0, int((probes or {}).get(k) or 0)) for k in keys]
    total = sum(counts) or 1
    x, y, bar_h = 8, 28, 22
    usable = width - 16
    chunks = []
    cursor = x
    for key, count in zip(keys, counts):
        w = usable * (count / total)
        if w <= 0:
            continue
        chunks.append(
            f'<rect x="{cursor:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" '
            f'fill="{colors[key]}"/>'
        )
        if w >= 28:
            chunks.append(
                f'<text x="{cursor + w / 2:.1f}" y="{y + 15:.1f}" text-anchor="middle" '
                f'font-family="{FONT_MONO}" font-size="9" fill="{WHITE if key != "empty" else INK}">'
                f"{count}</text>"
            )
        cursor += w
    legend = []
    lx = 8
    for key in keys:
        legend.append(
            f'<rect x="{lx}" y="{height - 18}" width="8" height="8" fill="{colors[key]}"/>'
            f'<text x="{lx + 11}" y="{height - 11}" font-family="{FONT}" font-size="8" '
            f'fill="{INK}">{escape_xml(labels[key])}</text>'
        )
        lx += 72
    attempted = int((probes or {}).get("attempted") or sum(counts))
    body = f"""  <text x="8" y="16" font-family="{FONT}" font-size="9" font-weight="600" fill="{MUTED}">Probe Outcomes · {attempted} attempted</text>
  {"".join(chunks)}
  {"".join(legend)}"""
    return svg_root(width, height, body)


def model_overlap_svg(
    unique: int,
    shared: int,
    *,
    width: int = 220,
    height: int = 160,
) -> str:
    """Donut of exclusive vs shared findings for this model."""
    unique = max(0, int(unique or 0))
    shared = max(0, int(shared or 0))
    total = unique + shared
    cx, cy, r, ri = width / 2, height / 2 + 6, 48, 28

    def _arc(start: float, frac: float, color: str) -> str:
        if frac <= 0 or total <= 0:
            return ""
        if frac >= 0.999:
            return (
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}"/>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{ri}" fill="{WHITE}"/>'
            )
        a0 = -math.pi / 2 + start * 2 * math.pi
        a1 = a0 + frac * 2 * math.pi
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        large = 1 if frac > 0.5 else 0
        xi0, yi0 = cx + ri * math.cos(a1), cy + ri * math.sin(a1)
        xi1, yi1 = cx + ri * math.cos(a0), cy + ri * math.sin(a0)
        return (
            f'<path d="M {x0:.1f},{y0:.1f} A {r},{r} 0 {large} 1 {x1:.1f},{y1:.1f} '
            f'L {xi0:.1f},{yi0:.1f} A {ri},{ri} 0 {large} 0 {xi1:.1f},{yi1:.1f} Z" '
            f'fill="{color}"/>'
        )

    u_frac = (unique / total) if total else 0
    s_frac = (shared / total) if total else 0
    body = f"""  <text x="8" y="14" font-family="{FONT}" font-size="9" font-weight="600" fill="{MUTED}">Exclusive vs Shared</text>
  {_arc(0, u_frac, INK)}
  {_arc(u_frac, s_frac, TEAL)}
  <text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="{FONT_MONO}" font-size="12" fill="{INK}">{total}</text>
  <rect x="8" y="{height - 16}" width="8" height="8" fill="{INK}"/>
  <text x="20" y="{height - 9}" font-family="{FONT}" font-size="8" fill="{INK}">Exclusive {unique}</text>
  <rect x="118" y="{height - 16}" width="8" height="8" fill="{TEAL}"/>
  <text x="130" y="{height - 9}" font-family="{FONT}" font-size="8" fill="{INK}">Shared {shared}</text>"""
    return svg_root(width, height, body)


def generate_dossier_graphics(dossiers: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """SVG strings keyed to ``dossier.charts``."""
    graphics: dict[str, str] = {}
    for row in dossiers or []:
        charts = row.get("charts") if isinstance(row.get("charts"), Mapping) else {}
        fp = charts.get("fingerprint")
        if fp:
            graphics[str(fp)] = model_fingerprint_svg(
                row.get("radar") or {},
                row.get("corpus_radar") or {},
            )
        mix = charts.get("mix")
        if mix:
            graphics[str(mix)] = model_mix_svg(row.get("bands") or {})
        probes = charts.get("probes")
        if probes:
            graphics[str(probes)] = model_probes_svg(row.get("probes") or {})
        overlap = charts.get("overlap")
        if overlap:
            graphics[str(overlap)] = model_overlap_svg(
                int(row.get("unique") or 0),
                int(row.get("shared") or 0),
            )
    return graphics
