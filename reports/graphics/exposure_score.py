"""Exposure radar + findings-by-LLM bar chart SVGs."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .style import (
    BAR_COLORS,
    BAR_LABELS,
    CREAM,
    CREAM_DEEP,
    FONT,
    INK,
    LETTER_H,
    LETTER_W,
    MUTED,
    RULE,
    TEAL,
    TEAL_DEEP,
    WHITE,
    escape_xml,
    svg_root,
    title_block,
    truncate,
)


def exposure_radar_svg(averages: Mapping[str, float], size: int = 380) -> str:
    """Radar chart with title above a cream panel that contains all axis labels."""
    axes = [
        ("specificity", "Specificity"),
        ("sensitivity", "Sensitivity"),
        ("corroboration", "Corroboration"),
        ("novelty", "Novelty"),
        ("confidence", "Confidence"),
    ]
    # Title sits outside the panel; radar + labels live inside it.
    # Panel shifted up one letter-height vs the title band.
    title_h = 52 - LETTER_H
    pad = 14
    panel_x = pad
    panel_y = title_h
    panel_w = size - 2 * pad
    panel_h = size - title_h - pad

    cx = size / 2
    cy = panel_y + panel_h / 2
    label_r = 34  # distance from ring edge to axis label center
    r_max = min(panel_w, panel_h) / 2 - label_r - 10
    n = len(axes)

    def _pt(i: int, level: float) -> tuple[float, float]:
        ang = -math.pi / 2 + (2 * math.pi * i / n)
        rr = r_max * (level / 5.0)
        return cx + rr * math.cos(ang), cy + rr * math.sin(ang)

    panel = (
        f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" '
        f'height="{panel_h}" rx="10" fill="{CREAM}" stroke="{RULE}"/>'
    )

    rings = []
    for level in (1, 2, 3, 4, 5):
        ring_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_pt(i, level) for i in range(n)))
        stroke = RULE if level < 5 else CREAM_DEEP
        width = 1 if level < 5 else 1.25
        rings.append(
            f'<polygon points="{ring_pts}" fill="none" stroke="{stroke}" stroke-width="{width}"/>'
        )
        # Tick labels on the top axis (specificity)
        tx, ty = _pt(0, level)
        rings.append(
            f'<text x="{tx + 4:.1f}" y="{ty + 3:.1f}" font-family="{FONT}" '
            f'font-size="8" fill="{MUTED}">{level}</text>'
        )

    spokes = []
    for i in range(n):
        x, y = _pt(i, 5)
        spokes.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{RULE}" stroke-width="1"/>'
        )

    values = [max(0.0, min(5.0, float(averages.get(key, 0)))) for key, _ in axes]
    poly_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (_pt(i, values[i]) for i in range(n)))
    polygon = (
        f'<polygon points="{poly_pts}" fill="{TEAL}" fill-opacity="0.42" '
        f'stroke="{TEAL_DEEP}" stroke-width="2.25"/>'
    )

    dots = []
    for i, val in enumerate(values):
        x, y = _pt(i, val)
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{WHITE}" '
            f'stroke="{TEAL_DEEP}" stroke-width="1.75"/>'
        )

    labels = []
    for i, (_, label) in enumerate(axes):
        ang = -math.pi / 2 + (2 * math.pi * i / n)
        lx = cx + (r_max + label_r) * math.cos(ang)
        ly = cy + (r_max + label_r) * math.sin(ang)
        # Keep vertical labels clear of the outer ring; Specificity one letter lower.
        if abs(math.cos(ang)) < 0.2:
            if math.sin(ang) < 0:
                ly += -4 + LETTER_H
            else:
                ly += 8
        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" font-weight="500" fill="{INK}">'
            f"{escape_xml(label)}</text>"
        )

    body = f"""  {title_block(cx, 20, "Finding Classification Profile")}
  {panel}
  {"".join(rings)}
  {"".join(spokes)}
  {polygon}
  {"".join(dots)}
  {"".join(labels)}"""
    return svg_root(size, size, body)


def _band_score(row: Mapping[str, Any], band: str) -> float:
    bands = row.get("bands") if isinstance(row.get("bands"), Mapping) else {}
    cell = bands.get(band) if isinstance(bands, Mapping) else None
    if isinstance(cell, Mapping):
        return float(cell.get("score", 0) or 0)
    return float(row.get(f"{band}_score", 0) or 0)


def _llm_row_score(row: Mapping[str, Any]) -> float:
    score = row.get("score")
    if score is not None:
        return max(0.0, float(score or 0))
    return sum(_band_score(row, k) for k in ("high", "medium", "low", "informational"))


def llm_findings_bars_svg(
    rows: Sequence[Mapping[str, Any]],
    width: int = 520,
    height: int = 320,
) -> str:
    """Bar chart of test LLMs scored by finding quantity and sensitivity.

    Bar height is the sum of finding sensitivities for that model. Stacks are
    colored by sensitivity band so both volume and severity are visible.
    """
    band_order = ("informational", "low", "medium", "high")  # bottom → top
    series = [dict(r) for r in (rows or []) if r.get("model")]
    series.sort(key=lambda r: (-_llm_row_score(r), str(r.get("model") or "")))
    peak = max((_llm_row_score(r) for r in series), default=0.0) or 1.0

    # Title + subtitle sit above the panel (subtitle is 16px below the title).
    left, right, top, bottom = 48, 20, 72, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    base = top + plot_h
    n = max(len(series), 1)
    gap = 10.0 if n >= 8 else (16.0 if n >= 5 else 24.0)
    bar_w = min(52.0, (plot_w - gap * (n - 1)) / n)
    total_w = n * bar_w + max(n - 1, 0) * gap
    x0 = left + max(0.0, (plot_w - total_w) / 2)

    panel_pad_x = 20
    panel_pad_top = 18
    panel_pad_bot = 48
    panel_left = left - panel_pad_x - LETTER_W
    panel_y = top - panel_pad_top
    panel = (
        f'<rect x="{panel_left}" y="{panel_y}" '
        f'width="{plot_w + 2 * panel_pad_x + LETTER_W}" '
        f'height="{plot_h + panel_pad_top + panel_pad_bot}" '
        f'rx="10" fill="{CREAM}" stroke="{RULE}"/>'
    )

    grid = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = base - frac * plot_h
        tick_v = int(round(peak * frac))
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{RULE}" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="9" fill="{MUTED}">{tick_v}</text>'
        )
    grid.append(
        f'<line x1="{left}" y1="{base}" x2="{left + plot_w}" y2="{base}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )

    bars: list[str] = []
    name_limit = 7 if n >= 9 else (9 if n >= 6 else 12)
    label_size = 9 if n >= 8 else 11
    for i, row in enumerate(series):
        x = x0 + i * (bar_w + gap)
        total = _llm_row_score(row)
        segments = [(band, _band_score(row, band)) for band in band_order]
        live = [(band, sc) for band, sc in segments if sc > 0]
        cx = x + bar_w / 2

        if not live or total <= 0:
            h = 3.0
            y = base - h
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                f'fill="{RULE}"/>'
            )
        else:
            y_cursor = base
            for j, (band, sc) in enumerate(live):
                h = (sc / peak) * plot_h
                if h <= 0:
                    continue
                y = y_cursor - h
                is_top = j == len(live) - 1
                color = BAR_COLORS[band]
                if is_top:
                    r = min(6.0, bar_w / 2, h / 2)
                    path = (
                        f"M {x:.1f},{y_cursor:.1f} "
                        f"L {x:.1f},{y + r:.1f} "
                        f"Q {x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
                        f"L {x + bar_w - r:.1f},{y:.1f} "
                        f"Q {x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},{y + r:.1f} "
                        f"L {x + bar_w:.1f},{y_cursor:.1f} Z"
                    )
                    bars.append(f'<path d="{path}" fill="{color}"/>')
                else:
                    bars.append(
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                        f'height="{h:.1f}" fill="{color}"/>'
                    )
                y_cursor = y
            label = str(int(round(total)))
            bars.append(
                f'<text x="{cx:.1f}" y="{y_cursor - 8:.1f}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="12" font-weight="600" fill="{INK}">'
                f"{escape_xml(label)}</text>"
            )

        model = truncate(str(row.get("model") or ""), name_limit)
        bars.append(
            f'<text x="{cx:.1f}" y="{base + 18}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="{label_size}" fill="{MUTED}">'
            f"{escape_xml(model)}</text>"
        )

    if not series:
        bars.append(
            f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h / 2:.1f}" '
            f'text-anchor="middle" font-family="{FONT}" font-size="12" fill="{MUTED}">'
            f"No findings</text>"
        )

    legend = []
    legend_y = height - 18
    items = [(k, BAR_LABELS[k]) for k in ("high", "medium", "low", "informational")]
    item_w = 72
    legend_w = item_w * len(items)
    lx = (width - legend_w) / 2 + 8
    for name, label in items:
        legend.append(
            f'<rect x="{lx:.1f}" y="{legend_y - 8:.1f}" width="10" height="10" rx="2" '
            f'fill="{BAR_COLORS[name]}"/>'
        )
        legend.append(
            f'<text x="{lx + 14:.1f}" y="{legend_y:.1f}" font-family="{FONT}" '
            f'font-size="10" fill="{MUTED}">{escape_xml(label)}</text>'
        )
        lx += item_w

    body = f"""  {title_block(width / 2, 20, "Findings by LLM", subtitle="Score = findings × sensitivity")}
  {panel}
  {"".join(grid)}
  {"".join(bars)}
  {"".join(legend)}"""
    return svg_root(width, height, body)
