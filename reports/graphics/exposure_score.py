"""Exposure radar + sensitivity distribution SVGs."""

from __future__ import annotations

import math
from typing import Mapping

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

    body = f"""  {title_block(cx, 20, "Exposure Radar")}
  {panel}
  {"".join(rings)}
  {"".join(spokes)}
  {polygon}
  {"".join(dots)}
  {"".join(labels)}"""
    return svg_root(size, size, body)


def sensitivity_bars_svg(bins: Mapping[str, int], width: int = 520, height: int = 304) -> str:
    """Bar chart with cream panel large enough for counts and category labels."""
    order = ["high", "medium", "low", "informational"]
    values = [int(bins.get(k, 0)) for k in order]
    peak = max(values) or 1

    # Plot sits lower so title/subtitle stay clear of the panel.
    left, right, top, bottom = 48, 24, 64 + 2 * LETTER_H, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    base = top + plot_h
    n = len(order)
    gap = 28
    bar_w = (plot_w - gap * (n - 1)) / n

    # Restored panel height (pad 28/36); still extended left by one letter-width.
    # Positioned lower than the pre-change y so title + subtitle remain visible.
    panel_pad_x = 20
    panel_pad_top = 28
    panel_pad_bot = 36
    panel_left = left - panel_pad_x - LETTER_W
    panel_y = top - panel_pad_top
    panel = (
        f'<rect x="{panel_left}" y="{panel_y}" '
        f'width="{plot_w + 2 * panel_pad_x + LETTER_W}" '
        f'height="{plot_h + panel_pad_top + panel_pad_bot}" '
        f'rx="10" fill="{CREAM}" stroke="{RULE}"/>'
    )

    # Horizontal grid + baseline
    grid = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = base - frac * plot_h
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{RULE}" stroke-width="1"/>'
        )
        tick_v = int(round(peak * frac))
        grid.append(
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="9" fill="{MUTED}">{tick_v}</text>'
        )
    grid.append(
        f'<line x1="{left}" y1="{base}" x2="{left + plot_w}" y2="{base}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )

    bars = []
    for i, (name, v) in enumerate(zip(order, values)):
        color = BAR_COLORS[name]
        label = BAR_LABELS[name]
        # Zero counts still show a hairline stub so the category is visible
        if v == 0:
            h = 3.0
            color = RULE
        else:
            h = max((v / peak) * plot_h, 8.0)
        x = left + i * (bar_w + gap)
        y = base - h
        # Rounded top via path (rect + top radius)
        r = min(6.0, bar_w / 2, h / 2)
        path = (
            f"M {x:.1f},{base:.1f} "
            f"L {x:.1f},{y + r:.1f} "
            f"Q {x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"L {x + bar_w - r:.1f},{y:.1f} "
            f"Q {x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},{y + r:.1f} "
            f"L {x + bar_w:.1f},{base:.1f} Z"
        )
        bars.append(f'<path d="{path}" fill="{color}"/>')
        bars.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="13" font-weight="600" fill="{INK}">{v}</text>'
        )
        bars.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{base + 22}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" fill="{MUTED}">{escape_xml(label)}</text>'
        )

    body = f"""  {title_block(width / 2, 20, "Sensitivity Distribution")}
  {panel}
  {"".join(grid)}
  {"".join(bars)}"""
    return svg_root(width, height, body)
