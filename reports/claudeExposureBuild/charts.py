"""Self-authored SVG charts for the claudeExposureBuild product.

Distinct chart forms from the default builder (donut, dimension lollipops,
model contribution bars, exposure ladder), on the shared MOYO palette. Charts
are returned as inline SVG strings so the renderer needs no asset files.
"""

from __future__ import annotations

import math

# MOYO palette (kept local so this product is self-contained).
TEAL = "#4FB0A2"
TEAL_DEEP = "#2F7A70"
TEAL_SOFT = "#A8D4CB"
CREAM = "#F2F1E8"
INK = "#1D2228"
MUTED = "#5C6570"
RULE = "#D9D4C8"
WHITE = "#FFFFFF"
INFO = "#C4BFB2"

FONT = "Inter, 'IBM Plex Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace"

SEV_ORDER = ["high", "medium", "low", "informational"]
SEV_COLOR = {
    "high": INK,
    "medium": TEAL_DEEP,
    "low": TEAL,
    "informational": INFO,
}
SEV_LABEL = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "informational": "Informational",
}


def _esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _truncate(text: object, n: int) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "\u2026"


def _svg(width: float, height: float, body: str, *, max_height: float | None = None) -> str:
    style = f"max-height:{max_height}mm;" if max_height else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="100%" preserveAspectRatio="xMidYMid meet" role="img" '
        f'style="display:block;{style}">'
        f'<rect width="100%" height="100%" fill="{WHITE}"/>{body}</svg>'
    )


def _text(x, y, s, *, size=12, fill=INK, weight="400", anchor="start", font=FONT, spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{font}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}"{ls}>{_esc(s)}</text>'
    )


def severity_donut_svg(bins: dict, *, compact: bool = False) -> str:
    """Ring of sensitivity bands with total in the center and a legend."""
    order = SEV_ORDER
    counts = {k: int(bins.get(k, 0) or 0) for k in order}
    total = sum(counts.values())

    width = 300
    height = 300 if not compact else 250
    cx, cy = 150, 138 if not compact else 118
    r = 92 if not compact else 78
    sw = 34 if not compact else 28
    circ = 2 * math.pi * r

    parts = [f'{_text(cx, 26, "Severity Distribution", size=14, weight="600", anchor="middle")}']
    # Track ring
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{CREAM}" stroke-width="{sw}"/>'
    )
    if total > 0:
        cumulative = 0.0
        for key in order:
            n = counts[key]
            if n <= 0:
                continue
            frac = n / total
            dash = frac * circ
            gap = circ - dash
            offset = -cumulative * circ
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
                f'stroke="{SEV_COLOR[key]}" stroke-width="{sw}" '
                f'stroke-dasharray="{dash:.2f} {gap:.2f}" stroke-dashoffset="{offset:.2f}" '
                f'transform="rotate(-90 {cx} {cy})"/>'
            )
            cumulative += frac
    parts.append(_text(cx, cy - 2, str(total), size=40, weight="700", anchor="middle"))
    parts.append(_text(cx, cy + 20, "Findings", size=12, fill=MUTED, anchor="middle",
                       spacing="0.08em"))

    # Legend (2 columns) under the ring
    lx0 = 40
    ly = cy + r + 26
    col_w = 120
    for i, key in enumerate(order):
        col = i % 2
        rowi = i // 2
        x = lx0 + col * col_w
        y = ly + rowi * 26
        parts.append(
            f'<rect x="{x}" y="{y - 9}" width="12" height="12" rx="2" fill="{SEV_COLOR[key]}"/>'
        )
        parts.append(_text(x + 18, y + 1, f"{SEV_LABEL[key]}", size=11, fill=INK))
        parts.append(_text(x + col_w - 16, y + 1, str(counts[key]), size=11,
                           weight="700", anchor="end"))
    return _svg(width, height, "".join(parts), max_height=(70 if compact else 95))


def dimension_bars_svg(averages: dict) -> str:
    """Horizontal lollipop bars for the averaged exposure dimensions (0-5)."""
    rows = [
        ("Sensitivity", float(averages.get("sensitivity", 0) or 0)),
        ("Specificity", float(averages.get("specificity", 0) or 0)),
        ("Novelty", float(averages.get("novelty", 0) or 0)),
        ("Confidence", float(averages.get("confidence", 0) or 0)),
        ("Corroboration", float(averages.get("corroboration", 0) or 0)),
    ]
    width = 520
    pad_l = 128
    pad_r = 46
    top = 46
    row_h = 40
    track_w = width - pad_l - pad_r
    height = top + row_h * len(rows) + 10

    parts = [_text(20, 26, "Exposure Dimensions", size=14, weight="600")]
    parts.append(_text(width - pad_r, 26, "scale 1\u20135", size=10, fill=MUTED, anchor="end"))
    # gridlines at 1..5
    for g in range(1, 6):
        gx = pad_l + track_w * (g / 5.0)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" y2="{top + row_h * len(rows) - 12}" '
            f'stroke="{RULE}" stroke-width="1" stroke-dasharray="2 4"/>'
        )
        parts.append(_text(gx, top + row_h * len(rows) + 2, str(g), size=9, fill=MUTED, anchor="middle"))

    for i, (label, val) in enumerate(rows):
        cy = top + i * row_h + row_h / 2 - 6
        frac = max(0.0, min(1.0, val / 5.0))
        bx = pad_l + track_w * frac
        parts.append(_text(pad_l - 12, cy + 4, label, size=12, anchor="end"))
        parts.append(
            f'<line x1="{pad_l}" y1="{cy}" x2="{pad_l + track_w}" y2="{cy}" '
            f'stroke="{CREAM}" stroke-width="6" stroke-linecap="round"/>'
        )
        parts.append(
            f'<line x1="{pad_l}" y1="{cy}" x2="{bx:.1f}" y2="{cy}" '
            f'stroke="{TEAL}" stroke-width="6" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{bx:.1f}" cy="{cy}" r="7" fill="{TEAL_DEEP}"/>')
        parts.append(_text(pad_l + track_w + 8, cy + 4, f"{val:.1f}", size=11,
                           weight="700", fill=INK))
    return _svg(width, height, "".join(parts), max_height=80)


def model_bars_svg(rows: list[dict], *, top_n: int = 10) -> str:
    """Horizontal bars ranking models by number of findings contributed."""
    rows = [r for r in rows if r.get("count")]
    rows = sorted(rows, key=lambda r: -int(r.get("count") or 0))[:top_n]
    width = 560
    pad_l = 150
    pad_r = 40
    top = 44
    row_h = 30
    track_w = width - pad_l - pad_r
    height = top + row_h * max(1, len(rows)) + 12
    max_count = max((int(r["count"]) for r in rows), default=1) or 1

    parts = [_text(20, 26, "Model Contribution", size=14, weight="600")]
    parts.append(_text(width - pad_r, 26, "findings", size=10, fill=MUTED, anchor="end"))
    if not rows:
        parts.append(_text(width / 2, height / 2, "No model data", size=12, fill=MUTED, anchor="middle"))
        return _svg(width, height, "".join(parts), max_height=90)

    for i, r in enumerate(rows):
        y = top + i * row_h
        cy = y + row_h / 2 - 4
        count = int(r["count"])
        frac = count / max_count
        bw = max(3.0, track_w * frac)
        parts.append(_text(pad_l - 12, cy + 4, _truncate(r.get("model", "?"), 22),
                           size=11, anchor="end"))
        parts.append(
            f'<rect x="{pad_l}" y="{cy - 8}" width="{track_w}" height="16" rx="3" fill="{CREAM}"/>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{cy - 8}" width="{bw:.1f}" height="16" rx="3" fill="{TEAL}"/>'
        )
        parts.append(_text(pad_l + bw + 8, cy + 4, str(count), size=11, weight="700"))
    return _svg(width, height, "".join(parts), max_height=95)

