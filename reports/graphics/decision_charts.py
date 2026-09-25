"""Decision-oriented charts: findings by model, reproduction matrix, dossier dot plot."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .style import (
    CREAM,
    FONT,
    FONT_MONO,
    INK,
    MUTED,
    RULE,
    TEAL_DEEP,
    TEAL_SOFT,
    WHITE,
    escape_xml,
    svg_root,
)

LABEL_SIZE = 13
SMALL_SIZE = 11


def _clip_words(text: str, n: int) -> str:
    words = " ".join(str(text or "").split())
    if len(words) <= n:
        return words
    return words[: n - 1].rsplit(" ", 1)[0].rstrip(",;:–—-") + "…"


def model_findings_bars_svg(rows: Sequence[Mapping[str, Any]], *, width: int = 640) -> str:
    """Per model: all findings and high-significance findings as two separate bars."""
    rows = list(rows or [])
    label_w, value_w = 104, 44
    bar_h, pair_gap, group_gap = 15, 3, 16
    top = 34
    group_h = 2 * bar_h + pair_gap
    height = top + max(1, len(rows)) * (group_h + group_gap) + 4
    usable = width - label_w - value_w
    peak = max([int(row.get("findings") or 0) for row in rows] or [1]) or 1

    parts = [
        f'<rect x="{label_w}" y="6" width="12" height="12" fill="{TEAL_SOFT}"/>'
        f'<text x="{label_w + 18}" y="16" font-family="{FONT}" font-size="{SMALL_SIZE}" fill="{INK}">All findings</text>'
        f'<rect x="{label_w + 118}" y="6" width="12" height="12" fill="{TEAL_DEEP}"/>'
        f'<text x="{label_w + 136}" y="16" font-family="{FONT}" font-size="{SMALL_SIZE}" fill="{INK}">High significance</text>'
    ]
    if not rows:
        parts.append(
            f'<text x="{label_w}" y="{top + 16}" font-family="{FONT}" font-size="{LABEL_SIZE}" '
            f'fill="{MUTED}">No model produced a finding.</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * (group_h + group_gap)
        total = int(row.get("findings") or 0)
        high = int(row.get("high") or 0)
        parts.append(
            f'<text x="{label_w - 10}" y="{y + group_h / 2 + 4.5:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="{LABEL_SIZE}" font-weight="600" fill="{INK}">'
            f"{escape_xml(str(row.get('model') or ''))}</text>"
        )
        for offset, value, color in ((0, total, TEAL_SOFT), (bar_h + pair_gap, high, TEAL_DEEP)):
            w = usable * value / peak
            parts.append(
                f'<rect x="{label_w}" y="{y + offset}" width="{max(w, 1.5):.1f}" '
                f'height="{bar_h}" fill="{color}"/>'
                f'<text x="{label_w + w + 6:.1f}" y="{y + offset + bar_h - 3.5:.1f}" '
                f'font-family="{FONT_MONO}" font-size="{SMALL_SIZE}" fill="{INK}">{value}</text>'
            )
    parts.append(
        f'<line x1="{label_w}" y1="{top - 6}" x2="{label_w}" y2="{height - 6}" '
        f'stroke="{RULE}" stroke-width="1"/>'
    )
    return svg_root(width, height, "".join(parts))


def reproduction_matrix_svg(rep: Mapping[str, Any] | None, *, width: int = 640) -> str:
    """Findings (columns) by model (rows). Filled = the model stated it."""
    rep = rep or {}
    models = list(rep.get("rows") or [])
    columns = list(rep.get("columns") or [])
    notes = list(rep.get("notes") or [])
    if not models or not columns:
        return svg_root(
            width,
            60,
            f'<text x="0" y="30" font-family="{FONT}" font-size="{LABEL_SIZE}" fill="{MUTED}">'
            "No findings to compare across models.</text>",
        )

    label_w, gap_split = 92, 10
    multi = [col for col in columns if col["n"] >= 2]
    single = [col for col in columns if col["n"] < 2]
    split = gap_split if multi and single else 0
    cw = min(26.0, (width - label_w - split - 4) / len(columns))
    ch = 22.0
    head_y, marker_y = 14.0, 56.0
    grid_y = 72.0

    def col_x(index: int) -> float:
        extra = split if index >= len(multi) else 0
        return label_w + index * cw + extra

    parts: list[str] = []
    for group, start, text in (
        (multi, 0, f"Stated by 2+ models ({len(multi)})"),
        (single, len(multi), f"One model only ({len(single)})"),
    ):
        if not group:
            continue
        x0 = col_x(start)
        x1 = col_x(start + len(group) - 1) + cw
        parts.append(
            f'<text x="{x0:.1f}" y="{head_y}" font-family="{FONT}" font-size="{SMALL_SIZE}" '
            f'font-weight="600" fill="{INK}">{escape_xml(text)}</text>'
            f'<line x1="{x0:.1f}" y1="{head_y + 6}" x2="{x1 - 1:.1f}" y2="{head_y + 6}" '
            f'stroke="{INK}" stroke-width="1"/>'
        )

    for r, model in enumerate(models):
        y = grid_y + r * ch
        parts.append(
            f'<text x="{label_w - 10}" y="{y + ch / 2 + 4.5:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="{LABEL_SIZE}" fill="{INK}">{escape_xml(model)}</text>'
        )
        for c, col in enumerate(columns):
            filled = model in col["models"]
            parts.append(
                f'<rect x="{col_x(c) + 0.75:.1f}" y="{y + 1:.1f}" width="{cw - 1.5:.1f}" '
                f'height="{ch - 2:.1f}" fill="{TEAL_DEEP if filled else CREAM}"/>'
            )

    grid_bottom = grid_y + len(models) * ch
    last_x, lifted = -100.0, False
    for c, col in enumerate(columns):
        marker = col.get("marker")
        if not marker:
            continue
        cx = col_x(c) + cw / 2
        lifted = (cx - last_x < 18) and not lifted
        my = marker_y - 18 if lifted else marker_y
        last_x = cx
        parts.append(
            f'<line x1="{cx:.1f}" y1="{my + 8:.1f}" x2="{cx:.1f}" y2="{grid_bottom:.1f}" '
            f'stroke="{INK}" stroke-width="1" stroke-dasharray="2 2" fill="none" opacity="0.55"/>'
            f'<circle cx="{cx:.1f}" cy="{my:.1f}" r="8" fill="{INK}"/>'
            f'<text x="{cx:.1f}" y="{my + 4:.1f}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="{SMALL_SIZE}" font-weight="700" fill="{WHITE}">{marker}</text>'
        )

    y = grid_bottom + 26
    for note in notes:
        parts.append(
            f'<circle cx="8" cy="{y - 4:.1f}" r="8" fill="{INK}"/>'
            f'<text x="8" y="{y:.1f}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="{SMALL_SIZE}" font-weight="700" fill="{WHITE}">{note["marker"]}</text>'
            f'<text x="24" y="{y:.1f}" font-family="{FONT}" font-size="{SMALL_SIZE + 1}" fill="{INK}">'
            f'<tspan font-weight="600">{escape_xml(note["why"])}.</tspan> '
            f'{escape_xml(_clip_words(note["label"], max(30, 94 - len(note["why"]))))}</text>'
        )
        y += 24
    height = y - 8
    return svg_root(width, height, "".join(parts))


DOT_METRICS = (
    ("specificity", "Specificity"),
    ("sensitivity", "Sensitivity"),
    ("novelty", "Novelty"),
    ("corroboration", "Corroboration"),
    ("confidence", "Confidence"),
)


def model_dotplot_svg(
    model_avgs: Mapping[str, float] | None,
    corpus_avgs: Mapping[str, float] | None,
    *,
    width: int = 360,
) -> str:
    """One row per metric on a 1–5 line: dot = this model, tick = corpus average."""
    label_w, value_w = 112, 44
    x0, x1 = label_w, width - value_w
    row_h, top = 30, 30
    height = top + len(DOT_METRICS) * row_h + 26

    def sx(v: float) -> float:
        v = max(1.0, min(5.0, float(v or 0) or 1.0))
        return x0 + (x1 - x0) * (v - 1) / 4

    parts = [
        f'<text x="0" y="14" font-family="{FONT}" font-size="{SMALL_SIZE}" font-weight="600" fill="{MUTED}">Metric</text>'
    ]
    for tick in (1, 3, 5):
        parts.append(
            f'<text x="{sx(tick):.1f}" y="14" text-anchor="middle" font-family="{FONT_MONO}" '
            f'font-size="10" fill="{MUTED}">{tick}</text>'
        )
    for i, (key, label) in enumerate(DOT_METRICS):
        cy = top + i * row_h + row_h / 2
        mine = float((model_avgs or {}).get(key) or 0)
        base = float((corpus_avgs or {}).get(key) or 0)
        parts.append(
            f'<text x="0" y="{cy + 4.5:.1f}" font-family="{FONT}" font-size="{LABEL_SIZE - 1}" fill="{INK}">{label}</text>'
            f'<line x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}" stroke="{RULE}" stroke-width="2"/>'
        )
        if base:
            bx = sx(base)
            parts.append(
                f'<line x1="{bx:.1f}" y1="{cy - 8:.1f}" x2="{bx:.1f}" y2="{cy + 8:.1f}" '
                f'stroke="{INK}" stroke-width="2"/>'
            )
        if mine:
            parts.append(
                f'<circle cx="{sx(mine):.1f}" cy="{cy:.1f}" r="6.5" fill="{TEAL_DEEP}"/>'
            )
        parts.append(
            f'<text x="{width}" y="{cy + 4:.1f}" text-anchor="end" font-family="{FONT_MONO}" '
            f'font-size="{SMALL_SIZE}" fill="{INK}">{mine:.1f}</text>'
        )
    ly = height - 8
    parts.append(
        f'<circle cx="{x0 + 6}" cy="{ly - 4}" r="5.5" fill="{TEAL_DEEP}"/>'
        f'<text x="{x0 + 16}" y="{ly}" font-family="{FONT}" font-size="{SMALL_SIZE}" fill="{INK}">This model</text>'
        f'<line x1="{x0 + 110}" y1="{ly - 11}" x2="{x0 + 110}" y2="{ly + 3}" stroke="{INK}" stroke-width="2"/>'
        f'<text x="{x0 + 118}" y="{ly}" font-family="{FONT}" font-size="{SMALL_SIZE}" fill="{INK}">Corpus average</text>'
    )
    return svg_root(width, height, "".join(parts))
