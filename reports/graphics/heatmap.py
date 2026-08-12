"""Model × finding heatmap SVG — sized for A4 graphic boxes."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .style import (
    CREAM,
    FONT,
    FONT_MONO,
    HEAT_SCALE,
    INK,
    MUTED,
    PRINT_MAX_HEIGHT,
    PRINT_MAX_WIDTH,
    RULE,
    WHITE,
    escape_xml,
    full_model_name,
    short_model_name,
    svg_root,
    title_block,
)


def _wrap_label(label: str, width: int = 20) -> list[str]:
    """Word-wrap ``label`` so each line is at most ``width`` characters."""
    label = " ".join(str(label).split()).strip()
    if not label:
        return [""]
    lines: list[str] = []
    remaining = label
    while len(remaining) > width:
        # Prefer breaking before a parenthetical tag when it fits on line 1.
        paren = remaining.find(" (")
        if 0 < paren <= width and not lines:
            lines.append(remaining[:paren].strip())
            remaining = remaining[paren:].strip()
            continue
        cut = remaining.rfind(" ", 0, width + 1)
        if cut <= 0:
            # Break after a hyphen inside long tokens (model ids).
            hyphen = remaining.rfind("-", 1, width)
            cut = hyphen + 1 if hyphen > 0 else width
        piece = remaining[:cut].strip()
        remaining = remaining[cut:].strip()
        if piece:
            lines.append(piece)
    if remaining:
        lines.append(remaining)
    return lines or [label]


def _multiline_text(
    lines: list[str],
    *,
    x: float,
    y: float,
    font_size: float,
    fill: str,
    anchor: str = "end",
    font: str = FONT,
    weight: str = "500",
    line_gap: float | None = None,
) -> str:
    """Horizontal multi-line ``<text>`` block centered on ``y``."""
    gap = font_size + 2 if line_gap is None else line_gap
    n = max(1, len(lines))
    # Center the block on y
    y0 = y - (n - 1) * gap / 2 + font_size * 0.35
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else gap
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape_xml(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y0:.1f}" text-anchor="{anchor}" '
        f'font-family="{font}" font-size="{font_size}" font-weight="{weight}" fill="{fill}">'
        f"{''.join(tspans)}</text>"
    )


def model_heatmap_svg(
    findings: Iterable[dict],
    *,
    aliases: dict[str, str] | None = None,
    max_findings: int = 20,
    max_width: float = PRINT_MAX_WIDTH,
    max_height: float = PRINT_MAX_HEIGHT,
) -> str:
    """Heatmap with claims across the top and models down the left."""
    aliases = aliases or {}
    all_findings = list(findings)

    # Rows = models (left); columns = claims (top).
    model_keys: list[str] = []
    model_labels: dict[str, str] = {}
    seen_m: set[str] = set()
    for f in all_findings:
        raw_models = f.get("source_models")
        if not isinstance(raw_models, list) or not raw_models:
            raw_models = [f.get("source_model") or ""]
        for raw in raw_models:
            key = short_model_name(str(raw or ""), aliases)
            if not key or key in seen_m:
                continue
            seen_m.add(key)
            model_keys.append(key)
            model_labels[key] = full_model_name(str(raw or ""))
            if len(model_keys) >= 10:
                break
        if len(model_keys) >= 10:
            break
    if not model_keys:
        model_keys = ["—"]
        model_labels = {"—": "—"}

    claims = all_findings[:max_findings]
    if not claims:
        claims = [{"claim_id": "—", "source_model": "", "sensitivity": 0}]

    model_lines = {k: _wrap_label(model_labels[k], 20) for k in model_keys}
    # Top axis: claim ids (wrapped at 20 if ever long).
    claim_lines = {
        f["claim_id"]: _wrap_label(str(f.get("claim_id") or "—"), 20) for f in claims
    }

    font_axis = 9
    max_model_lines = max(len(v) for v in model_lines.values())
    max_claim_line_len = max(
        max((len(line) for line in lines), default=1) for lines in claim_lines.values()
    )
    max_claim_lines = max(len(v) for v in claim_lines.values())

    # Left gutter fits wrapped model names; top band fits vertical claim labels.
    left = max(120.0, 16 + 20 * font_axis * 0.55)
    claim_label_h = max(
        56.0,
        min(140.0, 12.0 + max_claim_line_len * font_axis * 0.58),
    )
    # Extra top room when claim labels wrap to multiple parallel vertical lines.
    if max_claim_lines > 1:
        claim_label_h = min(150.0, claim_label_h + (max_claim_lines - 1) * (font_axis + 2))

    title_h, legend_h, right_pad, bottom_pad = 48, 34, 16, 12
    top = title_h + 8 + claim_label_h
    n_models = len(model_keys)
    n_claims = len(claims)

    avail_w = max_width - left - right_pad
    # Row height fits wrapped model labels (left axis sits beside each row).
    line_gap_model = font_axis + 1
    min_row = max(22.0, max_model_lines * line_gap_model + 6)
    cell_w = max(16.0, min(28.0, avail_w / max(1, n_claims)))
    cell_h = min_row

    # Drop claims (columns) if overflowing width; keep all models (SVG may grow taller).
    while left + n_claims * cell_w + right_pad > max_width and n_claims > 6:
        n_claims -= 1
        claims = claims[:n_claims]
        cell_w = max(16.0, min(28.0, (max_width - left - right_pad) / max(1, n_claims)))

    n_models = len(model_keys)
    width = left + n_claims * cell_w + right_pad
    height = top + n_models * cell_h + legend_h + bottom_pad
    font_cell = 9 if min(cell_w, cell_h) >= 22 else 8

    grid: dict[tuple[int, int], int] = defaultdict(int)
    claim_index = {f["claim_id"]: j for j, f in enumerate(claims)}
    for f in all_findings:
        cid = f.get("claim_id")
        if cid not in claim_index:
            continue
        j = claim_index[cid]
        sens = max(0, min(5, int(f.get("sensitivity", 1) or 0)))
        raw_models = f.get("source_models")
        if not isinstance(raw_models, list) or not raw_models:
            raw_models = [f.get("source_model") or ""]
        for raw in raw_models:
            key = short_model_name(str(raw or ""), aliases)
            if key in model_keys:
                i = model_keys.index(key)
                grid[(i, j)] = max(grid[(i, j)], sens)

    panel_x = left - 8
    panel_y = top - 8
    panel_w = n_claims * cell_w + 16
    panel_h = n_models * cell_h + 16
    panel = (
        f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" '
        f'height="{panel_h:.1f}" rx="8" fill="{CREAM}" stroke="{RULE}"/>'
    )

    cells = []
    for i in range(n_models):
        for j in range(n_claims):
            v = grid.get((i, j), 0)
            fill = HEAT_SCALE.get(v, HEAT_SCALE[0])
            x = left + j * cell_w
            y = top + i * cell_h
            cells.append(
                f'<rect x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" '
                f'width="{cell_w - 3:.1f}" height="{cell_h - 3:.1f}" '
                f'rx="3" fill="{fill}" stroke="{WHITE}" stroke-width="1"/>'
            )
            if v:
                ink = WHITE if v >= 3 else INK
                cells.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 3:.1f}" '
                    f'text-anchor="middle" font-family="{FONT_MONO}" font-size="{font_cell}" '
                    f'fill="{ink}">{v}</text>'
                )

    # Claims across the top — vertical, wrapped at 20 chars.
    claim_base_y = top - 10
    line_gap = font_axis + 3
    xlabels = []
    for j, f in enumerate(claims):
        lines = claim_lines[f["claim_id"]]
        tx = left + j * cell_w + cell_w / 2
        tspans = []
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else line_gap
            tspans.append(f'<tspan x="{tx:.1f}" dy="{dy}">{escape_xml(line)}</tspan>')
        xlabels.append(
            f'<text x="{tx:.1f}" y="{claim_base_y:.1f}" text-anchor="start" '
            f'transform="rotate(-90 {tx:.1f},{claim_base_y:.1f})" '
            f'font-family="{FONT}" font-size="{font_axis}" font-weight="500" fill="{INK}">'
            f"{''.join(tspans)}</text>"
        )

    # Models on the left — horizontal, wrapped at 20 chars.
    ylabels = []
    for i, key in enumerate(model_keys):
        cy = top + i * cell_h + cell_h / 2
        ylabels.append(
            _multiline_text(
                model_lines[key],
                x=left - 10,
                y=cy,
                font_size=font_axis,
                fill=INK,
                anchor="end",
                weight="500",
            )
        )

    legend_y = top + n_models * cell_h + 22
    legend = [
        f'<text x="{left}" y="{legend_y:.1f}" font-family="{FONT}" font-size="9" '
        f'fill="{MUTED}">Sensitivity</text>'
    ]
    lx = left + 70
    for v in range(1, 6):
        fill = HEAT_SCALE[v]
        legend.append(
            f'<rect x="{lx:.1f}" y="{legend_y - 10:.1f}" width="16" height="13" rx="2" '
            f'fill="{fill}" stroke="{RULE}"/>'
            f'<text x="{lx + 8:.1f}" y="{legend_y + 1:.1f}" text-anchor="middle" '
            f'font-family="{FONT_MONO}" font-size="8" '
            f'fill="{WHITE if v >= 3 else INK}">{v}</text>'
        )
        lx += 22
    legend.append(
        f'<text x="{lx + 6:.1f}" y="{legend_y:.1f}" font-family="{FONT}" font-size="9" '
        f'fill="{MUTED}">empty = none'
        f"{' · top ' + str(n_claims) if len(all_findings) > n_claims else ''}</text>"
    )

    body = f"""  {title_block(width / 2, 20, "Model Heatmap")}
  {panel}
  {"".join(cells)}
  {"".join(xlabels)}
  {"".join(ylabels)}
  {"".join(legend)}"""
    return svg_root(width, height, body)
