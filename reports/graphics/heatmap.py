"""Model × cluster heatmap SVG — sized for A4 graphic boxes."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .style import (
    FONT,
    HEAT_SCALE,
    INK,
    MUTED,
    PRINT_MAX_HEIGHT,
    PRINT_MAX_WIDTH,
    RULE,
    WHITE,
    escape_xml,
    full_model_name,
    models_with_results,
    raw_finding_models,
    short_model_name,
    svg_root,
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


def _column_id(finding: dict) -> str:
    return str(finding.get("cluster_id") or finding.get("claim_id") or "—").strip() or "—"


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
    models_probed: Iterable[str] | None = None,
    max_findings: int | None = 48,
    max_width: float = PRINT_MAX_WIDTH,
    max_height: float = PRINT_MAX_HEIGHT,
    full: bool = False,
) -> str:
    """Heatmap with clusters across the top and models down the left.

    When ``models_probed`` is provided, rows are the probed models that
    produced at least one finding. Empty rows for failed or silent models are
    omitted, and labels that were never probed never appear.

    When ``full`` is true, every cluster is kept and the SVG grows as wide as
    needed (companion asset for operators; not sized for the A4 graphic box).
    """
    aliases = aliases or {}
    all_findings = list(findings)
    if full:
        max_findings = None

    # Rows = models (left); columns = clusters (top).
    model_keys: list[str] = []
    model_labels: dict[str, str] = {}
    seen_m: set[str] = set()

    for raw in models_with_results(
        all_findings, models_probed=models_probed, aliases=aliases
    ):
        key = short_model_name(raw, aliases)
        if not key or key == "unknown" or key in seen_m:
            continue
        seen_m.add(key)
        model_keys.append(key)
        model_labels[key] = full_model_name(raw)

    if not model_keys:
        model_keys = ["—"]
        model_labels = {"—": "—"}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for f in all_findings:
        grouped[_column_id(f)].append(f)
    columns: list[dict] = []
    for col_id, members in grouped.items():
        if col_id == "—":
            continue
        sens = 0
        models_in_col: set[str] = set()
        for f in members:
            try:
                sens = max(sens, int(f.get("sensitivity", 0) or 0))
            except (TypeError, ValueError):
                pass
            for raw in raw_finding_models(f):
                key = short_model_name(str(raw or ""), aliases)
                if key and key != "unknown":
                    models_in_col.add(key)
        columns.append(
            {
                "id": col_id,
                "sensitivity": max(0, min(5, sens)),
                "models": models_in_col,
                # Cross-model agreement among rows shown on this chart.
                "agreement": len(models_in_col & set(model_keys)),
            }
        )
    # Deliverable heatmap prefers clusters with the most model overlap first.
    columns.sort(
        key=lambda row: (
            -int(row["agreement"]),
            -int(row["sensitivity"]),
            str(row["id"]),
        )
    )
    if max_findings is not None:
        columns = columns[:max_findings]
    if not columns:
        columns = [{"id": "—", "sensitivity": 0, "models": set(), "agreement": 0}]

    model_lines = {k: _wrap_label(model_labels[k], 20) for k in model_keys}
    # Top axis: cluster ids (wrapped at 20 if ever long).
    claim_lines = {col["id"]: _wrap_label(str(col["id"]), 20) for col in columns}

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

    title_h, legend_h, right_pad, bottom_pad = 8, 44, 16, 12
    top = title_h + 8 + claim_label_h
    n_models = len(model_keys)
    n_claims = len(columns)

    # Row height fits wrapped model labels (left axis sits beside each row).
    line_gap_model = font_axis + 1
    min_row = max(22.0, max_model_lines * line_gap_model + 6)
    cell_h = min_row

    if full:
        # Fixed cell width; SVG grows horizontally to fit every cluster.
        cell_w = 18.0
    else:
        avail_w = max_width - left - right_pad
        cell_w = max(16.0, min(28.0, avail_w / max(1, n_claims)))
        # Drop clusters (columns) if overflowing width; keep all models.
        while left + n_claims * cell_w + right_pad > max_width and n_claims > 6:
            n_claims -= 1
            columns = columns[:n_claims]
            cell_w = max(
                16.0, min(28.0, (max_width - left - right_pad) / max(1, n_claims))
            )

    n_models = len(model_keys)
    n_claims = len(columns)
    width = left + n_claims * cell_w + right_pad
    height = top + n_models * cell_h + legend_h + bottom_pad
    # max_height is unused for layout (rows already determine height); keep param
    # for API compatibility with print-boxed callers.
    del max_height

    # Presence: which models support each cluster. Color is that cluster's
    # peak sensitivity (one value per column).
    supported: set[tuple[int, int]] = set()
    claim_sens: list[int] = [int(col["sensitivity"]) for col in columns]
    for j, col in enumerate(columns):
        for key in col["models"]:
            if key in model_keys:
                supported.add((model_keys.index(key), j))

    panel_x = left - 8
    panel_y = top - 8
    panel_w = n_claims * cell_w + 16
    panel_h = n_models * cell_h + 16
    panel = (
        f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" '
        f'height="{panel_h:.1f}" fill="{WHITE}" stroke="{RULE}"/>'
    )

    cells = []
    for i in range(n_models):
        for j in range(n_claims):
            hit = (i, j) in supported
            v = claim_sens[j] if hit else 0
            fill = HEAT_SCALE.get(v, HEAT_SCALE[0])
            x = left + j * cell_w
            y = top + i * cell_h
            cells.append(
                f'<rect x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" '
                f'width="{cell_w - 3:.1f}" height="{cell_h - 3:.1f}" '
                f'rx="3" fill="{fill}" stroke="{WHITE}" stroke-width="1"/>'
            )

    # Clusters across the top — vertical, wrapped at 20 chars.
    claim_base_y = top - 10
    line_gap = font_axis + 3
    xlabels = []
    for j, col in enumerate(columns):
        lines = claim_lines[col["id"]]
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

    # Color key: empty + sensitivity 1–5.
    key_y = top + n_models * cell_h + 20
    key_items = [
        (0, "Empty"),
        (1, "1"),
        (2, "2"),
        (3, "3"),
        (4, "4"),
        (5, "5"),
    ]
    legend = [
        f'<text x="{left:.1f}" y="{key_y:.1f}" font-family="{FONT}" font-size="9" '
        f'font-weight="600" fill="{INK}">Sensitivity</text>'
    ]
    swatch_y = key_y + 8
    lx = left
    for level, label in key_items:
        fill = HEAT_SCALE.get(level, HEAT_SCALE[0])
        legend.append(
            f'<rect x="{lx:.1f}" y="{swatch_y:.1f}" width="12" height="12" rx="2" '
            f'fill="{fill}" stroke="{RULE}"/>'
            f'<text x="{lx + 16:.1f}" y="{swatch_y + 10:.1f}" font-family="{FONT}" '
            f'font-size="9" fill="{MUTED}">{escape_xml(label)}</text>'
        )
        lx += 52 if level == 0 else 36
    note = "Filled = model supported this cluster · color = that cluster's sensitivity"
    if full:
        note += f" · all {n_claims} clusters"
    elif len(grouped) > n_claims:
        note += f" · top {n_claims} by cross-model agreement"
    legend.append(
        f'<text x="{left:.1f}" y="{swatch_y + 26:.1f}" font-family="{FONT}" '
        f'font-size="9" fill="{MUTED}">{escape_xml(note)}</text>'
    )

    body = f"""  {panel}
  {"".join(cells)}
  {"".join(xlabels)}
  {"".join(ylabels)}
  {"".join(legend)}"""
    return svg_root(width, height, body)
