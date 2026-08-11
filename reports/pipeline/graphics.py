"""[4] Orchestrate SVG graphic generation into the build dir."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphics.exposure_score import exposure_radar_svg, sensitivity_bars_svg
from graphics.heatmap import model_heatmap_svg
from graphics.graph import evidence_graph_svg
from graphics.style import normalize_svg_for_embed

# Filenames under ``<run_dir>/assets/`` (editable before PDF rebuild).
ASSET_NAMES = {
    "exposure_radar": "exposure-radar.svg",
    "model_heatmap": "model-heatmap.svg",
    "sensitivity_distribution": "sensitivity-distribution.svg",
    "evidence_graph": "evidence-graph.svg",
}

DEFAULT_EMIT = [
    "exposure_radar",
    "model_heatmap",
    "sensitivity_distribution",
    "evidence_graph",
]


def assets_dir(run_dir: Path) -> Path:
    return run_dir / "assets"


def write_graphics_assets(run_dir: Path, graphics: dict[str, str]) -> list[Path]:
    """Write generated SVG strings into ``assets/*.svg`` for hand-editing."""
    out = assets_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "screenshots").mkdir(exist_ok=True)
    written: list[Path] = []
    for key, svg in graphics.items():
        filename = ASSET_NAMES.get(key)
        if not filename or not svg:
            continue
        path = out / filename
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    return written


def load_graphics_assets(
    run_dir: Path,
    *,
    emit: list[str] | None = None,
) -> dict[str, str]:
    """Load previously written ``assets/*.svg`` for PDF embed (no regeneration)."""
    emit = emit or DEFAULT_EMIT
    out = assets_dir(run_dir)
    graphics: dict[str, str] = {}
    missing: list[str] = []
    for key in emit:
        filename = ASSET_NAMES.get(key)
        if not filename:
            continue
        path = out / filename
        if not path.exists():
            missing.append(str(path))
            continue
        graphics[key] = normalize_svg_for_embed(path.read_text(encoding="utf-8"))
    if missing:
        raise FileNotFoundError(
            "Missing graphics under assets/ (run graphics stage first, or drop "
            f"--keep-graphics):\n  " + "\n  ".join(missing)
        )
    return graphics


def generate_graphics(
    report_data: dict[str, Any],
    out_dir: Path,
    *,
    emit: list[str] | None = None,
    aliases: dict[str, str] | None = None,
    write_files: bool = True,
    write_assets: bool = True,
) -> dict[str, str]:
    """Build SVG figures for PDF embedding.

    When ``write_assets`` is true, also writes kebab-case files under
    ``out_dir/assets/`` so operators can edit them and rebuild with
    ``--keep-graphics``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    emit = emit or list(DEFAULT_EMIT)
    aliases = aliases or {}
    graphics: dict[str, str] = {}

    if "exposure_radar" in emit:
        graphics["exposure_radar"] = exposure_radar_svg(report_data.get("radar_averages") or {})

    if "model_heatmap" in emit:
        graphics["model_heatmap"] = model_heatmap_svg(
            report_data.get("findings") or [], aliases=aliases
        )

    if "sensitivity_distribution" in emit:
        graphics["sensitivity_distribution"] = sensitivity_bars_svg(
            report_data.get("sensitivity_bins") or {}
        )

    if "evidence_graph" in emit:
        graphics["evidence_graph"] = evidence_graph_svg(
            report_data.get("findings") or [],
            report_data.get("chains") or [],
            aliases=aliases,
        )

    graphics = {k: normalize_svg_for_embed(v) for k, v in graphics.items()}

    if write_files:
        for name, svg in graphics.items():
            (out_dir / f"{name}.svg").write_text(svg, encoding="utf-8")

    if write_assets:
        write_graphics_assets(out_dir, graphics)

    report_data["graphics"] = {
        k: f"assets/{ASSET_NAMES[k]}" for k in graphics if k in ASSET_NAMES
    }
    return graphics
