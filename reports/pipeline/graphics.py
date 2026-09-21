"""[4] Orchestrate SVG graphic generation into the build dir."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphics.exposure_score import exposure_radar_svg, llm_findings_bars_svg
from graphics.heatmap import model_heatmap_svg
from graphics.graph import evidence_graph_svg, snapshot_graph_findings
from graphics.model_charts import generate_dossier_graphics
from graphics.style import normalize_svg_for_embed
from pipeline.model_dossiers import build_model_dossiers
from pipeline.score import aggregate_findings_by_llm

# Filenames under ``<run_dir>/assets/`` (editable before PDF rebuild).
ASSET_NAMES = {
    "exposure_radar": "exposure-radar.svg",
    "model_heatmap": "model-heatmap.svg",
    "model_heatmap_full": "model-heatmap-full.svg",
    "findings_by_llm": "findings-by-llm.svg",
    "evidence_graph": "evidence-graph.svg",
    "evidence_graph_snapshot": "evidence-graph-snapshot.svg",
}

DEFAULT_EMIT = [
    "exposure_radar",
    "model_heatmap",
    "findings_by_llm",
    "evidence_graph",
]


def _dossier_filename(key: str) -> str | None:
    if not str(key).startswith("d_"):
        return None
    rest = str(key)[2:]
    for kind in ("fingerprint", "mix", "probes", "overlap"):
        suffix = f"_{kind}"
        if rest.endswith(suffix):
            slug = rest[: -len(suffix)]
            if slug:
                return f"dossiers/{slug}__{kind}.svg"
    return None


def _load_dossier_assets(run_dir: Path) -> dict[str, str]:
    folder = assets_dir(run_dir) / "dossiers"
    graphics: dict[str, str] = {}
    if not folder.is_dir():
        return graphics
    for path in sorted(folder.glob("*__*.svg")):
        stem = path.stem
        if "__" not in stem:
            continue
        slug, kind = stem.rsplit("__", 1)
        graphics[f"d_{slug}_{kind}"] = normalize_svg_for_embed(
            path.read_text(encoding="utf-8")
        )
    return graphics


def assets_dir(run_dir: Path) -> Path:
    return run_dir / "assets"


def write_graphics_assets(run_dir: Path, graphics: dict[str, str]) -> list[Path]:
    """Write generated SVG strings into ``assets/*.svg`` for hand-editing."""
    out = assets_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "screenshots").mkdir(exist_ok=True)
    written: list[Path] = []
    for key, svg in graphics.items():
        filename = ASSET_NAMES.get(key) or _dossier_filename(key)
        if not filename or not svg:
            continue
        path = out / filename
        path.parent.mkdir(parents=True, exist_ok=True)
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
    snap_name = ASSET_NAMES.get("evidence_graph_snapshot")
    if snap_name:
        snap_path = out / snap_name
        if snap_path.exists():
            graphics["evidence_graph_snapshot"] = normalize_svg_for_embed(
                snap_path.read_text(encoding="utf-8")
            )
        elif "evidence_graph" in graphics:
            graphics["evidence_graph_snapshot"] = graphics["evidence_graph"]
    graphics.update(_load_dossier_assets(run_dir))
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
    chart_findings = list(
        report_data.get("findings_all") or report_data.get("findings") or []
    )
    chart_chains = list(report_data.get("chains") or [])
    graphics: dict[str, str] = {}

    if "exposure_radar" in emit:
        graphics["exposure_radar"] = exposure_radar_svg(report_data.get("radar_averages") or {})

    explore = report_data.get("explore_meta") or {}
    # Roster of models queried. Charts keep only those that produced findings.
    probed = [
        str(m).strip()
        for m in (
            explore.get("models_tested")
            or report_data.get("models_tested")
            or []
        )
        if str(m).strip()
    ]

    if "model_heatmap" in emit:
        graphics["model_heatmap"] = model_heatmap_svg(
            chart_findings,
            aliases=aliases,
            models_probed=probed or None,
        )

    # Full-width companion (all clusters); always written next to PDF assets.
    if write_assets or "model_heatmap_full" in emit:
        graphics["model_heatmap_full"] = model_heatmap_svg(
            chart_findings,
            aliases=aliases,
            models_probed=probed or None,
            full=True,
        )

    if "findings_by_llm" in emit:
        if chart_findings or probed:
            rows = aggregate_findings_by_llm(
                chart_findings,
                aliases,
                models_probed=probed or None,
            )
        else:
            rows = list(report_data.get("findings_by_llm") or [])
        graphics["findings_by_llm"] = llm_findings_bars_svg(rows)
        report_data["findings_by_llm"] = rows

    if "evidence_graph" in emit:
        graphics["evidence_graph"] = evidence_graph_svg(
            chart_findings,
            chart_chains,
            aliases=aliases,
            models_probed=probed or None,
        )
        snap_source = list(report_data.get("findings") or chart_findings)
        snap_findings = snapshot_graph_findings(snap_source)
        graphics["evidence_graph_snapshot"] = evidence_graph_svg(
            snap_findings or chart_findings,
            chart_chains,
            aliases=aliases,
            models_probed=probed or None,
            max_claims=max(len(snap_findings), 1) if snap_findings else 10,
        )

    dossiers = build_model_dossiers(
        list(report_data.get("findings") or chart_findings),
        corpus=list(report_data.get("response_corpus") or []),
        sources=list(report_data.get("sources") or []),
        radar_averages=report_data.get("radar_averages") or {},
        models_probed=probed or None,
        aliases=aliases,
    )
    report_data["model_dossiers"] = dossiers
    graphics.update(generate_dossier_graphics(dossiers))

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
