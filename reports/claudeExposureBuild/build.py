#!/usr/bin/env python3
"""claudeExposureBuild CLI — alternative Exposure Snapshot + one-page.

Renders this product from a run's ``report_data.json`` (produced by the shared
pipeline). If that artifact is missing and ``--exploration`` is given, the
shared pipeline is run through the ``synthesize`` stage first (no default-builder
PDFs are produced). Output PDFs are written to
``reports/build/<run-id>/claude/output/``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import yaml

REPORTS_ROOT = Path(__file__).resolve().parent.parent  # reports/
REPO_ROOT = REPORTS_ROOT.parent
for _p in (REPORTS_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Package-relative imports work because reports/ is on sys.path.
from claudeExposureBuild.content import build_content


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_run_dir(cfg: dict, run_id: str) -> Path:
    base = Path((cfg.get("output") or {}).get("dir") or "reports/build")
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / run_id


def _data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".gif": "image/gif",
    }.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _ensure_report_data(
    run_dir: Path, exploration: Path | None, config_path: Path, dry_run: bool
) -> dict:
    """Load ``report_data.json`` for the run, building it via the shared
    pipeline (through the synthesize stage) if necessary."""
    rd_path = run_dir / "report_data.json"
    if not rd_path.exists():
        if not exploration or not exploration.exists():
            raise SystemExit(
                f"{rd_path} not found. Provide --exploration to build it, or run "
                "reports/build_report.py first."
            )
        import build_report  # shared pipeline entry

        argv = [
            "-e", str(exploration),
            "--run-id", run_dir.name,
            "--config", str(config_path),
            "--stop-after", "synthesize",
        ]
        if dry_run:
            argv.append("--dry-run")
        print("[claudeExposureBuild] building report_data via shared pipeline…", file=sys.stderr)
        build_report.main(argv)
    return json.loads(rd_path.read_text(encoding="utf-8"))


def _render(html: str, out_pdf: Path) -> None:
    from weasyprint import HTML

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(out_pdf.parent.resolve()) + "/").write_pdf(str(out_pdf))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="claudeExposureBuild report builder")
    ap.add_argument("-e", "--exploration", type=Path, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--config", type=Path, default=REPORTS_ROOT / "config.yaml")
    ap.add_argument(
        "--report",
        choices=["snapshot", "onepage", "both"],
        default="both",
        help="Which product to render (default: both).",
    )
    ap.add_argument("--dry-run", action="store_true", help="If building data, use heuristic extract")
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    exploration = args.exploration
    if exploration is None and (cfg.get("input") or {}).get("exploration"):
        exploration = Path(cfg["input"]["exploration"])

    run_id = args.run_id or ((cfg.get("output") or {}).get("run_id"))
    if not run_id:
        if exploration:
            run_id = exploration.parent.name
        else:
            raise SystemExit("Provide --run-id (or --exploration to infer it).")

    run_dir = _resolve_run_dir(cfg, run_id)
    report_data = _ensure_report_data(run_dir, exploration, args.config, args.dry_run)

    # Enrich explore metadata (models/techniques/fuzz mode) if we can read it.
    if exploration and exploration.exists() and not report_data.get("explore_meta"):
        try:
            from pipeline.parse import exploration_run_meta

            report_data["explore_meta"] = exploration_run_meta(exploration)
        except Exception:
            pass

    render_cfg = cfg.get("render") or {}
    aliases = (cfg.get("graphics") or {}).get("model_aliases") or {}
    logo_src = REPORTS_ROOT / (render_cfg.get("logo") or "assets/branding/MoyoLogo.png")
    partner_src = REPORTS_ROOT / (
        render_cfg.get("partner_logo") or "assets/branding/SenTeGuardLogo.png"
    )

    content = build_content(
        report_data,
        report_date_cfg=render_cfg.get("report_date"),
        aliases=aliases,
    )

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from markupsafe import Markup
    except ImportError as e:
        raise SystemExit("jinja2 is required. Install with: pip install jinja2") from e

    from pipeline.textclean import find_markdown_residue, markdown_to_html, plain_text

    tpl_dir = Path(__file__).resolve().parent / "templates"
    css_dir = Path(__file__).resolve().parent / "css"
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["md"] = lambda value: Markup(markdown_to_html(value))
    env.filters["plain"] = plain_text

    common = {
        "content": content,
        "logo_uri": _data_uri(logo_src),
        "partner_logo_uri": _data_uri(partner_src),
    }

    out_dir = run_dir / "claude" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _warn_markdown(name: str, rendered: str) -> None:
        residue = find_markdown_residue(rendered)
        if residue:
            print(f"  warn: markdown syntax survived into {name}:", file=sys.stderr)
            for snippet in residue:
                print(f"    … {snippet}", file=sys.stderr)

    if args.report in ("snapshot", "both"):
        report_css = (css_dir / "report.css").read_text(encoding="utf-8")
        report_html = env.get_template("report.html.j2").render(**common, css_text=report_css)
        _warn_markdown("report.pdf", report_html)
        report_pdf = out_dir / "report.pdf"
        _render(report_html, report_pdf)
        written.append(report_pdf)

    if args.report in ("onepage", "both"):
        onepage_css = (css_dir / "onepage.css").read_text(encoding="utf-8")
        onepage_html = env.get_template("onepage.html.j2").render(**common, css_text=onepage_css)
        _warn_markdown("one-page.pdf", onepage_html)
        if "<a " in onepage_html.lower() or "<a>" in onepage_html.lower():
            raise SystemExit("one-page must not contain hyperlinks (<a>)")
        onepage_pdf = out_dir / "one-page.pdf"
        _render(onepage_html, onepage_pdf)
        written.append(onepage_pdf)

    for p in written:
        print(f"  → {p}", file=sys.stderr)
    print(f"Done → {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
