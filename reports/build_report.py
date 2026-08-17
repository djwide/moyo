#!/usr/bin/env python3
"""Exploration processor CLI: exploration.md → claims → report PDFs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

REPORTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REPORTS_ROOT.parent
if str(REPORTS_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORTS_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.parse import (
    exploration_run_meta,
    load_chunks_manifest,
    parse_exploration,
    prompts_from_exploration,
    remember_exploration_path,
    resolve_exploration_path,
    topic_from_exploration,
    write_chunks_manifest,
)
from pipeline.extract import extract_all, load_claims
from pipeline.citations import attach_chunk_citations
from pipeline.cluster import cluster_claims
from pipeline.score import score_report
from pipeline.synthesize import synthesize
from pipeline.graphics import ASSET_NAMES, generate_graphics, load_graphics_assets
from pipeline.content import write_content_package
from pipeline.email_alert import write_alert_email
from pipeline.textclean import find_markdown_residue, markdown_to_html, plain_text


STAGES = ["parse", "extract", "cluster", "score", "synthesize", "graphics", "render"]


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_run_dir(cfg: dict, run_id: str) -> Path:
    base = Path(cfg.get("output", {}).get("dir") or "reports/build")
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / run_id


def _format_date(raw: str | None) -> str:
    if raw:
        try:
            d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw.upper()
    else:
        d = datetime.now()
    return d.strftime("%-d %b %Y").upper()


def _design_system_root(cfg: dict) -> Path:
    render_cfg = cfg.get("render") or {}
    rel = render_cfg.get("design_system") or "design-system"
    path = REPORTS_ROOT / rel
    if not path.is_dir():
        raise SystemExit(f"Design system not found: {path}")
    return path


def _resolve_isvf_path(cfg: dict) -> Path | None:
    """Resolve the ISVF repo path (for Basis Report remediation)."""
    render_cfg = cfg.get("render") or {}
    rel = render_cfg.get("isvf_path") or "IdeaSecurityVerificationFramework"
    p = Path(rel)
    if not p.is_absolute():
        p = REPO_ROOT / rel
    return p if p.exists() else None


def _merge_collection_issues(report_data: dict, run_dir: Path) -> list[dict]:
    """Retrieval failures from exploration.md plus extract blanks from this run."""
    issues: list[dict] = []
    seen: set[tuple] = set()

    def _add(item: dict) -> None:
        key = (
            str(item.get("stage") or ""),
            str(item.get("source") or item.get("source_model") or ""),
            str(item.get("chunk_id") or ""),
            str(item.get("query") or ""),
            str(item.get("reason") or ""),
        )
        if key in seen:
            return
        seen.add(key)
        issues.append(item)

    for item in report_data.get("collection_issues") or []:
        if isinstance(item, dict):
            _add(item)
    meta = report_data.get("explore_meta") or {}
    for item in meta.get("collection_issues") or []:
        if isinstance(item, dict):
            _add(item)
    extract_path = run_dir / "extract_issues.json"
    if extract_path.is_file():
        try:
            extra = json.loads(extract_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            extra = []
        if isinstance(extra, list):
            for item in extra:
                if isinstance(item, dict):
                    _add(item)
    return issues


def render_pdfs(
    report_data: dict,
    graphics: dict[str, str],
    *,
    run_dir: Path,
    cfg: dict,
    keep_graphics: bool = False,
    report_type: str = "snapshot",
    include_remediation: bool = False,
) -> dict[str, Path]:
    """Render design-system templates → PDFs + alert email under ``output/``.

    ``report_type`` selects which products to build:

    - ``snapshot`` (default) — Exposure Snapshot: ``one-page.pdf`` + ``report.pdf``
      (+ alert email).
    - ``basis`` — Basis Report only: ``basis-report.pdf``.
    - ``both`` — Exposure Snapshot and Basis Report.

    ``include_remediation`` (default off) controls whether mitigations /
    remediations appear (ISVF controls, follow-up playbook, defensive action).

    Content (``report.md`` / ``report.yaml``) and SVG assets are written under
    the run directory; presentation comes from the shared design system.
    The one-pager PDF contains no hyperlinks. The alert email may include a CTA URL.

    When ``keep_graphics`` is true, existing ``assets/*.svg`` are not overwritten
    (edit charts by hand, then rebuild PDF only).
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from markupsafe import Markup
    except ImportError as e:
        raise SystemExit(
            "jinja2 is required for render. Install with: pip install jinja2"
        ) from e
    try:
        from weasyprint import HTML
    except ImportError as e:
        raise SystemExit(
            "weasyprint is required for PDF output. Install with: pip install weasyprint"
        ) from e

    want_snapshot = report_type in ("snapshot", "both")
    want_basis = report_type in ("basis", "both")

    render_cfg = cfg.get("render") or {}
    ds_root = _design_system_root(cfg)
    report_date = _format_date(render_cfg.get("report_date"))
    isvf_path = _resolve_isvf_path(cfg) if include_remediation else None
    logo_src = REPORTS_ROOT / (
        render_cfg.get("logo") or "assets/branding/moyo-logo-wordmark.svg"
    )
    partner_logo_src = REPORTS_ROOT / (
        render_cfg.get("partner_logo") or "assets/branding/SenTeGuardLogo.png"
    )
    favicon_src = REPORTS_ROOT / (
        render_cfg.get("favicon") or "assets/branding/favicon-32x32.png"
    )

    headline = render_cfg.get("headline") or report_data.get("headline")
    report_data = {**report_data, "headline": headline}

    content = write_content_package(
        report_data,
        run_dir,
        report_date=report_date,
        logo_src=logo_src if logo_src.exists() else None,
        partner_logo_src=partner_logo_src if partner_logo_src.exists() else None,
        graphics_svgs=graphics,
        aliases=(cfg.get("graphics") or {}).get("model_aliases") or {},
        overwrite_graphics=not keep_graphics,
        isvf_path=isvf_path,
        include_remediation=include_remediation,
        llm_config=(cfg.get("synthesize") or cfg.get("extract") or {}),
    )

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(
            [
                str(ds_root / "templates"),
                str(ds_root),
            ]
        ),
        autoescape=select_autoescape(["html", "xml"]),
    )
    # Markdown never reaches the page: `md` renders it as HTML formatting,
    # `plain` flattens it to prose.
    env.filters["md"] = lambda value: Markup(markdown_to_html(value))
    env.filters["plain"] = plain_text

    logo_uri = "assets/company-logo.svg"
    if (run_dir / "assets" / "company-logo.png").exists():
        logo_uri = "assets/company-logo.png"
    elif not (run_dir / logo_uri).exists() and logo_src.exists():
        # Non-SVG logo fallback for HTML img tag
        logo_uri = f"assets/screenshots/{logo_src.name}"
        (run_dir / "assets" / "screenshots").mkdir(parents=True, exist_ok=True)
        shutil.copy2(logo_src, run_dir / "assets" / "screenshots" / logo_src.name)

    partner_logo_uri = ""
    partner_dest = run_dir / "assets" / "partner-logo.png"
    if partner_dest.exists():
        partner_logo_uri = "assets/partner-logo.png"
    elif partner_logo_src.exists():
        (run_dir / "assets" / "screenshots").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            partner_logo_src,
            run_dir / "assets" / "screenshots" / partner_logo_src.name,
        )
        partner_logo_uri = f"assets/screenshots/{partner_logo_src.name}"

    favicon_uri = ""
    if favicon_src.exists():
        fav_dst = run_dir / "assets" / "screenshots" / favicon_src.name
        fav_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(favicon_src, fav_dst)
        favicon_uri = f"assets/screenshots/{favicon_src.name}"

    common = {
        "content": content,
        "graphics": graphics,
        "logo_uri": logo_uri,
        "partner_logo_uri": partner_logo_uri,
        "favicon_uri": favicon_uri,
        "dot_max": int((cfg.get("graphics") or {}).get("dot_max", 5)),
        "show_request_full_report": bool(render_cfg.get("show_request_full_report", True)),
    }

    outputs: dict[str, Path] = {}

    def _warn_markdown(name: str, html: str) -> None:
        residue = find_markdown_residue(html)
        if residue:
            print(
                f"  warn: markdown syntax survived into {name}:", file=sys.stderr
            )
            for snippet in residue:
                print(f"    … {snippet}", file=sys.stderr)

    onepage_html = report_html = basis_html = None
    if want_snapshot:
        onepage_html = env.get_template("onepage.html.j2").render(
            **common,
            css_href="css/onepage.css",
        )
        # Belt-and-suspenders: PDF snapshot must not ship clickable links
        if "<a " in onepage_html.lower() or "<a>" in onepage_html.lower():
            raise SystemExit("one-page template must not contain hyperlinks (<a>)")
        report_html = env.get_template("report.html.j2").render(
            **common,
            css_href="css/report.css",
        )
        _warn_markdown("one-page.pdf", onepage_html)
        _warn_markdown("report.pdf", report_html)
        email_paths = write_alert_email(
            content,
            output_dir,
            cta_text=str(
                render_cfg.get("email_cta_text")
                or "Request the full MOYO exposure assessment"
            ),
            cta_url=(render_cfg.get("email_cta_url") or None),
            include_remediation=include_remediation,
        )
        outputs["email_text"] = email_paths["text"]
        outputs["email_html"] = email_paths["html"]

    if want_basis:
        basis_html = env.get_template("basis.html.j2").render(
            **common,
            css_href="css/basis.css",
        )
        _warn_markdown("basis-report.pdf", basis_html)

    with tempfile.TemporaryDirectory(prefix="moyo-report-") as tmp:
        tmp_dir = Path(tmp)
        # Presentation (design-system CSS) + content assets for WeasyPrint
        shutil.copytree(ds_root / "css", tmp_dir / "css")
        if (run_dir / "assets").exists():
            shutil.copytree(run_dir / "assets", tmp_dir / "assets")
        base_url = str(tmp_dir.resolve()) + "/"

        if want_snapshot:
            onepage_pdf = output_dir / "one-page.pdf"
            report_pdf = output_dir / "report.pdf"
            (tmp_dir / "one-page.html").write_text(onepage_html, encoding="utf-8")
            (tmp_dir / "report.html").write_text(report_html, encoding="utf-8")
            HTML(filename=str(tmp_dir / "one-page.html"), base_url=base_url).write_pdf(
                str(onepage_pdf)
            )
            HTML(filename=str(tmp_dir / "report.html"), base_url=base_url).write_pdf(
                str(report_pdf)
            )
            outputs["onepage"] = onepage_pdf
            outputs["report"] = report_pdf
            onepage_html_path = output_dir / "one-page.html"
            report_html_path = output_dir / "report.html"
            onepage_html_path.write_text(onepage_html, encoding="utf-8")
            report_html_path.write_text(report_html, encoding="utf-8")
            outputs["onepage_html"] = onepage_html_path
            outputs["report_html"] = report_html_path

        if want_basis:
            basis_pdf = output_dir / "basis-report.pdf"
            (tmp_dir / "basis.html").write_text(basis_html, encoding="utf-8")
            HTML(filename=str(tmp_dir / "basis.html"), base_url=base_url).write_pdf(
                str(basis_pdf)
            )
            outputs["basis"] = basis_pdf
            basis_html_path = output_dir / "basis-report.html"
            basis_html_path.write_text(basis_html, encoding="utf-8")
            outputs["basis_html"] = basis_html_path

    # Drop legacy root-level PDF/HTML leftovers from older layout
    for stale in (
        "one-page.html",
        "onepage.html",
        "report.html",
        "onepage.pdf",
        "one-page.pdf",
        "report.pdf",
    ):
        p = run_dir / stale
        if p.exists():
            p.unlink()
    for name in ("exposure_radar", "model_heatmap", "sensitivity_distribution", "evidence_graph"):
        p = run_dir / f"{name}.svg"
        if p.exists():
            p.unlink()
    styles_dst = run_dir / "styles"
    if styles_dst.exists():
        shutil.rmtree(styles_dst)

    return outputs


def stage_index(name: str) -> int:
    try:
        return STAGES.index(name)
    except ValueError as e:
        raise SystemExit(f"Unknown stage {name!r}. Choose from: {', '.join(STAGES)}") from e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MOYO exploration processor")
    ap.add_argument("-e", "--exploration", type=Path, help="Path to exploration.md")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--config", type=Path, default=REPORTS_ROOT / "config.yaml")
    ap.add_argument("--from-stage", default="parse", choices=STAGES)
    ap.add_argument("--stop-after", default="render", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true", help="Heuristic extract; no LLM calls")
    ap.add_argument(
        "--test",
        action="store_true",
        help=(
            "Use fake deterministic LLM clients (no network / API keys). "
            "Implies --dry-run for extract/synthesize; also settable via "
            "MOYO_TEST_MODE=1."
        ),
    )
    ap.add_argument(
        "--report",
        choices=["snapshot", "basis", "both"],
        default="snapshot",
        help=(
            "Which product to render: 'snapshot' = Exposure Snapshot "
            "(one-page + report), 'basis' = Basis Report (comprehensive), "
            "'both'."
        ),
    )
    ap.add_argument(
        "--include-remediation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include mitigations/remediations (ISVF controls, follow-up "
            "playbook, defensive action) in snapshot and basis PDFs. "
            "Default: off (or render.include_remediation in config)."
        ),
    )
    ap.add_argument("--chunk-tokens", type=int, default=None)
    ap.add_argument("--extract-workers", type=int, default=None)
    ap.add_argument("--headline", type=str, default=None)
    ap.add_argument(
        "--keep-graphics",
        action="store_true",
        help=(
            "Reuse existing assets/*.svg instead of regenerating "
            "(edit charts, then: --from-stage render --keep-graphics)"
        ),
    )
    ap.add_argument(
        "--graphics-only",
        action="store_true",
        help=(
            "Regenerate assets/*.svg from report_data.json and exit "
            "(no PDF/email; same as --from-stage graphics --stop-after graphics)"
        ),
    )
    args = ap.parse_args(argv)

    if args.test:
        try:
            from moyo.llm.testing import enable_test_mode
            enable_test_mode()
        except Exception as exc:
            print(f"Warning: could not enable LLM test mode: {exc}", file=sys.stderr)
        args.dry_run = True

    if args.graphics_only:
        if args.keep_graphics:
            raise SystemExit("--graphics-only regenerates charts; drop --keep-graphics")
        args.from_stage = "graphics"
        args.stop_after = "graphics"

    cfg = _load_config(args.config)
    if args.chunk_tokens:
        cfg.setdefault("chunk", {})["target_tokens"] = args.chunk_tokens
        cfg["chunk"]["max_tokens"] = max(args.chunk_tokens, cfg["chunk"].get("max_tokens", 15000))
    if args.extract_workers:
        cfg.setdefault("extract", {})["workers"] = args.extract_workers
    if args.headline:
        cfg.setdefault("render", {})["headline"] = args.headline
    if args.include_remediation is not None:
        cfg.setdefault("render", {})["include_remediation"] = bool(
            args.include_remediation
        )

    exploration = args.exploration
    if exploration is None and cfg.get("input", {}).get("exploration"):
        exploration = Path(cfg["input"]["exploration"])

    run_id = args.run_id or (cfg.get("output") or {}).get("run_id")
    if not run_id:
        if exploration:
            run_id = exploration.parent.name
        else:
            raise SystemExit("Provide --run-id (or --exploration to infer it).")

    run_dir = _resolve_run_dir(cfg, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Recover exploration.md for mid-pipeline rebuilds (topic / cover prompt /
    # explore_meta) even when --exploration was omitted.
    exploration = resolve_exploration_path(
        exploration=exploration,
        run_id=run_id,
        run_dir=run_dir,
        repo_root=REPO_ROOT,
    )
    if exploration is not None:
        remember_exploration_path(run_dir, exploration, repo_root=REPO_ROOT)

    start = stage_index(args.from_stage)
    stop = stage_index(args.stop_after)
    if stop < start:
        raise SystemExit("--stop-after must be at or after --from-stage")

    def want(stage: str) -> bool:
        i = stage_index(stage)
        return start <= i <= stop

    chunk_cfg = cfg.get("chunk") or {}
    extract_cfg = cfg.get("extract") or {}
    # Post-cluster LLM stages (synthesize, Englishize) use Kimi — never Ollama.
    synthesize_cfg = cfg.get("synthesize") or extract_cfg
    cluster_cfg = cfg.get("cluster") or {}
    score_cfg = cfg.get("score") or {}
    graphics_cfg = cfg.get("graphics") or {}
    render_cfg = cfg.get("render") or {}

    chunks_path = run_dir / "chunks.jsonl"
    claims_path = run_dir / "claims.jsonl"
    report_data_path = run_dir / "report_data.json"

    chunks = []
    claims = []
    clusters = []
    report_data: dict = {}
    graphics: dict = {}

    if want("parse"):
        if not exploration or not exploration.exists():
            raise SystemExit(f"exploration.md not found: {exploration}")
        print(f"[1] parse {exploration}", file=sys.stderr)
        chunks = parse_exploration(
            exploration,
            max_tokens=int(chunk_cfg.get("max_tokens", 15000)),
            include_failed=bool(chunk_cfg.get("include_failed", False)),
        )
        write_chunks_manifest(chunks, chunks_path)
        print(f"  → {len(chunks)} chunks → {chunks_path}", file=sys.stderr)
    elif start > stage_index("parse") and chunks_path.exists():
        pass

    if want("extract"):
        if not chunks:
            if not exploration or not exploration.exists():
                raise SystemExit("extract needs --exploration (or prior parse)")
            chunks = parse_exploration(
                exploration,
                max_tokens=int(chunk_cfg.get("max_tokens", 15000)),
                include_failed=bool(chunk_cfg.get("include_failed", False)),
            )
        print(f"[2] extract ({'dry-run' if args.dry_run else extract_cfg.get('model')})", file=sys.stderr)
        prompt = REPORTS_ROOT / (extract_cfg.get("prompt") or "prompts/extract_claims.md")
        claims = extract_all(
            chunks,
            out_path=claims_path,
            prompt_path=prompt,
            config=extract_cfg,
            dry_run=args.dry_run,
            chunk_config=chunk_cfg,
        )
        print(f"  → {len(claims)} claims → {claims_path}", file=sys.stderr)
    elif start > stage_index("extract") and (
        want("cluster") or want("score")
    ):
        if not claims_path.exists():
            raise SystemExit(f"Missing {claims_path}; run extract first")
        claims = load_claims(claims_path)

    # Real-world citations live in exploration.md → chunks.jsonl → claims.jsonl.
    # Refresh them onto the claims whenever the chunk manifest is available, so
    # reports cite sources even when extraction ran before this stage.
    if claims and (want("cluster") or want("score")):
        if not chunks and chunks_path.exists():
            chunks = load_chunks_manifest(chunks_path)
        if chunks:
            attach_chunk_citations(claims, chunks, overwrite=True)
            cited = sum(1 for c in claims if c.get("citations"))
            with claims_path.open("w", encoding="utf-8") as f:
                for c in claims:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(
                f"  → citations on {cited}/{len(claims)} claims → {claims_path}",
                file=sys.stderr,
            )

    if want("cluster"):
        print("[3] cluster", file=sys.stderr)
        before_n = len(claims)
        prompt_rel = cluster_cfg.get("prompt") or "prompts/cluster_claims.md"
        claims, clusters = cluster_claims(
            claims,
            corroboration_min_sources=int(cluster_cfg.get("corroboration_min_sources", 2)),
            collapse=bool(cluster_cfg.get("collapse", True)),
            llm_config=cluster_cfg,
            prompt_path=REPORTS_ROOT / prompt_rel,
            dry_run=bool(args.dry_run),
        )
        with claims_path.open("w", encoding="utf-8") as f:
            for c in claims:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(
            f"  → {len(clusters)} clusters; "
            f"collapsed {before_n} → {len(claims)} claims",
            file=sys.stderr,
        )
    elif start > stage_index("cluster"):
        from collections import defaultdict

        groups: dict[str, list] = defaultdict(list)
        for c in claims:
            groups[c.get("cluster_id") or c["claim_id"]].append(c)
        clusters = [
            {
                "cluster_id": cid,
                "claim_ids": [x["claim_id"] for x in members],
                "models": sorted({x["source_model"] for x in members}),
                "representative_id": members[0]["claim_id"],
                "size": len(members),
            }
            for cid, members in groups.items()
        ]

    if want("score"):
        print("[3] score → report_data.json", file=sys.stderr)
        if exploration and exploration.exists():
            topic = topic_from_exploration(exploration)
            prompts = prompts_from_exploration(exploration)
        else:
            topic = run_id.replace("_", " ")
            prompts = [topic] if topic and topic != run_id else []
        report_data = score_report(
            claims,
            clusters,
            run_id=run_id,
            topic=topic,
            config=score_cfg,
            graphics_cfg=graphics_cfg,
        )
        if prompts:
            report_data["prompts"] = prompts
            report_data["topic"] = prompts[0]
        if render_cfg.get("headline"):
            report_data["headline"] = render_cfg["headline"]
        if exploration and exploration.exists():
            report_data["explore_meta"] = exploration_run_meta(exploration)
        report_data["collection_issues"] = _merge_collection_issues(
            report_data, run_dir
        )
        report_data_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  → {report_data_path}", file=sys.stderr)
    elif start > stage_index("score"):
        report_data = json.loads(report_data_path.read_text(encoding="utf-8"))

    if want("synthesize"):
        print("[3] synthesize", file=sys.stderr)
        if not report_data:
            report_data = json.loads(report_data_path.read_text(encoding="utf-8"))
        if exploration and exploration.exists() and not report_data.get("explore_meta"):
            report_data["explore_meta"] = exploration_run_meta(exploration)
        report_data["collection_issues"] = _merge_collection_issues(
            report_data, run_dir
        )
        if exploration and exploration.exists():
            prompts = prompts_from_exploration(exploration)
            if prompts:
                report_data["prompts"] = prompts
                report_data["topic"] = prompts[0]
        report_data = synthesize(
            report_data,
            prompts_dir=REPORTS_ROOT / "prompts",
            config=render_cfg,
            llm_config=synthesize_cfg,
            dry_run=args.dry_run,
        )
        report_data_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    emit = list(graphics_cfg.get("emit") or [])

    if want("graphics"):
        print("[4] graphics (SVG)", file=sys.stderr)
        if not report_data:
            report_data = json.loads(report_data_path.read_text(encoding="utf-8"))
        if args.keep_graphics:
            graphics = load_graphics_assets(run_dir, emit=emit or None)
            print(f"  → kept {len(graphics)} SVG figures from assets/", file=sys.stderr)
        else:
            graphics = generate_graphics(
                report_data,
                run_dir,
                emit=emit,
                aliases=graphics_cfg.get("model_aliases") or {},
                write_files=False,
                write_assets=True,
            )
            report_data_path.write_text(
                json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  → {len(graphics)} SVG figures → {run_dir / 'assets'}/", file=sys.stderr)
            for key in graphics:
                name = ASSET_NAMES.get(key)
                if name:
                    print(f"  → {run_dir / 'assets' / name}", file=sys.stderr)

    if want("render"):
        print("[5] render PDF", file=sys.stderr)
        if not report_data:
            report_data = json.loads(report_data_path.read_text(encoding="utf-8"))
        if exploration and exploration.exists() and not report_data.get("explore_meta"):
            report_data["explore_meta"] = exploration_run_meta(exploration)
        report_data["collection_issues"] = _merge_collection_issues(
            report_data, run_dir
        )
        if exploration and exploration.exists():
            prompts = prompts_from_exploration(exploration)
            if prompts and (
                not report_data.get("prompts")
                or report_data.get("topic") in {None, "", run_id}
            ):
                report_data["prompts"] = prompts
                report_data["topic"] = prompts[0]
        if args.keep_graphics:
            graphics = load_graphics_assets(run_dir, emit=emit or None)
            print(f"  → using assets/*.svg (not regenerating)", file=sys.stderr)
        elif not graphics:
            graphics = generate_graphics(
                report_data,
                run_dir,
                emit=emit,
                aliases=graphics_cfg.get("model_aliases") or {},
                write_files=False,
                write_assets=True,
            )
        outputs = render_pdfs(
            report_data,
            graphics,
            run_dir=run_dir,
            cfg=cfg,
            keep_graphics=args.keep_graphics,
            report_type=args.report,
            include_remediation=bool(
                (cfg.get("render") or {}).get("include_remediation", False)
            ),
        )
        label = {
            "onepage": "Exposure Snapshot (one-page)",
            "report": "Exposure Snapshot (report)",
            "basis": "Basis Report",
            "email_text": "alert email (text)",
            "email_html": "alert email (html)",
        }
        for key in ("onepage", "report", "basis", "email_text", "email_html"):
            if key in outputs:
                print(f"  → {label[key]}: {outputs[key]}", file=sys.stderr)

    print(f"Done → {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
