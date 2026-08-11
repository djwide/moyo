# claudeExposureBuild

An alternative **Exposure Snapshot + one-page** builder, authored from scratch
as a comparison against the default `reports/build_report.py` product. It keeps
the same house style (grey title banner, MOYO + SenTeGuard logos, "What AI
Systems Reveal", abridged sections, prose executive summary) but ships its
**own charts, templates, and CSS**.

It renders from the **same scored data** (`report_data.json`) as the default
builder, so the two products are directly comparable on the same findings.

## Run

```bash
# From an existing run (report_data.json already produced by build_report.py)
python reports/claudeExposureBuild/build.py --run-id VicenteGonzalezOppo --report both

# From scratch (builds report_data.json via the shared pipeline first)
python reports/claudeExposureBuild/build.py \
  -e data/public_sources/<slug>/exploration.md --report both
```

`--report` is `snapshot` (report.pdf), `onepage` (one-page.pdf), or `both`
(default). Add `--dry-run` to use heuristic extraction when building data.

## Output

Written to a separate folder so it never overwrites the default builder:

```
reports/build/<run-id>/claude/output/
├── report.pdf      # multi-page Exposure Snapshot
└── one-page.pdf    # landscape one-pager (no hyperlinks)
```

## What's different

- **Own charts** (inline SVG, no external assets): severity donut, exposure
  dimension lollipops, model-contribution bars, and a stepped exposure ladder —
  in place of the default builder's bars / radar / heatmap / sankey.
- **Self-contained render**: CSS is inlined and logos are embedded as data URIs,
  so each PDF is produced from a single HTML string (no temp asset copying).
- **Same data contract**: consumes `report_data.json` fields (findings,
  clusters, chains, `radar_averages`, `sensitivity_bins`, `model_exposure`,
  `explore_meta`) and reuses only pure data helpers (`parse_executive_payload`,
  `format_source_cite`, `short_model_name`).

## Files

- `charts.py` — SVG chart generators + palette.
- `content.py` — content assembly (meta, executive prose, abridged sets, charts).
- `templates/` — `report.html.j2`, `one-page` (`onepage.html.j2`).
- `css/` — `report.css`, `onepage.css`.
- `build.py` — CLI entry point.
