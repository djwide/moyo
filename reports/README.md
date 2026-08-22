# MOYO exploration processor

Turns `exploration.md` into scored claims, **SVG** brand figures, and
print-ready PDFs via the **MOYO Report Design System v1**.

**Content** (`report.md` / `report.yaml`) is separate from **presentation**
(`design-system/` templates + CSS). Charts are SVG, not PNG.

Operator guide: [`docs/exploration_processor.md`](../docs/exploration_processor.md)  
Design system: [`design-system/DESIGN.md`](design-system/DESIGN.md)  
Reader glossary (HTML): [`design-system/terminology.html`](design-system/terminology.html)

```bash
# From repo root
pip install -e ".[reports]"   # jinja2 + weasyprint

python reports/build_report.py \
  --exploration data/public_sources/what_is_the_recipe_for_coca_cola/exploration.md \
  --run-id what_is_the_recipe_for_coca_cola

# No LLM / smoke test
python reports/build_report.py -e path/to/exploration.md --dry-run
```

## Per-run package

```text
reports/build/<run-id>/
├── report.md                 # narrative content (LLM / human editable)
├── report.yaml               # structured content for templates
├── report_data.json          # pipeline intermediate
├── claims.jsonl
├── assets/
│   ├── company-logo.svg
│   ├── exposure-radar.svg
│   ├── model-heatmap.svg
│   ├── findings-by-llm.svg
│   ├── evidence-graph.svg
│   └── screenshots/          # optional raster only
└── output/
    ├── report.pdf            # Exposure Snapshot: full (abridged) assessment
    ├── one-page.pdf          # Exposure Snapshot: one-pager (no hyperlinks)
    ├── basis-report.pdf      # Basis Report (comprehensive) — only when built
    ├── alert-email.txt       # short sales/alert email (plain)
    ├── alert-email.html
    └── alert-email.subject.txt
```

Shared presentation (not copied per run): `reports/design-system/`.

## Report types

Two products share one pipeline (`parse → extract → cluster → score →
synthesize → graphics → render`), selected with `--report`:

- **Exposure Snapshot** (`--report snapshot`, default) — `one-page.pdf` +
  `report.pdf` (abridged: top findings/claims) + alert email.
- **Basis Report** (`--report basis`) — `basis-report.pdf`: full findings with
  evidence, complete prioritized exposure inventory, severity + rationale +
  confidence, what was inferred/recovered, multiple corroborating model
  outputs, "exactly how MOYO reached X" derivation, source/evidence graph, full
  exposure chain, and exploitation implications. Remediation is optional
  (off by default; see below).
- `--report both` builds both.

```bash
# Comprehensive Basis Report from an exploration.md
python reports/build_report.py \
  -e data/public_sources/<slug>/exploration.md --report basis

# Both products, re-rendering only (reuse existing run artifacts + charts)
python reports/build_report.py --run-id <id> \
  -e data/public_sources/<slug>/exploration.md \
  --report both --from-stage render --keep-graphics

# Opt in to mitigations / remediations (ISVF + follow-up playbook)
python reports/build_report.py -e path/to/exploration.md \
  --report both --include-remediation
```

Mitigations/remediations are **off by default**. Enable with
`--include-remediation` (or `render.include_remediation: true` in
`config.yaml`). When enabled, Basis Report remediation is sourced from the
**Idea Security Verification Framework** control catalog via
`render.isvf_path` (default `IdeaSecurityVerificationFramework`; set `null` to
disable catalog lookup). A **Build Report** tab in the GUI (`moyo-gui`) exposes
the same options.

Rebuild charts only (overwrites `assets/*.svg` from `report_data.json`):

```bash
python reports/build_report.py --run-id <id> --graphics-only
```

Rebuild PDF after editing content (`report.yaml` / `report.md` / `report_data.json`):

```bash
python reports/build_report.py --run-id <id> --from-stage render --keep-content --keep-graphics
```

Rebuild charts only (overwrites `assets/*.svg` from `report_data.json`):

Edit charts by hand under `assets/*.svg`, then rebuild PDF without regenerating them:

```bash
python reports/build_report.py --run-id <id> --from-stage render --keep-graphics
```

(`exposure-radar.svg`, `model-heatmap.svg`, `findings-by-llm.svg`, `evidence-graph.svg`)
