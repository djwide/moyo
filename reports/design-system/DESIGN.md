# MOYO Report Design System v1

Content is separate from presentation.

- **Content** (per run, LLM-populated): `report.md` + `report.yaml`
- **Presentation** (shared): this design system — templates, CSS, SVG charts
- **Assets** (per run): `assets/*.svg` (charts, logo) — prefer SVG over PNG
- **Output**: `output/report.pdf`, `output/one-page.pdf`

```text
reports/build/<run-id>/
├── report.md              # narrative content
├── report.yaml            # structured content + meta
├── assets/
│   ├── company-logo.svg
│   ├── exposure-radar.svg
│   ├── model-heatmap.svg
│   ├── sensitivity-distribution.svg
│   ├── evidence-graph.svg
│   └── screenshots/       # optional raster evidence
└── output/
    ├── report.pdf
    └── one-page.pdf
```

Shared presentation lives in `reports/design-system/` (not copied per run).

## Fonts

| Role | Primary | Fallback |
|------|---------|----------|
| Headings / UI | Inter / Geist | IBM Plex Sans, Helvetica Neue, Arial |
| Body prose | Source Serif 4 | Georgia, Times New Roman, serif |
| Mono / IDs | IBM Plex Mono | ui-monospace, Menlo, Consolas |

## Brand color

| Token | Hex |
|-------|-----|
| Teal | `#4FB0A2` |
| Cream | `#F2F1E8` |
| Ink | `#1D2228` |
| Black | `#000000` |
| High | `#C0392B` |
| Medium | `#D68910` |
| Low | `#1E8449` |

## Page types

| Page | Purpose |
|------|---------|
| Cover | Brand, topic, date, classification |
| Executive Summary | Narrative + key metrics |
| Risk Overview | Severity distribution + exposure radar |
| Finding | Finding card(s) with scores + status |
| Evidence | Evidence boxes + graph |
| Model Comparison | Heatmap + model exposure ranks |
| Methodology | How the assessment was run |
| Appendix | Claim index / chains |

## Components

| Component | Class / macro | Notes |
|-----------|---------------|-------|
| Severity badge | `severity-badge` | high / medium / low / info |
| Finding card | `finding-card` | claim + scores + status |
| Evidence box | `evidence-box` | raw excerpt + line refs |
| Quote box | `quote-box` | pull quote |
| Metric card | `metric-card` | big number + label |
| Risk matrix | `risk-matrix` | optional 2×2; bins chart often used |
| Model comparison chart | SVG asset | heatmap / exposure dots |
| Confidence indicator | `confidence` | 1–5 dots or bar |
| Remediation box | `remediation-box` | follow-up method + action |

## LLM population contract

The synthesize stage (and optional human edit) fills `report.yaml` / `report.md`.
Templates **must not** invent findings — they only render fields from content.
Charts are generated as SVG into `assets/` and referenced by path in `report.yaml`.

## Terminology companion (web)

Public glossary for report readers (standalone HTML, embeddable):

[`terminology.html`](terminology.html)

Covers findings/IDs, status labels, score dimensions, severity bands, source
citations (`Model + N`), charts, exposure metrics, and pipeline stages.
