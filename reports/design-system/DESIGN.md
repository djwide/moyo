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
│   ├── findings-by-llm.svg
│   ├── evidence-graph.svg
│   └── screenshots/       # optional raster evidence
└── output/
    ├── report.pdf
    └── one-page.pdf
```

Shared presentation lives in `reports/design-system/` (not copied per run).

**Density:** Findings, evidence, next steps, sources/citations, and glossary
mix dense tables with a few spacious callouts. Major sections start on a
new A4 page (`break-before: page`); sources and glossary may share a sheet.
Do not lay out the report as a web-style infinite scroll of identical cards.

## Fonts

| Role | Primary | Fallback |
|------|---------|----------|
| Headings / body | IBM Plex Serif (bundled) | Georgia, Times New Roman, serif |
| UI / labels | IBM Plex Sans (bundled) | Helvetica Neue, Arial |
| Mono / IDs | IBM Plex Mono (bundled) | ui-monospace, Menlo, Consolas |

WeasyPrint embeds the WOFF files in `design-system/fonts/`. Do not fall back to Inter or Geist.

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
| Cover | Brand, prompt, date, models |
| What the models disclosed | Narrative + stat strip |
| Which models disclosed the most | Findings-by-LLM bars + exposure radar |
| Findings that carry this exposure | Lead finding + compact / band variants |
| Verbatim excerpts | Evidence boxes + graph |
| Where the models validate and surface the same information, and where they don't | Commonality vs differences, then heatmap |
| Cluster index | Compact findings table keyed by cluster |
| What the models cited | Source table with repeating headers |
| How to read the scores | Glossary |

## Components

| Component | Class / macro | Notes |
|-----------|---------------|-------|
| Severity mark | `severity-badge` | Color tick + sentence-case label; not a filled chip |
| Finding | `finding_block` | lead / compact / band via Jinja loop |
| Evidence box | `evidence-box` | transcript excerpt + line refs |
| Quote box | `quote-box` | pull quote as a hairline band |
| Stat strip | `stat-strip` | lead number + compact counts, not a card grid |
| Risk matrix | `risk-matrix` | optional; findings-by-LLM chart is preferred |
| Model comparison chart | SVG asset | heatmap / exposure dots |
| Confidence indicator | `confidence` | 1–5 dots |
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
