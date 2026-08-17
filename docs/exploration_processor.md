# Exploration Processor — Operator Guide

This document explains how a human runs a full `exploration.md` through the
MOYO **exploration processor**, and which knobs they can turn without changing
code.

The processor turns a multi-LLM exploration report into scored claims, graphics,
and print-ready HTML/PDF (one-pager + full report). It does **not** replace
`moyo-gather explore` (retrieval). It starts **after** you already have an
`exploration.md`.

---

## Pipeline at a glance

```text
exploration.md
      │
      ▼
[1] deterministic parser / chunker
      │
      ├── preserve ALL raw evidence
      │
      ▼
[2] per-chunk LLM extraction
      │
      ▼
claims.jsonl
      │
      ▼
[3] dedupe + cluster + score
      │
      ▼
report_data.json
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
[4] charts       [4] graphs      [4] tables
  SVG              SVG
      └──────────────┴──────────────┘
                     │
                     ▼
            [5] Jinja HTML template
                     │
             �──────────────┘
                     │
                     ▼
            [5] Jinja HTML template
                     │
             ┌───────┴───────┐
             ▼               ▼
       one-page.html      report.html
             │               │
             ▼               ▼
       one-page.pdf      report.pdf
```

**Renderer stack (not XeLaTeX):**

```text
Markdown / JSON data → Jinja HTML + CSS → WeasyPrint → PDF
```

Repository layout (under `reports/`):

```text
reports/
├── build_report.py          # CLI entry: process one exploration.md
├── config.yaml              # human-tunable knobs (see below)
├── pipeline/                # parse → extract → cluster → score → synthesize → graphics
├── schemas/                 # claim + report JSON schemas
├── prompts/                 # extractor / cluster / executive / follow-up prompts
├── templates/               # onepage.html.j2, report.html.j2, appendix.html.j2
├── styles/                  # base.css, onepage.css, report.css (MOYO brand)
├── graphics/                # heatmap, graph, exposure radar generators
├── assets/branding/         # logos + favicons
└── build/                   # outputs for the latest (or named) run
```

---

## What you need before starting

| Requirement | Why |
|-------------|-----|
| A completed `exploration.md` | Input document (from `moyo-gather explore`) |
| Optional `summary.md` | Improves executive synthesis if present; not required for lossless claim extraction |
| Working LLM for extraction | Default: local Ollama (`llama3.1:8b`) or override in `reports/config.yaml` |
| Python env with WeasyPrint + Jinja2 | HTML → PDF |
| Disk space under `reports/build/` | Intermediate JSON + SVG + HTML + PDF |

You do **not** need XeLaTeX.

---

## Human workflow (push a document through)

### Step A — Place or point at the exploration

1. Finish an explore run (or copy an existing folder), e.g.  
   `data/public_sources/what_is_the_recipe_for_coca_cola/exploration.md`
2. Decide a **run id** (slug). Default = parent folder name  
   (`what_is_the_recipe_for_coca_cola`).

### Step B — Review / set knobs (optional but common)

Edit `reports/config.yaml` (see [Knobs](#knobs-human-tunable-inputs)) **or** pass
CLI overrides. Typical pre-run checks:

- Chunk size still in the 5–15k token band for your extractor model context
- Sensitivity / interestingness thresholds for the one-pager “HIGH” badges
- Which models count toward “LLMs tested” (exclude failed probes if desired)
- Brand date format and whether to force a report date

### Step C — Run the processor

```bash
# From repo root (once build_report.py is wired)
python reports/build_report.py \
  --exploration data/public_sources/what_is_the_recipe_for_coca_cola/exploration.md \
  --run-id what_is_the_recipe_for_coca_cola
```

Useful variants:

```bash
# Only parse + extract (inspect claims.jsonl before clustering)
python reports/build_report.py ... --stop-after extract

# Re-cluster / re-score without re-calling the extractor
python reports/build_report.py ... --from-stage cluster

# Rebuild HTML/PDF only from existing report_data.json
python reports/build_report.py ... --from-stage render
```

### Step D — Human QA on claims (recommended)

Open `reports/build/<run-id>/claims.jsonl` and spot-check:

1. **Outliers and contested claims are still present** (never dropped for being
   unusual, disputed, sensitive, or single-source).
2. `raw_excerpt` + `raw_start_line` / `raw_end_line` still match `exploration.md`.
3. Categories and statuses look right:
   - `CORROBORATED` — ≥2 distinct sources
   - `CONTESTED` — sources disagree on a material point
   - `OUTLIER` — surprising / extreme vs consensus
   - `UNVERIFIED` — plausible but weakly grounded
   - `MODEL-SPECIFIC` — unique to one model family

If extraction quality is poor, adjust prompts under `reports/prompts/` or
extraction temperature / chunk size, then re-run from `--from-stage extract`.
Interrupted extracts resume automatically (finished chunk ids are recorded in
`extract_done.jsonl` beside `claims.jsonl`). To start over, delete both files.

### Step E — Review scored report data

Open `reports/build/<run-id>/report_data.json`:

- Exposure tallies (high / medium / low / informational)
- Top findings for the one-pager
- Model exposure ranks
- Evidence chains

**Human can edit this JSON** (numbers, titles, chain labels) before render if
you want editorial control without re-extracting. Then run `--from-stage render`.

### Step F — Render HTML + PDF

The render stage builds one of two products, chosen with `--report`:

| `--report` | Output | Role |
|------------|--------|------|
| `snapshot` (default) | `one-page.pdf` + `report.pdf` | **Exposure Snapshot** — shareable one-pager + abridged assessment |
| `basis` | `basis-report.pdf` | **Basis Report** — comprehensive assessment |
| `both` | all of the above | |

The **Basis Report** reuses the same run artifacts (no re-extraction) and adds:
full findings with evidence, a complete prioritized exposure inventory,
severity + rationale + confidence per finding, what was inferred/recovered,
multiple corroborating model outputs, an "exactly how MOYO reached X"
derivation, the source/evidence graph, the full exposure chain, and
exploitation implications. Mitigations/remediations (ISVF control catalog +
follow-up playbook) are **off by default**; pass `--include-remediation` (or
set `render.include_remediation: true`) to include them. Catalog path:
`render.isvf_path` (set `null` to disable lookup when remediation is on).

```bash
python reports/build_report.py -e path/to/exploration.md --report basis
python reports/build_report.py -e path/to/exploration.md --report basis \
  --include-remediation
```

The **Build Report** tab in `moyo-gui` exposes the same choices (exploration
path, run id, report type, from-stage, dry-run, keep-graphics,
include-remediation).

Open the HTML in a browser first; PDF via WeasyPrint should match closely.

### Step G — Ship / archive

Copy or commit the `reports/build/<run-id>/` folder (or export only the PDFs).
Keep `claims.jsonl` + `report_data.json` with the PDFs so findings stay auditable.

---

## Stage-by-stage: what is automatic vs human

| Stage | Automatic | Human input |
|-------|-----------|-------------|
| **[1] Parse / chunk** | Split by language → query → model response; keep line offsets; never drop text | Chunk target size; max chunk tokens; whether to include pruned stubs |
| **[2] Extract** | Per-chunk LLM → claim objects into `claims.jsonl` (resumes via `extract_done.jsonl` if interrupted) | Extractor model, temperature, prompt file, concurrency; optional “must-include themes”. Delete `claims.jsonl` + `extract_done.jsonl` to force a full re-extract |
| **[3] Dedupe / cluster / score** | Local Ollama groups same-fact claims; collapse into one; union citations + source_models; keep `member_scores`; sensitivity = max; confidence = # LLMs | Ollama model/URL; batch size; collapse on/off; corroboration minimum |
| **[4] Graphics** | SVG radar, heatmap, sensitivity bars, evidence graph | Which graphics to emit; color scale; model alias map for short names |
| **[5] Templates → PDF** | Jinja fill + WeasyPrint | Template copy/tone; logo choice; forced headline; date; hide/show “REQUEST FULL REPORT” |

---

## Claim object contract (lossless evidence)

Every extracted finding should look like:

```json
{
  "claim_id": "C0142",
  "claim": "Model supplied a reconstructed Merchandise 7X formula",
  "source_model": "Kimi",
  "query_id": "Q18",
  "category": "proprietary_adjacent",
  "sensitivity": 4,
  "specificity": 5,
  "novelty": 5,
  "confidence": 3,
  "corroboration": 1,
  "source_count": 0,
  "interestingness": 5,
  "status": "OUTLIER",
  "raw_excerpt": "...exact source text...",
  "raw_start_line": 741,
  "raw_end_line": 759,
  "citations": ["https://example.org/report"]
}
```

After clustering, similar claims are **collapsed** into one survivor by a
**local Ollama** model (same-fact paraphrase detection). `corroboration` /
`confidence` equal the number of distinct LLMs on the merged claim (clamped
1–5), `member_scores` keeps each member's individual dims, union
`sensitivity` is the highest member sensitivity, and `specificity` gains +1
when the claim text contains exact numbers.

**Hard rule for operators and prompts:** never omit a finding solely because it
is unusual, disputed, sensitive, or single-model. Classify it instead:

| Status | Meaning |
|--------|---------|
| `CORROBORATED` | Multiple distinct LLMs agree |
| `CONTESTED` | Material disagreement across sources |
| `OUTLIER` | Diverges sharply from consensus / unusually specific |
| `UNVERIFIED` | Not well grounded / weak support |
| `MODEL-SPECIFIC` | Distinctive to one model (or language-tagged instance) |

Score dimensions are typically **1–5 integers** (tunable ranges in config).

---

## One-pager content map

The snapshot layout operators should expect:

```text
┌─────────────────────────────────────────────────────────┐
│ MOYO                                         8 AUG 2026 │
│ EXPOSURE SNAPSHOT                                       │
│ headline from topic / optional override                 │
├─────────────────────────────────────────────────────────┤
│ FINDINGS COUNT │ LLMs TESTED │ HIGH-SENSITIVITY COUNT   │
├───────────────────────────────┬─────────────────────────┤
│ MOST SIGNIFICANT FINDING      │ MODEL EXPOSURE (dots)   │
├───────────────────────────────┴─────────────────────────┤
│ EXPOSURE CHAIN (3–5 steps)                              │
├───────────────────────────────┬─────────────────────────┤
│ WHAT ELSE WE FOUND            │ FULL ASSESSMENT teaser  │
└───────────────────────────────┴─────────────────────────┘
```

**Standard graphics (every run):**

1. **Exposure radar** — specificity / sensitivity / corroboration / novelty / confidence  
2. **Model heatmap** — findings × models  
3. **Sensitivity distribution** — high / medium / low / informational  
4. **Evidence graph** — responses → claims → chains  

---

## Knobs (human-tunable inputs)

Primary file: [`reports/config.yaml`](../reports/config.yaml).

### Input / paths

| Knob | Default idea | Effect |
|------|--------------|--------|
| `input.exploration` | CLI `--exploration` | Source markdown |
| `input.summary` | sibling `summary.md` if present | Extra context for synthesis |
| `output.run_id` | folder slug | Namespace under `reports/build/` |
| `output.dir` | `reports/build/<run_id>` | Where artifacts land |

### Chunking `[1]`

| Knob | Typical | Effect |
|------|---------|--------|
| `chunk.target_tokens` | `8000` | Aim for 5–15k; raise if extractor context is large |
| `chunk.max_tokens` | `15000` | Hard ceiling before forced split |
| `chunk.min_tokens` | `120` | Skip thinner stubs **before** paid extract |
| `chunk.languages` | `null` | Optional allowlist, e.g. `[English]`, to cut multilingual extract cost |
| `chunk.skip_refusals` | `true` | Skip refusal / empty-hedge chunks before extract |
| `chunk.include_failed` | `false` | Whether “retrieval failed” blocks become chunks |
| `chunk.include_pruned_stubs` | `false` | Foreign-language pruned stubs |

Extract also strips leading `#####` model headers and trailing Sources/URL
lists from the **prompt text only** (line offsets still match `exploration.md`).

### Extraction `[2]`

| Knob | Typical | Effect |
|------|---------|--------|
| `extract.provider` / `model` | `custom` / `kimi-k2.6` | Who extracts claims (Moonshot Kimi by default) |
| `extract.base_url` | `https://api.moonshot.ai/v1` | OpenAI-compatible endpoint |
| `extract.api_key` | `$MOONSHOT_API_KEY` | Env var (or literal) for the extractor |
| `extract.temperature` | `0.2` | Lower = stabler JSON |
| `extract.max_tokens` | `2500` | Per-chunk completion budget |
| `extract.workers` | `4` | Parallel chunk extractions |
| `extract.prompt` | `prompts/extract_claims.md` | Instruction text (`query_text` injected) |
| `extract.require_raw_excerpt` | `true` | Reject claims missing evidence span |

**Phase 2 (planned, not shipped):** cheap triage before paid extract via
`extract.triage` (`enabled`, `backend: heuristic|ollama`, `keep: [dense, sparse]`)
writing `triage.jsonl` under the run dir for audit. Use chunk gates above until
then.

### Clustering / scoring `[3]`

| Knob | Typical | Effect |
|------|---------|--------|
| `cluster.provider` | `ollama` locally; Cloud Run overlays a hosted utility LLM | Same-fact grouping |
| `cluster.model` | `llama3.1:8b` locally; OpenRouter `meta-llama/llama-3.1-8b-instruct` in cloud | Utility model |
| `cluster.base_url` | `http://localhost:11434` locally; OpenRouter in cloud | Utility endpoint |
| `cluster.batch_size` | `35` | Claims per grouping call |
| `cluster.collapse` | `true` | Merge similar claims into one (false = annotate only) |
| `cluster.corroboration_min_sources` | `2` | Threshold for `CORROBORATED` |
| `score.weights.sensitivity` | `0.25` | Rank weight for one-pager |
| `score.weights.specificity` | `0.25` | |
| `score.weights.novelty` | `0.20` | |
| `score.weights.interestingness` | `0.20` | |
| `score.weights.confidence` | `0.10` | |
| `score.high_sensitivity_min` | `4` | Counts toward “HIGH-SENSITIVITY” |
| `score.top_finding_count` | `1` | Featured on one-pager |
| `score.chain_count` | `3` | Chains in snapshot + executive |

### Graphics `[4]`

| Knob | Typical | Effect |
|------|---------|--------|
| `graphics.emit` | list of names | Enable/disable radar, heatmap, etc. |
| `graphics.model_aliases` | map | Short labels on heatmap (`ChatGPT` → `GPT`) |
| `graphics.dot_max` | `5` | Model exposure dots ●●●○○ |

### Render / brand `[5]`

| Knob | Typical | Effect |
|------|---------|--------|
| `render.headline` | `null` | Override auto headline |
| `render.report_date` | `null` (= today) | Force date on one-pager |
| `render.isvf_path` | `IdeaSecurityVerificationFramework` | ISVF repo for remediation when enabled (`null` disables) |
| `render.include_remediation` | `false` | Include mitigations/remediations in snapshot + basis (CLI: `--include-remediation`) |
| `render.partner_logo` | `assets/branding/SenTeGuardLogo.png` | Second header logo (title page) |
| `render.logo` | `assets/branding/moyo-logo-wordmark.png` | Header mark |
| `render.favicon` | `assets/branding/favicon-32x32.png` | HTML favicon |
| `render.show_request_full_report` | `true` | CTA on one-pager |
| `render.page_size` | `A4` | WeasyPrint page |
| `render.base_css` / `onepage_css` / `report_css` | paths | Visual system |

### Brand colors (CSS variables)

Aligned with MOYO logos under `reports/assets/branding/`:

| Token | Approx hex | Use |
|-------|------------|-----|
| `--moyo-teal` | `#4FB0A2` | Accents, M / O ring |
| `--moyo-cream` | `#F2F1E8` | Light text / surfaces |
| `--moyo-ink` | `#1D2228` | Dark type / Y |
| `--moyo-black` | `#000000` | Header bars, PDF chrome |

---

## CLI cheat sheet (target interface)

```bash
# Full pipeline
python reports/build_report.py -e path/to/exploration.md

# Knobs without editing YAML
python reports/build_report.py -e path/to/exploration.md \
  --chunk-tokens 10000 \
  --extract-workers 6 \
  --headline "What AI systems reveal about Coca-Cola's secret formula"

# Editorial pass: edit report_data.json, then
python reports/build_report.py --run-id <id> --from-stage render
```

---

## Quality checklist (human)

Before sending PDFs outward:

- [ ] One-pager headline is accurate and non-sensational beyond the evidence  
- [ ] High-sensitivity count matches findings with `sensitivity >=` threshold  
- [ ] Most significant finding still has `raw_excerpt` traceable in `exploration.md`  
- [ ] Contested / outlier / model-specific items appear (not silently dropped)  
- [ ] Model exposure dots match the models that actually returned content  
- [ ] Evidence chain steps are reproducible (each step tied to claim ids)  
- [ ] HTML and PDF both open; logos and date correct  
- [ ] Full report includes methodology note (white-box / multi-LLM fan-out) and remediation section  

---

## Relationship to existing `moyo-gather` commands

| Command | Role vs this processor |
|---------|------------------------|
| `moyo-gather explore` | Creates `exploration.md` (upstream) |
| `moyo-gather summarize` | Optional `summary.md` claims brief (local Ollama) |
| `moyo-gather deliverable` | Earlier Grok HTML deliverable (related narrative product) |
| **`reports/build_report.py`** | Structured, lossless claim pipeline → one-pager + full report PDF |

Use gather for acquisition; use the exploration processor for **auditable scoring and print products**.

---

## Failure modes operators hit most

| Symptom | Likely knob / action |
|---------|----------------------|
| Extractor returns empty / truncated JSON | Lower `chunk.target_tokens`; raise `extract.num_ctx` / `max_tokens` |
| Too many near-duplicate claims | Ensure Ollama is up; lower `cluster.batch_size` if merges are weak; re-run `--from-stage cluster` |
| One-pager feels “safe” / generic | Raise weight on `specificity` + `interestingness`; check outliers not filtered |
| PDF missing fonts / broken layout | Check WeasyPrint install; simplify CSS; verify image paths absolute or relative to HTML |
| Run too slow | Raise `extract.workers`; exclude failed/pruned chunks; `--from-stage` to skip work |

---

## Status of this package

- **Specified:** pipeline, schemas, prompts, templates, graphics, brand assets, operator knobs.  
- **Scaffold:** `reports/` tree, `config.yaml`, JSON schemas, prompt stubs, CSS brand tokens, `build_report.py` entrypoint.  
- **Implement next:** wire `pipeline/*.py` stages end-to-end against a real `exploration.md` (Coca-Cola folder is a good first fixture).
