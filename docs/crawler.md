# moyo Public Sources: Crawl and Explore

## Overview

The public-side gather package has two complementary modes:

1. **Crawl** — aggregate open-source documents by topic or tokens (adapters for patents, press, git, conferences, etc.).
2. **Explore** — take a naive plain-language prompt, reword it into retrieval seeds, fan out to every configured retrieval LLM, and write `exploration.md` only.

Explore is the path for non-technical “give me everything you know about X” questions. Crawl remains the path for building a public FAISS corpus from external sources.

## Explore (naive prompt → multi-LLM report)

### CLI

```bash
# Basic scan: rotate paraphrase / translate / summarize (n=3 => each once)
moyo-gather explore --prompt "What is the recipe for Coca-Cola?" --fuzz-mode basic

# Multiple prompts (repeat --prompt, and/or load a file — one prompt per line)
moyo-gather explore \
  --prompt "What is the recipe for Coca-Cola?" \
  --prompt "Who killed JFK?" \
  --fuzz-mode basic
moyo-gather explore --prompts-file prompts.txt --fuzz-mode basic

# Multilingual: full strategy set per language (English + ES / FR / Mandarin Chinese)
moyo-gather explore --prompt "..." --fuzz-mode multilingual --seeds 3

# Extra languages (added on top of the defaults)
moyo-gather explore --prompt "..." --fuzz-mode multilingual -l German -l Japanese

# Optional typo strategy (not in mode defaults)
moyo-gather explore --prompt "..." -S paraphrase -S translate -S summarize -S typo

# Probe retrieval LLMs only (same preflight table as explore; no report)
moyo-gather check-llms
moyo-gather check-llms --json

# Optional: summary.md from an existing exploration.md (not written by explore)
moyo-gather summarize --dir data/public_sources/what_is_the_recipe_for_coca_cola
moyo-gather summarize -e path/to/exploration.md -o path/to/summary.md

# Optional: deliverable from exploration.md + summary.md via Grok (xAI)
moyo-gather deliverable --dir data/public_sources/what_is_the_recipe_for_coca_cola
moyo-gather summarize --dir data/... --with-deliverable

# Structured exploration processor (claims → one-pager + full report PDF)
# See docs/exploration_processor.md
python reports/build_report.py \
  --exploration data/public_sources/what_is_the_recipe_for_coca_cola/exploration.md \
  --run-id what_is_the_recipe_for_coca_cola
```

Claims / narrative synthesis uses **local Ollama** (`llama3.1:8b` by default), not the remote `MOYO_LLM_*` default. Ollama’s default context window is only ~2048–4096 tokens even though Llama 3.1 can go much higher; moyo raises it via `MOYO_SUMMARY_NUM_CTX` (default `32768`). Larger `num_ctx` needs more RAM/VRAM.

Outputs land under `data/public_sources/<slug>/`:

| File | Contents |
|------|----------|
| `exploration.md` | Sources, seeds, detailed findings (language → query → model) — **written by explore** |
| `summary.md` | Optional claims brief — only if you run `moyo-gather summarize` |
| `deliverable.html` | Optional Deliverable via Grok (needs summary.md) |
| `deliverable.md` | Short sidecar pointing at the HTML report |

### Fuzz modes

| Mode | Seeds |
|------|--------|
| **basic** | Rotate `paraphrase → translate → summarize`. `--seeds 3` does each once; `6` does each twice. Add `-S typo` a la carte. |
| **multilingual** | For English and each target language, rotate `paraphrase / abstract / summarize`. `--seeds` is **per language group**. Defaults: Spanish, French, Mandarin Chinese (+ `--language`). |

Seed generation uses the **local Ollama fuzzer** (`llama3.1:8b`), not the remote default LLM.

### Pipeline

1. **Preflight** — probe each retrieval LLM; print `name / status / reason` before the scan.
2. **Reword** — local fuzzer builds labeled seeds (strategy + language).
3. **Raw retrieval** — every seed × every retrieval LLM in parallel (stable `seed_index` / `llm_index`).
4. **Compile** — sort, organise, label; translate foreign answers to English locally (Ollama). No analysis yet.
5. **Analyze** — default LLM synthesises narrative summary + claims brief (outliers, model distinctions, language-annotated attributions).
6. **Render** — `exploration.md` writes compiled findings first, then the summary.

Foreign-language sections are annotated in headers (e.g. `Kimi (Mandarin Chinese)`). Original-language text is kept in a collapsible block when translation ran.

### LLM configuration

| Role | Where | Used for |
|------|--------|----------|
| **Utility fuzzer** | Desktop: Ollama at `127.0.0.1:11434`. Cloud Run: OpenRouter Llama 3.1 8B Instruct | Seed rewording + response translation |
| **Default LLM** | `MOYO_LLM_*` in `.env` | Summary / claims synthesis |
| **Retrieval LLMs** | `config/retrieval_llms.json` | Fan-out targets (ChatGPT, Claude, Grok, Gemini, Qwen, Kimi, Perplexity, OpenRouter, …) |

Copy `config/retrieval_llms.example.json` → `config/retrieval_llms.json` and put API keys in `.env` (see `.env.example`). Providers without a key fail that source only; the run continues.

See [`docs/configuration_and_monitoring_summary.md`](configuration_and_monitoring_summary.md) for Ollama/WSL setup and key wiring.

### GUI

On the **Gather Public Sources** tab, choose **Naive prompt (AI explore)**, pick fuzz mode, optionally add extra languages (multilingual), and run. Progress (including the LLM preflight table) streams into the log.

### Python API

```python
from moyo.publicside.gatherpublicsources.explorer import explore_and_save

result = explore_and_save(
    "What is the recipe for Coca-Cola?",
    fuzz_mode="basic",          # or "multilingual"
    num_seeds=3,
    extra_languages=["German"], # multilingual only
)
print(result.output_path, result.summary_path)
```

## Crawl (topic / tokens → sources.json)

### Components

- `moyo/publicside/gatherpublicsources/crawler.py` — orchestrator:
  - `crawl(topic)` — crawl by topic string
  - `crawl_with_tokens(tokens)` — crawl by token list
- `schema.py` — Pydantic models for sources, configs, jobs
- `sources/` — adapters against public APIs (PatentsView/Google Patents, GDELT, GitHub, arXiv/OpenAlex, NVD/GHSA)
- `parsers/` / `enrichers/` — HTML/PDF/text, classification, dedupe

### CLI

```bash
moyo-gather crawl --topic "artificial intelligence"
moyo-gather crawl-tokens --tokens "neural networks,transformers,LLM"
moyo-gather crawl --topic "machine learning" --output results/ml_sources
```

Adapters (no placeholder hosts):

| Source type | Public API | Optional env |
|---|---|---|
| Patents | USPTO [PatentsView](https://patentsview.org/) or Google Patents xhr | `PATENTSVIEW_API_KEY` |
| Press / news | [GDELT 2.0 DOC API](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/) (PR Newswire, Business Wire, …) | — |
| Git commits | [GitHub commit search](https://docs.github.com/en/rest/search/search#search-commits); GitLab if token set | `GITHUB_TOKEN`, `GITLAB_TOKEN` |
| Papers / talks | [arXiv API](https://info.arxiv.org/help/api/index.html), [OpenAlex](https://docs.openalex.org/) | `OPENALEX_MAILTO` |
| Advisories | [NVD CVE 2.0](https://nvd.nist.gov/developers/vulnerabilities), [GitHub Advisories](https://docs.github.com/en/rest/advisories) | `NVD_API_KEY`, `GITHUB_TOKEN` |

GitHub dorks, paste-site scrapers, IEEE/ACM HTML, and `example.com` stubs were removed. Credential harvesting is not implemented.

### Token-driven crawling

Use centroid/topic tokens from the private corpus to focus queries:

```python
from pathlib import Path
from moyo.privateside.mapcorpus import tokens_for_corpus
from moyo.publicside.gatherpublicsources.crawler import PublicSourcesCrawler

centroids, topic_tokens, labels, texts = tokens_for_corpus(
    Path("data/private/corpus.txt"), top_k=8
)
tokens = [t for cluster in topic_tokens for t in cluster][:25]

crawler = PublicSourcesCrawler()
result = crawler.crawl_with_tokens(tokens)
print(result.message, result.output_path)
```

Outputs under `data/public_sources/<topic>/` when persistence is enabled:

- `sources.json` — normalized sources
- `summary.json` — counts, types, date ranges

## Related docs

- [`configuration_and_monitoring_summary.md`](configuration_and_monitoring_summary.md) — `.env`, retrieval LLMs, Ollama
- [`gui.md`](gui.md) — Gather Public Sources tab
- [`architecture.md`](architecture.md) — package layout
