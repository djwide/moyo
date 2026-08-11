# moyo Architecture

## Overview

moyo is an experimental tool for corpus mapping and information-barrier analysis. It shares low-level utilities with sente via the vendored `shared_utils` package.

## Repository Structure

```
.                               ← repo root
├── moyo/                       ← Python package
│   ├── cli.py                  # Main CLI entry point
│   ├── llm/                    # Shared LLMClient + default/retrieval registry
│   ├── privateside/
│   │   ├── datainput/          # GUI bridge and file/text input
│   │   └── mapcorpus/          # Corpus building and centroids
│   ├── publicside/
│   │   ├── gatherpublicsources/  # Crawler + naive-prompt explorer
│   │   └── barrierprobe/       # Barrier analysis and LLM search
│   ├── redteam/                # LLM red-teaming (whitebox + blackbox)
│   ├── gui/                    # PyQt5 desktop GUI (moyo-gui → gui/app.py)
│   ├── config/                 # Pydantic settings (MOYO_* env)
│   ├── metrics.py / metrics_server.py / cli_metrics.py
│   └── logging.py
├── config/                     # retrieval_llms.json, model_config.json
├── indexes/                    # FAISS indexes only (per-corpus names)
├── shared_utils/               # Vendored shared utilities
│   ├── embeddings.py, faiss_index.py, chunking.py, model_config.py, ...
│   └── models/                 # miniLM_fp32.onnx, miniLM_int8.onnx
├── .env.example
└── docs/
```

## Components

### Private Side (`privateside/`)

#### GUI Bridge (`datainput/gui_bridge.py`)
Receives data from GUI applications (text or files), validates and preprocesses it, then passes it to the corpus builder.

#### Corpus Builder (`mapcorpus/builder.py`)
- Text normalisation and deduplication
- Multi-granularity chunking (sections, sentences, list/bullet items) via
  `shared_utils.chunking`
- Embedding generation (sentence-transformers)
- FAISS index creation under `indexes/` with per-corpus filenames

#### Centroids (`mapcorpus/centroids.py`)
Derives topic tokens from the private corpus for use as crawl seeds on the public side.

### Public Side (`publicside/`)

#### Gather (`gatherpublicsources/`)
- `crawler.py` — topic / token crawl into `sources.json`
- `explorer.py` — naive-prompt multi-LLM explore → `exploration.md`
  (preflight → reword → parallel retrieve → compile/localize → analyze → render)
- CLI: `moyo-gather crawl`, `crawl-tokens`, `explore`, `summarize`, `deliverable`, `check-llms`

Source adapters: patents, press releases, git commits, conference talks, arXiv/PubMed, generic web search.

#### Barrier Probe (`barrierprobe/`)
Analyses information barriers between private and public FAISS indexes:
- `barrier_analyzer.py` – cosine distance and Sobolev norm analysis
- `llm_fuzzer.py` – LLM-assisted fuzzing; explore rewording; fuzz modes
  `basic` / `multilingual` (local Ollama default for generation)
- `iterative_llm_search.py` – iterative refinement of closest matches
- `two_layer_fuzzer.py` – two-layer architecture (real document graph + hypothesis graph)
- `unified_fuzzing_engine.py` – unified entry point
- `advanced_fuzzing_techniques.py` – grammar/mutational/random-walk/differential/authority fuzzers
- `public_index_builder.py` – public FAISS index with multi-granular chunks + metadata

The main CLI is `moyo-probe` (`cli.py`: `fuzz`, `search`, `analyze`,
`test-llm`, `analyze-corpus`). The advanced and two-layer fuzzers have their
own standalone Click groups (`cli_advanced_fuzzing.py`,
`cli_two_layer_fuzzer.py`) invoked with `python -m`.

### Shared LLM layer (`moyo/llm/`)
Provider-agnostic `LLMClient` / `LLMSpec` (OpenAI, Anthropic, Ollama, custom
OpenAI-compatible, echo). Registry resolves the default LLM from `MOYO_LLM_*`
and retrieval LLMs from `config/retrieval_llms.json`.

### Red Team (`redteam/`)
Probes a *target* LLM for proprietary-information leakage, exposed as
`moyo-redteam` (`whitebox`, `blackbox`, `report`). White-box uses a known
secret inventory (`SecretStore`) to plan attacks; black-box does
hypothesis-driven blind probing. A separate helper LLM generates probes and
never sees the target's responses. See `docs/threat_model.md`.

### Desktop GUI (`gui/`)
`moyo/gui/app.py` is a PyQt5 app (`moyo-gui`) with tabs that orchestrate the
private/public/probe/fuzz flows above and a 2D FAISS visualization tab.

## Data Flows

### Private Data Ingestion
```
Input (text / file) → GUI Bridge → Chunking → Embeddings → Private FAISS Index
```

### Public Data Collection
```
Private centroids → Token-driven crawler → Source adapters → Parsers/enrichers → Public FAISS Index
```

### Naive-prompt exploration
```
Prompt → Local fuzzer seeds → Retrieval LLMs (parallel)
      → Compile / label / translate → Summary (default LLM) → exploration.md
```

### Barrier Analysis
```
Private Index + Public Index → Cosine distance → Sobolev norms → Risk assessment
```

### Iterative LLM Enhancement
```
Closest matches → Text fuzzing → LLM queries → Semantic search → Refined results
```

## Key Technologies
- Python 3.10+, Click, PyQt5
- FAISS (CPU and GPU), sentence-transformers
- OpenAI / Anthropic / xAI / Gemini / Qwen / Kimi / OpenRouter (OpenAI-compatible) + local Ollama
- Pydantic settings (`MOYO_*`), project `config/` + `.env`
- Prometheus for metrics (`metrics.py`, `metrics_server.py`, `cli_metrics.py`)

## Development

```bash
# Install from the repo root (shared_utils is vendored and included automatically)
pip install -e .

# Run tests
python -m pytest tests/

# Code quality
flake8 moyo/
black moyo/
```

## Configuration

```bash
# LLM API keys (required for barrier probing with LLMs)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Embedding model
export MOYO_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

Configuration is defined in `moyo/config/settings.py` (Pydantic). Settings are
read from defaults, a `.env` file, and `MOYO_*` environment variables (see
`.env.example`). See `docs/configuration_and_monitoring_summary.md` for the
full list of options and Prometheus monitoring.
