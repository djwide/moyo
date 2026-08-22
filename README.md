# moyo

Experimental tooling for corpus mapping and barrier probing. moyo helps organizations understand information barriers between their private data and public sources.

## Overview

moyo provides:

- **Private side processing**: Ingest local data and build a FAISS-backed vector corpus
- **Public side analysis**: Gather and index open-source information
- **Barrier assessment**: Identify how closely public information approaches private data, using cosine distance and LLM-assisted fuzzing
- **Corpus management**: Build, query, and maintain knowledge corpora

## Architecture

```
.                               ← repo root (git clone goes here)
├── moyo/                       ← Python package
│   ├── cli.py                  # Main CLI (moyo setup/info/version)
│   ├── config/                 # Configuration management
│   ├── privateside/
│   │   ├── datainput/          # Data input and GUI bridge
│   │   └── mapcorpus/          # Corpus building and centroids
│   │       ├── builder.py      # FAISS indexer
│   │       ├── centroids.py    # Topic token extraction
│   │       └── schema.py
│   ├── publicside/
│   │   ├── gatherpublicsources/  # Crawler and source adapters
│   │   │   ├── crawler.py      # crawl() and crawl_with_tokens()
│   │   │   ├── sources/        # Patent, git, arXiv, web, etc.
│   │   │   ├── parsers/
│   │   │   └── enrichers/
│   │   └── barrierprobe/       # Barrier analysis
│   │       ├── barrier_analyzer.py
│   │       ├── llm_fuzzer.py
│   │       ├── iterative_llm_search.py
│   │       └── two_layer_fuzzer.py
│   ├── redteam/                # LLM red-teaming (moyo-redteam)
│   │   ├── whitebox/           # known-secret probing + private-index refinement
│   │   ├── blackbox/           # blind hypothesis-driven probing
│   │   └── probe_paths.py      # loads probe_paths/ seed lists
│   ├── gui/                    # PyQt5 desktop GUI (moyo-gui)
│   ├── config/                 # Pydantic settings
│   ├── metrics.py, metrics_server.py, cli_metrics.py   # Prometheus monitoring
│   └── logging.py              # Structured logging
├── shared_utils/               # Vendored utilities (embeddings, FAISS, ingest)
├── probe_paths/                # Target-customer secret lists for blind probing
├── examples/
└── docs/
```

## Installation

```bash
# From the repo root — shared_utils is vendored and installed automatically
pip install -e .

# Optional extras
pip install -e ".[monitoring]"   # psutil, requests
pip install -e ".[gui]"          # PyQt5, matplotlib, scikit-learn
```

## Quick Start

```bash
# Initial setup
moyo setup
moyo info

# Process private data
moyo-datainput process "Your confidential text here"
moyo-datainput process --file document.txt

# Build a corpus
moyo-corpus build /path/to/documents --dedupe --normalize

# Probe information barriers
moyo-probe search -c corpus_dir -q "security incident" -k 10
moyo-probe fuzz -p "data breach" -t "confidential information" -i corpus.index
```

## CLI Reference

### `moyo`
```bash
moyo --help
moyo setup       # Create data directory structure
moyo info        # System information
moyo version     # Version
```

### `moyo-datainput`
```bash
moyo-datainput process --help
moyo-datainput process --file document.txt
moyo-datainput process --files f1.txt f2.txt --chunk-size 256 --model all-MiniLM-L6-v2
```

### `moyo-corpus`
```bash
moyo-corpus build /path/to/documents
moyo-corpus build --chunk-size 512 --dedupe --normalize --save-chunks
moyo-corpus build-text "Text 1" "Text 2"
```

### `moyo-gather`
```bash
moyo-gather crawl --topic "artificial intelligence safety"
moyo-gather crawl-tokens --tokens "neural networks,transformers,LLM"
moyo-gather crawl --topic "machine learning" --output data/public_sources/ml
```

### `moyo-probe`
```bash
moyo-probe search -c corpus_dir -q "query" -k 10
moyo-probe fuzz -p "phrase" -t "target" -i corpus.index \
  --llm-provider openai --model gpt-4o --max-iterations 10
moyo-probe analyze -p indexes/public -r indexes/private \
  --output-json report.json --output-html report.html
moyo-probe test-llm --llm-provider openai --model gpt-4o

# Local LLM via Ollama (no API key)
moyo-probe fuzz -p "phrase" -t "target" -i corpus.index \
  --llm-provider ollama --model llama3.1:8b
```

`--llm-provider` accepts `openai`, `anthropic`, `ollama`, `custom`, and `local`.
Use `custom` with `--base-url` to point at any OpenAI-compatible server (vLLM,
LM Studio, Together, Groq, OpenRouter, DeepSeek, llama.cpp server, ...):

```bash
moyo-probe fuzz -p "phrase" -t "target" -i corpus.index \
  --llm-provider custom --base-url http://localhost:8000/v1 --model my-model
```

### `moyo-redteam`
```bash
# White-box: probe a target LLM against a known secret inventory
moyo-redteam whitebox --secrets-file secrets.json --target-provider openai --target-model gpt-4o

# White-box with private-corpus grounding + iterative refinement:
# probes are grounded in the private index's mapcorpus centroids and refined
# each round toward the protected passages a response comes closest to.
moyo-redteam whitebox --secrets-file secrets.json \
  --target-provider openai --target-model gpt-4o \
  --private-index indexes/private --refine-rounds 3

# Black-box: hypothesis-driven blind probing
moyo-redteam blackbox --domain "acme corp" --rounds 3 --hypothesis-source llm

# Black-box seeded from a bundled probe path (list of target-valuable secrets):
moyo-redteam blackbox --domain "state campaign" --rounds 8 \
  --probe-path political_opposition_research

# Feed black-box hypotheses into moyo-gather explore (explore CLI unchanged):
moyo-redteam blackbox-explore -d "state campaign" \
  --probe-path political_opposition_research --prompts-only -f /tmp/bb.txt
moyo-gather explore -f /tmp/bb.txt

moyo-redteam report --input results.json --format text
```

Probe paths live in [`probe_paths/`](probe_paths/README.md) — one subdirectory per
target customer, each a `.txt` list of secrets valuable to know (e.g.
`political_opposition_research`, `pharmaceutical_rd`, `tech_company_ma`). They seed
the black-box hypothesis engine.
See [docs/threat_model.md](docs/threat_model.md) for the red-team threat model.

### `moyo-gui`
```bash
moyo-gui                       # launch the PyQt5 desktop app
```

## Python API

```python
from moyo.privateside.mapcorpus import CorpusBuilder
from moyo.publicside.barrierprobe import BarrierAnalyzer

# Build corpus from private data
builder = CorpusBuilder()
builder.add_text("Your private text here")
result = builder.build_index()   # CorpusBuildResult

# Analyse barriers between a public and private index.
# Index paths are supplied via BarrierProbeConfig; analyze_barriers(top_k=...)
# loads them and returns a BarrierProbeResult.
from moyo.publicside.barrierprobe import BarrierProbeConfig

analyzer = BarrierAnalyzer(BarrierProbeConfig(
    public_index_path="indexes/public",
    private_index_path="indexes/private",
))
results = analyzer.analyze_barriers(top_k=10)
print(results.breach_count, results.high_risk_breaches)
```

### Token-driven crawling

```python
from pathlib import Path
from moyo.privateside.mapcorpus import tokens_for_corpus
from moyo.publicside.gatherpublicsources.crawler import PublicSourcesCrawler

centroids, topic_tokens, labels, texts = tokens_for_corpus(
    Path('data/private/corpus.txt'), top_k=8
)
tokens = [t for cluster in topic_tokens for t in cluster][:25]

crawler = PublicSourcesCrawler()
result = crawler.crawl_with_tokens(tokens)
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="..."      # Required for LLM fuzzing / OpenAI embeddings
export ANTHROPIC_API_KEY="..."   # Alternative LLM provider
export MOYO_EMBEDDING_MODEL_NAME="BAAI/bge-base-en-v1.5"
export MOYO_EMBEDDING_DEVICE="auto"   # auto | cuda | cpu
export MOYO_CHUNK_SIZE="512"
```

Embedding model tiers, GPU setup, and GUI options: [`docs/embeddings.md`](docs/embeddings.md).

### Data Directory Structure

After `moyo setup`:

```
.
├── indexes/
│   ├── private/    # Private corpus indexes
│   └── public/     # Public corpus indexes
├── data/
│   ├── private/    # Private data files
│   └── public/     # Public data files
└── logs/
```

## Dependencies

- `shared_utils` (vendored under `shared_utils/` in the repo root) – text processing, embeddings, FAISS
- `pydantic>=2` + `pydantic-settings` – schemas and validation
- `click` – CLI framework
- `faiss-cpu` (or `faiss-gpu`) – vector similarity search
- `openai >= 1.0.0` – LLM fuzzing
- `anthropic >= 0.7.0` – alternative LLM provider

## Development

```bash
# Install in editable mode with all extras
pip install -e ".[monitoring,gui]"

python -m pytest tests/
flake8 moyo/
black moyo/
```

## GUI

A PyQt5 desktop GUI ships inside the package at `moyo/gui/app.py`, exposed as
the `moyo-gui` console script:

```bash
pip install -e ".[gui]"
moyo-gui                       # preferred
# equivalently:
python -m moyo.gui.app
```

The GUI has tabs for private data input, private index creation, gathering
public sources, building the public corpus, barrier probing, LLM fuzzing
(local Ollama models and any custom OpenAI-compatible endpoint), and 2D index
visualization. See
[docs/gui.md](docs/gui.md) for details, including local LLM setup.

## Documentation

| File | Description |
|------|-------------|
| [docs/architecture.md](docs/architecture.md) | Component architecture and data flows |
| [docs/barrier_analysis_guide.md](docs/barrier_analysis_guide.md) | Using barrier analysis |
| [docs/crawler.md](docs/crawler.md) | Public sources crawler |
| [docs/iterative_llm_search_guide.md](docs/iterative_llm_search_guide.md) | Iterative LLM search |
| [docs/two_layer_fuzzing_architecture.md](docs/two_layer_fuzzing_architecture.md) | Two-layer fuzzing design |
| [docs/runbook.md](docs/runbook.md) | Operational runbook |
| [docs/gui.md](docs/gui.md) | Desktop GUI (`moyo-gui`) guide |
| [docs/gui_bridge_guide.md](docs/gui_bridge_guide.md) | GUI bridge / data-input API |
| [docs/threat_model.md](docs/threat_model.md) | Red-team threat model |
| [docs/configuration_and_monitoring_summary.md](docs/configuration_and_monitoring_summary.md) | Config and Prometheus metrics |
