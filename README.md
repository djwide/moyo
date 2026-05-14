# Moyo

Experimental tooling for corpus mapping and barrier probing. Moyo helps organisations understand information barriers between their private data and public sources.

## Overview

Moyo provides:

- **Private side processing**: Ingest local data and build a FAISS-backed vector corpus
- **Public side analysis**: Gather and index open-source information
- **Barrier assessment**: Identify how closely public information approaches private data, using cosine distance, Sobolev norms, and LLM-assisted fuzzing
- **Corpus management**: Build, query, and maintain knowledge corpora

## Architecture

```
moyo/
├── moyo/
│   ├── cli.py                    # Main CLI (moyo setup/info/version)
│   ├── config/                   # Configuration management
│   ├── interfaces/               # Common interfaces
│   ├── privateside/
│   │   ├── datainput/            # Data input and GUI bridge
│   │   └── mapcorpus/            # Corpus building and centroids
│   │       ├── builder.py        # FAISS indexer
│   │       ├── centroids.py      # Topic token extraction
│   │       └── schema.py
│   └── publicside/
│       ├── gatherpublicsources/  # Crawler and source adapters
│       │   ├── crawler.py        # crawl() and crawl_with_tokens()
│       │   ├── sources/          # Patent, git, arXiv, web, etc.
│       │   ├── parsers/
│       │   └── enrichers/
│       └── barrierprobe/         # Barrier analysis
│           ├── barrier_analyzer.py
│           ├── llm_fuzzer.py
│           ├── iterative_llm_search.py
│           └── two_layer_fuzzer.py
├── examples/
└── docs/
```

## Installation

```bash
# From the monorepo root
pip install -e shared_utils/
pip install -e moyo/
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

### `moyo-probe`
```bash
moyo-probe search -c corpus_dir -q "query" -k 10
moyo-probe fuzz -p "phrase" -t "target" -i corpus.index \
  --llm-provider openai --model gpt-4 --max-iterations 10
moyo-probe test-llm --llm-provider openai --model gpt-4
```

## Python API

```python
from moyo.privateside.mapcorpus import CorpusBuilder
from moyo.publicside.barrierprobe import BarrierAnalyzer

# Build corpus from private data
builder = CorpusBuilder()
builder.add_text("Your private text here")
index = builder.build_index()

# Analyse barriers
analyzer = BarrierAnalyzer()
results = analyzer.analyze_barriers(private_index=index)
```

### Token-driven crawling

```python
from pathlib import Path
from moyo.privateside.mapcorpus import tokens_for_corpus
from moyo.publicside.gatherpublicsources.crawler import PublicSourcesCrawler

centroids, topic_tokens, labels, texts = tokens_for_corpus(
    Path('moyo/data/private/corpus.txt'), top_k=8
)
tokens = [t for cluster in topic_tokens for t in cluster][:25]

crawler = PublicSourcesCrawler()
result = crawler.crawl_with_tokens(tokens)
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="..."      # Required for LLM fuzzing
export ANTHROPIC_API_KEY="..."   # Alternative LLM provider
export MOYO_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export MOYO_CHUNK_SIZE="512"
```

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

- `shared_utils` (vendored under `moyo/shared_utils/`) – text processing, embeddings, FAISS
- `pydantic` – schemas and validation
- `click` – CLI framework
- `faiss-cpu` (or `faiss-gpu`) – vector similarity search
- `openai >= 1.0.0` – LLM fuzzing
- `anthropic >= 0.7.0` – alternative LLM provider

## Development

```bash
cd moyo
pip install -e .
python -m pytest tests/
flake8 moyo/
black moyo/
```

## GUI

A PyQt5 desktop GUI is available in `moyoGUI/`:

```bash
cd moyo/moyoGUI
python run_moyo_gui.py
```

See [moyoGUI/README.md](moyoGUI/README.md) for details.

## Documentation

| File | Description |
|------|-------------|
| [docs/architecture.md](docs/architecture.md) | Component architecture and data flows |
| [docs/barrier_analysis_guide.md](docs/barrier_analysis_guide.md) | Using barrier analysis |
| [docs/crawler.md](docs/crawler.md) | Public sources crawler |
| [docs/iterative_llm_search_guide.md](docs/iterative_llm_search_guide.md) | Iterative LLM search |
| [docs/two_layer_fuzzing_architecture.md](docs/two_layer_fuzzing_architecture.md) | Two-layer fuzzing design |
| [docs/runbook.md](docs/runbook.md) | Operational runbook |
| [docs/configuration_and_monitoring_summary.md](docs/configuration_and_monitoring_summary.md) | Config and Prometheus metrics |
