# Moyo Architecture

## Overview

Moyo is an experimental tool for corpus mapping and information-barrier analysis. It sits alongside sente in the SenTe monorepo and shares utilities via `shared_utils`.

## Repository Structure

```
SenTe/
├── shared_utils/               # Shared utilities and ONNX models
│   └── shared_utils/
│       ├── embeddings.py, faiss_index.py, chunking.py, text_processing.py, ...
│       └── models/             # miniLM_fp32.onnx, miniLM_int8.onnx
├── moyo/                       # Moyo package
│   ├── moyo/
│   │   ├── cli.py              # Main CLI entry point
│   │   ├── config/             # Configuration management
│   │   ├── interfaces/         # Common interfaces
│   │   ├── privateside/
│   │   │   ├── datainput/      # GUI bridge and file/text input
│   │   │   └── mapcorpus/      # Corpus building and centroids
│   │   └── publicside/
│   │       ├── gatherpublicsources/  # Crawler orchestrator and adapters
│   │       └── barrierprobe/         # Barrier analysis and LLM search
│   ├── examples/
│   ├── docs/
│   └── shared_utils/           # Vendored shared utilities (embeddings, FAISS, ingest, etc.)
```

## Components

### Private Side (`privateside/`)

#### GUI Bridge (`datainput/gui_bridge.py`)
Receives data from GUI applications (text or files), validates and preprocesses it, then passes it to the corpus builder.

#### Corpus Builder (`mapcorpus/builder.py`)
- Text normalisation and deduplication
- Sentence-aware chunking with overlap
- Embedding generation (sentence-transformers)
- FAISS index creation and persistence

#### Centroids (`mapcorpus/centroids.py`)
Derives topic tokens from the private corpus for use as crawl seeds on the public side.

### Public Side (`publicside/`)

#### Crawler (`gatherpublicsources/crawler.py`)
Orchestrator with two modes:
- `crawl(topics)` – search by topic string
- `crawl_with_tokens(tokens)` – token-driven crawling seeded from private-corpus centroids

Source adapters: patents, press releases, git commits, conference talks, arXiv/PubMed, generic web search.

Parsers and enrichers handle HTML/PDF extraction, classification, and deduplication.

#### Barrier Probe (`barrierprobe/`)
Analyses information barriers between private and public FAISS indexes:
- `barrier_analyzer.py` – cosine distance and Sobolev norm analysis
- `llm_fuzzer.py` – LLM-assisted phrase fuzzing (OpenAI and Anthropic)
- `iterative_llm_search.py` – iterative refinement of closest matches
- `two_layer_fuzzer.py` – two-layer architecture (real document graph + hypothesis graph)
- `unified_fuzzing_engine.py` – unified entry point

## Data Flows

### Private Data Ingestion
```
Input (text / file) → GUI Bridge → Chunking → Embeddings → Private FAISS Index
```

### Public Data Collection
```
Private centroids → Token-driven crawler → Source adapters → Parsers/enrichers → Public FAISS Index
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
- Python 3.10+, Click
- FAISS (CPU and GPU), sentence-transformers
- OpenAI and Anthropic SDKs (for LLM fuzzing)
- Pydantic for schemas and configuration
- Prometheus for metrics (`cli_metrics.py`, `metrics_server.py`)

## Development

```bash
# Install from the monorepo root
pip install -e shared_utils/
pip install -e moyo/

# Run tests
python -m pytest moyo/tests/

# Code quality
flake8 moyo/moyo/
black moyo/moyo/
```

## Configuration

```bash
# LLM API keys (required for barrier probing with LLMs)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Embedding model
export MOYO_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

See `moyo/config/` for YAML-based configuration and `docs/configuration_and_monitoring_summary.md` for Prometheus monitoring details.
