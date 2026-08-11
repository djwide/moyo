# Configuration and Monitoring Implementation Summary

## Overview

This document summarizes the implementation of centralized configuration, structured logging, and Prometheus metrics for the moyo project.

## What Was Implemented

### 1. **Centralized Configuration System**

**Files Created/Modified:**
- `moyo/config/settings.py` - Comprehensive Pydantic configuration classes
- `.env.example` - Documented sample of the commonly-overridden settings
- Monitoring dependencies are declared as the `[monitoring]` extra in `pyproject.toml`

**Key Features:**
- **Environment Variable Support**: All settings are set via defaults, a `.env` file, and `MOYO_*` environment variables, including nested prefixes like `MOYO_LOG_`, `MOYO_EMBEDDING_`, `MOYO_FAISS_`, `MOYO_LLM_`.
- **Structured Configuration**: Type-safe configuration with validation using Pydantic
- **Component-Specific Settings**: Separate configuration classes for logging, Prometheus, pipeline, embedding, FAISS, and LLM settings

> **Note:** configuration is environment-based only. An earlier YAML config
> file (`config.yaml`) and its (never-wired) loader were removed; copy
> `.env.example` to `.env` to persist overrides.

**Usage:**
```python
from moyo.config.settings import get_settings

settings = get_settings()
print(f"Environment: {settings.environment}")
print(f"Embedding model: {settings.embedding.model_name}")
```

### 2. **Structured Logging System**

**Files Created:**
- `moyo/logging.py` - Comprehensive structured logging implementation

**Key Features:**
- **JSON Structured Logging**: Machine-readable log format with context
- **Context Management**: Automatic operation timing and context tracking
- **Configurable Output**: Console and file logging with rotation
- **Environment Integration**: Automatic environment and component tagging
- **Operation Context**: Built-in timing and status tracking for operations

**Usage:**
```python
from moyo.logging import get_logger

logger = get_logger("my-component")
logger.info("Processing document", document_id="123", status="success")

with logger.operation_context("data_processing", batch_size=1000):
    # Operations are automatically timed and logged
    process_data()
```

### 3. **Prometheus Metrics System**

**Files Created:**
- `moyo/metrics.py` - Comprehensive metrics registry
- `moyo/metrics_server.py` - HTTP server for metrics exposure
- `moyo/cli_metrics.py` - CLI commands for metrics management

**Key Features:**
- **Pipeline Metrics**: Timing and throughput for all pipeline operations
- **Document Processing**: Success rates and timing by document type
- **FAISS Operations**: Query performance and index statistics
- **LLM Integration**: Request timing and success rates for hypothesis generation
- **Fuzzing Campaigns**: Comprehensive metrics for barrier analysis
- **System Metrics**: Resource utilization tracking
- **HTTP Server**: Built-in metrics endpoint for Prometheus scraping

**Usage:**
```python
from moyo.metrics import get_metrics_registry, pipeline_timer

metrics = get_metrics_registry()

# Manual metrics recording
metrics.record_document_processed("corpus_builder", "pdf", "success")

# Automatic timing with decorator
@pipeline_timer("document_processing", "corpus_builder")
def process_documents():
    # Function is automatically timed
    pass
```

### 4. **CLI Integration**

**Files Modified:**
- `moyo/cli.py` - Registers the `metrics` command group (optional import)

**New Commands:**
```bash
# Metrics management
moyo metrics start          # Start metrics server
moyo metrics stop           # Stop metrics server
moyo metrics show           # Display current metrics
moyo metrics health         # Check server health
moyo metrics export         # Export metrics to file

# Configuration
moyo --verbose              # Enable verbose logging
moyo --debug                # Enable debug logging
```

### 5. **Example and Documentation**

**Files Created:**
- `examples/config_and_monitoring_example.py` - Comprehensive usage examples
- `docs/configuration_and_monitoring_summary.md` - This summary

## Configuration Options

### Environment Variables

All configuration can be set via environment variables:

```bash
# Core settings
export MOYO_ENVIRONMENT=production
export MOYO_DEBUG=true

# Logging
export MOYO_LOG_LEVEL=DEBUG
export MOYO_LOG_STRUCTURED=true
export MOYO_LOG_FILE_PATH=logs/moyo.log

# Prometheus
export MOYO_PROMETHEUS_ENABLED=true
export MOYO_PROMETHEUS_PORT=8000

# Pipeline
export MOYO_PIPELINE_BATCH_SIZE=2000
export MOYO_PIPELINE_MAX_WORKERS=8

# Embedding
export MOYO_EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
export MOYO_EMBEDDING_DEVICE=auto   # or cuda | cpu

# FAISS
export MOYO_FAISS_INDEX_TYPE=IVF100
export MOYO_FAISS_DIMENSION=768

# Default LLM (hot-swappable project-wide)
export MOYO_LLM_PROVIDER=openai     # openai | anthropic | ollama | custom | echo
export MOYO_LLM_MODEL=gpt-4o
export MOYO_LLM_API_KEY=your-api-key
# export MOYO_LLM_BASE_URL=http://127.0.0.1:11434   # ollama / custom
```

See [`docs/embeddings.md`](embeddings.md) for embedding model tier recommendations
(MiniLM → MPNet/BGE → multilingual → OpenAI) and GPU setup.

### Persisting configuration with `.env`

For persistent overrides, copy `.env.example` to `.env` in the working
directory. Values there are loaded automatically (environment variables still
take precedence):

```bash
MOYO_ENVIRONMENT=production
MOYO_LOG_LEVEL=INFO
MOYO_PROMETHEUS_PORT=8000
MOYO_EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
MOYO_EMBEDDING_DEVICE=cuda
```

API keys for hosted LLM providers (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`XAI_API_KEY`, `GEMINI_API_KEY`, `DASHSCOPE_API_KEY`, `MOONSHOT_API_KEY`,
`PERPLEXITY_API_KEY`, `OPENROUTER_API_KEY`, …) also live in `.env`. See
`.env.example` for the full list. The LLM layer loads `.env` into the process environment so those keys
are available to `config/retrieval_llms.json` (`$VAR` references).

### Default LLM vs retrieval LLMs vs local fuzzer

Three related configs for `moyo-gather explore`:

| Role | Where | Used for |
|------|--------|----------|
| **Local fuzzer** | Ollama (`llama3.1:8b` @ `127.0.0.1:11434`) | Seed rewording + translating foreign answers while compiling the report |
| **Summary LLM** | Ollama via `MOYO_SUMMARY_*` (default `llama3.1:8b`, `num_ctx=32768`) | Narrative summary + claims brief (`summary.md`); prefers points of precision |
| **Deliverable LLM** | Grok / xAI via `MOYO_DELIVERABLE_*` (default `grok-4.5`, key `XAI_API_KEY`) | Formal `deliverable.md` (exposure, evidence graph, findings, mitigation) |
| **Default LLM** | `MOYO_LLM_*` in `.env` | Other moyo paths; explore retrieval override via CLI `--provider` |
| **Retrieval LLMs** | `config/retrieval_llms.json` (or `MOYO_RETRIEVAL_LLMS`) | Fan-out: each seed is sent to every listed model |

Ollama’s default context window is typically **2048–4096 tokens** even when a
model (e.g. Llama 3.1) supports up to **128k**. Summarisation raises this with
`MOYO_SUMMARY_NUM_CTX` (default `32768`); larger values use more RAM/VRAM.
Copy `config/retrieval_llms.example.json` to `config/retrieval_llms.json` and
edit entries to match the providers you have keys for. Explore prints a
preflight `name / status / reason` table at scan start; providers without a key
(or an unreachable Ollama) fail that source only and do not stop the run.

**Fuzz modes:** `basic` (paraphrase / translate / summarize) or
`multilingual` (paraphrase / abstract / summarize per language; defaults
Spanish, French, Mandarin Chinese). ``typo`` is optional a la carte. See
[`docs/crawler.md`](crawler.md).

**Embedding model selection** persists in `config/model_config.json` (not under
`data/`). The shared synonym JSON map has been removed; local transformers use
built-in synonym tables.

### Local Ollama setup

Ollama is the local generative LLM path (`provider: ollama`). It needs **no API
key**. moyo talks to it over HTTP at `http://127.0.0.1:11434` by default.

**1. Install inside WSL (recommended if moyo runs in WSL)**

Windows Ollama is not reachable from WSL's `127.0.0.1` under default NAT
networking. Install the Linux binary in WSL instead:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

**2. Keep the server running**

Something must listen on port `11434`. Manual start:

```bash
ollama serve
# or backgrounded:
ollama serve > /tmp/ollama.log 2>&1 &
```

Verify:

```bash
curl -sf http://127.0.0.1:11434/api/version
ollama list
```

**3. Auto-start on shell login (no systemd required)**

If WSL does not support `[boot] systemd=true` (older WSL prints
`Unknown key 'boot.systemd'`), add this to `~/.bashrc` so a new shell starts
Ollama when it is not already up:

```bash
if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  nohup ollama serve > /tmp/ollama.log 2>&1 &
fi
```

Open a new terminal (or `source ~/.bashrc`), then re-check `curl` / `ollama list`.

**4. Wire it into moyo**

`config/retrieval_llms.json` should include an Ollama entry (already present in
the example):

```json
{
  "provider": "ollama",
  "model": "llama3.1:8b",
  "base_url": "http://localhost:11434",
  "label": "Local Ollama (llama3.1:8b)"
}
```

Change `model` to any tag from `ollama list`. To make Ollama the **default**
LLM (rewording + summary), set in `.env`:

```bash
MOYO_LLM_PROVIDER=ollama
MOYO_LLM_MODEL=llama3.1:8b
MOYO_LLM_BASE_URL=http://127.0.0.1:11434
```

**5. Smoke test**

```bash
# Retrieval LLM preflight only (config/retrieval_llms.json; no explore)
moyo-gather check-llms

# Or through the fuzzer / default-LLM path
moyo-probe test-llm --llm-provider ollama --model llama3.1:8b
```

If the API is down, explore still runs other retrieval LLMs; preflight marks
Ollama `fail`, and that source’s sections in `exploration.md` record the error.

## Metrics Available

### Pipeline Metrics
- `moyo_pipeline_duration_seconds` - Operation timing
- `moyo_pipeline_operations_total` - Operation counts

### Document Processing
- `moyo_documents_processed_total` - Document processing counts
- `moyo_document_processing_duration_seconds` - Processing timing

### FAISS Operations
- `moyo_faiss_queries_total` - Query counts
- `moyo_faiss_query_duration_seconds` - Query timing
- `moyo_faiss_index_size` - Index sizes

### LLM Integration
- `moyo_llm_requests_total` - Request counts
- `moyo_llm_request_duration_seconds` - Request timing

### Fuzzing Campaigns
- `moyo_fuzzing_campaigns_total` - Campaign counts
- `moyo_hypotheses_generated_total` - Hypothesis generation
- `moyo_public_documents_discovered_total` - Document discovery

### System Metrics
- `moyo_memory_usage_bytes` - Memory usage
- `moyo_cpu_usage_percent` - CPU usage

## Getting Started

### 1. Install Dependencies
```bash
pip install -e ".[monitoring]"
```

### 2. Run the Example
```bash
cd moyo
python examples/config_and_monitoring_example.py
```

### 3. Start Metrics Server
```bash
moyo metrics start
```

### 4. View Metrics
```bash
# In another terminal
moyo metrics show
# Or visit http://localhost:8000/metrics
```

### 5. Use in Your Code
```python
from moyo.config.settings import get_settings
from moyo.logging import get_logger
from moyo.metrics import get_metrics_registry

# Load configuration
settings = get_settings()

# Setup logging
logger = get_logger("my-component")

# Record metrics
metrics = get_metrics_registry()
metrics.record_document_processed("my-component", "pdf", "success")
```

## Value for SenTe Project

This implementation provides significant value for the SenTe project:

1. **Performance Optimization**: Data-driven optimization of barrier analysis pipelines
2. **Cost Management**: Track and optimize LLM API usage costs
3. **Quality Assurance**: Monitor the effectiveness of barrier detection
4. **Operational Excellence**: Proactive problem detection and resolution
5. **Scalability**: Infrastructure monitoring for large-scale processing

The monitoring system is essential for maintaining trust in barrier analysis results and ensuring reliable operation at scale.
