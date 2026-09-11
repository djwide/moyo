# moyo Runbook

Operational guidance for running moyo components and managing corpus mapping and barrier probing systems.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Core Operations](#core-operations)
4. [Private Side Operations](#private-side-operations)
5. [Public Side Operations](#public-side-operations)
6. [Barrier Analysis](#barrier-analysis)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Emergency Procedures](#emergency-procedures)
10. [Security Considerations](#security-considerations)

## System Overview

moyo is an experimental tooling system for corpus mapping and barrier probing. It provides comprehensive tools for building knowledge corpora and assessing information barriers between private and public data sources.

### Key Components

- **Private Side**: Ingest local data and map into FAISS-backed corpus
- **Public Side**: Gather open-source information and probe barriers between corpora
- **Barrier Analysis**: Cosine pair ranking plus a JS / entropy / margin layer; LLM-assisted fuzzing to probe information boundaries
- **Corpus Management**: Build, maintain, and query knowledge corpora

### Architecture

```
moyo/
├── privateside/          # Private data processing
│   ├── datainput/       # Data ingestion and validation
│   └── mapcorpus/       # Corpus building and management
├── publicside/          # Public data analysis
│   ├── gatherpublicsources/  # Public source crawling
│   └── barrierprobe/    # Barrier analysis and probing
└── shared/              # Common utilities and interfaces
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- pip package manager
- Git (for development)
- At least 4GB RAM (8GB+ recommended)
- 10GB+ disk space for indexes and data

### Step 1: Install Dependencies

```bash
# Install moyo from the repo root
# (vendored shared_utils is installed automatically)
pip install -e .
```

### Step 2: Initial Setup

```bash
# Set up directory structure
moyo setup

# Verify installation
moyo info
```

### Step 3: Configuration

```bash
# Check current configuration
moyo info

# Persistent keys + default LLM (recommended)
cp .env.example .env
# Edit .env: OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, … and MOYO_LLM_*

# Explore fan-out targets
cp config/retrieval_llms.example.json config/retrieval_llms.json
# Edit entries for providers you have keys for
```

See [`docs/configuration_and_monitoring_summary.md`](configuration_and_monitoring_summary.md).

## Core Operations

### System Status Check

```bash
# Check system health
moyo info

# Verify all components are available
moyo version
```

### Directory Structure

After setup, the following structure should exist:

```
moyo/
├── indexes/             # legacy global FAISS (prefer projects/<slug>/indexes/)
│   ├── private/
│   └── public/
├── config/              # retrieval_llms.json, model_config.json, …
├── projects/            # per-engagement phrases + FAISS + public sources
├── .env                 # API keys + MOYO_LLM_* (from .env.example)
└── logs/
```

## Private Side Operations

### Data Input Processing

#### Process Text Directly

```bash
# Process a single text input
moyo-datainput process "Your confidential text here"

# Process with custom configuration
moyo-datainput process "Text content" \
  --chunk-size 256 \
  --chunk-overlap 25 \
  --model all-MiniLM-L6-v2 \
  --index-type flat
```

#### Process Files

```bash
# Process a single file
moyo-datainput process --file document.txt

# Process multiple files (repeat --files for each file)
moyo-datainput process --files file1.txt --files file2.txt --files file3.txt

# Name the index explicitly (otherwise it is named after the corpus/file)
moyo-datainput process --file document.txt --name my_corpus
```

Indexes are always written under `indexes/` and named after the corpus
(e.g. `indexes/private/<name>/<name>.faiss`). There is no shared `index.faiss`.
The CLI defaults to the most recently built index or lets you choose.

Chunking uses multi-granularity splitting (sections, sentences, list/bullet
items) so long prose does not dilute embeddings; see `shared_utils/chunking.py`.

#### Advanced Processing

```bash
# Process with JSON output for programmatic use
moyo-datainput process --file document.txt --json

# Process without saving index (for testing)
moyo-datainput process --file document.txt --no-save

# Verbose processing with debug information
moyo-datainput -v --debug process --file document.txt
```

### Corpus Building

#### Build from GUI Bridge Data

```python
from moyo.privateside.mapcorpus.builder import build_corpus_from_gui_bridge

# Build corpus from GUI bridge processing results
result = build_corpus_from_gui_bridge(gui_bridge_data, config)
```

#### Build from Files

```python
from moyo.privateside.mapcorpus.builder import build_corpus_from_files

# Build corpus from file paths
file_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
result = build_corpus_from_files(file_paths, config)
```

#### Build from Texts

```python
from moyo.privateside.mapcorpus.builder import build_corpus_from_texts

# Build corpus from text inputs
texts = ["Text 1", "Text 2", "Text 3"]
result = build_corpus_from_texts(texts, config)
```

### Corpus Management

#### Normalize Corpus

```python
from moyo.privateside.mapcorpus.builder import CorpusBuilder

builder = CorpusBuilder()
# ... add data ...

# Apply normalization
builder.normalize_corpus()

# Or use the new normalize_chunks method
normalized_count = builder.normalize_chunks()
```

#### Deduplicate Corpus

```python
# Remove duplicates
duplicates_removed = builder.deduplicate_corpus()
print(f"Removed {duplicates_removed} duplicates")
```

#### Build Index

```python
# Build FAISS index
result = builder.build_index()

if result.success:
    print(f"Index built successfully: {result.vectors_created} vectors")
    print(f"Processing time: {result.processing_time:.2f}s")
else:
    print(f"Build failed: {result.message}")
```

## Public Side Operations

### Naive-prompt exploration (multi-LLM)

`moyo-gather explore` rewords a plain-language prompt (local Ollama on the
desktop; OpenRouter Llama 3.1 8B Instruct on Cloud Run via
`OPENROUTER_API_KEY`), fans out each seed to every retrieval LLM in
`config/retrieval_llms.json`, compiles/labels/translates responses, then
writes `exploration.md` only.

```bash
# basic (default): English seeds; default strategies paraphrase/translate/summarize
moyo-gather explore --prompt "What is the recipe for Coca-Cola?" --fuzz-mode basic
# Multiple prompts: repeat --prompt and/or use --prompts-file (one per line)
moyo-gather explore -p "What is the recipe for Coca-Cola?" -p "Who killed JFK?"

# A la carte strategies (-S overrides the mode's default set; mode still
# controls language fan-out). Include typo or shuffle explicitly when wanted:
moyo-gather explore -p "..." --fuzz-mode basic -S paraphrase -S summarize -S typo
moyo-gather explore -p "..." --fuzz-mode basic -S paraphrase -S shuffle

# multilingual: EN + ES / FR / Mandarin Chinese (extend with -l)
moyo-gather explore --prompt "..." --fuzz-mode multilingual --seeds 3 -l German

# Preflight prints name/status/reason for each retrieval LLM at scan start
```

Fuzz modes: **basic** (default) | **multilingual** (legacy aliases `full` /
`full-multilingual` still normalize). Strategies are a la carte via repeatable
``--strategy`` / ``-S`` (`paraphrase`, `translate`, `summarize`, `typo`,
`abstract`, `shuffle`). Mode defaults omit ``typo`` and ``shuffle``; the GUI
shows the same default
(`basic`) and strategy checkboxes pre-checked to the mode’s set.

**Claim-friendly explore (for cheaper report extraction):** ask for dated /
numbered public-record facts, not “where would I look?” meta-prompts; prefer
``--fuzz-mode basic`` unless you need multilingual coverage; drop low-value
strategies with ``-S``; if answers truncate mid-bullet, raise retrieval
``max_tokens`` on the LLM spec. The report extractor skips refusals / tiny
stubs and can limit languages via ``reports/config.yaml`` ``chunk.*``.

Rewording, foreign-response translation, clustering, and claims/narrative
summary synthesis use a cheap utility LLM: Ollama `llama3.1:8b` on the desktop,
or OpenRouter `meta-llama/llama-3.1-8b-instruct` on Cloud Run (override with
`MOYO_UTILITY_*`; Moonshot Kimi is the fallback if `OPENROUTER_API_KEY` is
missing). Retrieval fan-out still uses `config/retrieval_llms.json`. ``--workers`` caps concurrent
retrieval *and* foreign-response translation (default: one per configured
retrieval LLM). Full pipeline details: [`docs/crawler.md`](crawler.md).

### Public source crawling

Use the `moyo-gather` CLI or the `PublicSourcesCrawler` Python API:

```bash
# Crawl by topic string
moyo-gather crawl --topic "artificial intelligence"

# Token-driven crawl (comma-separated list)
moyo-gather crawl-tokens --tokens "neural networks,transformers,LLM"

# Save results to a custom output directory
moyo-gather crawl --topic "machine learning" --output results/ml_sources
```

Or call the Python API directly:

```python
from moyo.publicside.gatherpublicsources.crawler import PublicSourcesCrawler

crawler = PublicSourcesCrawler()

# Crawl by topic string
results = crawler.crawl("artificial intelligence")

# Token-driven crawl seeded from private-corpus centroids
results = crawler.crawl_with_tokens(tokens)
```

### Public Index Building

```python
from moyo.publicside.barrierprobe.public_index_builder import PublicIndexBuilder

# Create builder
builder = PublicIndexBuilder(config)

# Add sources
sources_processed = builder.add_sources(sources)

# Apply processing
if config.normalization_enabled:
    normalized_count = builder.normalize_chunks()

if config.deduplication_enabled:
    duplicates_removed = builder.deduplicate_chunks()

# Build index
result = builder.build_index("Public AI Index", "AI-related public sources")
```

## Barrier Analysis

Public and private indexes must share chunk size, overlap, and embedding model.
`moyo-probe analyze` ranks pairs by cosine nearest-neighbor distance, then adds
a distribution layer on the same matrix. See
[`docs/barrier_analysis_guide.md`](barrier_analysis_guide.md) for interpretation
detail.

| Level | Question | What to read |
| --- | --- | --- |
| Pair | Which public passage is closest to this private chunk? | Cosine NN distance; Pairwise Exposure |
| Neighborhood | Is that match specific, or generic topic overlap? | Top-1/top-2 margin; normalized entropy over `k=20` public neighbors; Concentrated Matches |
| Corpus | How much semantic territory is shared? | **Semantic Separation** (JS distance over joint cluster occupancy) |

Headline on CLI, GUI, JSON, and HTML:

```
Semantic Separation: 0.63
Pairwise Exposure: High
Concentrated Matches: 7
```

High Semantic Separation is **not** barrier integrity. One leaked fact can
barely move global occupancy — use Pairwise Exposure and Concentrated Matches
for that. Directional KL (private→public / public→private) is a diagnostic in
`distribution_diagnostics` only; it is not part of the headline score.

Pairwise Exposure bands (closest private→public NN): High `≤0.1`, Medium
`≤0.3`, Low `≤0.5`, else None. A match is concentrated when `d₁ ≤ 0.30` and
either the top-1/top-2 margin is `≥0.08` or normalized top-k entropy is
`≤0.70`.

### Calibrate the cosine-distance cutoff

Absolute distances do not transfer across embedding models. Re-run after any
index rebuild that changes the model.

```bash
# Profiles: strict = closest 5%; balanced = 10%; recall = 25%
moyo-probe calibrate \
  --public-index indexes/public \
  --private-index indexes/private \
  --profile balanced \
  --output cache/barrierprobe/calibration.json
```

Use the recommended value as `--similarity-threshold` on `analyze` (that flag
is a cosine-**distance** cutoff: smaller = closer). The default `0.8` is
intentionally loose.

### Analyze public vs private indexes

```bash
moyo-probe analyze \
  --public-index indexes/public \
  --private-index indexes/private \
  --similarity-threshold 0.8 \
  --top-k 10 \
  --neighborhood-k 20

# Write JSON / HTML (includes margin, entropy, concentrated per row)
moyo-probe analyze \
  -p indexes/public \
  -r indexes/private \
  --similarity-threshold 0.7 \
  --top-k 20 \
  --neighborhood-k 20 \
  --llm-top-k 5 \
  --output-json cache/barrierprobe/report.json \
  --output-html cache/barrierprobe/report.html
```

`--neighborhood-k` is the public-neighbor window for margin and entropy
(default 20). Do not compute entropy over the entire public corpus — corpus
size would change the score. `analyze` still runs a round of iterative LLM
refinement on the suspicious pairs (`--llm-top-k`).

GUI: Barrier Probe tab → **Calibrate Threshold**, then **Run Barrier Analysis**.
The summary line is the headline trio plus breach counts; the table adds Margin,
Entropy, and Concentrated columns.

### LLM-Assisted Fuzzing

#### Fuzz Phrases

```bash
# Basic phrase fuzzing (repeat -p for each phrase)
moyo-probe fuzz \
  -p "data breach" -p "security incident" \
  -t "confidential information" \
  -i indexes/private/corpus.index \
  -o results/fuzzing_results.json

# Advanced fuzzing with custom parameters
moyo-probe fuzz \
  -f phrases.txt \
  -t "trade secrets" \
  -i indexes/private/corpus.index \
  --llm-provider openai \
  --model gpt-4o \
  --max-iterations 10 \
  --target-similarity 0.95 \
  --search-k 20

# Broader search: call each operator three times; keep 5 live nodes
moyo-probe fuzz \
  -p "data breach" \
  -t "confidential information" \
  -i indexes/private \
  --calls-per-strategy 3 \
  --keep-k 5

# Fuzz with a local LLM via Ollama (no API key required)
moyo-probe fuzz \
  -p "data breach" \
  -t "confidential information" \
  -i indexes/private \
  --llm-provider ollama \
  --model llama3.1:8b
```

Default call plan is paraphrase once, plus translate once each into
Spanish, Chinese, French, and Japanese. Override with `-S` / `--strategy`,
`-l` / `--language`, `--calls-per-strategy`, and `--keep-k`. After every
round the orchestrator prunes answers to the live node set.

`--llm-provider` accepts `openai`, `anthropic`, `ollama` (local Ollama
server), and `local` (embedding-only synonym transformer, no LLM). Default
models per provider: OpenAI `gpt-4o`, Anthropic `claude-sonnet-4-6`, Ollama
`llama3.1:8b`. See the GUI README for local LLM (Ollama) setup.

#### Test LLM Configuration

```bash
# Test LLM connectivity and configuration
moyo-probe test-llm \
  --llm-provider openai \
  --model gpt-4o \
  --api-key $OPENAI_API_KEY

# Test a local Ollama model
moyo-probe test-llm --llm-provider ollama --model llama3.1:8b
```

### Corpus Search

```bash
# Search private corpus
moyo-probe search \
  --corpus-dir indexes/private \
  --query "security vulnerability" \
  --k 10 \
  --similarity-threshold 0.8

# Search with custom parameters
moyo-probe search \
  --corpus-dir indexes/private \
  --query "confidential data" \
  --k 20 \
  --similarity-threshold 0.7
```

### Iterative LLM Search

```python
from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.schema import BarrierProbeConfig
from moyo.publicside.barrierprobe.iterative_llm_search import refine_suspicious_pairs

# refine_suspicious_pairs takes a BarrierProbeResult and the analyzer that
# produced it (indexes already loaded), and returns a refined result.
config = BarrierProbeConfig(
    public_index_path="indexes/public",
    private_index_path="indexes/private",
)
analyzer = BarrierAnalyzer(config)
result = analyzer.analyze_barriers(top_k=10)
refined = refine_suspicious_pairs(result, analyzer, top_k=5)
```

## Monitoring and Maintenance

### System Health Checks

#### Daily Checks

```bash
# Check system status
moyo info

# Verify index integrity
find indexes/ -name "*.index" -exec echo "Checking {}" \; -exec python -c "import faiss; faiss.read_index('{}')" \;

# Check disk space
df -h projects/ logs/
```

#### Weekly Checks

```bash
# Analyze corpus statistics
python -c "
from moyo.privateside.mapcorpus.builder import CorpusBuilder
builder = CorpusBuilder()
stats = builder._get_statistics()
print('Corpus Statistics:', stats)
"

# Check for outdated indexes
find indexes/ -name "*.index" -mtime +7 -exec echo "Old index: {}" \;
```

### Log Management

#### Log Rotation

```bash
# Set up log rotation (if not using system logrotate)
logrotate /etc/logrotate.d/moyo

# Manual log cleanup
find logs/ -name "*.log" -mtime +30 -delete
```

#### Log Analysis

```bash
# Check for errors in recent logs
grep -i error logs/*.log | tail -20

# Check processing statistics
grep "Processing complete" logs/*.log | tail -10
```

### Performance Monitoring

#### Index Performance

```bash
# Check index sizes
du -sh indexes/*/

# Monitor search performance
time moyo-probe search --corpus-dir indexes/private --query "test" --k 10
```

#### Memory Usage

```bash
# Monitor memory usage during processing
ps aux | grep moyo
```

## Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'moyo'` when running `moyo` or `moyo-datainput`

**Cause**: A console script under `~/.local/bin/` (often installed as root with `#!/usr/bin/python3`) is on your `PATH` before the scripts from your pyenv virtualenv. `python -c "import moyo"` may still work.

**Solution**:
```bash
cd /path/to/moyo
pyenv activate sente   # or your env where moyo is installed

# Reinstall into *this* Python and remove stale ~/.local/bin scripts
bash scripts/fix-cli-path.sh

# Verify — should NOT point at ~/.local/bin
which moyo-datainput
head -1 "$(which moyo-datainput)"

# Workaround (always uses the active python)
python -m moyo.privateside.datainput.cli process "secret text"
```

If removal fails with permission denied: `sudo rm -f ~/.local/bin/moyo*`, then run `python -m pip install -e .` again.

**Problem**: `ModuleNotFoundError: No module named 'shared_utils'`

**Solution**:
```bash
# shared_utils is vendored inside the moyo package — reinstall from the repo root
python -m pip install -e .

# Verify installation
python -c "import shared_utils; print('shared_utils available')"
```

#### FAISS Index Errors

**Problem**: `RuntimeError: Error loading FAISS index`

**Solution**:
```bash
# Check index file integrity
python -c "import faiss; faiss.read_index('path/to/index')"

# Rebuild index if corrupted (remove the old index directory first, then reprocess)
rm -rf indexes/private/corrupted_index
moyo-datainput process --file source.txt
```

#### Barrier analysis scores look wrong

**Problem**: Semantic Separation is high (corpora look separated) but you still see High Pairwise Exposure or Concentrated Matches.

**Cause**: JS occupancy is a corpus-level mix score. One leaked passage barely moves cluster histograms.

**Action**: Treat Pairwise Exposure and Concentrated Matches as the leak signal. Review those rows (distance + margin + entropy). Do not treat Semantic Separation as barrier integrity.

**Problem**: `analyze` / `calibrate` distances look unlike a previous run.

**Cause**: Public and private indexes were built with different embedding models, chunk size, or overlap — or you reused a MiniLM cutoff on MPNet/BGE.

**Action**: Rebuild both indexes with the same model and packing, then re-run `moyo-probe calibrate` before `analyze`.

#### LLM API Errors

**Problem**: `OpenAI API error: Invalid API key`

**Solution**:
```bash
# Verify API key
echo $OPENAI_API_KEY

# Test LLM connection
moyo-probe test-llm --llm-provider openai --model gpt-4o
```

#### Memory Issues

**Problem**: `MemoryError during processing`

**Solution**:
```bash
# Use a smaller chunk size
moyo-datainput process --file large_file.txt --chunk-size 256
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Enable debug mode
moyo-datainput --debug process --file document.txt

# Check debug logs
tail -f logs/debug.log
```

### Performance Issues

#### Slow Processing

```bash
# Check system resources
htop
iostat -x 1

# Optimize configuration
moyo-datainput process --file document.txt \
  --chunk-size 512 \
  --index-type flat
```

#### Large Index Sizes

```bash
# Use a compressed index type (ivf or hnsw)
moyo-datainput process --file document.txt --index-type ivf
moyo-datainput process --file document.txt --index-type hnsw
```

## Emergency Procedures

### System Recovery

#### Index Corruption

```bash
# Stop all processing
pkill -f moyo

# Backup corrupted index
cp indexes/private/corrupted.index indexes/private/corrupted.index.backup

# Rebuild from source
moyo-datainput process --file source.txt
```

#### Data Loss

```bash
# Check for backups
ls -la backups/

# Restore from backup
cp backups/index_20231201.index indexes/private/

# Verify restoration
moyo-probe search --corpus-dir indexes/private --query "test" --k 1
```

### Emergency Shutdown

```bash
# Graceful shutdown
pkill -TERM -f moyo

# Force shutdown if needed
pkill -KILL -f moyo

# Verify shutdown
ps aux | grep moyo
```

### Data Export

```bash
# Export corpus data
python -c "
from moyo.privateside.mapcorpus.builder import CorpusBuilder
builder = CorpusBuilder()
builder.load_corpus('indexes/private/corpus.index')
builder.export_corpus('emergency_export.json')
"
```

## Security Considerations

### Data Protection

#### Private Data Handling

- Always process private data on secure, isolated systems
- Use encrypted storage for sensitive corpora
- Implement access controls for index files
- Regularly audit data access logs

#### API Key Management

```bash
# Use environment variables for API keys
export OPENAI_API_KEY="your-secure-key"
export ANTHROPIC_API_KEY="your-secure-key"

# Never commit API keys to version control
echo "*.key" >> .gitignore
echo "api_keys.txt" >> .gitignore
```

#### Network Security

- Use VPN for remote access to processing systems
- Implement firewall rules for API access
- Monitor network traffic for unusual patterns
- Use HTTPS for all external API calls

### Access Control

#### User Permissions

```bash
# Set appropriate file permissions
chmod 600 indexes/private/*
chmod 644 indexes/public/*

# Use dedicated user for processing
sudo useradd -r -s /bin/false moyo
sudo chown -R moyo:moyo projects/ logs/
```

#### Audit Logging

```bash
# Enable audit logging
echo "moyo ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/moyo

# Monitor access
tail -f /var/log/auth.log | grep moyo
```

### Compliance

#### Data Retention

```bash
# Implement data retention policies
find projects/ -mtime +365 -exec echo "Old data: {}" \;

# Archive old data
tar -czf archive_$(date +%Y%m%d).tar.gz projects/old/
```

#### Privacy Impact Assessment

- Document all data processing activities
- Assess privacy risks for each corpus
- Implement data minimization principles
- Regular privacy impact reviews

## Appendices

### Configuration Reference

#### Processing Configuration

```python
from moyo.privateside.datainput.gui_bridge import ProcessingConfig

config = ProcessingConfig(
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    index_type="flat",
    save_index=True,
    output_dir="indexes/private"
)
```

#### Corpus Configuration

```python
from moyo.privateside.mapcorpus.schema import CorpusConfig

config = CorpusConfig(
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="all-MiniLM-L6-v2",
    index_type="flat",
    min_chunk_length=10,
    max_chunk_length=2000,
    deduplication_enabled=True,
    save_metadata=True
)
```

#### LLM Configuration

```python
from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzerConfig

config = LLMFuzzerConfig(
    llm_provider="ollama",   # openai | anthropic | ollama | custom | local
    model_name="llama3.1:8b",
    api_key=None,            # unused for ollama/local
    base_url="http://localhost:11434",
    whitebox_strategies=["paraphrase", "translate"],
    translate_languages=["Spanish", "Chinese", "French", "Japanese"],
    calls_per_strategy=1,    # 2 or 3 for a broader search
    keep_k=3,
    max_iterations=5,
    target_similarity=0.95,
    search_k=10,
    similarity_threshold=0.8,
)
```

Project-wide defaults:

- Embedding selection: `config/model_config.json` (via `shared_utils.model_config`)
- Default generative LLM: `MOYO_LLM_*` in `.env`
- Explore fan-out: `config/retrieval_llms.json`

### Command Reference

#### Main Commands

```bash
# System commands
moyo version          # Show version
moyo info            # System information
moyo setup           # Initial setup

# Data input commands
moyo-datainput process [text|--file|--files]  # Process data (indexes under indexes/)

# Barrier probe commands
moyo-probe calibrate # Cosine-distance cutoff from unlabeled NN distances
moyo-probe analyze   # Pair + neighborhood + corpus layer (JS / margin / entropy)
moyo-probe fuzz      # LLM fuzz orchestrator (--calls-per-strategy, --keep-k, -S, -l)
moyo-probe search    # Corpus search (text preview + source from metadata)
moyo-probe test-llm  # Test LLM configuration

# Public source gathering
moyo-gather crawl --topic <topic>
moyo-gather crawl-tokens --tokens <comma-separated-tokens>
moyo-gather explore --prompt "..." --fuzz-mode basic|multilingual
moyo-gather check-llms          # retrieval LLM preflight only (no explore)
moyo-gather summarize --dir <explore-dir>   # exploration.md -> summary.md only
moyo-gather deliverable --dir <explore-dir> # summary+exploration -> deliverable.md (Grok)

# Exploration processor (claims → report PDFs). Two products via --report:
#   snapshot (default) = Exposure Snapshot (one-page.pdf + report.pdf)
#   basis              = Basis Report (comprehensive)
#   both
# Remediations off by default; add --include-remediation to opt in.
# Finished local runs also upload to gs://senteguard-website-moyo-reports/
# reports/<storageFolder>/ (same layout as Cloud Run). Pass --no-upload to skip.
# See docs/exploration_processor.md for knobs and human QA steps.
python reports/build_report.py \
  --exploration projects/<slug>/public_sources/exploration.md \
  --run-id <slug> --report snapshot
# Comprehensive Basis Report:
python reports/build_report.py -e projects/<slug>/public_sources/exploration.md --report basis
# With mitigations / remediations:
python reports/build_report.py -e projects/<slug>/public_sources/exploration.md \
  --report both --include-remediation
# Offline / no Ollama: add --dry-run. GUI: moyo-gui → "Build Report" tab.
```

#### Common Options

```bash
--verbose, -v        # Enable verbose output
--debug             # Enable debug logging
--json              # Output results as JSON
--help              # Show help information
```

### Error Codes

| Code | Description | Action |
|------|-------------|--------|
| E001 | Module not found | Install missing dependencies |
| E002 | API key invalid | Check API key configuration |
| E003 | Index corrupted | Rebuild index from source |
| E004 | Memory error | Reduce chunk size |
| E005 | File not found | Check file paths and permissions |
| E006 | Network error | Check network connectivity |
| E007 | Permission denied | Check file permissions |
| E008 | Invalid configuration | Review configuration parameters |

### Support Contacts

- **Development Team**: moyo-dev@company.com
- **Operations Team**: moyo-ops@company.com
- **Security Team**: moyo-security@company.com
- **Emergency Contact**: moyo-emergency@company.com

---

**Last Updated**: August 2026  
**Version**: 1.3  
**Maintainer**: moyo Operations Team
