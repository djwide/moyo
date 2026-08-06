# moyo GUI

A PyQt5 desktop interface for the moyo barrier-probing pipeline. The GUI lives
inside the main package at `moyo/gui/app.py` and is launched via the `moyo-gui`
console script.

## Installation

```bash
pip install -e ".[gui]"
```

This installs `PyQt5`, `matplotlib`, and `scikit-learn`. Optional extras:

```bash
pip install umap-learn          # enables UMAP in the Visualize Indices tab
pip install scipy               # enables KDE density contours
```

## Running

Pick one of:

```bash
moyo-gui                          # console script (preferred)
python -m moyo.gui.app            # module entry point
```

## Tabs

The main window exposes 7 tabs, mirroring the moyo CLI commands.

| Tab                       | Backed by                                              |
| ------------------------- | ------------------------------------------------------ |
| Private Data Input        | `moyo.privateside.datainput`                           |
| Create Private Index      | `shared_utils.embeddings` + `shared_utils.faiss_index` |
| Gather Public Sources     | `moyo.publicside.gatherpublicsources.PublicSourcesCrawler` |
| Build Public Corpus       | `moyo.publicside.barrierprobe.PublicIndexBuilder`      |
| Barrier Probe             | `moyo.publicside.barrierprobe.BarrierAnalyzer`         |
| LLM Fuzzer                | `moyo.publicside.barrierprobe.LLMFuzzer`               |
| Visualize Indices         | matplotlib + sklearn (+ optional UMAP / scipy)         |

### Private Data Input

Chunk and prepare text from direct input, a single file, or a folder. Inputs
are written to JSON that the **Create Private Index** tab can read.

### Create Private Index

Embed prepared chunks and write a FAISS index (`flat`, `ivf`, or `hnsw`).

Choose an **embedding model** from the shared catalog (MiniLM, MPNet, BGE,
E5, multilingual, OpenAI) and a **device** (`Auto` / `CUDA` / `CPU`). See
[`docs/embeddings.md`](embeddings.md) for tier recommendations. Public and
private indices must use the same model.

### Gather Public Sources

Crawl public corpora (patents, press releases, git commits, conferences,
leaks) by topic *or* token list. Output JSON files can be fed directly into
the next tab. Parameters: max per source, max total, request delay,
source-type filters, output directory.

> **Note:** the bundled source adapters ship with placeholder endpoints and
> return no results until pointed at real services. See `docs/crawler.md`.

### Build Public Corpus

Loads `sources.json` files emitted by the previous tab and builds a FAISS
index. Exposes the full `IndexConfig`: embedding model + device, chunk
size/overlap, min/max chunk length, source-type / relevance / confidence
filters, deduplication and normalization toggles, and index type. Model
tiers are documented in [`docs/embeddings.md`](embeddings.md).

### Barrier Probe

Compares a public and a private index. For each private chunk it finds the
nearest public neighbour and assigns a risk level by cosine distance
(`≤0.1` high, `≤0.3` medium, otherwise low). Results land in a sortable,
colour-coded table, with one-click export to JSON or HTML. The probe must
be configured with matching chunk granularity and the same embedding model
on both sides.

### LLM Fuzzer

Iteratively rewrites input phrases toward a target concept and probes how
close they land to corpus content. Supports four providers:

- `local` — embedding-only synonym shuffler (no API key, no server)
- `ollama` — a **real local LLM** served by [Ollama](https://ollama.com);
  defaults to `llama3.1:8b`; set **Base URL** if not on
  `http://localhost:11434`. No API key required. See "Local LLM setup" below.
- `openai` — defaults to `gpt-4o`; reads `OPENAI_API_KEY`
- `anthropic` — defaults to `claude-sonnet-4-6`; reads `ANTHROPIC_API_KEY`

A **Test LLM Connection** button verifies the provider before a full run.
Parameters: max iterations, target similarity, search-K, similarity
threshold, temperature.

#### Local LLM setup (Ollama)

The `ollama` provider runs a genuine instruction-tuned LLM entirely on your
machine, offloading to an NVIDIA GPU via CUDA when available.

```bash
# 1. Install Ollama (Linux / WSL2):
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the server (leave running in its own terminal):
ollama serve

# 3. Pull a model that fits your GPU. For an 8 GB card, a 7–8B model at the
#    default Q4 quantisation is the sweet spot (~4.7 GB VRAM):
ollama pull llama3.1:8b        # general default
#   alternatives: qwen2.5:7b, mistral:7b, gemma2:9b (tighter fit)
#   smaller / faster: llama3.2:3b, phi3:mini

# 4. Verify from moyo:
moyo-probe test-llm --llm-provider ollama --model llama3.1:8b
```

In the GUI, pick **Provider = ollama**, leave **Base URL** blank for the
default endpoint, and click **Test LLM Connection**. On the CLI, pass
`--llm-provider ollama --model llama3.1:8b` (and `--base-url` if remote).

VRAM guidance for an 8 GB GPU:

| Model            | Quant | ~VRAM | Notes                              |
| ---------------- | ----- | ----- | ---------------------------------- |
| `llama3.2:3b`    | Q4    | ~3 GB | fastest, lightest                  |
| `phi3:mini`      | Q4    | ~3 GB | strong for its size                |
| `mistral:7b`     | Q4    | ~4.4 GB | solid all-rounder                |
| `llama3.1:8b`    | Q4    | ~4.7 GB | recommended default              |
| `qwen2.5:7b`     | Q4    | ~4.7 GB | good at instruction-following    |
| `gemma2:9b`      | Q4    | ~6.5 GB | tight fit; close other GPU apps  |

If a model is larger than free VRAM, Ollama still runs it by splitting layers
to CPU/RAM — slower but functional (you have 32 GB system RAM to spare).

### Visualize Indices

Loads a private + public FAISS index pair once, then lets you switch between
plot types without reloading.

**Dimensionality reduction** (used by Scatter and Density Contours):

- **MDS** — preserves pairwise distances. Best for showing relative
  semantic proximity. Slow on large corpora (>1k chunks).
- **PCA** — fast linear projection. Good for a first look at how spread
  out the embeddings are.
- **t-SNE** — emphasises local neighbourhoods. Good for spotting tight
  clusters but distances between clusters are *not* meaningful.
- **UMAP** *(requires `umap-learn`)* — preserves both local and global
  structure. Often the most readable plot for >500 points.

**Plot types**:

1. **Scatter (2D projection)** — colour-coded private/public scatter, with
   optional nearest-neighbour distance lines and KMeans cluster colouring.
2. **Distance histogram** — overlays the distribution of *all* public-vs-
   private distances against the per-private *nearest* distances; the
   left tail flags potential leakage.
3. **Nearest-neighbour CDF** — cumulative curve answering "what fraction
   of my private chunks have a public chunk within distance `x`?". The
   steeper the early rise, the higher the barrier-breach risk.
4. **Density contours (KDE)** *(requires `scipy`)* — KDE contours per
   corpus on the chosen projection, showing where each side concentrates.
5. **Cross-distance heatmap** — 2D binned heatmap of mean public–private
   distance across the projected space; hot regions highlight zones with
   strong cross-corpus proximity.
6. **Pairwise distance matrix** — full N×N cosine-distance matrix with a
   white line dividing the private vs public blocks. Useful as a sanity
   check and for reports.

Each plot can be exported to PNG/PDF/SVG via the **Export Current Plot**
button.

## Architecture

```
moyo/gui/
├── __init__.py        # re-exports `main`
└── app.py             # MoyoGUI + all tab classes + BackgroundWorker
```

Long-running operations (crawling, embedding, analysis, fuzzing) are
dispatched through `BackgroundWorker` (a `QThread` wrapping any callable) so
the UI thread stays responsive. Stdout/stderr from the worker is streamed
into each tab's log pane.

## Future improvements

- Persist last-used paths/options via `QSettings`.
- Move each tab into its own module under `moyo/gui/tabs/`.
- Add a unified Project workflow (a wizard that runs gather → build → probe
  → fuzz sequentially on one topic).
- Add a Stop button to in-flight workers.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `moyo-gui: command not found` | Run `pip install -e ".[gui]"`, then `pyenv rehash` (if using pyenv). |
| Black canvas / no plot | Click **Load Indices** first; the **Generate Plot** button is disabled until then. |
| `UMAP not installed` | `pip install umap-learn` |
| `scipy not available` | `pip install scipy` |
| `LLM client not initialised` | Either install `openai`/`anthropic`, or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, or use the `local` provider. |
| Device shows "CUDA not available" | See [`docs/embeddings.md`](embeddings.md) § GPU setup (`nvidia-smi`, `/dev/nvidia*`, WSL restart, CUDA PyTorch build). |
