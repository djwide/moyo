# Embedding models

moyo embeds private and public corpus chunks with
[`sentence-transformers`](https://www.sbert.net/) (local) or the OpenAI
Embeddings API. Public and private indices **must** use the same model
(and the same chunk size/overlap) for barrier analysis distances to be
meaningful. Switching models means **re-embedding both sides**.

Configuration:

| Setting | Env / config | Default |
| ------- | ------------ | ------- |
| Model   | `MOYO_EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` |
| Device  | `MOYO_EMBEDDING_DEVICE` | `auto` (`cuda` when PyTorch sees a GPU, else `cpu`) |
| Batch   | `MOYO_EMBEDDING_BATCH_SIZE` | `32` |
| Normalize | `MOYO_EMBEDDING_NORMALIZE` | `true` (required for FlatIP = cosine) |

The GUI (**Create Private Index**, **Build Public Corpus**) exposes the full
catalog plus a device selector (`Auto` / `CUDA` / `CPU`). Catalog keys live in
`shared_utils/model_config.py` (`EMBEDDING_CATALOG`). The selected model key is
persisted in `config/model_config.json` (override directory with `MOYO_CONFIG_DIR`).

## GPU setup

With `MOYO_EMBEDDING_DEVICE=auto` (recommended), local models load on CUDA
when available:

```bash
# Verify toolkit + runtime (WSL2)
nvcc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If `nvcc` works but `torch.cuda.is_available()` is `False`:

1. Confirm `/dev/nvidia*` exists inside WSL (`ls /dev/nvidia*`).
2. Confirm Windows has a recent NVIDIA driver with WSL support.
3. Restart WSL (`wsl --shutdown` from PowerShell, then reopen).
4. Ensure PyTorch was installed with a CUDA build (e.g. `cu118` / `cu124` /
   `cu128`), not CPU-only.

FAISS in this repo uses `faiss-cpu` by default; only the **embedding** step
is GPU-accelerated. That is usually the bottleneck for index builds.

## Model tier recommendations

| Tier | Keys (GUI) | Model | Dims | When to use |
| ---- | ---------- | ----- | ---- | ----------- |
| **Fast** | `mini` | `all-MiniLM-L6-v2` | 384 | Prototyping and CPU-only iteration. English only. |
| **Fast+** | `mini-l12` | `all-MiniLM-L12-v2` | 384 | Modest quality bump; **same 384d layout** as L6 (re-embed, no FAISS dim change). Still CPU-friendly. |
| **Balanced** | `mpnet` | `all-mpnet-base-v2` | 768 | Strong local STS model. Prefer GPU for bulk builds. |
| **Default / retrieval** | `bge-base` | `BAAI/bge-base-en-v1.5` | 768 | Default. Best local match for short private phrases vs public text. English. |
| **Balanced / retrieval** | `e5-base` | `intfloat/e5-base-v2` | 768 | Strong retrieval; best with `query:` / `passage:` prefixes (not applied automatically today). |
| **Multilingual** | `multilingual` | `paraphrase-multilingual-mpnet-base-v2` | 768 | Non-English (or mixed) public/private corpora. |
| **API** | `openai-small` | `text-embedding-3-small` | 1536 | Highest convenience / quality via API. **Private text leaves the machine.** Needs `OPENAI_API_KEY`. |
| **API** | `openai-large` | `text-embedding-3-large` | 3072 | Max API quality; costlier; same privacy caveat. |

### Practical guidance for moyo

1. **Use `bge-base`** (the default) for private and public indexes.
2. Drop to **MiniLM-L6** only for CPU-only prototyping; rebuild both sides
   before comparing distances.
3. Use **`multilingual`** only if language coverage requires it.
4. Prefer **local models for private-side** work; use OpenAI only if you
   accept data leaving the host and recalibrate distance thresholds after
   switching.
5. After any model change, **rebuild both public and private indices** and
   re-run barrier baselines — absolute cosine distances are not comparable
   across embedding spaces. Private and public indexes **must** share the
   same embedding model.

### Cost sketch (local)

Approximate FAISS float32 RAM for the vector store alone
(`vectors × dims × 4 bytes`):

| Model class | Dims | 100k vectors | 1M vectors |
| ----------- | ---- | ------------ | ---------- |
| MiniLM      | 384  | ~150 MB      | ~1.5 GB    |
| MPNet / BGE / E5 | 768 | ~300 MB | ~3 GB |
| OpenAI small | 1536 | ~600 MB     | ~6 GB      |

Bulk embed time scales roughly with model size; on GPU, MPNet/BGE are
typically comfortable. On CPU-only hosts, MiniLM remains the pragmatic
default for large corpora.

## Chunking, normalization, and compute time

Public and private indexes must share **embedding model**, **L2-normalization**,
and (for document corpora) **chunk_size / overlap / max_tokens**. Phrase-level
private indexes only need the model and normalization to match; the public
side already emits sentence/item vectors that line up with short secrets.

`max_tokens` is taken from the model catalog (MiniLM 256, MPNet 384, BGE/E5
512) so a larger encoder is not still packing MiniLM-sized windows.

Overlap defaults to ~10% of `chunk_size` (`MOYO_PIPELINE_OVERLAP=50` at 512).
Section chunks shorter than `MOYO_PIPELINE_MIN_CHUNK_LENGTH` (50) are dropped
as boilerplate; sentence/item chunks and atomic private secrets are kept.
Dedup stays on by default.

### Compute tradeoffs

Embed time is linear in **vector count × sequence length × model size**.
Barrier analysis is `O(N_private × N_public × dim)` in the current exact
NumPy pass.

| Choice | Effect on vector count | Effect on embed time | Effect on barrier pass |
| ------ | ---------------------- | -------------------- | ---------------------- |
| Multi-granularity (section + sentence + item) | ~2–4× vs sections only | ~2–4× | ~2–4× public N |
| Smaller `chunk_size` | more section windows | up (more encodes) | up |
| Larger `max_tokens` (MPNet 384 vs MiniLM 256) | fewer, longer sections | each encode is slower; fewer of them | slight drop in N, higher dim if you also switch model |
| MiniLM → MPNet/BGE | same N if chunking is unchanged | ~3–5× on GPU, more on CPU; 2× RAM at 768d | 2× in the dim term |
| L2-normalize | none | negligible (one pass over each vector) | none (required for cosine) |
| Dedup + min section length | fewer near-duplicate/boilerplate vectors | down | down |
| `moyo-probe calibrate` | none | none | one extra NN pass, same cost as analyze |

Use GPU (`MOYO_EMBEDDING_DEVICE=auto`) for the embed step — that dominates
index builds. FAISS FlatIP search is not the bottleneck at typical corpus
sizes.

After any model change, rebuild **both** indexes and re-run
`moyo-probe calibrate -p <public> -r <private>` (or **Calibrate Threshold**
in the Barrier Probe tab). MiniLM distances are not a valid cutoff on
MPNet/BGE.

## CLI / code

```python
from shared_utils import embed, resolve_device, get_device_info

print(get_device_info())
# {'cuda_available': True, 'resolved_device': 'cuda', ...}

vectors = embed(
    ["secret formula fragment"],
    model_name="BAAI/bge-base-en-v1.5",  # or catalog key "bge-base"
    device="auto",
)
```

Environment override:

```bash
export MOYO_EMBEDDING_DEVICE=cuda
export MOYO_EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
```
