# Embedding models

moyo embeds private and public corpus chunks with
[`sentence-transformers`](https://www.sbert.net/) (local) or the OpenAI
Embeddings API. Public and private indices **must** use the same model
(and the same chunk size/overlap) for barrier analysis distances to be
meaningful. Switching models means **re-embedding both sides**.

Configuration:

| Setting | Env / config | Default |
| ------- | ------------ | ------- |
| Model   | `MOYO_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` |
| Device  | `MOYO_EMBEDDING_DEVICE` | `auto` (`cuda` when PyTorch sees a GPU, else `cpu`) |
| Batch   | `MOYO_EMBEDDING_BATCH_SIZE` | `32` |

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
| **Fast** | `mini` | `all-MiniLM-L6-v2` | 384 | Default for prototyping and pipeline iteration. English only. Fine on CPU. |
| **Fast+** | `mini-l12` | `all-MiniLM-L12-v2` | 384 | Modest quality bump; **same 384d layout** as L6 (re-embed, no FAISS dim change). Still CPU-friendly. |
| **Balanced** | `mpnet` | `all-mpnet-base-v2` | 768 | Strong local default once you care about barrier precision. Prefer GPU for bulk builds. |
| **Balanced / retrieval** | `bge-base` | `BAAI/bge-base-en-v1.5` | 768 | Often beats MPNet on retrieval benchmarks. English. Same hardware profile as MPNet. |
| **Balanced / retrieval** | `e5-base` | `intfloat/e5-base-v2` | 768 | Strong retrieval; best with `query:` / `passage:` prefixes (not applied automatically today). |
| **Multilingual** | `multilingual` | `paraphrase-multilingual-mpnet-base-v2` | 768 | Non-English (or mixed) public/private corpora. |
| **API** | `openai-small` | `text-embedding-3-small` | 1536 | Highest convenience / quality via API. **Private text leaves the machine.** Needs `OPENAI_API_KEY`. |
| **API** | `openai-large` | `text-embedding-3-large` | 3072 | Max API quality; costlier; same privacy caveat. |

### Practical guidance for moyo

1. **Stay on MiniLM-L6** while building and tuning the pipeline.
2. When real corpora matter for barrier precision, **benchmark `mpnet` or
   `bge-base`** on a sample: compare nearest-neighbour rankings and distance
   distributions, not only public MTEB scores.
3. Use **`multilingual`** only if language coverage requires it.
4. Prefer **local models for private-side** work; use OpenAI only if you
   accept data leaving the host and recalibrate distance thresholds after
   switching.
5. After any model change, **rebuild both public and private indices** and
   re-run barrier baselines — absolute cosine distances are not comparable
   across embedding spaces.

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

## CLI / code

```python
from shared_utils import embed, resolve_device, get_device_info

print(get_device_info())
# {'cuda_available': True, 'resolved_device': 'cuda', ...}

vectors = embed(
    ["secret formula fragment"],
    model_name="all-mpnet-base-v2",  # or catalog key "mpnet"
    device="auto",
)
```

Environment override:

```bash
export MOYO_EMBEDDING_DEVICE=cuda
export MOYO_EMBEDDING_MODEL_NAME=all-mpnet-base-v2
```
