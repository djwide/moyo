"""Shared embedding utilities for sente and moyo projects."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Global variables for model caching
_transformer = None
_model_name: Optional[str] = None
_device: Optional[str] = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.error(
        "sentence-transformers not available. "
        "Please install with: pip install sentence-transformers"
    )


def cuda_available() -> bool:
    """Return True if PyTorch can see a CUDA device."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve ``auto`` / ``cuda`` / ``cpu`` to a concrete device string.

    Preference order when ``device`` is None:
    1. ``MOYO_EMBEDDING_DEVICE`` env var
    2. ``moyo.config.settings`` EmbeddingConfig.device (if importable)
    3. ``auto``
    """
    if device is None:
        device = os.environ.get("MOYO_EMBEDDING_DEVICE")
        if not device:
            try:
                from moyo.config.settings import get_settings
                device = get_settings().embedding.device
            except Exception:
                device = "auto"

    device = (device or "auto").strip().lower()
    if device in ("auto", ""):
        resolved = "cuda" if cuda_available() else "cpu"
    elif device in ("cuda", "gpu"):
        if cuda_available():
            resolved = "cuda"
        else:
            logger.warning(
                "CUDA requested but not available (PyTorch sees no GPU). "
                "Falling back to CPU. On WSL2, confirm NVIDIA driver GPU "
                "passthrough (`nvidia-smi` and /dev/nvidia*) then restart WSL."
            )
            resolved = "cpu"
    elif device == "cpu":
        resolved = "cpu"
    else:
        logger.warning("Unknown embedding device %r; using CPU", device)
        resolved = "cpu"

    return resolved


def get_device_info() -> dict:
    """Snapshot of embedding device status for GUI/CLI display."""
    info = {
        "cuda_available": cuda_available(),
        "resolved_device": resolve_device("auto"),
        "configured_device": os.environ.get("MOYO_EMBEDDING_DEVICE", "auto"),
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    if info["cuda_available"]:
        try:
            import torch
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 1)
        except Exception as exc:
            logger.debug("Could not read GPU properties: %s", exc)
    return info


def _openai_api_model_name(model_name: str) -> str:
    aliases = {
        "openai-small": "text-embedding-3-small",
        "openai-large": "text-embedding-3-large",
    }
    return aliases.get(model_name, model_name)


def l2_normalize(vectors: List[List[float]]) -> List[List[float]]:
    """L2-normalize rows so inner product equals cosine similarity."""
    if not vectors:
        return vectors
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0, 1.0, norms)
    return arr.tolist()


def resolve_normalize(normalize: Optional[bool] = None) -> bool:
    """Resolve embedding L2-normalization. Default True (required for FlatIP)."""
    if normalize is not None:
        return bool(normalize)
    env = os.environ.get("MOYO_EMBEDDING_NORMALIZE")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}
    try:
        from moyo.config.settings import get_settings
        return bool(get_settings().embedding.normalize)
    except Exception:
        return True


def _embed_openai(
    texts: List[str],
    model_name: str,
    batch_size: int,
    normalize: bool = True,
) -> List[List[float]]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package required for OpenAI embedding models. "
            "Install with: pip install openai"
        ) from e

    api_model = _openai_api_model_name(model_name)
    client = OpenAI()
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=api_model, input=batch)
        # API returns data sorted by index, but be defensive
        ordered = sorted(response.data, key=lambda d: d.index)
        vectors.extend([list(item.embedding) for item in ordered])

    return l2_normalize(vectors) if normalize else vectors


def clear_embedding_cache() -> None:
    """Drop the cached sentence-transformers model (e.g. after device change)."""
    global _transformer, _model_name, _device
    _transformer = None
    _model_name = None
    _device = None


def get_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> "SentenceTransformer":
    """Get or create embedding model singleton.

    Args:
        model_name: Name of the model to load. If None, uses last/default.
        device: ``auto``, ``cuda``, or ``cpu``. Reloads if device changes.

    Returns:
        SentenceTransformer instance
    """
    global _transformer, _model_name, _device

    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not available. "
            "Please install with: pip install sentence-transformers"
        )

    from shared_utils.model_config import is_openai_model, resolve_model_name

    resolved_name = resolve_model_name(model_name) if model_name else (_model_name or resolve_model_name(None))
    if is_openai_model(resolved_name):
        raise ValueError(
            f"Model {resolved_name!r} is an OpenAI API model; use embed() instead of "
            "get_embedding_model()."
        )

    resolved_device = resolve_device(device)

    needs_reload = (
        _transformer is None
        or _model_name != resolved_name
        or _device != resolved_device
    )

    if needs_reload:
        try:
            logger.info(
                "Loading embedding model %s on device=%s",
                resolved_name,
                resolved_device,
            )
            _transformer = SentenceTransformer(resolved_name, device=resolved_device)
            _model_name = resolved_name
            _device = resolved_device
            logger.info("Loaded embedding model: %s (%s)", resolved_name, resolved_device)
        except Exception as e:
            error_msg = f"Failed to load embedding model {resolved_name}: {e}"
            logger.error(error_msg)
            clear_embedding_cache()
            raise RuntimeError(error_msg) from e

    return _transformer


def embed(
    texts: Iterable[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
    normalize: Optional[bool] = None,
    device: Optional[str] = None,
) -> List[List[float]]:
    """Embed texts using sentence-transformers or the OpenAI embeddings API.

    Args:
        texts: Iterable of text strings to embed
        model_name: Model key or Hugging Face / OpenAI model name
        batch_size: Batch size for processing
        normalize: L2-normalize embeddings so FlatIP = cosine. Defaults to
            ``MOYO_EMBEDDING_NORMALIZE`` / True. Leave on for barrier analysis.
        device: ``auto`` / ``cuda`` / ``cpu`` (ignored for OpenAI models)

    Returns:
        List of embedding vectors
    """
    from shared_utils.model_config import is_openai_model, resolve_model_name

    texts_list = list(texts)
    if not texts_list:
        return []

    if not all(isinstance(text, str) for text in texts_list):
        raise ValueError("All texts must be strings")

    resolved_name = resolve_model_name(model_name)
    do_normalize = resolve_normalize(normalize)
    if not do_normalize:
        logger.warning(
            "Embedding L2-normalization is OFF. IndexFlatIP inner products "
            "will not equal cosine similarity; barrier distances will drift."
        )

    if is_openai_model(resolved_name):
        try:
            return _embed_openai(
                texts_list, resolved_name, batch_size, normalize=do_normalize
            )
        except Exception as e:
            if isinstance(e, (ImportError, RuntimeError, ValueError)):
                raise
            error_msg = f"OpenAI embedding failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    model = get_embedding_model(resolved_name, device=device)

    try:
        embeddings = model.encode(
            texts_list,
            batch_size=batch_size,
            normalize_embeddings=do_normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(f"Expected 2D embeddings, got {embeddings.ndim}D")

        expected_dim = model.get_sentence_embedding_dimension()
        actual_dim = embeddings.shape[1]
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )

        return embeddings.tolist()

    except Exception as e:
        if isinstance(e, (ImportError, RuntimeError, ValueError)):
            raise
        error_msg = f"Embedding failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_embedding_dimension(model_name: Optional[str] = None) -> int:
    """Get the dimension of embeddings for a given model."""
    from shared_utils.model_config import get_dimensions, is_openai_model, resolve_model_name

    resolved = resolve_model_name(model_name)
    if is_openai_model(resolved):
        return get_dimensions(resolved)

    # Prefer catalog (avoids loading the model just for dims)
    catalog_dim = get_dimensions(resolved, default=0)
    if catalog_dim:
        return catalog_dim

    model = get_embedding_model(resolved)
    return model.get_sentence_embedding_dimension()


def get_active_device() -> Optional[str]:
    """Return the device the cached local model is currently on, if any."""
    return _device
