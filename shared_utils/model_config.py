"""Centralized model configuration for embedding selection.

Moved from senteGUI.model_config to shared_utils so all packages can share it.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional

# Repo root (shared_utils/..) and project config directory.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = pathlib.Path(os.environ.get("MOYO_CONFIG_DIR", _REPO_ROOT / "config"))
CONFIG_FILE = CONFIG_DIR / "model_config.json"

# Legacy alias — some callers still import DATA_DIR for corpus paths.
DATA_DIR = pathlib.Path(os.environ.get("SENTE_DATA_DIR", "data"))

# Full catalog: key → metadata. GUI and CLI should prefer this over hard-coded lists.
# ``backend``: "local" uses sentence-transformers; "openai" uses the OpenAI API.
EMBEDDING_CATALOG: Dict[str, Dict[str, Any]] = {
    "mini": {
        "model_name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "backend": "local",
        "tier": "fast",
        "label": "MiniLM-L6 — fast (384d)",
        "description": "Default for prototyping. Fast on CPU/GPU; English only.",
    },
    "mini-l12": {
        "model_name": "all-MiniLM-L12-v2",
        "dimensions": 384,
        "backend": "local",
        "tier": "fast",
        "label": "MiniLM-L12 — fast+ (384d)",
        "description": "Modest quality bump over L6; same 384d index layout.",
    },
    "mpnet": {
        "model_name": "all-mpnet-base-v2",
        "dimensions": 768,
        "backend": "local",
        "tier": "balanced",
        "label": "MPNet — balanced (768d)",
        "description": "Strong local default for barrier analysis. Prefer GPU for bulk builds.",
    },
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768,
        "backend": "local",
        "tier": "balanced",
        "label": "BGE-base — retrieval (768d)",
        "description": "Often beats MPNet on retrieval benchmarks. English.",
    },
    "e5-base": {
        "model_name": "intfloat/e5-base-v2",
        "dimensions": 768,
        "backend": "local",
        "tier": "balanced",
        "label": "E5-base — retrieval (768d)",
        "description": "Strong retrieval model. Prefers 'query: '/'passage: ' prefixes for best results.",
    },
    "multilingual": {
        "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dimensions": 768,
        "backend": "local",
        "tier": "balanced",
        "label": "Multilingual MPNet (768d)",
        "description": "Use when public or private corpora are not English-only.",
    },
    "openai-small": {
        "model_name": "text-embedding-3-small",
        "dimensions": 1536,
        "backend": "openai",
        "tier": "api",
        "label": "OpenAI small (1536d)",
        "description": "API quality; private data leaves the machine. Requires OPENAI_API_KEY.",
    },
    "openai-large": {
        "model_name": "text-embedding-3-large",
        "dimensions": 3072,
        "backend": "openai",
        "tier": "api",
        "label": "OpenAI large (3072d)",
        "description": "Highest API quality; costlier. Requires OPENAI_API_KEY.",
    },
}

# Backward-compatible key → HF/API model name map
EMBEDDING_MODELS: Dict[str, str] = {
    key: meta["model_name"] for key, meta in EMBEDDING_CATALOG.items()
}

DEFAULT_MODEL_KEY = "mini"
DEFAULT_MODEL_NAME = EMBEDDING_MODELS[DEFAULT_MODEL_KEY]

OPENAI_MODEL_NAMES = {
    meta["model_name"]
    for meta in EMBEDDING_CATALOG.values()
    if meta["backend"] == "openai"
} | {"openai-small", "openai-large"}


def list_embedding_choices() -> List[Dict[str, Any]]:
    """Return catalog entries with their keys for GUI/CLI population."""
    return [{"key": key, **meta} for key, meta in EMBEDDING_CATALOG.items()]


def get_catalog_entry(model_key_or_name: str) -> Optional[Dict[str, Any]]:
    """Look up catalog metadata by key or by model_name."""
    if model_key_or_name in EMBEDDING_CATALOG:
        return {"key": model_key_or_name, **EMBEDDING_CATALOG[model_key_or_name]}
    for key, meta in EMBEDDING_CATALOG.items():
        if meta["model_name"] == model_key_or_name:
            return {"key": key, **meta}
    return None


def get_dimensions(model_key_or_name: str, default: int = 384) -> int:
    entry = get_catalog_entry(model_key_or_name)
    return int(entry["dimensions"]) if entry else default


def is_openai_model(model_key_or_name: str) -> bool:
    entry = get_catalog_entry(model_key_or_name)
    if entry:
        return entry["backend"] == "openai"
    return model_key_or_name in OPENAI_MODEL_NAMES


def resolve_model_name(model_key_or_name: Optional[str] = None) -> str:
    """Resolve a GUI key or HF name to the canonical model_name string."""
    if not model_key_or_name:
        return get_current_model_name()
    if model_key_or_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_key_or_name]
    return model_key_or_name


def get_model_config() -> dict:
    if not CONFIG_FILE.exists():
        config = {"model_key": DEFAULT_MODEL_KEY, "model_name": DEFAULT_MODEL_NAME, "last_updated": None}
        write_model_config(config)
        return config
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        config = {"model_key": DEFAULT_MODEL_KEY, "model_name": DEFAULT_MODEL_NAME, "last_updated": None}
        write_model_config(config)
        return config


def write_model_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_current_model_name() -> str:
    return get_model_config().get("model_name", DEFAULT_MODEL_NAME)


def get_current_model_key() -> str:
    return get_model_config().get("model_key", DEFAULT_MODEL_KEY)


def set_model(model_key: str) -> None:
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Invalid model key: {model_key}. Available keys: {list(EMBEDDING_MODELS.keys())}")
    write_model_config({"model_key": model_key, "model_name": EMBEDDING_MODELS[model_key], "last_updated": None})


def get_model_name_for_key(model_key: str) -> str:
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Invalid model key: {model_key}. Available keys: {list(EMBEDDING_MODELS.keys())}")
    return EMBEDDING_MODELS[model_key]


def get_model_key_for_name(model_name: str) -> str:
    for key, name in EMBEDDING_MODELS.items():
        if name == model_name:
            return key
    raise ValueError(f"Invalid model name: {model_name}. Available models: {list(EMBEDDING_MODELS.values())}")
