"""Centralized model configuration for embedding selection.

Moved from senteGUI.model_config to shared_utils so all packages can share it.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Optional

# Default data directory (respects SENTE_DATA_DIR if set)
DATA_DIR = pathlib.Path(os.environ.get("SENTE_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = DATA_DIR / "model_config.json"

EMBEDDING_MODELS = {
    "mini": "all-MiniLM-L6-v2",
    "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "openai-large": "text-embedding-3-large",
    "openai-small": "text-embedding-3-small",
}

DEFAULT_MODEL_KEY = "mini"
DEFAULT_MODEL_NAME = EMBEDDING_MODELS[DEFAULT_MODEL_KEY]


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


