"""Centralized data directory configuration for sente components.

This module ensures that all sente components (GUI, API server, linter engines)
use the same data directory for FAISS indices, corpus files, and configuration.
"""

from __future__ import annotations

import os
import pathlib
import json
from typing import Optional, Dict, Any


# Model naming conventions (kept for path helpers; prefer shared_utils.model_config)
MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "name": "minilm_l6_v2",
        "dimensions": 384,
        "description": "MiniLM L6 v2 - Fast, lightweight model"
    },
    "all-MiniLM-L12-v2": {
        "name": "minilm_l12_v2",
        "dimensions": 384,
        "description": "MiniLM L12 v2 - Better quality than L6"
    },
    "all-mpnet-base-v2": {
        "name": "mpnet_base_v2",
        "dimensions": 768,
        "description": "MPNet Base v2 - Balanced performance"
    },
    "BAAI/bge-base-en-v1.5": {
        "name": "bge_base_en_v15",
        "dimensions": 768,
        "description": "BGE base English v1.5 - Strong retrieval"
    },
    "intfloat/e5-base-v2": {
        "name": "e5_base_v2",
        "dimensions": 768,
        "description": "E5 base v2 - Strong retrieval"
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "name": "multilingual_mpnet_base_v2",
        "dimensions": 768,
        "description": "Multilingual MPNet base v2"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "multilingual_minilm_l12_v2",
        "dimensions": 384,
        "description": "Multilingual MiniLM L12 v2"
    },
    "text-embedding-3-small": {
        "name": "openai_text_embedding_3_small",
        "dimensions": 1536,
        "description": "OpenAI Text Embedding 3 Small"
    },
    "text-embedding-3-large": {
        "name": "openai_text_embedding_3_large",
        "dimensions": 3072,
        "description": "OpenAI Text Embedding 3 Large"
    },
    "openai-small": {
        "name": "openai_small",
        "dimensions": 1536,
        "description": "OpenAI Small - OpenAI API model"
    },
    "openai-large": {
        "name": "openai_large",
        "dimensions": 3072,
        "description": "OpenAI Large - OpenAI API model"
    }
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, {
        "name": model_name.lower().replace("-", "_").replace(".", "_"),
        "dimensions": 384,  # Default fallback
        "description": f"Custom model: {model_name}"
    })


def get_model_key(model_name: str) -> str:
    """Get the standardized key for a model name."""
    config = get_model_config(model_name)
    return config["name"]


def get_sente_data_dir() -> pathlib.Path:
    """Get the centralized data directory for all sente components.
    
    Priority order:
    1. SENTE_DATA_DIR environment variable
    2. Base data/sente directory (new centralized location)
    3. Fallback to current working directory / data
    
    Returns:
        Path to the data directory
    """
    # 1. Check environment variable first
    env_data_dir = os.environ.get("SENTE_DATA_DIR")
    if env_data_dir:
        data_dir = pathlib.Path(env_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    # 2. Try to find the new centralized data/sente directory
    try:
        # Look for data/sente relative to shared_utils location
        shared_utils_path = pathlib.Path(__file__).resolve()
        # Navigate up to find the monorepo root
        for parent in [shared_utils_path] + list(shared_utils_path.parents)[:6]:
            candidate = parent / "data" / "sente"
            if candidate.exists():
                return candidate
    except Exception:
        pass
    
    # 3. Fallback to current working directory / data
    fallback_dir = pathlib.Path.cwd() / "data"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir


# Global data directory that all components should use
SENTE_DATA_DIR = get_sente_data_dir()


def get_faiss_index_path(component: str = "sente", model_name: str = "all-MiniLM-L6-v2", index_type: str = "combined") -> pathlib.Path:
    """
    Get the path for a FAISS index file.
    
    Args:
        component: Either 'sente' or 'moyo'
        model_name: The embedding model name (e.g., 'all-MiniLM-L6-v2')
        index_type: Type of index ('combined', 'safe', 'unsafe', 'public', 'private')
    
    Returns:
        Path to the FAISS index file
    """
    if component not in ["sente", "moyo"]:
        raise ValueError(f"Component must be 'sente' or 'moyo', got: {component}")
    
    model_key = get_model_key(model_name)
    base_dir = SENTE_DATA_DIR if component == "sente" else SENTE_DATA_DIR  # For now, both use same directory
    
    # Create sentence_transformers subdirectory with model-specific subfolder
    sentence_transformers_dir = base_dir / "sentence_transformers" / model_key
    sentence_transformers_dir.mkdir(parents=True, exist_ok=True)
    
    # Naming convention: {index_type}_index.faiss (within model subfolder)
    return sentence_transformers_dir / f"{index_type}_index.faiss"


def get_metadata_path(component: str = "sente", model_name: str = "all-MiniLM-L6-v2", index_type: str = "combined") -> pathlib.Path:
    """
    Get the path for a metadata file.
    
    Args:
        component: Either 'sente' or 'moyo'
        model_name: The embedding model name (e.g., 'all-MiniLM-L6-v2')
        index_type: Type of index ('combined', 'safe', 'unsafe', 'public', 'private')
    
    Returns:
        Path to the metadata file
    """
    if component not in ["sente", "moyo"]:
        raise ValueError(f"Component must be 'sente' or 'moyo', got: {component}")
    
    model_key = get_model_key(model_name)
    base_dir = SENTE_DATA_DIR if component == "sente" else SENTE_DATA_DIR  # For now, both use same directory
    
    # Create sentence_transformers subdirectory with model-specific subfolder
    sentence_transformers_dir = base_dir / "sentence_transformers" / model_key
    sentence_transformers_dir.mkdir(parents=True, exist_ok=True)
    
    # Naming convention: {index_type}_metadata.json (within model subfolder)
    return sentence_transformers_dir / f"{index_type}_metadata.json"


def get_index_metadata_path(model_name: str = "all-MiniLM-L6-v2", index_type: str = "combined") -> pathlib.Path:
    """Get the path to the index metadata file with proper naming convention."""
    return get_metadata_path("sente", model_name, index_type)


def get_regex_rules_path(component: str = "sente") -> pathlib.Path:
    """
    Get the path to the regex rules file.
    
    Args:
        component: Either 'sente' or 'moyo'
    
    Returns:
        Path to the regex rules file
    """
    if component not in ["sente", "moyo"]:
        raise ValueError(f"Component must be 'sente' or 'moyo', got: {component}")
    
    return SENTE_DATA_DIR / "static_hits" / "regex_rules_master.json"


def get_corpus_path(component: str = "sente", model_name: str = "all-MiniLM-L6-v2", corpus_type: str = "combined") -> pathlib.Path:
    """
    Get the path for a corpus file.
    
    Args:
        component: Either 'sente' or 'moyo'
        model_name: The embedding model name (e.g., 'all-MiniLM-L6-v2')
        corpus_type: Type of corpus ('combined', 'safe', 'dangerous', 'unsafe')
    
    Returns:
        Path to the corpus file
    """
    if component not in ["sente", "moyo"]:
        raise ValueError(f"Component must be 'sente' or 'moyo', got: {component}")
    
    model_key = get_model_key(model_name)
    base_dir = SENTE_DATA_DIR if component == "sente" else SENTE_DATA_DIR  # For now, both use same directory
    
    # Create component directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Naming convention: {model_key}_{corpus_type}_corpus.txt
    return base_dir / f"{model_key}_{corpus_type}_corpus.txt"


def get_combined_corpus_path(model_name: str = "all-MiniLM-L6-v2") -> pathlib.Path:
    """Get the path to the combined encoded corpus file with proper naming convention."""
    return get_corpus_path("sente", model_name, "combined")


def get_safe_corpus_path(model_name: str = "all-MiniLM-L6-v2") -> pathlib.Path:
    """Get the path to the safe corpus file with proper naming convention."""
    return get_corpus_path("sente", model_name, "safe")


def get_fixed_dangerous_corpus_path() -> pathlib.Path:
    """Get the path to the fixed dangerous corpus file (model-independent)."""
    return SENTE_DATA_DIR / "dangerous_corpus.txt"


def get_fixed_safe_corpus_path() -> pathlib.Path:
    """Get the path to the fixed safe corpus file (model-independent)."""
    return SENTE_DATA_DIR / "safe_corpus.txt"


def get_safe_index_path(component: str = "sente", model_name: str = "all-MiniLM-L6-v2") -> pathlib.Path:
    """
    Get the path for a safe FAISS index file.
    
    Args:
        component: Either 'sente' or 'moyo'
        model_name: The embedding model name (e.g., 'all-MiniLM-L6-v2')
    
    Returns:
        Path to the safe FAISS index file
    """
    return get_faiss_index_path(component, model_name, "safe")


def get_safe_metadata_path(component: str = "sente", model_name: str = "all-MiniLM-L6-v2") -> pathlib.Path:
    """
    Get the path for a safe index metadata file.
    
    Args:
        component: Either 'sente' or 'moyo'
        model_name: The embedding model name (e.g., 'all-MiniLM-L6-v2')
    
    Returns:
        Path to the safe metadata file
    """
    return get_metadata_path(component, model_name, "safe")


def get_allowlist_path() -> pathlib.Path:
    """Get the path to the allowlist file."""
    return SENTE_DATA_DIR / "allowlist.txt"


def get_current_model_name() -> str:
    """Get the current default model name."""
    # Check if there's a model config file
    config_file = SENTE_DATA_DIR.parent / "model_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get("default_model", "all-MiniLM-L6-v2")
        except Exception:
            pass
    
    return "all-MiniLM-L6-v2"


def set_current_model_name(model_name: str) -> None:
    """Set the current default model name."""
    config_file = SENTE_DATA_DIR.parent / "model_config.json"
    
    config = {
        "default_model": model_name,
        "model_configs": MODEL_CONFIGS
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def get_most_recently_used_model(component: str = "sente") -> str:
    """
    Get the model name that was most recently used to build an index.
    
    Args:
        component: Either 'sente' or 'moyo'
    
    Returns:
        The model name that was most recently used
    """
    if component not in ["sente", "moyo"]:
        raise ValueError(f"Component must be 'sente' or 'moyo', got: {component}")
    
    base_dir = SENTE_DATA_DIR  # For shared_utils, we only handle sente
    
    if not base_dir.exists():
        return "all-MiniLM-L6-v2"  # Default fallback
    
    # Look for metadata files in sentence_transformers subdirectories and find the most recent one
    most_recent_model = None
    most_recent_time = 0
    
    sentence_transformers_dir = base_dir / "sentence_transformers"
    if sentence_transformers_dir.exists():
        for model_dir in sentence_transformers_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            for metadata_file in model_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if this metadata has a build timestamp
                    if "build_timestamp" in metadata and "model_name" in metadata:
                        build_time = metadata["build_timestamp"]
                        if build_time > most_recent_time:
                            most_recent_time = build_time
                            most_recent_model = metadata["model_name"]
                except Exception:
                    continue
    
    if most_recent_model:
        return most_recent_model
    
    # Fallback: check if there are any FAISS indexes in sentence_transformers subdirectories
    if sentence_transformers_dir.exists():
        for model_dir in sentence_transformers_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            for index_file in model_dir.glob("*.faiss"):
                # Extract model name from directory name
                model_key = model_dir.name
                # Use the MODEL_CONFIGS to find the model name
                for model_name, config in MODEL_CONFIGS.items():
                    if config["name"] == model_key:
                        return model_name
    
    # Final fallback
    return "all-MiniLM-L6-v2"


def get_available_indexes(component: str = "sente") -> list[dict[str, Any]]:
    """
    Get a list of available FAISS indexes with their metadata.
    
    Args:
        component: Either 'sente' or 'moyo'
    
    Returns:
        List of dictionaries containing index information
    """
    if component not in ["sente", "moyo"]:
        raise ValueError(f"Component must be 'sente' or 'moyo', got: {component}")
    
    base_dir = SENTE_DATA_DIR  # For shared_utils, we only handle sente
    
    if not base_dir.exists():
        return []
    
    indexes = []
    most_recent_time = 0
    most_recent_index = None
    
    # Find all metadata files in sentence_transformers subdirectories
    sentence_transformers_dir = base_dir / "sentence_transformers"
    if not sentence_transformers_dir.exists():
        return []
    
    for model_dir in sentence_transformers_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        for metadata_file in model_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Extract model name and index type from filename
                filename = metadata_file.stem
                
                # Find the model name and index type
                model_name = metadata.get("model_name", "Unknown")
                index_type = "combined"  # Default assumption
                
                # Try to extract index type from filename
                if "safe_metadata" in filename:
                    index_type = "safe"
                elif "combined_metadata" in filename:
                    index_type = "combined"
                
                # Check if corresponding FAISS index exists
                index_path = get_faiss_index_path("sente", model_name, index_type)
                if not index_path.exists():
                    continue
                
                # Get build timestamp
                build_timestamp = metadata.get("build_timestamp", 0)
                
                # Create index info
                index_info = {
                    "model_name": model_name,
                    "index_type": index_type,
                    "filename": index_path.name,
                    "build_timestamp": build_timestamp,
                    "unsafe_lines": metadata.get("unsafe_lines", 0),
                    "safe_lines": metadata.get("safe_lines", 0),
                    "total_lines": metadata.get("unsafe_lines", 0) + metadata.get("safe_lines", 0),
                    "is_most_recent": False  # Will be set below
                }
                
                indexes.append(index_info)
                
                # Track the most recent index
                if build_timestamp > most_recent_time:
                    most_recent_time = build_timestamp
                    most_recent_index = index_info
                    
            except Exception as e:
                print(f"Warning: Could not read metadata file {metadata_file}: {e}")
                continue
    
    # Mark the most recent index
    if most_recent_index:
        most_recent_index["is_most_recent"] = True
    
    # Sort by build timestamp (most recent first)
    indexes.sort(key=lambda x: x["build_timestamp"], reverse=True)
    
    return indexes


def migrate_existing_indexes() -> list[dict[str, Any]]:
    """
    Migrate existing indexes to the new sentence_transformers directory structure.
    
    Returns:
        List of migration operations performed
    """
    migrations = []
    
    # Look for old-style index files and migrate them
    old_patterns = [
        "combined_index.faiss",
        "safe_index.faiss", 
        "combined_metadata.json",
        "safe_metadata.json"
    ]
    
    # Also look for model-specific files in the old format
    for model_name, config in MODEL_CONFIGS.items():
        model_key = config["name"]
        old_patterns.extend([
            f"{model_key}_combined_index.faiss",
            f"{model_key}_safe_index.faiss",
            f"{model_key}_combined_metadata.json",
            f"{model_key}_safe_metadata.json"
        ])
    
    for pattern in old_patterns:
        old_files = list(SENTE_DATA_DIR.glob(pattern))
        for old_file in old_files:
            try:
                # Determine the model name and index type from the old file
                if "combined" in old_file.name:
                    index_type = "combined"
                elif "safe" in old_file.name:
                    index_type = "safe"
                else:
                    continue
                
                # Determine model name from filename
                if old_file.name.startswith("combined_") or old_file.name.startswith("safe_"):
                    model_name = "all-MiniLM-L6-v2"  # Default assumption for old files
                else:
                    # Extract model key from filename and find corresponding model name
                    model_key = old_file.name.split("_")[0]
                    model_name = None
                    for name, config in MODEL_CONFIGS.items():
                        if config["name"] == model_key:
                            model_name = name
                            break
                    if not model_name:
                        model_name = "all-MiniLM-L6-v2"  # Fallback
                
                # Create new path in sentence_transformers directory
                model_key = get_model_key(model_name)
                sentence_transformers_dir = SENTE_DATA_DIR / "sentence_transformers" / model_key
                sentence_transformers_dir.mkdir(parents=True, exist_ok=True)
                
                if old_file.suffix == ".faiss":
                    new_name = f"{index_type}_index.faiss"
                else:
                    new_name = f"{index_type}_metadata.json"
                
                new_path = sentence_transformers_dir / new_name
                
                # Only migrate if new file doesn't exist
                if not new_path.exists():
                    old_file.rename(new_path)
                    migrations.append({
                        "operation": "migrate",
                        "old_file": str(old_file),
                        "new_file": str(new_path),
                        "model": model_name,
                        "type": index_type
                    })
                    
            except Exception as e:
                print(f"Failed to migrate {old_file}: {e}")
    
    return migrations


def initialize_data_directories() -> None:
    """Initialize the data directory structure and migrate existing indexes."""
    print("🔧 Initializing centralized data directory structure...")
    
    # Create base directories
    SENTE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Migrate existing indexes
    migrations = migrate_existing_indexes()
    
    if migrations:
        print(f"✅ Migrated {len(migrations)} existing indexes")
    else:
        print("ℹ️ No existing indexes found to migrate")
    
    # Set default model if not already set
    if not (SENTE_DATA_DIR.parent / "model_config.json").exists():
        set_current_model_name("all-MiniLM-L6-v2")
        print("✅ Set default model to all-MiniLM-L6-v2")


def ensure_data_directory() -> None:
    """Ensure the data directory exists and has required subdirectories."""
    SENTE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create common subdirectories
    subdirs = ["static_hits", "semantic_hits", "semantic_misses", "stress", "gui_data", "synthetic_cases"]
    for subdir in subdirs:
        (SENTE_DATA_DIR / subdir).mkdir(exist_ok=True)


# Initialize the data directory when this module is imported
ensure_data_directory()


if __name__ == "__main__":
    initialize_data_directories()
