"""Secret store for white-box mode.

Loads the organization's known secrets from various file formats and
pre-computes embeddings so they can be used for response evaluation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Secret:
    """A single organizational secret or piece of proprietary information."""

    id: str
    content: str
    label: str = ""
    tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "label": self.label,
            "tags": self.tags,
        }


class SecretStore:
    """Loads and manages the organization's known secrets.

    Supported file formats:
    - .json  – list of objects with at least a "content" field
    - .jsonl – one JSON object per line
    - .yaml / .yml – list of mappings
    - .txt  – one secret per non-empty line (content only)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self._secrets: List[Secret] = []

    def load_from_file(self, path: str) -> List[Secret]:
        """Load secrets from file and pre-compute their embeddings."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Secrets file not found: {path}")

        suffix = p.suffix.lower()
        if suffix == ".json":
            self._secrets = self._load_json(p)
        elif suffix == ".jsonl":
            self._secrets = self._load_jsonl(p)
        elif suffix in (".yaml", ".yml"):
            self._secrets = self._load_yaml(p)
        elif suffix == ".txt":
            self._secrets = self._load_txt(p)
        else:
            raise ValueError(f"Unsupported secrets file format: {suffix}")

        logger.info(f"Loaded {len(self._secrets)} secrets from {path}")
        self._embed_secrets()
        return self._secrets

    def load_from_list(self, items: List[str]) -> List[Secret]:
        """Load secrets from a plain list of strings."""
        self._secrets = [
            Secret(id=f"secret_{i}", content=item, label=item[:60])
            for i, item in enumerate(items)
        ]
        self._embed_secrets()
        return self._secrets

    @property
    def secrets(self) -> List[Secret]:
        return self._secrets

    def get_embeddings(self) -> np.ndarray:
        """Return a matrix of shape (N, D) of secret embeddings."""
        embs = [s.embedding for s in self._secrets if s.embedding is not None]
        if not embs:
            return np.empty((0,))
        return np.vstack(embs)

    def _embed_secrets(self) -> None:
        """Compute and store embeddings for all loaded secrets."""
        try:
            from shared_utils import embed
            texts = [s.content for s in self._secrets]
            embeddings = embed(texts, self.embedding_model)
            for secret, emb in zip(self._secrets, embeddings):
                secret.embedding = np.array(emb, dtype=np.float32)
            logger.info(f"Embedded {len(self._secrets)} secrets with model '{self.embedding_model}'")
        except Exception as exc:
            logger.warning(f"Could not embed secrets: {exc}. Cosine similarity evaluation will be skipped.")

    def _load_json(self, path: Path) -> List[Secret]:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            data = [data]
        return [self._item_to_secret(i, item) for i, item in enumerate(data)]

    def _load_jsonl(self, path: Path) -> List[Secret]:
        secrets = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    secrets.append(self._item_to_secret(i, json.loads(line)))
        return secrets

    def _load_yaml(self, path: Path) -> List[Secret]:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML secrets files: pip install pyyaml")
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, list):
            data = [data]
        return [self._item_to_secret(i, item) for i, item in enumerate(data)]

    def _load_txt(self, path: Path) -> List[Secret]:
        secrets = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    secrets.append(Secret(id=f"secret_{i}", content=line, label=line[:60]))
        return secrets

    @staticmethod
    def _item_to_secret(index: int, item: Any) -> Secret:
        if isinstance(item, str):
            return Secret(id=f"secret_{index}", content=item, label=item[:60])
        content = item.get("content", item.get("text", item.get("secret", str(item))))
        return Secret(
            id=item.get("id", f"secret_{index}"),
            content=content,
            label=item.get("label", item.get("name", content[:60])),
            tags=item.get("tags", []),
        )
