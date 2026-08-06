"""Evaluators for barrier probe results.

Provides cosine-similarity-based semantic distance scoring using MiniLM
embeddings (sentence-transformers). Generation elsewhere uses Ollama.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate(candidate: str, embedding_model: str = "all-MiniLM-L6-v2") -> float:
    """Return a semantic relevance score for candidate text in [0, 1].

    Uses MiniLM embeddings; returns a capped L2-norm proxy for density.
    """
    try:
        import numpy as np
        from shared_utils import embed

        embs = embed([candidate], embedding_model)
        if not embs:
            return 0.0
        vec = np.array(embs[0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return min(1.0, norm) if norm else 0.0
    except Exception as exc:
        logger.warning("evaluate() failed: %s", exc)
        return 0.0


def evaluate_similarity(
    text_a: str, text_b: str, embedding_model: str = "all-MiniLM-L6-v2"
) -> float:
    """Return cosine similarity between two texts using MiniLM embeddings."""
    try:
        import numpy as np
        from shared_utils import embed

        embs = embed([text_a, text_b], embedding_model)
        if len(embs) < 2:
            return 0.0
        a = np.array(embs[0], dtype=np.float32)
        b = np.array(embs[1], dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception as exc:
        logger.warning("evaluate_similarity() failed: %s", exc)
        return 0.0
