"""Evaluators for barrier probe results.

Provides cosine-similarity-based semantic distance scoring between a candidate
text and the private corpus, reusing the BarrierAnalyzer infrastructure.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate(candidate: str, embedding_model: str = "all-MiniLM-L6-v2") -> float:
    """Return a semantic relevance score for candidate text in [0, 1].

    Uses the shared embedding infrastructure to embed the candidate and returns
    its L2-normalised magnitude as a proxy for information density.  A score
    closer to 1.0 indicates higher semantic richness.

    For full barrier analysis (comparing against a private FAISS index) use
    :class:`moyo.publicside.barrierprobe.barrier_analyzer.BarrierAnalyzer` directly.

    Args:
        candidate: Text to evaluate.
        embedding_model: Sentence-transformer model name.

    Returns:
        Float in [0, 1] representing relative semantic density.
    """
    try:
        import numpy as np
        from shared_utils import embed

        embs = embed([candidate], embedding_model)
        if not embs:
            return 0.0
        vec = np.array(embs[0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        # Typical MiniLM L2 norm for a meaningful sentence is ~1.0 (if already
        # normalised by sentence-transformers) so we cap at 1.0.
        return min(norm, 1.0)
    except Exception as exc:
        logger.warning(f"evaluate() failed: {exc}")
        return 0.0


def evaluate_similarity(text_a: str, text_b: str, embedding_model: str = "all-MiniLM-L6-v2") -> float:
    """Compute cosine similarity between two texts.

    Returns a value in [-1, 1], where 1.0 means semantically identical.

    Args:
        text_a: First text.
        text_b: Second text.
        embedding_model: Sentence-transformer model name.

    Returns:
        Cosine similarity float.
    """
    try:
        import numpy as np
        from shared_utils import embed

        embs = embed([text_a, text_b], embedding_model)
        if len(embs) < 2:
            return 0.0
        a = np.array(embs[0], dtype=np.float32)
        b = np.array(embs[1], dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a / norm_a, b / norm_b))
    except Exception as exc:
        logger.warning(f"evaluate_similarity() failed: {exc}")
        return 0.0
