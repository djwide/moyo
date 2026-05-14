"""Helpers for corpus centroid computation and topic token extraction.

This module provides utilities to:
- Load a private corpus (list of strings or a file path)
- Compute cluster centroids over sentence embeddings
- Extract representative topic tokens per centroid using TF-IDF
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Re-exported helpers from shared_utils top-level package
from shared_utils import embed


CorpusInput = Union[Sequence[str], str, Path]


def _load_corpus_texts(corpus: CorpusInput) -> List[str]:
    """Load texts from a corpus input.

    Accepts:
    - List/sequence of strings (returned as-is, stripped and filtered)
    - Path or str to a text file. If the file follows the GUI format
      (blocks with lines starting with "Text:"), those are extracted; otherwise
      each non-empty line is a document.
    """
    if isinstance(corpus, (list, tuple)):
        return [t.strip() for t in corpus if isinstance(t, str) and t.strip()]

    path = Path(corpus)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="ignore")

    texts: List[str] = []
    # Try GUI corpus format first
    if "\nText:" in raw or raw.startswith("Text:"):
        for line in raw.splitlines():
            if line.startswith("Text:"):
                text = line[len("Text:"):].strip()
                if text:
                    texts.append(text)
    # Fallback: each non-empty line is one item
    if not texts:
        texts = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    return texts


def _auto_select_k(embeddings: np.ndarray, max_clusters: int = 10, random_state: int = 42) -> int:
    """Heuristically select number of clusters using silhouette score.

    Returns at least 1. If not enough samples for clustering, falls back to 1.
    """
    num_samples = embeddings.shape[0]
    if not SKLEARN_AVAILABLE or num_samples < 3:
        return 1

    upper = min(max_clusters, num_samples - 1)
    if upper < 2:
        return 1

    best_k = 2
    best_score = -1.0
    for k in range(2, upper + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k if best_score >= 0 else 1


def compute_corpus_centroids(
    corpus: CorpusInput,
    embedding_model: Optional[str] = None,
    num_clusters: Optional[int] = None,
    max_clusters: int = 10,
    random_state: int = 42,
) -> Dict[str, object]:
    """Compute cluster centroids for a private corpus.

    Args:
        corpus: List of texts or path to a corpus file.
        embedding_model: Name of the embedding model to use (defaults to shared_utils default).
        num_clusters: If provided, use this many clusters; otherwise auto-select.
        max_clusters: Upper bound for auto-selected clusters.
        random_state: Random seed for clustering.

    Returns:
        Dict with keys:
        - texts: List[str]
        - embeddings: np.ndarray of shape (N, D)
        - labels: np.ndarray of shape (N,)
        - centroids: np.ndarray of shape (K, D)
        - model: embedding_model name (str)
    """
    texts = _load_corpus_texts(corpus)
    if not texts:
        return {"texts": [], "embeddings": np.zeros((0, 0)), "labels": np.array([]), "centroids": np.zeros((0, 0)), "model": embedding_model or "default"}

    vectors = embed(texts, model_name=embedding_model, normalize=True)
    embeddings = np.array(vectors, dtype=np.float32)

    if num_clusters is None:
        num_clusters = _auto_select_k(embeddings, max_clusters=max_clusters, random_state=random_state)

    if not SKLEARN_AVAILABLE or num_clusters <= 1:
        # Single centroid (mean)
        centroid = embeddings.mean(axis=0, keepdims=True)
        labels = np.zeros((embeddings.shape[0],), dtype=int)
        centroids = centroid
    else:
        km = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        centroids = km.cluster_centers_

    return {
        "texts": texts,
        "embeddings": embeddings,
        "labels": labels,
        "centroids": centroids,
        "model": embedding_model or "default",
    }


def extract_topic_tokens(
    texts: Sequence[str],
    labels: Sequence[int],
    top_k: int = 10,
    stop_words: Optional[Union[str, List[str]]] = "english",
) -> List[List[str]]:
    """Extract representative topic tokens for each cluster using TF-IDF.

    Args:
        texts: The corpus texts aligned with labels.
        labels: Cluster labels for each text.
        top_k: Number of tokens to return per cluster.
        stop_words: Stop-word strategy for TfidfVectorizer.

    Returns:
        List of token lists, one per cluster index 0..K-1.
    """
    if not SKLEARN_AVAILABLE or not texts:
        return [[]]

    labels = np.array(labels)
    k = int(labels.max()) + 1 if labels.size else 1

    # Fit TF-IDF on entire corpus
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    topics: List[List[str]] = []
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            topics.append([])
            continue
        # Average TF-IDF scores within the cluster
        cluster_matrix = tfidf[mask]
        mean_scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_indices = np.argsort(mean_scores)[::-1][:top_k]
        tokens = vocab[top_indices].tolist()
        topics.append(tokens)

    return topics


def tokens_for_corpus(
    corpus: CorpusInput,
    embedding_model: Optional[str] = None,
    num_clusters: Optional[int] = None,
    top_k: int = 10,
    max_clusters: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[List[str]], np.ndarray, List[str]]:
    """High-level helper: compute centroids and topic tokens for a corpus.

    Returns:
        centroids, topic_tokens, labels, texts
    """
    res = compute_corpus_centroids(
        corpus,
        embedding_model=embedding_model,
        num_clusters=num_clusters,
        max_clusters=max_clusters,
        random_state=random_state,
    )
    texts: List[str] = res["texts"]
    labels: np.ndarray = res["labels"]
    centroids: np.ndarray = res["centroids"]
    topic_tokens = extract_topic_tokens(texts, labels, top_k=top_k)
    return centroids, topic_tokens, labels, texts


