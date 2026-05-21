"""Shared embedding utilities for sente and moyo projects."""

from typing import Iterable, List, Optional
import logging

logger = logging.getLogger(__name__)

# Global variables for model caching
_transformer = None
_model_name = "all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.error("sentence-transformers not available. Please install with: pip install sentence-transformers")


def get_embedding_model(model_name: Optional[str] = None) -> "SentenceTransformer":
    """Get or create embedding model singleton.
    
    Args:
        model_name: Name of the model to load. If None, uses default.
        
    Returns:
        SentenceTransformer instance
        
    Raises:
        ImportError: If sentence-transformers is not available
        RuntimeError: If model loading fails
    """
    global _transformer, _model_name
    
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not available. "
            "Please install with: pip install sentence-transformers"
        )
    
    if model_name:
        _model_name = model_name
    
    if _transformer is None:
        try:
            _transformer = SentenceTransformer(_model_name)
            logger.info(f"Loaded embedding model: {_model_name}")
        except Exception as e:
            error_msg = f"Failed to load embedding model {_model_name}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    return _transformer


def embed(texts: Iterable[str], 
          model_name: Optional[str] = None, 
          batch_size: int = 32,
          normalize: bool = True) -> List[List[float]]:
    """Embed texts using sentence-transformers.
    
    Args:
        texts: Iterable of text strings to embed
        model_name: Name of the model to use
        batch_size: Batch size for processing
        normalize: Whether to normalize embeddings
        
    Returns:
        List of embedding vectors
        
    Raises:
        ImportError: If sentence-transformers is not available
        RuntimeError: If embedding fails or dimension mismatch occurs
        ValueError: If texts is empty or invalid
    """
    # Convert to list if it's an iterator
    texts_list = list(texts)
    if not texts_list:
        return []
    
    # Validate input
    if not all(isinstance(text, str) for text in texts_list):
        raise ValueError("All texts must be strings")
    
    model = get_embedding_model(model_name)
    
    try:
        embeddings = model.encode(
            texts_list,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        # Validate embedding dimensions
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
        else:
            error_msg = f"Embedding failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def get_embedding_dimension(model_name: Optional[str] = None) -> int:
    """Get the dimension of embeddings for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Embedding dimension
        
    Raises:
        ImportError: If sentence-transformers is not available
        RuntimeError: If model loading fails
    """
    model = get_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()
