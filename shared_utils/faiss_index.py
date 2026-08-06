"""Shared FAISS index utilities for sente and moyo projects."""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None
    logger.warning("FAISS not available")


class StringStore:
    """Simple id→string store for corpus text retrieval.
    
    This provides a persistent mapping between integer IDs and original text strings,
    enabling retrieval of the original corpus text after FAISS search returns indices.
    """
    
    def __init__(self):
        self._store: Dict[int, str] = {}
        self._next_id: int = 0
    
    def add(self, text: str) -> int:
        """Add a string and return its ID.
        
        Args:
            text: The text string to store
            
        Returns:
            The assigned integer ID
        """
        id_ = self._next_id
        self._store[id_] = text
        self._next_id += 1
        return id_
    
    def add_batch(self, texts: List[str]) -> List[int]:
        """Add multiple strings and return their IDs.
        
        Args:
            texts: List of text strings to store
            
        Returns:
            List of assigned integer IDs (same order as input)
        """
        ids = []
        for text in texts:
            ids.append(self.add(text))
        return ids
    
    def get(self, id_: int) -> Optional[str]:
        """Get string by ID.
        
        Args:
            id_: The integer ID to look up
            
        Returns:
            The original text string, or None if not found
        """
        return self._store.get(id_)
    
    def get_batch(self, ids: List[int]) -> List[Optional[str]]:
        """Get multiple strings by IDs.
        
        Args:
            ids: List of integer IDs to look up
            
        Returns:
            List of text strings (None for any not found)
        """
        return [self._store.get(id_) for id_ in ids]
    
    def __len__(self) -> int:
        return len(self._store)
    
    def __contains__(self, id_: int) -> bool:
        return id_ in self._store
    
    def save(self, path: Union[str, Path]) -> None:
        """Save store to JSON file.
        
        Args:
            path: File path to save to
        """
        path = Path(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"store": self._store, "next_id": self._next_id}, f, ensure_ascii=False)
        logger.info(f"Saved string store with {len(self._store)} entries to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "StringStore":
        """Load store from JSON file.
        
        Args:
            path: File path to load from
            
        Returns:
            StringStore instance with loaded data
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        instance = cls()
        # JSON keys are strings; convert back to int
        instance._store = {int(k): v for k, v in data["store"].items()}
        instance._next_id = data["next_id"]
        logger.info(f"Loaded string store with {len(instance._store)} entries from {path}")
        return instance


class FAISSIndex:
    """Wrapper for FAISS index with metadata tracking and persistence."""
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            index_type: Type of index ("flat", "ivf", "hnsw")
        """
        if faiss is None:
            raise RuntimeError("FAISS not available")
        
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.metadata: List[Dict[str, Any]] = []
        self.is_trained = False
        self.string_store = StringStore()  # Store for original text strings
        
    def _create_index(self):
        """Create FAISS index based on type with GPU support."""
        # Check for GPU availability
        gpu_available = hasattr(faiss, 'GpuIndexFlatIP')
        
        if self.index_type == "flat":
            if gpu_available:
                # Create GPU index
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(self.dimension)
                return faiss.GpuIndexFlatIP(res, cpu_index)
            else:
                return faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
        elif self.index_type == "ivf":
            # IVF with 100 clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            cpu_index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            if gpu_available:
                res = faiss.StandardGpuResources()
                return faiss.GpuIndexIVFFlat(res, cpu_index)
            else:
                return cpu_index
        elif self.index_type == "hnsw":
            cpu_index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
            if gpu_available:
                res = faiss.StandardGpuResources()
                return faiss.GpuIndexHNSWFlat(res, cpu_index)
            else:
                return cpu_index
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]] = None) -> None:
        """Add vectors and metadata to the index.
        
        Args:
            vectors: List of embedding vectors
            metadata: Optional metadata for each vector
        """
        if not vectors:
            return
            
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # For IVF index, need to train first
        if self.index_type == "ivf" and not self.is_trained:
            self.index.train(vectors_array)
            self.is_trained = True
        
        # Add vectors to index
        self.index.add(vectors_array)
        
        # Add metadata
        if metadata:
            self.metadata.extend(metadata)
        else:
            # Create default metadata
            start_idx = len(self.metadata)
            for i in range(len(vectors)):
                self.metadata.append({
                    "id": start_idx + i,
                    "chunk_index": i,
                    "vector_index": start_idx + i
                })
        
        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
    
    def add_vectors_with_texts(
        self, 
        vectors: List[List[float]], 
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """Add vectors with their original texts. Returns assigned text IDs.
        
        This is the preferred method for adding vectors when you need to retrieve
        the original text after search. The texts are stored in a StringStore
        and linked via text_id in the metadata.
        
        Args:
            vectors: List of embedding vectors
            texts: List of original text strings (must match vectors length)
            metadata: Optional metadata for each vector (text_id will be added)
            
        Returns:
            List of assigned text IDs
            
        Raises:
            ValueError: If vectors and texts have different lengths
        """
        if not vectors or len(vectors) != len(texts):
            raise ValueError("vectors and texts must have same non-zero length")
        
        # Store texts and get IDs
        text_ids = self.string_store.add_batch(texts)
        
        # Build metadata with text IDs
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]
        elif len(metadata) < len(vectors):
            # Extend metadata if shorter than vectors
            metadata = list(metadata) + [{} for _ in range(len(vectors) - len(metadata))]
        
        for i, (text_id, text) in enumerate(zip(text_ids, texts)):
            metadata[i]["text_id"] = text_id
            metadata[i]["text_preview"] = text[:100] if len(text) > 100 else text
        
        # Add to FAISS
        self.add_vectors(vectors, metadata)
        return text_ids
    
    def get_text_by_id(self, id_: int) -> Optional[str]:
        """Retrieve original text by ID.
        
        Args:
            id_: The text ID (from metadata["text_id"])
            
        Returns:
            The original text string, or None if not found
        """
        return self.string_store.get(id_)
    
    def get_texts_by_ids(self, ids: List[int]) -> List[Optional[str]]:
        """Retrieve multiple original texts by IDs.
        
        Args:
            ids: List of text IDs
            
        Returns:
            List of original text strings (None for any not found)
        """
        return self.string_store.get_batch(ids)
    
    def search(self, query_vector: List[float], k: int = 10) -> Tuple[List[float], List[int], List[Dict[str, Any]]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if self.index.ntotal == 0:
            return [], [], []
        
        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Get metadata for returned indices
        metadata = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                metadata.append(self.metadata[idx])
            else:
                metadata.append({"id": idx, "error": "metadata_not_found"})
        
        return distances[0].tolist(), indices[0].tolist(), metadata
    
    def search_with_texts(
        self, query_vector: List[float], k: int = 10
    ) -> Tuple[List[float], List[int], List[Optional[str]], List[Dict[str, Any]]]:
        """Search for similar vectors and return original texts.
        
        This extends the basic search() method by also retrieving the original
        text strings from the StringStore for each result.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, texts, metadata)
            - distances: Similarity scores
            - indices: FAISS vector indices
            - texts: Original text strings (None if not in string store)
            - metadata: Metadata dicts for each result
        """
        distances, indices, metadata = self.search(query_vector, k)
        
        # Fetch original texts from string store
        text_ids = [m.get("text_id") for m in metadata]
        texts = []
        for text_id in text_ids:
            if text_id is not None:
                texts.append(self.string_store.get(text_id))
            else:
                texts.append(None)
        
        return distances, indices, texts, metadata
    
    def get_vector_count(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal
    
    def save(self, directory: Union[str, Path], name: str = "index") -> Path:
        """Save index and metadata to directory.

        Args:
            directory: Directory to write the index into.
            name: Base name for the FAISS file, i.e. ``<name>.faiss``. Use the
                corpus name so indexes are identifiable and never collide on a
                single ``index.faiss``.

        Returns:
            Path to the written ``<name>.faiss`` file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index, named after the corpus it was built from
        index_path = directory / f"{name}.faiss"
        
        # Handle GPU indices - convert to CPU before saving
        if 'Gpu' in type(self.index).__name__:
            # This is a GPU index, we need to create a CPU version for saving
            # First, get the vectors from the GPU index
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            
            # Create a new CPU index of the same type
            if 'FlatIP' in type(self.index).__name__:
                cpu_index = faiss.IndexFlatIP(self.dimension)
            elif 'IVF' in type(self.index).__name__:
                quantizer = faiss.IndexFlatIP(self.dimension)
                cpu_index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                if self.is_trained:
                    cpu_index.train(vectors)
            elif 'HNSW' in type(self.index).__name__:
                cpu_index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                # Fallback to flat index
                cpu_index = faiss.IndexFlatIP(self.dimension)
            
            # Add vectors to CPU index
            cpu_index.add(vectors)
            
            # Save the CPU index
            faiss.write_index(cpu_index, str(index_path))
        else:
            # This is already a CPU index
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = directory / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save string store if it has entries
        if len(self.string_store) > 0:
            string_store_path = directory / "string_store.json"
            self.string_store.save(string_store_path)
        
        # Save index info
        info_path = directory / "index_info.json"
        info = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "vector_count": self.index.ntotal,
            "is_trained": self.is_trained,
            "has_string_store": len(self.string_store) > 0
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Saved index to {index_path}")
        return index_path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "FAISSIndex":
        """Load index and metadata.

        Args:
            path: Either a ``<name>.faiss`` file or a directory containing a
                single ``*.faiss`` file (``index.faiss`` is preferred when
                present, for backward compatibility).
        """
        path = Path(path)

        if path.is_file() and path.suffix == ".faiss":
            index_path = path
        elif (path / "index.faiss").exists():
            index_path = path / "index.faiss"
        else:
            # A single corpus directory, or a root containing per-corpus
            # subdirectories. Prefer a top-level .faiss, else the most recently
            # built nested one.
            faiss_files = sorted(path.glob("*.faiss"))
            if not faiss_files:
                faiss_files = sorted(
                    path.rglob("*.faiss"), key=lambda p: p.stat().st_mtime, reverse=True
                )
            if not faiss_files:
                raise FileNotFoundError(f"No .faiss file found in: {path}")
            index_path = faiss_files[0]

        # Companion files live alongside the .faiss file
        directory = index_path.parent
        metadata_path = directory / "metadata.json"
        info_path = directory / "index_info.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load index info
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Create instance
        instance = cls(dimension=info["dimension"], index_type=info["index_type"])
        instance.is_trained = info["is_trained"]
        
        # Load FAISS index
        instance.index = faiss.read_index(str(index_path))
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                instance.metadata = json.load(f)
        
        # Load string store if available
        string_store_path = directory / "string_store.json"
        if string_store_path.exists():
            instance.string_store = StringStore.load(string_store_path)
        
        logger.info(f"Loaded index from {directory}")
        return instance
    
    def clear(self) -> None:
        """Clear all vectors, metadata, and string store."""
        self.index = self._create_index()
        self.metadata = []
        self.is_trained = False
        self.string_store = StringStore()
        logger.info("Cleared index")


def build_index(vectors: List[List[float]], dimension: int = 384) -> FAISSIndex:
    """Build a FAISS index from vectors."""
    if faiss is None:
        raise RuntimeError("FAISS not available")
    
    index = FAISSIndex(dimension=dimension)
    index.add_vectors(vectors)
    return index


def build_index_from_text(lines: List[str], index_path: Path, model_name: str) -> None:
    """Build a FAISS index from text lines using sentence transformers or OpenAI API.
    
    Args:
        lines: List of text lines to embed
        index_path: Path to save the index
        model_name: Name of the embedding model to use
    """
    if faiss is None:
        raise RuntimeError("FAISS not available")
    
    # Check if this is an OpenAI model
    openai_models = {"text-embedding-3-large", "text-embedding-3-small", "openai-small", "openai-large"}
    
    if model_name in openai_models:
        # Use the linter's load_model function which has OpenAI API support
        try:
            import sys
            import pathlib
            
            # Add the sente packages to the path
            sente_path = pathlib.Path(__file__).parent.parent.parent / "sente" / "sente" / "packages" / "sentesdk"
            if sente_path.exists():
                sys.path.insert(0, str(sente_path))
                from sentesdk.linter import load_model
                
                # Load model using the linter's function (handles OpenAI API)
                model = load_model(model_name=model_name)
                
                # Generate embeddings
                embeddings = model.encode(lines, normalize_embeddings=True)
                
            else:
                raise RuntimeError("sente packages not found")
                
        except ImportError as e:
            logger.warning(f"Could not import sente linter: {e}")
            raise RuntimeError(f"OpenAI model {model_name} requires sente linter for API access")
            
    else:
        # Use sentence-transformers for Hugging Face models
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers not available")
        
        # Load model
        from shared_utils.embeddings import resolve_device
        model = SentenceTransformer(model_name, device=resolve_device())
        
        # Generate embeddings
        embeddings = model.encode(lines, normalize_embeddings=True)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype=np.float32))
    
    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "embedding_dimension": embeddings.shape[1],
        "num_vectors": len(lines),
        "build_timestamp": time.time()
    }
    
    metadata_path = index_path.parent / "index_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Built index with {len(lines)} vectors using model {model_name}")


def load_index(directory: Union[str, Path]) -> FAISSIndex:
    """Load a FAISS index from directory."""
    return FAISSIndex.load(directory)
