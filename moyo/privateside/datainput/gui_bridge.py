"""Bridge for GUI applications to input data and build FAISS indexes."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import time

from shared_utils import (
    embed,
    get_embedding_model,
    chunk_text,
    chunk_text_multi_granularity,
    chunk_lines,
    FAISSIndex,
    ensure_directory,
)
from .validators import validate_text, validate_file_path, validate_file_content, get_file_info
from .loaders import load_file_by_type, get_supported_extensions

logger = logging.getLogger(__name__)

# All private-side FAISS indexes live here, one subdirectory per corpus.
PRIVATE_INDEX_ROOT = "indexes/private"


def slugify_corpus_name(name: str) -> str:
    """Turn an arbitrary source label into a safe corpus/index name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return slug or f"corpus_{datetime.now():%Y%m%d_%H%M%S}"


@dataclass
class ProcessingResult:
    """Result of processing text or files."""
    success: bool
    message: str
    chunks_created: int = 0
    vectors_created: int = 0
    index_path: Optional[str] = None
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProcessingConfig:
    """Configuration for text processing and indexing."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    save_index: bool = True
    output_dir: str = PRIVATE_INDEX_ROOT  # root for per-corpus index subdirectories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GUIBridge:
    """Bridge class for GUI applications to process data and build indexes."""
    
    def __init__(self, config: ProcessingConfig = None):
        """Initialize the GUI bridge.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.current_index: Optional[FAISSIndex] = None
        self.processing_stats = {
            "total_files_processed": 0,
            "total_chunks_created": 0,
            "total_vectors_created": 0,
            "total_processing_time": 0.0
        }
    
    def process_text(
        self,
        text: str,
        source_name: str = "text_input",
        index_name: Optional[str] = None,
    ) -> ProcessingResult:
        """Process text input and build/update index.
        
        Args:
            text: Text to process
            source_name: Name for the source (for metadata)
            index_name: Corpus name used for the on-disk index files. Defaults
                to a slug of ``source_name`` (or a timestamp for raw text).
            
        Returns:
            ProcessingResult with details
        """
        corpus_name = slugify_corpus_name(index_name or source_name)
        start_time = time.time()
        result = ProcessingResult(success=False, message="")
        
        try:
            # Validate text
            is_valid, error = validate_text(text)
            if not is_valid:
                result.message = f"Text validation failed: {error}"
                return result
            
            # Chunk at multiple granularities so list items, bullets, and short
            # idea lines become their own vectors (not just diluted section text).
            granular = chunk_text_multi_granularity(
                text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            if not granular:
                result.message = "No valid chunks created from text"
                return result

            # Map local indexes → absolute vector positions for parent linking
            # once this batch is appended to the current index.
            base_offset = (
                self.current_index.get_vector_count() if self.current_index else 0
            )
            index_to_pos = {gc.index: base_offset + i for i, gc in enumerate(granular)}

            chunks = [gc.text for gc in granular]
            metadata = []
            for gc in granular:
                parent_pos = (
                    index_to_pos.get(gc.parent_index)
                    if gc.parent_index is not None
                    else None
                )
                metadata.append({
                    "source": source_name,
                    "chunk_index": gc.index,
                    "chunk_size": len(gc.text),
                    "level": gc.level,
                    "parent_index": gc.parent_index,
                    "parent_vector": parent_pos,
                    "text": gc.text,
                    "text_preview": gc.text[:100] + "..." if len(gc.text) > 100 else gc.text,
                })
            
            # Embed chunks
            embeddings = embed(chunks, self.config.embedding_model, self.config.batch_size)
            
            # Create or update index
            if self.current_index is None:
                self.current_index = FAISSIndex(
                    dimension=len(embeddings[0]) if embeddings else 384,
                    index_type=self.config.index_type
                )
            
            # Store full text via the string store + metadata so search can
            # always surface the matched content.
            self.current_index.add_vectors_with_texts(embeddings, chunks, metadata)
            
            # Update stats
            self.processing_stats["total_chunks_created"] += len(chunks)
            self.processing_stats["total_vectors_created"] += len(embeddings)
            
            # Save index if requested
            index_path = None
            if self.config.save_index and self.current_index:
                index_path = self._save_index(corpus_name)
            
            processing_time = time.time() - start_time
            self.processing_stats["total_processing_time"] += processing_time
            
            result.success = True
            result.message = f"Successfully processed text: {len(chunks)} chunks created"
            result.chunks_created = len(chunks)
            result.vectors_created = len(embeddings)
            result.index_path = index_path
            result.processing_time = processing_time
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            result.message = f"Error processing text: {str(e)}"
            result.errors.append(str(e))
        
        return result
    
    def process_file(
        self, file_path: Union[str, Path], index_name: Optional[str] = None
    ) -> ProcessingResult:
        """Process a single file and build/update index.
        
        Args:
            file_path: Path to file to process
            index_name: Corpus name for the on-disk index. Defaults to the
                file's stem so the index is named after its corpus.
            
        Returns:
            ProcessingResult with details
        """
        file_path = Path(file_path)
        start_time = time.time()
        result = ProcessingResult(success=False, message="")
        
        try:
            # Validate file
            is_valid, error = validate_file_path(file_path, get_supported_extensions())
            if not is_valid:
                result.message = f"File validation failed: {error}"
                return result
            
            # Load file content
            is_valid, error, content = validate_file_content(file_path)
            if not is_valid:
                result.message = f"File content validation failed: {error}"
                return result
            
            # Load and convert to text
            text = load_file_by_type(file_path)
            
            # Process as text with file name as source
            source_name = f"file:{file_path.name}"
            corpus_name = index_name or file_path.stem
            result = self.process_text(text, source_name, index_name=corpus_name)
            
            # Update file-specific stats
            if result.success:
                self.processing_stats["total_files_processed"] += 1
                result.processing_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            result.message = f"Error processing file: {str(e)}"
            result.errors.append(str(e))
        
        return result
    
    def process_files(
        self, file_paths: List[Union[str, Path]], index_name: Optional[str] = None
    ) -> List[ProcessingResult]:
        """Process multiple files into a single combined index.
        
        All files accumulate into one index. When ``index_name`` is omitted the
        corpus is named after the files' common parent directory (or the first
        file's stem), so the combined index gets one meaningful ``.faiss`` name.
        
        Args:
            file_paths: List of file paths to process
            index_name: Corpus name for the combined on-disk index
            
        Returns:
            List of ProcessingResult for each file
        """
        results = []
        corpus_name = index_name or self._derive_corpus_name(file_paths)
        
        for file_path in file_paths:
            result = self.process_file(file_path, index_name=corpus_name)
            results.append(result)
            
            # Log progress
            if result.success:
                logger.info(f"Processed {file_path}: {result.chunks_created} chunks")
            else:
                logger.warning(f"Failed to process {file_path}: {result.message}")
        
        return results
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index.
        
        Returns:
            Dictionary with index information
        """
        if self.current_index is None:
            return {
                "has_index": False,
                "vector_count": 0,
                "dimension": 0,
                "index_type": None
            }
        
        return {
            "has_index": True,
            "vector_count": self.current_index.get_vector_count(),
            "dimension": self.current_index.dimension,
            "index_type": self.current_index.index_type,
            "is_trained": self.current_index.is_trained
        }
    
    def search_index(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Search the current index with a text query.
        
        Args:
            query: Text query to search for
            k: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        if self.current_index is None:
            return {
                "success": False,
                "message": "No index available",
                "results": []
            }
        
        try:
            # Embed the query
            query_embeddings = embed([query], self.config.embedding_model)
            if not query_embeddings:
                return {
                    "success": False,
                    "message": "Failed to embed query",
                    "results": []
                }
            
            # Search the index
            distances, indices, metadata = self.current_index.search(query_embeddings[0], k)
            
            # Format results
            results = []
            for i, (distance, idx, meta) in enumerate(zip(distances, indices, metadata)):
                results.append({
                    "rank": i + 1,
                    "distance": float(distance),
                    "index": int(idx),
                    "metadata": meta
                })
            
            return {
                "success": True,
                "message": f"Found {len(results)} results",
                "query": query,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return {
                "success": False,
                "message": f"Search error: {str(e)}",
                "results": []
            }
    
    def clear_index(self) -> ProcessingResult:
        """Clear the current index.
        
        Returns:
            ProcessingResult with details
        """
        result = ProcessingResult(success=False, message="")
        
        try:
            if self.current_index:
                self.current_index.clear()
                result.success = True
                result.message = "Index cleared successfully"
            else:
                result.message = "No index to clear"
                
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            result.message = f"Error clearing index: {str(e)}"
            result.errors.append(str(e))
        
        return result
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            **self.processing_stats,
            "index_info": self.get_index_info()
        }
    
    @staticmethod
    def _derive_corpus_name(file_paths: List[Union[str, Path]]) -> str:
        """Pick a corpus name for a group of files."""
        paths = [Path(p) for p in file_paths]
        if not paths:
            return slugify_corpus_name("corpus")
        parents = {p.resolve().parent for p in paths}
        if len(paths) > 1 and len(parents) == 1:
            return slugify_corpus_name(paths[0].resolve().parent.name)
        return slugify_corpus_name(paths[0].stem)

    def _save_index(self, corpus_name: str) -> Optional[str]:
        """Save the current index to disk, named after its corpus.
        
        The index is written to ``<output_dir>/<corpus_name>/<corpus_name>.faiss``
        (plus companion metadata), keeping every corpus in its own subdirectory
        under the single index root.
        
        Returns:
            Path to the saved ``.faiss`` file, or None if failed
        """
        if self.current_index is None:
            return None
        
        try:
            index_dir = Path(self.config.output_dir) / corpus_name
            ensure_directory(index_dir)
            
            index_path = self.current_index.save(index_dir, name=corpus_name)
            return str(index_path)
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return None
    
    def load_index(self, index_path: Union[str, Path]) -> ProcessingResult:
        """Load an existing index from disk.
        
        Args:
            index_path: Path to index directory
            
        Returns:
            ProcessingResult with details
        """
        result = ProcessingResult(success=False, message="")
        
        try:
            index_path = Path(index_path)
            self.current_index = FAISSIndex.load(index_path)
            
            result.success = True
            result.message = f"Index loaded from {index_path}"
            result.index_path = str(index_path)
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            result.message = f"Error loading index: {str(e)}"
            result.errors.append(str(e))
        
        return result


def launch_gui() -> None:
    """Launch the sente GUI when available."""
    try:
        from .sente_adapters import get_index_utils
        index_utils = get_index_utils()
        if index_utils is None:
            raise RuntimeError("sente GUI not available")
        # Placeholder: would invoke GUI components
        logger.info("sente GUI integration not yet implemented")
    except Exception as e:
        logger.error(f"Failed to launch GUI: {e}")
        raise RuntimeError(f"GUI not available: {e}")


# Convenience functions for direct use
def process_text_and_build_index(
    text: str,
    config: ProcessingConfig = None,
    source_name: str = "text_input",
    index_name: Optional[str] = None,
) -> ProcessingResult:
    """Convenience function to process text and build index."""
    bridge = GUIBridge(config)
    return bridge.process_text(text, source_name, index_name=index_name)


def process_files_and_build_index(
    file_paths: List[Union[str, Path]],
    config: ProcessingConfig = None,
    index_name: Optional[str] = None,
) -> List[ProcessingResult]:
    """Convenience function to process files and build index."""
    bridge = GUIBridge(config)
    return bridge.process_files(file_paths, index_name=index_name)
