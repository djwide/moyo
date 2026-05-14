"""Bridge for GUI applications to input data and build FAISS indexes."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import time

from shared_utils import embed, get_embedding_model, chunk_text, chunk_lines, FAISSIndex, ensure_directory
from .validators import validate_text, validate_file_path, validate_file_content, get_file_info
from .loaders import load_file_by_type, get_supported_extensions

logger = logging.getLogger(__name__)


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
    output_dir: str = "indexes/private"
    
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
    
    def process_text(self, text: str, source_name: str = "text_input") -> ProcessingResult:
        """Process text input and build/update index.
        
        Args:
            text: Text to process
            source_name: Name for the source (for metadata)
            
        Returns:
            ProcessingResult with details
        """
        start_time = time.time()
        result = ProcessingResult(success=False, message="")
        
        try:
            # Validate text
            is_valid, error = validate_text(text)
            if not is_valid:
                result.message = f"Text validation failed: {error}"
                return result
            
            # Chunk the text
            chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
            if not chunks:
                result.message = "No valid chunks created from text"
                return result
            
            # Create metadata for chunks
            metadata = []
            for i, chunk in enumerate(chunks):
                metadata.append({
                    "source": source_name,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                })
            
            # Embed chunks
            embeddings = embed(chunks, self.config.embedding_model, self.config.batch_size)
            
            # Create or update index
            if self.current_index is None:
                self.current_index = FAISSIndex(
                    dimension=len(embeddings[0]) if embeddings else 384,
                    index_type=self.config.index_type
                )
            
            # Add to index
            self.current_index.add_vectors(embeddings, metadata)
            
            # Update stats
            self.processing_stats["total_chunks_created"] += len(chunks)
            self.processing_stats["total_vectors_created"] += len(embeddings)
            
            # Save index if requested
            index_path = None
            if self.config.save_index and self.current_index:
                index_path = self._save_index()
            
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
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a single file and build/update index.
        
        Args:
            file_path: Path to file to process
            
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
            result = self.process_text(text, source_name)
            
            # Update file-specific stats
            if result.success:
                self.processing_stats["total_files_processed"] += 1
                result.processing_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            result.message = f"Error processing file: {str(e)}"
            result.errors.append(str(e))
        
        return result
    
    def process_files(self, file_paths: List[Union[str, Path]]) -> List[ProcessingResult]:
        """Process multiple files and build/update index.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ProcessingResult for each file
        """
        results = []
        
        for file_path in file_paths:
            result = self.process_file(file_path)
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
    
    def _save_index(self) -> Optional[str]:
        """Save the current index to disk.
        
        Returns:
            Path where index was saved, or None if failed
        """
        if self.current_index is None:
            return None
        
        try:
            output_dir = Path(self.config.output_dir)
            ensure_directory(output_dir)
            
            self.current_index.save(output_dir)
            return str(output_dir)
            
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
    source_name: str = "text_input"
) -> ProcessingResult:
    """Convenience function to process text and build index."""
    bridge = GUIBridge(config)
    return bridge.process_text(text, source_name)


def process_files_and_build_index(
    file_paths: List[Union[str, Path]],
    config: ProcessingConfig = None
) -> List[ProcessingResult]:
    """Convenience function to process files and build index."""
    bridge = GUIBridge(config)
    return bridge.process_files(file_paths)
