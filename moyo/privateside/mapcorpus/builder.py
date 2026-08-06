"""Corpus builder for creating FAISS indexes from processed data."""

import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

from shared_utils import embed, get_embedding_model, chunk_text, FAISSIndex, ensure_directory, generate_id

from .schema import (
    DocumentChunk, MappedDocument, CorpusConfig, CorpusBuildResult, 
    CorpusInfo, SearchResult
)
from shared_utils import deduplicate_texts, normalize_text, TextNormalizationConfig, DeduplicationConfig
from .normalize import normalize_chunks, filter_chunks_by_length, get_text_statistics
from .dedupe import remove_duplicates, get_duplicate_statistics

logger = logging.getLogger(__name__)


class CorpusBuilder:
    """Main corpus builder class for creating FAISS indexes."""
    
    def __init__(self, config: Optional[CorpusConfig] = None):
        """Initialize the corpus builder.
        
        Args:
            config: Configuration for corpus building
        """
        self.config = config or CorpusConfig()
        self.index: Optional[FAISSIndex] = None
        self.chunks: List[DocumentChunk] = []
        self.corpus_id: str = generate_id("corpus")
        
    def add_text(self, text: str, source_name: str = "text_input", 
                 metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Add text content to the corpus.
        
        Args:
            text: Text content to add
            source_name: Name of the source
            metadata: Additional metadata
            
        Returns:
            List of created chunks
        """
        if not text or not text.strip():
            return []
        
        # Chunk the text
        text_chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        
        # Create DocumentChunk objects
        chunks = []
        for i, chunk_text_content in enumerate(text_chunks):
            chunk_id = generate_id("chunk")
            chunk = DocumentChunk(
                id=chunk_id,
                text=chunk_text_content,
                chunk_index=len(self.chunks) + i,
                source_document=source_name,
                chunk_size=len(chunk_text_content),
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks from '{source_name}'")
        return chunks
    
    def add_file(self, file_path: Union[str, Path], 
                 metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Add file content to the corpus.
        
        Args:
            file_path: Path to the file
            metadata: Additional metadata
            
        Returns:
            List of created chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Add with file metadata
            file_metadata = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            return self.add_text(content, source_name=f"file:{file_path.name}", 
                               metadata=file_metadata)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
    
    def add_files(self, file_paths: List[Union[str, Path]], 
                  metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Add multiple files to the corpus.
        
        Args:
            file_paths: List of file paths
            metadata: Additional metadata for all files
            
        Returns:
            List of all created chunks
        """
        all_chunks = []
        
        for file_path in file_paths:
            chunks = self.add_file(file_path, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Added {len(all_chunks)} chunks from {len(file_paths)} files")
        return all_chunks
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add pre-existing chunks to the corpus.
        
        Args:
            chunks: List of document chunks
        """
        # Update chunk indices
        start_index = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = start_index + i
        
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} existing chunks")
    
    def normalize_corpus(self) -> None:
        """Apply normalization to all chunks in the corpus."""
        if not self.chunks:
            return
        
        logger.info("Applying text normalization...")
        
        # Apply normalization
        self.chunks = normalize_chunks(self.chunks, {
            'lowercase': True,
            'normalize_unicode': True,
            'normalize_whitespace': True,
            'remove_urls': True,
            'remove_emails': True,
            'normalize_punctuation': True,
            'keep_punctuation': True
        })
        
        # Filter by length
        self.chunks = filter_chunks_by_length(
            self.chunks, 
            self.config.min_chunk_length, 
            self.config.max_chunk_length
        )
        
        logger.info(f"Normalization complete: {len(self.chunks)} chunks remaining")
    
    def normalize_chunks(self) -> int:
        """Apply text normalization to chunks.
        
        Returns:
            Number of chunks normalized
        """
        if not self.chunks:
            return 0
        
        logger.info("Applying text normalization to chunks...")
        
        # Apply normalization
        original_count = len(self.chunks)
        self.chunks = normalize_chunks(self.chunks, {
            'lowercase': True,
            'normalize_unicode': True,
            'normalize_whitespace': True,
            'remove_urls': True,
            'remove_emails': True,
            'normalize_punctuation': True,
            'keep_punctuation': True
        })
        
        # Filter by length
        self.chunks = filter_chunks_by_length(
            self.chunks, 
            self.config.min_chunk_length, 
            self.config.max_chunk_length
        )
        
        normalized_count = original_count - len(self.chunks)
        logger.info(f"Normalized {normalized_count} chunks")
        return normalized_count
    
    def deduplicate_corpus(self) -> int:
        """Remove duplicates from the corpus.
        
        Returns:
            Number of duplicates removed
        """
        if not self.chunks:
            return 0
        
        logger.info("Removing duplicates...")
        
        # Remove duplicates
        self.chunks, duplicates_removed = remove_duplicates(
            self.chunks,
            exact_duplicates=self.config.deduplication_enabled,
            similar_duplicates=False,  # Could be configurable
            similarity_threshold=0.9
        )
        
        logger.info(f"Deduplication complete: {duplicates_removed} duplicates removed")
        return duplicates_removed
    
    def build_index(self) -> CorpusBuildResult:
        """Build the FAISS index from the corpus.
        
        Returns:
            CorpusBuildResult with build information
        """
        start_time = time.time()
        result = CorpusBuildResult(success=False, message="")
        
        try:
            if not self.chunks:
                result.message = "No chunks available for indexing"
                return result
            
            logger.info(f"Building index from {len(self.chunks)} chunks...")
            
            # Extract text for embedding
            texts = [chunk.text for chunk in self.chunks]
            
            # Generate embeddings
            embeddings = embed(
                texts, 
                self.config.embedding_model, 
                self.config.batch_size
            )
            
            if not embeddings:
                result.message = "Failed to generate embeddings"
                return result
            
            # Create FAISS index
            self.index = FAISSIndex(
                dimension=len(embeddings[0]) if embeddings else 384,
                index_type=self.config.index_type
            )
            
            # Prepare metadata for index
            metadata = []
            for i, chunk in enumerate(self.chunks):
                chunk_metadata = {
                    "id": chunk.id,
                    "source": chunk.source_document,
                    "chunk_index": chunk.chunk_index,
                    "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                    **chunk.metadata
                }
                metadata.append(chunk_metadata)
            
            # Add vectors to index
            self.index.add_vectors(embeddings, metadata)
            
            # Save index if requested
            index_path = None
            if self.config.save_metadata:
                index_path = self._save_corpus()
            
            processing_time = time.time() - start_time
            
            result.success = True
            result.message = f"Successfully built index with {len(self.chunks)} chunks"
            result.documents_processed = len(set(chunk.source_document for chunk in self.chunks))
            result.chunks_created = len(self.chunks)
            result.vectors_created = len(embeddings)
            result.processing_time = processing_time
            result.index_path = index_path
            
            logger.info(f"Index built successfully: {len(embeddings)} vectors in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            result.message = f"Error building index: {str(e)}"
            result.errors.append(str(e))
        
        return result
    
    def search(self, query: str, k: int = 10) -> SearchResult:
        """Search the corpus.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            SearchResult with search information
        """
        start_time = time.time()
        
        if not self.index:
            return SearchResult(
                query=query,
                results=[],
                total_results=0,
                search_time=0.0,
                metadata={"error": "No index available"}
            )
        
        try:
            # Embed the query
            query_embeddings = embed([query], self.config.embedding_model)
            if not query_embeddings:
                return SearchResult(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    metadata={"error": "Failed to embed query"}
                )
            
            # Search the index
            distances, indices, metadata = self.index.search(query_embeddings[0], k)
            
            # Format results
            results = []
            for i, (distance, idx, meta) in enumerate(zip(distances, indices, metadata)):
                results.append({
                    "rank": i + 1,
                    "distance": float(distance),
                    "index": int(idx),
                    "metadata": meta
                })
            
            search_time = time.time() - start_time
            
            return SearchResult(
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time
            )
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return SearchResult(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def get_corpus_info(self) -> CorpusInfo:
        """Get information about the corpus.
        
        Returns:
            CorpusInfo with corpus details
        """
        vector_count = self.index.get_vector_count() if self.index else 0
        dimension = self.index.dimension if self.index else 0
        
        return CorpusInfo(
            corpus_id=self.corpus_id,
            created_at=datetime.now(),
            document_count=len(set(chunk.source_document for chunk in self.chunks)),
            chunk_count=len(self.chunks),
            vector_count=vector_count,
            embedding_dimension=dimension,
            index_type=self.config.index_type,
            embedding_model=self.config.embedding_model,
            metadata={
                "config": self.config.dict(),
                "statistics": self._get_statistics()
            }
        )
    
    def _corpus_name(self) -> str:
        """Derive a corpus name for on-disk naming from the source documents."""
        sources = {chunk.source_document for chunk in self.chunks}
        stems = {Path(s.split("file:", 1)[-1]).stem for s in sources if s}
        if len(stems) == 1:
            raw = next(iter(stems))
        else:
            raw = self.corpus_id
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip()).strip("._-")
        return slug or self.corpus_id

    def _save_corpus(self) -> Optional[str]:
        """Save the corpus and index to disk.
        
        Each corpus is written to its own subdirectory under the index root,
        with the FAISS file named after the corpus (never a shared index.faiss).
        
        Returns:
            Path to the saved ``.faiss`` file, or None if failed
        """
        if not self.index:
            return None
        
        try:
            corpus_name = self._corpus_name()
            output_dir = Path(self.config.output_directory) / corpus_name
            ensure_directory(output_dir)
            
            # Save FAISS index, named after the corpus
            index_path = self.index.save(output_dir, name=corpus_name)
            
            # Save chunks if requested
            if self.config.save_chunks:
                chunks_file = output_dir / "chunks.json"
                chunks_data = [chunk.dict() for chunk in self.chunks]
                with open(chunks_file, 'w') as f:
                    json.dump(chunks_data, f, indent=2)
            
            # Save corpus info
            info_file = output_dir / "corpus_info.json"
            corpus_info = self.get_corpus_info()
            with open(info_file, 'w') as f:
                json.dump(corpus_info.dict(), f, indent=2, default=str)
            
            logger.info(f"Corpus saved to {output_dir}")
            return str(index_path)
            
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")
            return None
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the corpus."""
        stats = {
            "text_statistics": get_text_statistics(self.chunks),
            "duplicate_statistics": get_duplicate_statistics(self.chunks),
            "source_distribution": {}
        }
        
        # Source distribution
        source_counts = {}
        for chunk in self.chunks:
            source = chunk.source_document
            source_counts[source] = source_counts.get(source, 0) + 1
        
        stats["source_distribution"] = source_counts
        
        return stats
    
    def clear(self) -> None:
        """Clear the corpus and index."""
        self.chunks = []
        self.index = None
        self.corpus_id = generate_id("corpus")
        logger.info("Corpus cleared")
    
    @classmethod
    def load_from_path(cls, index_path: str) -> Optional['CorpusBuilder']:
        """Load a corpus from disk.
        
        Args:
            index_path: Path to the index directory
            
        Returns:
            CorpusBuilder with loaded corpus or None
        """
        try:
            index_dir = Path(index_path)
            
            # Load corpus info
            info_path = index_dir / "corpus_info.json"
            if not info_path.exists():
                logger.error(f"Corpus info file not found: {info_path}")
                return None
                
            with open(info_path, 'r', encoding='utf-8') as f:
                corpus_info = json.load(f)
            
            # Create config from corpus info
            config_data = corpus_info.get("metadata", {}).get("config", {})
            config = CorpusConfig(**config_data)
            
            # Create builder
            builder = cls(config)
            builder.corpus_id = corpus_info.get("corpus_id", generate_id("corpus"))
            
            # Load chunks if available
            chunks_path = index_dir / "chunks.json"
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                builder.chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
            
            # Load FAISS index if available (named after the corpus)
            if list(index_dir.glob("*.faiss")):
                builder.index = FAISSIndex.load(str(index_dir))
            
            logger.info(f"Loaded corpus from {index_path}")
            return builder
            
        except Exception as e:
            logger.error(f"Error loading corpus from {index_path}: {e}")
            return None


def build_corpus_from_gui_bridge(gui_bridge_data: List[Dict[str, Any]], 
                                config: Optional[CorpusConfig] = None) -> CorpusBuildResult:
    """Build corpus from GUI bridge data.
    
    Args:
        gui_bridge_data: Data from GUI bridge processing
        config: Corpus configuration
        
    Returns:
        CorpusBuildResult with build information
    """
    builder = CorpusBuilder(config)
    
    # Process GUI bridge data
    for item in gui_bridge_data:
        if "text" in item:
            builder.add_text(item["text"], item.get("source", "gui_input"))
        elif "file_path" in item:
            builder.add_file(item["file_path"])
    
    # Build the corpus
    return builder.build_index()


def build_corpus_from_files(file_paths: List[Union[str, Path]], 
                           config: Optional[CorpusConfig] = None) -> CorpusBuildResult:
    """Build corpus from file paths.
    
    Args:
        file_paths: List of file paths
        config: Corpus configuration
        
    Returns:
        CorpusBuildResult with build information
    """
    builder = CorpusBuilder(config)
    
    # Add files
    builder.add_files(file_paths)
    
    # Normalize and deduplicate
    builder.normalize_corpus()
    builder.deduplicate_corpus()
    
    # Build index
    return builder.build_index()


def build_corpus_from_texts(texts: List[str], 
                           config: Optional[CorpusConfig] = None) -> CorpusBuildResult:
    """Build corpus from text inputs.
    
    Args:
        texts: List of text inputs
        config: Corpus configuration
        
    Returns:
        CorpusBuildResult with build information
    """
    builder = CorpusBuilder(config)
    
    # Add texts
    for i, text in enumerate(texts):
        builder.add_text(text, f"text_{i}")
    
    # Normalize and deduplicate
    builder.normalize_corpus()
    builder.deduplicate_corpus()
    
    # Build index
    return builder.build_index()
