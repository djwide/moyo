"""Public index builder for creating FAISS indexes from crawled public sources."""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import pickle

import faiss
import numpy as np

from .schema import (
    IndexConfig, PublicChunk, PublicIndex, IndexBuildResult, 
    SearchResult, IndexType
)
from ..gatherpublicsources.schema import PublicSource, SourceType
from shared_utils import (
    embed,
    chunk_text,
    chunk_text_multi_granularity,
    FAISSIndex,
    ensure_directory,
    generate_id,
)
from shared_utils.chunking import keep_granular_chunk, resolve_chunk_max_tokens
from shared_utils.index_spec import spec_from_config

logger = logging.getLogger(__name__)


class PublicIndexBuilder:
    """Builder for creating FAISS indexes from public sources."""
    
    def __init__(self, config: IndexConfig):
        """Initialize the public index builder.
        
        Args:
            config: Index configuration
        """
        self.config = config
        self.chunks: List[PublicChunk] = []
        self.faiss_index: Optional[FAISSIndex] = None
        
    def add_sources(self, sources: List[PublicSource]) -> int:
        """Add public sources to the index builder.
        
        Args:
            sources: List of public sources to add
            
        Returns:
            Number of sources processed
        """
        processed_count = 0
        
        for source in sources:
            try:
                # Apply source filtering
                if not self._should_include_source(source):
                    continue
                
                # Process the source
                chunks_added = self._process_source(source)
                processed_count += 1
                
                logger.debug(f"Processed source {source.id}: {chunks_added} chunks")
                
            except Exception as e:
                logger.error(f"Error processing source {source.id}: {e}")
                continue
        
        logger.info(f"Added {processed_count} sources with {len(self.chunks)} total chunks")
        return processed_count
    
    def _should_include_source(self, source: PublicSource) -> bool:
        """Check if a source should be included based on filters.
        
        Args:
            source: Source to check
            
        Returns:
            True if source should be included
        """
        # Check source type filter
        if self.config.source_types and source.source_type not in self.config.source_types:
            return False
        
        # Check date range filter
        if source.published_date:
            if self.config.date_from and source.published_date < self.config.date_from:
                return False
            if self.config.date_to and source.published_date > self.config.date_to:
                return False
        
        # Check organization filter
        if self.config.organizations and source.organization:
            if source.organization not in self.config.organizations:
                return False
        
        # Check relevance score filter
        if source.relevance_score is not None:
            if source.relevance_score < self.config.min_relevance_score:
                return False
        
        # Check confidence score filter
        if source.confidence_score is not None:
            if source.confidence_score < self.config.min_confidence_score:
                return False
        
        return True
    
    def _process_source(self, source: PublicSource) -> int:
        """Process a single source and create chunks.
        
        Args:
            source: Source to process
            
        Returns:
            Number of chunks created
        """
        chunks_created = 0
        
        # Chunk the content at multiple granularities (variable-size section
        # chunks plus their sentence sub-chunks) so short/paraphrased queries can
        # match a fine-grained unit instead of a diluted whole-chunk vector.
        max_tokens = resolve_chunk_max_tokens(
            self.config.embedding_model, getattr(self.config, "max_tokens", None)
        )
        self.config.max_tokens = max_tokens
        granular_chunks = chunk_text_multi_granularity(
            source.content,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            max_tokens=max_tokens,
        )
        
        source_metadata = {
            "source_title": source.title,
            "source_url": str(source.url) if source.url else None,
            "source_author": source.author,
            "source_organization": source.organization,
            "source_published_date": source.published_date.isoformat() if source.published_date else None,
            "source_relevance_score": source.relevance_score,
            "source_confidence_score": source.confidence_score,
            "source_tags": source.tags,
            **source.metadata
        }
        
        # Map each granular chunk's local index to the PublicChunk id we assign,
        # so sentence chunks can reference their parent section chunk.
        child_parents = {
            gc.parent_index for gc in granular_chunks if gc.parent_index is not None
        }
        index_to_id: Dict[int, str] = {}
        for gc in granular_chunks:
            if not keep_granular_chunk(
                gc.level,
                gc.text,
                min_section_chars=self.config.min_chunk_length,
                max_section_chars=self.config.max_chunk_length,
                has_finer_children=gc.index in child_parents,
                keep_short_atomic=False,
            ):
                continue
            
            # Positions are best-effort: sentence text is whitespace-normalized
            # and may not be a verbatim substring of the raw source.
            start_char = source.content.find(gc.text)
            end_char = start_char + len(gc.text) if start_char >= 0 else -1
            
            parent_id = index_to_id.get(gc.parent_index) if gc.parent_index is not None else None
            
            chunk = PublicChunk(
                id=generate_id(f"chunk_{source.id}_{gc.index}"),
                content=gc.text,
                source_id=source.id,
                source_type=source.source_type,
                chunk_index=gc.index,
                start_char=start_char,
                end_char=end_char,
                level=gc.level,
                parent_id=parent_id,
                metadata=dict(source_metadata),
            )
            
            index_to_id[gc.index] = chunk.id
            self.chunks.append(chunk)
            chunks_created += 1
        
        return chunks_created
    
    def normalize_chunks(self) -> int:
        """Apply text normalization to chunks.
        
        Returns:
            Number of chunks normalized
        """
        if not self.config.normalization_enabled:
            return 0
        
        from moyo.privateside.mapcorpus.normalize import apply_text_normalization
        
        normalized_count = 0
        
        for chunk in self.chunks:
            try:
                original_content = chunk.content
                normalized_content = apply_text_normalization(original_content)
                
                if normalized_content != original_content:
                    chunk.content = normalized_content
                    normalized_count += 1
                    
            except Exception as e:
                logger.error(f"Error normalizing chunk {chunk.id}: {e}")
                continue
        
        logger.info(f"Normalized {normalized_count} chunks")
        return normalized_count
    
    def deduplicate_chunks(self) -> int:
        """Remove duplicate chunks.
        
        Returns:
            Number of duplicates removed
        """
        if not self.config.deduplication_enabled:
            return 0
        
        # Simple deduplication by content
        try:
            seen_contents = set()
            unique_chunks = []
            
            for chunk in self.chunks:
                if chunk.content not in seen_contents:
                    seen_contents.add(chunk.content)
                    unique_chunks.append(chunk)
            
            duplicates_removed = len(self.chunks) - len(unique_chunks)
            self.chunks = unique_chunks
            
            logger.info(f"Removed {duplicates_removed} duplicate chunks")
            return duplicates_removed
            
        except Exception as e:
            logger.error(f"Error deduplicating chunks: {e}")
            return 0
    
    def build_index(self, name: str, description: str = "") -> IndexBuildResult:
        """Build the FAISS index from processed chunks.
        
        Args:
            name: Name for the index
            description: Description of the index
            
        Returns:
            IndexBuildResult with build information
        """
        start_time = time.time()
        result = IndexBuildResult(success=False, message="")
        
        try:
            logger.info(f"Building index '{name}' with {len(self.chunks)} chunks")
            
            if not self.chunks:
                result.message = "No chunks to index"
                return result
            
            # Create embeddings for chunks
            logger.info("Creating embeddings...")
            chunk_texts = [chunk.content for chunk in self.chunks]
            embeddings = embed(
                chunk_texts,
                model_name=self.config.embedding_model,
                batch_size=32,
                normalize=self.config.normalize_embeddings,
                device=getattr(self.config, "embedding_device", "auto"),
            )
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(self.chunks, embeddings):
                chunk.embedding = embedding
            
            # Create FAISS index
            logger.info("Creating FAISS index...")
            self.faiss_index = FAISSIndex(
                dimension=len(embeddings[0]),
                index_type=self.config.index_type.value
            )
            
            # Add vectors to index WITH their text and metadata so search
            # results carry the matched text and can link back to the parent
            # section chunk (previously add_vectors() stored no text at all).
            vectors = np.array(embeddings, dtype=np.float32)
            vector_metadata = [
                {
                    "text": chunk.content,
                    "level": chunk.level,
                    "parent_id": chunk.parent_id,
                    "chunk_id": chunk.id,
                    "source_id": chunk.source_id,
                    "source_document": chunk.metadata.get("source_title") or chunk.source_id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in self.chunks
            ]
            self.faiss_index.add_vectors_with_texts(
                embeddings,
                [chunk.content for chunk in self.chunks],
                vector_metadata,
            )
            
            # Create index metadata
            index_id = generate_id(f"public_index_{name}")
            public_index = PublicIndex(
                id=index_id,
                name=name,
                description=description,
                config=self.config,
                source_count=len(set(chunk.source_id for chunk in self.chunks)),
                chunk_count=len(self.chunks),
                vector_count=len(embeddings),
                index_size_bytes=vectors.nbytes,
                metadata={
                    "embedding_model": self.config.embedding_model,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "source_types": [st.value for st in set(chunk.source_type for chunk in self.chunks)],
                    "organizations": list(set(chunk.metadata.get("source_organization") for chunk in self.chunks if chunk.metadata.get("source_organization")))
                }
            )
            
            # Save index and metadata
            output_path = self._save_index(public_index)
            
            processing_time = time.time() - start_time
            
            result.success = True
            result.message = f"Successfully built index '{name}'"
            result.index_id = index_id
            result.index_path = output_path
            result.sources_processed = public_index.source_count
            result.chunks_created = public_index.chunk_count
            result.vectors_created = public_index.vector_count
            result.processing_time = processing_time
            
            logger.info(f"Index built successfully: {public_index.chunk_count} chunks, {public_index.vector_count} vectors in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            result.message = f"Error building index: {str(e)}"
            result.errors.append(str(e))
        
        return result
    
    def _save_index(self, public_index: PublicIndex) -> str:
        """Save the index and metadata to disk.
        
        Args:
            public_index: Index metadata to save
            
        Returns:
            Path where index was saved
        """
        try:
            # Create output directory
            output_dir = Path(self.config.output_directory)
            ensure_directory(output_dir)
            
            # Create index-specific directory
            index_dir = output_dir / public_index.id
            ensure_directory(index_dir)
            
            # Save FAISS index, named after the public index
            if self.faiss_index:
                self.faiss_index.save(
                    index_dir,
                    name=public_index.id,
                    extra_info=spec_from_config(self.config).to_dict(),
                )
            
            # Save chunks
            chunks_path = index_dir / "chunks.json"
            chunks_data = [chunk.dict() for chunk in self.chunks]
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, default=str)
            
            # Save index metadata
            metadata_path = index_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(public_index.dict(), f, indent=2, default=str)
            
            logger.info(f"Index saved to {index_dir}")
            return str(index_dir)
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return ""
    
    def search(self, query: str, k: int = 10) -> SearchResult:
        """Search the built index.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            SearchResult with search results
        """
        start_time = time.time()
        result = SearchResult(query=query)
        
        try:
            if not self.faiss_index or not self.chunks:
                result.message = "No index available for search"
                return result
            
            # Create query embedding
            query_embedding = embed(
                [query],
                model_name=self.config.embedding_model,
                normalize=self.config.normalize_embeddings,
                device=getattr(self.config, "embedding_device", "auto"),
            )[0]
            
            # Search index
            distances, indices = self.faiss_index.search([query_embedding], k)
            
            # Format results
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    result.results.append({
                        "rank": i + 1,
                        "distance": float(distance),
                        "chunk_id": chunk.id,
                        "content": chunk.content,
                        "source_id": chunk.source_id,
                        "source_type": chunk.source_type.value,
                        "metadata": chunk.metadata
                    })
            
            result.total_results = len(result.results)
            result.search_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            result.message = f"Error searching index: {str(e)}"
        
        return result


def build_public_index_from_sources(
    sources: List[PublicSource],
    name: str,
    description: str = "",
    config: Optional[IndexConfig] = None
) -> IndexBuildResult:
    """Convenience function to build a public index from sources.
    
    Args:
        sources: List of public sources
        name: Name for the index
        description: Description of the index
        config: Index configuration (optional)
        
    Returns:
        IndexBuildResult with build information
    """
    if config is None:
        config = IndexConfig()
    
    builder = PublicIndexBuilder(config)
    
    # Add sources
    builder.add_sources(sources)
    
    # Apply processing
    if config.normalization_enabled:
        builder.normalize_chunks()
    
    if config.deduplication_enabled:
        builder.deduplicate_chunks()
    
    # Build index
    return builder.build_index(name, description)


def load_public_index(index_path: str) -> Optional[PublicIndexBuilder]:
    """Load a public index from disk.
    
    Args:
        index_path: Path to the index directory
        
    Returns:
        PublicIndexBuilder with loaded index or None
    """
    try:
        index_dir = Path(index_path)
        
        # Load metadata
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load chunks
        chunks_path = index_dir / "chunks.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Create builder
        config = IndexConfig(**metadata["config"])
        builder = PublicIndexBuilder(config)
        
        # Load chunks
        builder.chunks = [PublicChunk(**chunk_data) for chunk_data in chunks_data]
        
        # Load FAISS index (named after the public index)
        if list(index_dir.glob("*.faiss")):
            builder.faiss_index = FAISSIndex.load(str(index_dir))
        
        logger.info(f"Loaded index from {index_path}")
        return builder
        
    except Exception as e:
        logger.error(f"Error loading index from {index_path}: {e}")
        return None
