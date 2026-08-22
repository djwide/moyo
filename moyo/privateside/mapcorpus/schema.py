from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from shared_utils.model_config import DEFAULT_MODEL_NAME


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""
    id: str
    text: str
    chunk_index: int
    source_document: str
    chunk_size: int
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    level: str = "section"  # section | sentence | item | phrase
    parent_id: Optional[str] = None


class MappedDocument(BaseModel):
    """Document with embedding for indexing."""
    id: str
    text: str
    embedding: List[float]
    source_path: Optional[str] = None
    source_type: str = "text"  # "text", "file", "web", etc.
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CorpusConfig(BaseModel):
    """Configuration for corpus building."""
    chunk_size: int = 512
    chunk_overlap: int = 50  # ~10% of chunk_size; keep matched with the public index
    max_tokens: Optional[int] = None  # derived from embedding model when unset
    embedding_model: str = DEFAULT_MODEL_NAME
    embedding_device: str = "auto"
    batch_size: int = 32
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    normalize_embeddings: bool = True  # required for FlatIP = cosine
    deduplication_enabled: bool = True
    normalization_enabled: bool = True
    min_chunk_length: int = 50  # section-level only; sentence/item/atomic secrets kept
    max_chunk_length: int = 2000
    output_directory: str = "indexes/private"
    save_chunks: bool = True
    save_metadata: bool = True
    granularity: str = "multi"


class CorpusBuildResult(BaseModel):
    """Result of corpus building operation."""
    success: bool
    message: str
    documents_processed: int = 0
    chunks_created: int = 0
    vectors_created: int = 0
    duplicates_removed: int = 0
    processing_time: float = 0.0
    index_path: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CorpusInfo(BaseModel):
    """Information about a built corpus."""
    corpus_id: str
    created_at: datetime
    document_count: int
    chunk_count: int
    vector_count: int
    embedding_dimension: int
    index_type: str
    embedding_model: str
    total_size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result of a corpus search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
