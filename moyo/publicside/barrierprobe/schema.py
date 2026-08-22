"""Schema definitions for barrierprobe public information indexing."""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from ..gatherpublicsources.schema import PublicSource, SourceType
from shared_utils.model_config import DEFAULT_MODEL_NAME


class IndexType(str, Enum):
    """Types of FAISS indexes."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    PQ = "pq"


class IndexConfig(BaseModel):
    """Configuration for building FAISS indexes."""
    index_type: IndexType = IndexType.FLAT
    embedding_model: str = DEFAULT_MODEL_NAME
    embedding_device: str = "auto"  # auto | cuda | cpu (unused for Ollama; kept for compat)
    chunk_size: int = 512
    chunk_overlap: int = 50  # ~10% of chunk_size; must match the private index
    max_tokens: Optional[int] = None  # derived from embedding_model when unset
    normalize_embeddings: bool = True
    save_metadata: bool = True
    output_directory: str = "indexes/public"
    
    # Index-specific parameters
    nlist: int = 100  # For IVF
    m: int = 32  # For HNSW
    ef_construction: int = 200  # For HNSW
    nbits: int = 8  # For PQ
    
    # Filtering — min applies to section chunks only (sentence/item stay)
    min_chunk_length: int = 50
    max_chunk_length: int = 2000
    deduplication_enabled: bool = True
    normalization_enabled: bool = True
    
    # Source filtering
    source_types: List[SourceType] = Field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    organizations: List[str] = Field(default_factory=list)
    min_relevance_score: float = 0.0
    min_confidence_score: float = 0.0


class PublicChunk(BaseModel):
    """A chunk of public information."""
    id: str
    content: str
    source_id: str
    source_type: SourceType
    chunk_index: int
    start_char: int
    end_char: int
    level: str = "section"  # granularity: "section", "sentence", or "item"
    parent_id: Optional[str] = None  # parent section chunk id (for sentences)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)


class PublicIndex(BaseModel):
    """A FAISS index of public information."""
    id: str
    name: str
    description: str
    config: IndexConfig
    source_count: int = 0
    chunk_count: int = 0
    vector_count: int = 0
    index_size_bytes: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndexBuildResult(BaseModel):
    """Result of building a public information index."""
    success: bool
    message: str
    index_id: Optional[str] = None
    index_path: Optional[str] = None
    sources_processed: int = 0
    chunks_created: int = 0
    vectors_created: int = 0
    duplicates_removed: int = 0
    processing_time: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Result of searching the public index."""
    query: str
    total_results: int = 0
    results: List[Dict[str, Any]] = Field(default_factory=list)
    search_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BarrierProbeConfig(BaseModel):
    """Configuration for barrier probing."""
    public_index_path: str
    private_index_path: str
    similarity_threshold: float = 0.8
    max_comparisons: int = 1000
    output_directory: str = "data/barrierprobe/results"
    save_detailed_results: bool = True
    include_metadata: bool = True
    # Distribution layer (JS / entropy / margin) sits on top of cosine NN.
    neighborhood_k: int = 20
    softmax_temperature: float = 0.10
    cluster_count: Optional[int] = None


class BarrierProbeResult(BaseModel):
    """Result of a barrier probe analysis."""
    probe_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    public_index_info: Dict[str, Any]
    private_index_info: Dict[str, Any]
    similarity_threshold: float
    potential_breaches: List[Dict[str, Any]] = Field(default_factory=list)
    breach_count: int = 0
    high_risk_breaches: int = 0
    medium_risk_breaches: int = 0
    low_risk_breaches: int = 0
    processing_time: float = 0.0
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Distribution layer headline. High semantic_separation is not integrity.
    semantic_separation: Optional[float] = None
    pairwise_exposure: str = "None"
    concentrated_matches: int = 0
    distribution_diagnostics: Dict[str, Any] = Field(default_factory=dict)
