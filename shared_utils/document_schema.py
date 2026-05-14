"""Normalized document schema for sente and Moyo projects.

This module defines a unified document schema that can be used across
both private and public data processing pipelines.
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
import mimetypes

from .ids import (
    generate_stable_document_id,
    generate_content_hash,
    generate_chunk_id,
    generate_fingerprint
)


class DocumentType(str, Enum):
    """Types of documents supported by the system."""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    EXCEL = "excel"
    WORD = "word"
    POWERPOINT = "powerpoint"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


class DocumentSource(str, Enum):
    """Source types for documents."""
    FILE = "file"
    URL = "url"
    DATABASE = "database"
    API = "api"
    MANUAL = "manual"
    CRAWLER = "crawler"
    GIT = "git"
    PATENT = "patent"
    CONFERENCE = "conference"
    NEWS = "news"
    RESEARCH = "research"
    UNKNOWN = "unknown"


class TextChunk(BaseModel):
    """A chunk of text from a document."""
    id: Optional[str] = None
    text: str
    chunk_index: int
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    chunk_size: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('id', pre=True, always=True)
    def generate_chunk_id(cls, v, values):
        """Generate chunk ID if not provided."""
        if v is None or v == "":
            document_id = values.get('document_id', 'unknown')
            chunk_index = values.get('chunk_index', 0)
            chunk_hash = generate_content_hash(values.get('text', ''))[:8]
            return generate_chunk_id(document_id, chunk_index, chunk_hash)
        return v
    
    @validator('chunk_size', pre=True, always=True)
    def set_chunk_size(cls, v, values):
        """Set chunk size based on text length if not provided."""
        if v is None or v == 0:
            return len(values.get('text', ''))
        return v


class NormalizedDocument(BaseModel):
    """Normalized document schema for unified document representation.
    
    This schema provides a consistent way to represent documents across
    different sources and processing pipelines.
    """
    
    # Core identification
    id: Optional[str] = None
    source: str  # Source identifier (file path, URL, etc.)
    source_type: DocumentSource = DocumentSource.UNKNOWN
    
    # Content information
    mime_type: str = "text/plain"
    text_chunks: List[TextChunk] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    # Content properties
    title: Optional[str] = None
    author: Optional[str] = None
    language: str = "en"
    encoding: str = "utf-8"
    
    # Processing information
    content_hash: Optional[str] = None
    fingerprint: Optional[str] = None
    processing_status: str = "pending"
    processing_errors: List[str] = Field(default_factory=list)
    
    # Size information
    original_size_bytes: Optional[int] = None
    processed_size_bytes: Optional[int] = None
    
    # URLs and references
    url: Optional[HttpUrl] = None
    original_url: Optional[HttpUrl] = None
    
    # Classification
    document_type: DocumentType = DocumentType.TEXT
    tags: List[str] = Field(default_factory=list)
    
    # Quality metrics
    confidence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    @validator('id', pre=True, always=True)
    def generate_document_id(cls, v, values):
        """Generate stable document ID if not provided."""
        if v is None or v == "":
            source = values.get('source', 'unknown')
            content_hash = values.get('content_hash')
            timestamp = values.get('created_at')
            return generate_stable_document_id(source, content_hash, timestamp)
        return v
    
    @validator('content_hash', pre=True, always=True)
    def generate_content_hash(cls, v, values):
        """Generate content hash from text chunks if not provided."""
        if v is None or v == "":
            text_chunks = values.get('text_chunks', [])
            if text_chunks:
                full_text = " ".join(chunk.text for chunk in text_chunks)
                return generate_content_hash(full_text)
        return v
    
    @validator('fingerprint', pre=True, always=True)
    def generate_fingerprint(cls, v, values):
        """Generate fingerprint for deduplication if not provided."""
        if v is None or v == "":
            text_chunks = values.get('text_chunks', [])
            metadata = values.get('metadata', {})
            if text_chunks:
                full_text = " ".join(chunk.text for chunk in text_chunks)
                return generate_fingerprint(full_text, metadata)
        return v
    
    @validator('mime_type', pre=True, always=True)
    def set_mime_type(cls, v, values):
        """Set MIME type based on document type or source if not provided."""
        if v is None or v == "text/plain":
            source = values.get('source', '')
            document_type = values.get('document_type')
            
            if document_type:
                mime_map = {
                    DocumentType.PDF: "application/pdf",
                    DocumentType.HTML: "text/html",
                    DocumentType.MARKDOWN: "text/markdown",
                    DocumentType.JSON: "application/json",
                    DocumentType.XML: "application/xml",
                    DocumentType.CSV: "text/csv",
                    DocumentType.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    DocumentType.WORD: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    DocumentType.POWERPOINT: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    DocumentType.IMAGE: "image/*",
                    DocumentType.AUDIO: "audio/*",
                    DocumentType.VIDEO: "video/*"
                }
                return mime_map.get(document_type, "text/plain")
            
            # Try to guess from source
            if source:
                guessed_type, _ = mimetypes.guess_type(source)
                if guessed_type:
                    return guessed_type
        
        return v or "text/plain"
    
    @validator('processed_size_bytes', pre=True, always=True)
    def calculate_processed_size(cls, v, values):
        """Calculate processed size from text chunks if not provided."""
        if v is None:
            text_chunks = values.get('text_chunks', [])
            if text_chunks:
                return sum(len(chunk.text.encode('utf-8')) for chunk in text_chunks)
        return v
    
    def get_full_text(self) -> str:
        """Get the full text content from all chunks."""
        return " ".join(chunk.text for chunk in self.text_chunks)
    
    def get_chunk_count(self) -> int:
        """Get the number of text chunks."""
        return len(self.text_chunks)
    
    def get_total_size(self) -> int:
        """Get the total size of all text chunks."""
        return sum(len(chunk.text) for chunk in self.text_chunks)
    
    def add_chunk(self, text: str, chunk_index: Optional[int] = None, 
                  start_position: Optional[int] = None, end_position: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> TextChunk:
        """Add a new text chunk to the document."""
        if chunk_index is None:
            chunk_index = len(self.text_chunks)
        
        chunk = TextChunk(
            text=text,
            chunk_index=chunk_index,
            start_position=start_position,
            end_position=end_position,
            metadata=metadata or {},
            document_id=self.id
        )
        
        self.text_chunks.append(chunk)
        self.modified_at = datetime.now()
        
        # Recalculate processed size
        self.processed_size_bytes = self.calculate_processed_size(None, self.dict())
        
        return chunk
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk by ID."""
        for i, chunk in enumerate(self.text_chunks):
            if chunk.id == chunk_id:
                del self.text_chunks[i]
                self.modified_at = datetime.now()
                self.processed_size_bytes = self.calculate_processed_size(None, self.dict())
                return True
        return False
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update document metadata."""
        self.metadata[key] = value
        self.modified_at = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the document."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.modified_at = datetime.now()
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the document."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.modified_at = datetime.now()
            return True
        return False
    
    def is_processed(self) -> bool:
        """Check if the document has been processed."""
        return self.processing_status in ["completed", "success"]
    
    def has_errors(self) -> bool:
        """Check if the document has processing errors."""
        return len(self.processing_errors) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "source": self.source,
            "source_type": self.source_type.value,
            "mime_type": self.mime_type,
            "text_chunks": [chunk.dict() for chunk in self.text_chunks],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "encoding": self.encoding,
            "content_hash": self.content_hash,
            "fingerprint": self.fingerprint,
            "processing_status": self.processing_status,
            "processing_errors": self.processing_errors,
            "original_size_bytes": self.original_size_bytes,
            "processed_size_bytes": self.processed_size_bytes,
            "url": str(self.url) if self.url else None,
            "original_url": str(self.original_url) if self.original_url else None,
            "document_type": self.document_type.value,
            "tags": self.tags,
            "confidence_score": self.confidence_score,
            "relevance_score": self.relevance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormalizedDocument':
        """Create document from dictionary representation."""
        # Handle text chunks
        if 'text_chunks' in data and isinstance(data['text_chunks'], list):
            data['text_chunks'] = [TextChunk(**chunk) for chunk in data['text_chunks']]
        
        # Handle timestamps
        for field in ['created_at', 'modified_at', 'processed_at']:
            if field in data and data[field] and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class DocumentCollection(BaseModel):
    """A collection of normalized documents."""
    
    id: str
    name: str
    description: Optional[str] = None
    documents: List[NormalizedDocument] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)
    
    def add_document(self, document: NormalizedDocument) -> None:
        """Add a document to the collection."""
        self.documents.append(document)
        self.modified_at = datetime.now()
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the collection by ID."""
        for i, doc in enumerate(self.documents):
            if doc.id == document_id:
                del self.documents[i]
                self.modified_at = datetime.now()
                return True
        return False
    
    def get_document(self, document_id: str) -> Optional[NormalizedDocument]:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        return None
    
    def get_documents_by_source(self, source: str) -> List[NormalizedDocument]:
        """Get all documents from a specific source."""
        return [doc for doc in self.documents if doc.source == source]
    
    def get_documents_by_type(self, document_type: DocumentType) -> List[NormalizedDocument]:
        """Get all documents of a specific type."""
        return [doc for doc in self.documents if doc.document_type == document_type]
    
    def get_total_documents(self) -> int:
        """Get the total number of documents."""
        return len(self.documents)
    
    def get_total_chunks(self) -> int:
        """Get the total number of text chunks across all documents."""
        return sum(doc.get_chunk_count() for doc in self.documents)
    
    def get_total_size(self) -> int:
        """Get the total size of all documents."""
        return sum(doc.get_total_size() for doc in self.documents)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics for the collection."""
        total_docs = len(self.documents)
        processed_docs = sum(1 for doc in self.documents if doc.is_processed())
        error_docs = sum(1 for doc in self.documents if doc.has_errors())
        
        return {
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "error_documents": error_docs,
            "total_chunks": self.get_total_chunks(),
            "total_size": self.get_total_size(),
            "processing_rate": processed_docs / total_docs if total_docs > 0 else 0
        }
