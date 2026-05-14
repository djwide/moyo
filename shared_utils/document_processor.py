"""Document processing utilities for normalized document schema."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from datetime import datetime
import mimetypes

from .document_schema import (
    NormalizedDocument, 
    TextChunk, 
    DocumentType, 
    DocumentSource
)
from .ids import (
    generate_stable_document_id,
    generate_content_hash,
    generate_fingerprint
)
from .chunking import chunk_text, chunk_text_simple
from .text_processing import normalize_text

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processor for converting various document formats to normalized schema."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_path: Union[str, Path], 
                    source_type: DocumentSource = DocumentSource.FILE,
                    metadata: Optional[Dict[str, Any]] = None) -> NormalizedDocument:
        """Process a file and convert to normalized document.
        
        Args:
            file_path: Path to the file
            source_type: Type of source
            metadata: Additional metadata
            
        Returns:
            NormalizedDocument instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine document type
        document_type = self._guess_document_type(file_path)
        
        # Read file content
        content = self._read_file_content(file_path, document_type)
        
        # Create normalized document
        doc = NormalizedDocument(
            source=str(file_path),
            source_type=source_type,
            document_type=document_type,
            mime_type=mimetypes.guess_type(str(file_path))[0] or "text/plain",
            original_size_bytes=file_path.stat().st_size,
            metadata=metadata or {}
        )
        
        # Process content into chunks
        self._process_content_into_chunks(doc, content)
        
        return doc
    
    def process_text(self, text: str, source: str = "text_input",
                    source_type: DocumentSource = DocumentSource.MANUAL,
                    metadata: Optional[Dict[str, Any]] = None) -> NormalizedDocument:
        """Process text content and convert to normalized document.
        
        Args:
            text: Text content
            source: Source identifier
            source_type: Type of source
            metadata: Additional metadata
            
        Returns:
            NormalizedDocument instance
        """
        # Create normalized document
        doc = NormalizedDocument(
            source=source,
            source_type=source_type,
            document_type=DocumentType.TEXT,
            mime_type="text/plain",
            original_size_bytes=len(text.encode('utf-8')),
            metadata=metadata or {}
        )
        
        # Process content into chunks
        self._process_content_into_chunks(doc, text)
        
        return doc
    
    def process_url(self, url: str, content: str,
                   source_type: DocumentSource = DocumentSource.URL,
                   metadata: Optional[Dict[str, Any]] = None) -> NormalizedDocument:
        """Process URL content and convert to normalized document.
        
        Args:
            url: URL of the content
            content: Content from URL
            source_type: Type of source
            metadata: Additional metadata
            
        Returns:
            NormalizedDocument instance
        """
        # Determine document type from URL
        document_type = self._guess_document_type_from_url(url)
        
        # Create normalized document
        doc = NormalizedDocument(
            source=url,
            source_type=source_type,
            document_type=document_type,
            url=url,
            original_url=url,
            original_size_bytes=len(content.encode('utf-8')),
            metadata=metadata or {}
        )
        
        # Process content into chunks
        self._process_content_into_chunks(doc, content)
        
        return doc
    
    def _guess_document_type(self, file_path: Path) -> DocumentType:
        """Guess document type from file extension."""
        suffix = file_path.suffix.lower()
        
        type_map = {
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.json': DocumentType.JSON,
            '.xml': DocumentType.XML,
            '.csv': DocumentType.CSV,
            '.xlsx': DocumentType.EXCEL,
            '.xls': DocumentType.EXCEL,
            '.docx': DocumentType.WORD,
            '.doc': DocumentType.WORD,
            '.pptx': DocumentType.POWERPOINT,
            '.ppt': DocumentType.POWERPOINT,
            '.pdf': DocumentType.PDF,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.png': DocumentType.IMAGE,
            '.gif': DocumentType.IMAGE,
            '.mp3': DocumentType.AUDIO,
            '.wav': DocumentType.AUDIO,
            '.mp4': DocumentType.VIDEO,
            '.avi': DocumentType.VIDEO
        }
        
        return type_map.get(suffix, DocumentType.UNKNOWN)
    
    def _guess_document_type_from_url(self, url: str) -> DocumentType:
        """Guess document type from URL."""
        url_lower = url.lower()
        
        if any(ext in url_lower for ext in ['.pdf', '.PDF']):
            return DocumentType.PDF
        elif any(ext in url_lower for ext in ['.html', '.htm', '.HTML', '.HTM']):
            return DocumentType.HTML
        elif any(ext in url_lower for ext in ['.json', '.JSON']):
            return DocumentType.JSON
        elif any(ext in url_lower for ext in ['.xml', '.XML']):
            return DocumentType.XML
        elif any(ext in url_lower for ext in ['.csv', '.CSV']):
            return DocumentType.CSV
        else:
            return DocumentType.TEXT
    
    def _read_file_content(self, file_path: Path, document_type: DocumentType) -> str:
        """Read file content based on document type."""
        try:
            if document_type in [DocumentType.TEXT, DocumentType.MARKDOWN, DocumentType.HTML, 
                               DocumentType.JSON, DocumentType.XML, DocumentType.CSV]:
                # Read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # For binary files, return placeholder or extract text if possible
                logger.warning(f"Binary file type {document_type} not fully supported: {file_path}")
                return f"[Binary file: {file_path.name}]"
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return f"[Error reading file: {file_path.name}]"
    
    def _process_content_into_chunks(self, doc: NormalizedDocument, content: str) -> None:
        """Process content into text chunks and add to document."""
        if not content.strip():
            logger.warning(f"Empty content for document: {doc.source}")
            return
        
        # Normalize text
        normalized_content = normalize_text(content)
        
        # Chunk the text
        chunks = chunk_text_simple(normalized_content, self.chunk_size, self.chunk_overlap)
        
        # Create text chunks
        for i, chunk_text in enumerate(chunks):
            chunk = TextChunk(
                text=chunk_text,
                chunk_index=i,
                start_position=i * (self.chunk_size - self.chunk_overlap),
                end_position=i * (self.chunk_size - self.chunk_overlap) + len(chunk_text),
                chunk_size=len(chunk_text),
                metadata={"chunk_method": "sliding_window"}
            )
            doc.text_chunks.append(chunk)
        
        # Update document processing status
        doc.processing_status = "completed"
        doc.processed_at = datetime.now()
        
        logger.info(f"Processed document {doc.source}: {len(chunks)} chunks created")


class DocumentConverter:
    """Converter for converting between different document formats."""
    
    @staticmethod
    def from_legacy_chunk(legacy_chunk: Dict[str, Any], 
                         document_source: str,
                         source_type: DocumentSource = DocumentSource.UNKNOWN) -> NormalizedDocument:
        """Convert legacy DocumentChunk to NormalizedDocument.
        
        Args:
            legacy_chunk: Legacy chunk dictionary
            document_source: Source identifier
            source_type: Type of source
            
        Returns:
            NormalizedDocument instance
        """
        # Create normalized document
        doc = NormalizedDocument(
            source=document_source,
            source_type=source_type,
            document_type=DocumentType.TEXT,
            mime_type="text/plain",
            metadata=legacy_chunk.get('metadata', {})
        )
        
        # Create text chunk
        chunk = TextChunk(
            text=legacy_chunk.get('text', ''),
            chunk_index=legacy_chunk.get('chunk_index', 0),
            start_position=legacy_chunk.get('start_position'),
            end_position=legacy_chunk.get('end_position'),
            chunk_size=legacy_chunk.get('chunk_size', len(legacy_chunk.get('text', ''))),
            metadata=legacy_chunk.get('metadata', {})
        )
        
        doc.text_chunks.append(chunk)
        doc.processing_status = "completed"
        doc.processed_at = datetime.now()
        
        return doc
    
    @staticmethod
    def from_public_source(public_source: Dict[str, Any]) -> NormalizedDocument:
        """Convert public source to NormalizedDocument.
        
        Args:
            public_source: Public source dictionary
            
        Returns:
            NormalizedDocument instance
        """
        # Determine source type
        source_type_map = {
            'patent': DocumentSource.PATENT,
            'press_release': DocumentSource.NEWS,
            'git_commit': DocumentSource.GIT,
            'conference_talk': DocumentSource.CONFERENCE,
            'web_search': DocumentSource.URL,
            'research_paper': DocumentSource.RESEARCH,
            'news_article': DocumentSource.NEWS
        }
        
        source_type = source_type_map.get(
            public_source.get('source_type', 'unknown'), 
            DocumentSource.UNKNOWN
        )
        
        # Create normalized document
        doc = NormalizedDocument(
            source=public_source.get('id', 'unknown'),
            source_type=source_type,
            document_type=DocumentType.TEXT,
            title=public_source.get('title'),
            author=public_source.get('author'),
            url=public_source.get('url'),
            original_url=public_source.get('source_url'),
            published_date=public_source.get('published_date'),
            language=public_source.get('language', 'en'),
            metadata=public_source.get('metadata', {}),
            tags=public_source.get('tags', []),
            confidence_score=public_source.get('confidence_score'),
            relevance_score=public_source.get('relevance_score')
        )
        
        # Add content as single chunk
        content = public_source.get('content', '')
        if content:
            chunk = TextChunk(
                text=content,
                chunk_index=0,
                chunk_size=len(content),
                metadata={"source_type": "public_source"}
            )
            doc.text_chunks.append(chunk)
        
        doc.processing_status = "completed"
        doc.processed_at = datetime.now()
        
        return doc
    
    @staticmethod
    def to_legacy_format(normalized_doc: NormalizedDocument) -> List[Dict[str, Any]]:
        """Convert NormalizedDocument to legacy format.
        
        Args:
            normalized_doc: NormalizedDocument instance
            
        Returns:
            List of legacy chunk dictionaries
        """
        legacy_chunks = []
        
        for chunk in normalized_doc.text_chunks:
            legacy_chunk = {
                'id': chunk.id,
                'text': chunk.text,
                'chunk_index': chunk.chunk_index,
                'source_document': normalized_doc.source,
                'chunk_size': chunk.chunk_size,
                'start_position': chunk.start_position,
                'end_position': chunk.end_position,
                'metadata': {**chunk.metadata, **normalized_doc.metadata}
            }
            legacy_chunks.append(legacy_chunk)
        
        return legacy_chunks


def create_document_from_file(file_path: Union[str, Path], 
                            chunk_size: int = 512,
                            chunk_overlap: int = 50,
                            metadata: Optional[Dict[str, Any]] = None) -> NormalizedDocument:
    """Convenience function to create a normalized document from a file.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata
        
    Returns:
        NormalizedDocument instance
    """
    processor = DocumentProcessor(chunk_size, chunk_overlap)
    return processor.process_file(file_path, metadata=metadata)


def create_document_from_text(text: str, 
                            source: str = "text_input",
                            chunk_size: int = 512,
                            chunk_overlap: int = 50,
                            metadata: Optional[Dict[str, Any]] = None) -> NormalizedDocument:
    """Convenience function to create a normalized document from text.
    
    Args:
        text: Text content
        source: Source identifier
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata
        
    Returns:
        NormalizedDocument instance
    """
    processor = DocumentProcessor(chunk_size, chunk_overlap)
    return processor.process_text(text, source, metadata=metadata)


def batch_process_files(file_paths: List[Union[str, Path]], 
                       chunk_size: int = 512,
                       chunk_overlap: int = 50) -> List[NormalizedDocument]:
    """Process multiple files in batch.
    
    Args:
        file_paths: List of file paths
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of NormalizedDocument instances
    """
    processor = DocumentProcessor(chunk_size, chunk_overlap)
    documents = []
    
    for file_path in file_paths:
        try:
            doc = processor.process_file(file_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            # Create error document
            error_doc = NormalizedDocument(
                source=str(file_path),
                source_type=DocumentSource.FILE,
                processing_status="error",
                processing_errors=[str(e)]
            )
            documents.append(error_doc)
    
    return documents
