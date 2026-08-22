"""Text normalization utilities for corpus building."""

import re
import unicodedata
from typing import List, Dict, Any
import logging

from .schema import DocumentChunk

logger = logging.getLogger(__name__)


def apply_unicode_normalization(text: str) -> str:
    """Normalize Unicode characters."""
    # Normalize to NFC form
    text = unicodedata.normalize('NFC', text)
    return text


def apply_whitespace_normalization(text: str) -> str:
    """Normalize whitespace characters."""
    # Replace various whitespace characters with standard spaces
    text = re.sub(r'[\t\n\r\f\v]+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def apply_case_normalization(text: str, lowercase: bool = True) -> str:
    """Normalize text case."""
    if lowercase:
        return text.lower()
    return text


def apply_punctuation_normalization(text: str) -> str:
    """Normalize punctuation marks."""
    # Replace multiple punctuation marks with single
    text = re.sub(r'[.!?]+', '.', text)
    text = re.sub(r'[,;]+', ',', text)
    text = re.sub(r'[-_]+', '-', text)
    
    # Normalize quotes - fix regex syntax
    text = re.sub(r'["""'']', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    return text


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """Remove or normalize special characters."""
    if keep_punctuation:
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', text)
    else:
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text)
    
    return text


def apply_url_removal(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)


def apply_email_removal(text: str) -> str:
    """Remove email addresses from text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(email_pattern, '', text)


def apply_text_normalization(text: str, 
                            lowercase: bool = True,
                            normalize_unicode: bool = True,
                            normalize_whitespace: bool = True,
                            remove_urls: bool = True,
                            remove_emails: bool = True,
                            normalize_punctuation: bool = True,
                            keep_punctuation: bool = True) -> str:
    """Apply comprehensive text normalization.
    
    Args:
        text: Input text to normalize
        lowercase: Whether to convert to lowercase
        normalize_unicode: Whether to normalize Unicode characters
        normalize_whitespace: Whether to normalize whitespace
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses
        normalize_punctuation: Whether to normalize punctuation
        keep_punctuation: Whether to keep punctuation marks
        
    Returns:
        Normalized text
    """
    if not text:
        return text
    
    # Apply normalization steps
    if normalize_unicode:
        text = apply_unicode_normalization(text)
    
    if remove_urls:
        text = apply_url_removal(text)
    
    if remove_emails:
        text = apply_email_removal(text)
    
    if normalize_punctuation:
        text = apply_punctuation_normalization(text)
    
    if not keep_punctuation:
        text = remove_special_characters(text, keep_punctuation=False)
    
    if normalize_whitespace:
        text = apply_whitespace_normalization(text)
    
    if lowercase:
        text = apply_case_normalization(text, lowercase=True)
    
    return text


def normalize_chunks(chunks: List[DocumentChunk], 
                     config: Dict[str, Any] = None) -> List[DocumentChunk]:
    """Normalize a list of document chunks.
    
    Args:
        chunks: List of document chunks to normalize
        config: Normalization configuration
        
    Returns:
        List of normalized chunks
    """
    if not chunks:
        return chunks
    
    config = config or {}
    
    normalized_chunks = []
    for chunk in chunks:
        normalized_text = apply_text_normalization(
            chunk.text,
            lowercase=config.get('lowercase', True),
            normalize_unicode=config.get('normalize_unicode', True),
            normalize_whitespace=config.get('normalize_whitespace', True),
            remove_urls=config.get('remove_urls', True),
            remove_emails=config.get('remove_emails', True),
            normalize_punctuation=config.get('normalize_punctuation', True),
            keep_punctuation=config.get('keep_punctuation', True)
        )
        
        # Create new chunk with normalized text
        normalized_chunk = DocumentChunk(
            id=chunk.id,
            text=normalized_text,
            chunk_index=chunk.chunk_index,
            source_document=chunk.source_document,
            chunk_size=len(normalized_text),
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            embedding=chunk.embedding,
            level=getattr(chunk, "level", "section") or "section",
            parent_id=getattr(chunk, "parent_id", None),
            metadata={**chunk.metadata, "normalized": True}
        )
        
        normalized_chunks.append(normalized_chunk)
    
    logger.info(f"Normalized {len(chunks)} chunks")
    return normalized_chunks


def filter_chunks_by_length(chunks: List[DocumentChunk], 
                           min_length: int = 50,
                           max_length: int = 2000) -> List[DocumentChunk]:
    """Filter chunks based on text length.

    Minimum length applies to *section* chunks only so sentence/item/phrase
    units (the ones that match short secrets) are not dropped as boilerplate.
    """
    if not chunks:
        return chunks
    
    filtered_chunks = []
    removed_count = 0
    
    for chunk in chunks:
        text_length = len(chunk.text)
        level = getattr(chunk, "level", "section") or "section"
        if text_length > max_length and level == "section":
            removed_count += 1
            continue
        if level == "section" and text_length < min_length:
            # Keep a short section that is the only representation of a secret.
            has_siblings = any(
                getattr(other, "parent_id", None) == chunk.id
                for other in chunks
                if other is not chunk
            )
            if has_siblings:
                removed_count += 1
                continue
        filtered_chunks.append(chunk)
    
    logger.info(f"Filtered chunks by length: {len(chunks)} -> {len(filtered_chunks)} ({removed_count} removed)")
    return filtered_chunks


def get_text_statistics(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Get statistics about text content in chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "average_length": 0.0,
            "min_length": 0,
            "max_length": 0
        }
    
    lengths = [len(chunk.text) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_characters": sum(lengths),
        "average_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "length_distribution": {
            "short": len([l for l in lengths if l < 100]),
            "medium": len([l for l in lengths if 100 <= l < 500]),
            "long": len([l for l in lengths if l >= 500])
        }
    }
