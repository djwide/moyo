"""Shared text processing utilities for sente and Moyo projects."""

import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextNormalizationConfig:
    """Configuration for text normalization."""
    lowercase: bool = True
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_punctuation: bool = True
    keep_punctuation: bool = True
    remove_special_chars: bool = False


@dataclass
class DeduplicationConfig:
    """Configuration for text deduplication."""
    exact_duplicates: bool = True
    similar_duplicates: bool = False
    similarity_threshold: float = 0.9
    case_sensitive: bool = False


def apply_unicode_normalization(text: str) -> str:
    """Normalize Unicode characters."""
    return unicodedata.normalize('NFC', text)


def apply_whitespace_normalization(text: str) -> str:
    """Normalize whitespace characters."""
    # Replace various whitespace characters with standard spaces
    text = re.sub(r'[\t\n\r\f\v]+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


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
    
    # Normalize quotes
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


def normalize_text(text: str, config: Optional[TextNormalizationConfig] = None) -> str:
    """Apply comprehensive text normalization.
    
    Args:
        text: Input text to normalize
        config: Normalization configuration
        
    Returns:
        Normalized text
    """
    if not text:
        return text
    
    if config is None:
        config = TextNormalizationConfig()
    
    # Apply normalization steps
    if config.normalize_unicode:
        text = apply_unicode_normalization(text)
    
    if config.normalize_whitespace:
        text = apply_whitespace_normalization(text)
    
    if config.remove_urls:
        text = apply_url_removal(text)
    
    if config.remove_emails:
        text = apply_email_removal(text)
    
    if config.normalize_punctuation:
        text = apply_punctuation_normalization(text)
    
    if config.remove_special_chars:
        text = remove_special_characters(text, config.keep_punctuation)
    
    if config.lowercase:
        text = apply_case_normalization(text, lowercase=True)
    
    # Final whitespace normalization
    text = apply_whitespace_normalization(text)
    
    return text


def calculate_text_hash(text: str) -> str:
    """Calculate a hash for text content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings."""
    return SequenceMatcher(None, text1, text2).ratio()


def find_exact_duplicates(texts: List[str], case_sensitive: bool = False) -> List[Tuple[int, int]]:
    """Find exact duplicate texts.
    
    Args:
        texts: List of text strings
        case_sensitive: Whether to consider case in comparison
        
    Returns:
        List of (index1, index2) pairs of duplicates
    """
    duplicates = []
    seen_hashes = {}
    
    for i, text in enumerate(texts):
        if not case_sensitive:
            text = text.lower()
        
        text_hash = calculate_text_hash(text)
        
        if text_hash in seen_hashes:
            duplicates.append((seen_hashes[text_hash], i))
        else:
            seen_hashes[text_hash] = i
    
    return duplicates


def find_similar_texts(texts: List[str], 
                      similarity_threshold: float = 0.9,
                      case_sensitive: bool = False) -> List[Tuple[int, int, float]]:
    """Find similar texts based on text similarity.
    
    Args:
        texts: List of text strings
        similarity_threshold: Threshold for similarity detection
        case_sensitive: Whether to consider case in comparison
        
    Returns:
        List of (index1, index2, similarity) tuples
    """
    similar_pairs = []
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            text1 = texts[i]
            text2 = texts[j]
            
            if not case_sensitive:
                text1 = text1.lower()
                text2 = text2.lower()
            
            similarity = calculate_text_similarity(text1, text2)
            if similarity >= similarity_threshold:
                similar_pairs.append((i, j, similarity))
    
    return similar_pairs


def deduplicate_texts(texts: List[str], 
                     config: Optional[DeduplicationConfig] = None) -> Tuple[List[str], int]:
    """Remove duplicate texts from the list.
    
    Args:
        texts: List of text strings
        config: Deduplication configuration
        
    Returns:
        Tuple of (deduplicated_texts, duplicates_removed_count)
    """
    if not texts:
        return texts, 0
    
    if config is None:
        config = DeduplicationConfig()
    
    original_count = len(texts)
    deduplicated_texts = texts.copy()
    duplicates_removed = 0
    
    # Remove exact duplicates
    if config.exact_duplicates:
        exact_dups = find_exact_duplicates(deduplicated_texts, config.case_sensitive)
        if exact_dups:
            # Sort in reverse order to remove from end first
            indices_to_remove = sorted(set(idx for pair in exact_dups for idx in pair[1:]), reverse=True)
            for idx in indices_to_remove:
                del deduplicated_texts[idx]
            duplicates_removed += len(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} exact duplicates")
    
    # Remove similar duplicates
    if config.similar_duplicates and len(deduplicated_texts) > 1:
        similar_pairs = find_similar_texts(
            deduplicated_texts, 
            config.similarity_threshold,
            config.case_sensitive
        )
        if similar_pairs:
            # Keep the first occurrence of each similar pair
            indices_to_remove = set()
            for i, j, similarity in similar_pairs:
                indices_to_remove.add(j)  # Remove the second occurrence
            
            # Remove in reverse order
            for idx in sorted(indices_to_remove, reverse=True):
                del deduplicated_texts[idx]
            duplicates_removed += len(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} similar duplicates (threshold: {config.similarity_threshold})")
    
    return deduplicated_texts, duplicates_removed


def get_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """Get statistics about a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with text statistics
    """
    if not texts:
        return {
            "count": 0,
            "total_length": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "total_words": 0,
            "avg_words": 0
        }
    
    lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    
    return {
        "count": len(texts),
        "total_length": sum(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_words": sum(word_counts),
        "avg_words": sum(word_counts) / len(word_counts)
    }


def filter_texts_by_length(texts: List[str], 
                          min_length: int = 0,
                          max_length: Optional[int] = None) -> List[str]:
    """Filter texts by length constraints.
    
    Args:
        texts: List of text strings
        min_length: Minimum text length
        max_length: Maximum text length (None for no limit)
        
    Returns:
        Filtered list of texts
    """
    filtered = []
    
    for text in texts:
        length = len(text)
        if length >= min_length and (max_length is None or length <= max_length):
            filtered.append(text)
    
    return filtered
