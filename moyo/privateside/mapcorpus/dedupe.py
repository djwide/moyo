"""Deduplication utilities for corpus building."""

import hashlib
from typing import List, Dict, Any, Set, Tuple
from difflib import SequenceMatcher
import logging

from .schema import DocumentChunk

logger = logging.getLogger(__name__)


def calculate_text_hash(text: str) -> str:
    """Calculate a hash for text content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings."""
    return SequenceMatcher(None, text1, text2).ratio()


def find_exact_duplicates(chunks: List[DocumentChunk]) -> List[Tuple[int, int]]:
    """Find exact duplicate chunks based on text content."""
    duplicates = []
    seen_hashes = {}
    
    for i, chunk in enumerate(chunks):
        text_hash = calculate_text_hash(chunk.text)
        
        if text_hash in seen_hashes:
            duplicates.append((seen_hashes[text_hash], i))
        else:
            seen_hashes[text_hash] = i
    
    return duplicates


def find_similar_chunks(chunks: List[DocumentChunk], similarity_threshold: float = 0.9) -> List[Tuple[int, int, float]]:
    """Find similar chunks based on text similarity."""
    similar_pairs = []
    
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            similarity = calculate_similarity(chunks[i].text, chunks[j].text)
            if similarity >= similarity_threshold:
                similar_pairs.append((i, j, similarity))
    
    return similar_pairs


def remove_duplicates(chunks: List[DocumentChunk], 
                     exact_duplicates: bool = True, 
                     similar_duplicates: bool = False,
                     similarity_threshold: float = 0.9) -> Tuple[List[DocumentChunk], int]:
    """Remove duplicate chunks from the list.
    
    Args:
        chunks: List of document chunks
        exact_duplicates: Whether to remove exact duplicates
        similar_duplicates: Whether to remove similar duplicates
        similarity_threshold: Threshold for similarity detection
        
    Returns:
        Tuple of (deduplicated_chunks, duplicates_removed_count)
    """
    if not chunks:
        return chunks, 0
    
    original_count = len(chunks)
    deduplicated_chunks = chunks.copy()
    duplicates_removed = 0
    
    # Remove exact duplicates
    if exact_duplicates:
        exact_dups = find_exact_duplicates(deduplicated_chunks)
        if exact_dups:
            # Sort in reverse order to remove from end first
            indices_to_remove = sorted(set(idx for pair in exact_dups for idx in pair[1:]), reverse=True)
            for idx in indices_to_remove:
                del deduplicated_chunks[idx]
            duplicates_removed += len(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} exact duplicates")
    
    # Remove similar duplicates
    if similar_duplicates and len(deduplicated_chunks) > 1:
        similar_pairs = find_similar_chunks(deduplicated_chunks, similarity_threshold)
        if similar_pairs:
            # Keep the first occurrence of each similar pair
            indices_to_remove = set()
            for i, j, similarity in similar_pairs:
                indices_to_remove.add(j)  # Remove the second occurrence
            
            # Remove in reverse order
            for idx in sorted(indices_to_remove, reverse=True):
                del deduplicated_chunks[idx]
            duplicates_removed += len(indices_to_remove)
            logger.info(f"Removed {len(indices_to_remove)} similar duplicates (threshold: {similarity_threshold})")
    
    # Update chunk indices after deduplication
    for i, chunk in enumerate(deduplicated_chunks):
        chunk.chunk_index = i
    
    total_removed = original_count - len(deduplicated_chunks)
    logger.info(f"Deduplication complete: {original_count} -> {len(deduplicated_chunks)} chunks ({total_removed} removed)")
    
    return deduplicated_chunks, total_removed


def analyze_duplicates(chunks: List[DocumentChunk], 
                      similarity_threshold: float = 0.9) -> Dict[str, Any]:
    """Analyze duplicates in a chunk list without removing them.
    
    Returns:
        Dictionary with duplicate analysis information
    """
    analysis = {
        "total_chunks": len(chunks),
        "exact_duplicates": [],
        "similar_duplicates": [],
        "duplicate_groups": [],
        "summary": {}
    }
    
    if not chunks:
        return analysis
    
    # Find exact duplicates
    exact_dups = find_exact_duplicates(chunks)
    analysis["exact_duplicates"] = exact_dups
    
    # Find similar duplicates
    similar_pairs = find_similar_chunks(chunks, similarity_threshold)
    analysis["similar_duplicates"] = similar_pairs
    
    # Group duplicates
    duplicate_groups = {}
    
    # Group exact duplicates
    for i, j in exact_dups:
        group_key = calculate_text_hash(chunks[i].text)
        if group_key not in duplicate_groups:
            duplicate_groups[group_key] = {"type": "exact", "indices": [i], "text": chunks[i].text}
        duplicate_groups[group_key]["indices"].append(j)
    
    # Group similar duplicates
    for i, j, similarity in similar_pairs:
        group_key = f"similar_{i}_{j}"
        duplicate_groups[group_key] = {
            "type": "similar", 
            "indices": [i, j], 
            "similarity": similarity,
            "texts": [chunks[i].text, chunks[j].text]
        }
    
    analysis["duplicate_groups"] = list(duplicate_groups.values())
    
    # Summary
    analysis["summary"] = {
        "exact_duplicate_count": len(exact_dups),
        "similar_duplicate_count": len(similar_pairs),
        "unique_chunks_after_exact_dedup": len(chunks) - len(set(idx for pair in exact_dups for idx in pair)),
        "duplicate_groups_count": len(duplicate_groups)
    }
    
    return analysis


def get_duplicate_statistics(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Get statistics about duplicates in the chunk list."""
    if not chunks:
        return {"total_chunks": 0, "unique_chunks": 0, "duplicate_ratio": 0.0}
    
    # Calculate unique chunks based on text content
    unique_texts = set(chunk.text for chunk in chunks)
    unique_count = len(unique_texts)
    total_count = len(chunks)
    
    return {
        "total_chunks": total_count,
        "unique_chunks": unique_count,
        "duplicate_ratio": (total_count - unique_count) / total_count if total_count > 0 else 0.0,
        "duplicates_removed": total_count - unique_count
    }
