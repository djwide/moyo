"""Shared text chunking utilities for sente and moyo projects."""

import re
from typing import List, Iterator, Optional


def chunk_text(text: str, 
               chunk_size: int = 512, 
               overlap: int = 50,
               preserve_sentences: bool = True) -> List[str]:
    """Split text into overlapping chunks with sentence boundary awareness.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        preserve_sentences: Whether to try to preserve sentence boundaries
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    if not preserve_sentences:
        return chunk_text_simple(text, chunk_size, overlap)
    
    # Split into sentences without breaking decimal numbers (e.g., 3.14)
    # Rule: split only on sentence-ending punctuation followed by whitespace,
    # and never when the dot is part of a numeric decimal pattern
    sentences = re.split(r'(?<!\d)(?<=[.!?])\s+(?!\d)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        # Fallback to character-based chunking
        return chunk_text_simple(text, chunk_size, overlap)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_text_simple(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple character-based chunking with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    
    return chunks


def chunk_lines(lines: Iterator[str], 
                max_lines_per_chunk: int = 10,
                max_chunk_size: Optional[int] = None) -> Iterator[str]:
    """Chunk text by grouping lines together.
    
    Args:
        lines: Iterator of text lines
        max_lines_per_chunk: Maximum number of lines per chunk
        max_chunk_size: Maximum character size per chunk (optional)
        
    Yields:
        Text chunks as strings
    """
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_size = len(line)
        
        # Check if adding this line would exceed limits
        if (len(current_chunk) >= max_lines_per_chunk or 
            (max_chunk_size and current_size + line_size > max_chunk_size)):
            if current_chunk:
                yield "\n".join(current_chunk)
                current_chunk = []
                current_size = 0
        
        current_chunk.append(line)
        current_size += line_size
    
    # Yield remaining lines
    if current_chunk:
        yield "\n".join(current_chunk)


def chunk_by_tokens(text: str, 
                   tokenizer=None,
                   max_tokens: int = 512,
                   overlap_tokens: int = 50) -> List[str]:
    """Chunk text by tokens using a tokenizer.
    
    Args:
        text: Input text to chunk
        tokenizer: Tokenizer to use (e.g., from transformers)
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap
        
    Returns:
        List of text chunks
    """
    if not tokenizer:
        # Fallback to character-based chunking
        return chunk_text(text, max_tokens * 4, overlap_tokens * 4)
    
    try:
        tokens = tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    except Exception:
        # Fallback to character-based chunking
        return chunk_text(text, max_tokens * 4, overlap_tokens * 4)
