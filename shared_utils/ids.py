"""Shared ID generation utilities for sente and moyo projects."""

import uuid
import time
import hashlib
import base64
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime


def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with optional prefix.
    
    Args:
        prefix: Prefix for the ID
        
    Returns:
        Unique ID string
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def generate_timestamp_id(prefix: str = "id") -> str:
    """Generate a unique ID with timestamp and optional prefix.
    
    Args:
        prefix: Prefix for the ID
        
    Returns:
        Unique ID string with timestamp
    """
    timestamp = int(time.time() * 1000)  # Milliseconds
    random_part = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{random_part}"


def generate_uuid() -> str:
    """Generate a full UUID string.
    
    Returns:
        Full UUID string
    """
    return str(uuid.uuid4())


def generate_short_uuid() -> str:
    """Generate a short UUID string.
    
    Returns:
        Short UUID string (8 characters)
    """
    return uuid.uuid4().hex[:8]


def generate_numeric_id() -> int:
    """Generate a numeric ID based on timestamp.
    
    Returns:
        Numeric ID
    """
    return int(time.time() * 1000000)  # Microseconds


def generate_session_id() -> str:
    """Generate a session ID.
    
    Returns:
        Session ID string
    """
    return f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def generate_file_id(filename: str) -> str:
    """Generate a file ID based on filename and timestamp.
    
    Args:
        filename: Original filename
        
    Returns:
        File ID string
    """
    # Clean filename
    clean_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_name = clean_name.replace(' ', '_')
    
    timestamp = int(time.time())
    return f"file_{clean_name}_{timestamp}_{uuid.uuid4().hex[:4]}"


# Stable ID Policy for Normalized Documents

def generate_stable_document_id(
    source: str,
    content_hash: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    namespace: str = "doc"
) -> str:
    """Generate a stable document ID based on source and content.
    
    Stable IDs are deterministic and reproducible for the same content,
    making them suitable for deduplication and cross-system references.
    
    Args:
        source: Document source identifier (file path, URL, etc.)
        content_hash: SHA256 hash of document content (optional)
        timestamp: Document timestamp (optional)
        namespace: ID namespace prefix
        
    Returns:
        Stable document ID string
    """
    # Create a deterministic string from source
    source_normalized = str(source).lower().strip()
    
    # If content hash is provided, use it for stability
    if content_hash:
        # Use first 12 chars of content hash for stability
        hash_part = content_hash[:12]
        return f"{namespace}_{hash_part}_{hashlib.md5(source_normalized.encode()).hexdigest()[:8]}"
    
    # Fallback to source-based hash
    source_hash = hashlib.sha256(source_normalized.encode()).hexdigest()[:16]
    timestamp_part = ""
    if timestamp:
        timestamp_part = f"_{int(timestamp.timestamp())}"
    
    return f"{namespace}_{source_hash}{timestamp_part}"


def generate_content_hash(content: str, algorithm: str = "sha256") -> str:
    """Generate a hash of document content.
    
    Args:
        content: Document content to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Content hash string
    """
    if algorithm == "sha256":
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def generate_chunk_id(
    document_id: str,
    chunk_index: int,
    chunk_hash: Optional[str] = None
) -> str:
    """Generate a stable chunk ID.
    
    Args:
        document_id: Parent document ID
        chunk_index: Chunk index within document
        chunk_hash: Hash of chunk content (optional)
        
    Returns:
        Stable chunk ID string
    """
    if chunk_hash:
        return f"chunk_{document_id}_{chunk_index}_{chunk_hash[:8]}"
    else:
        return f"chunk_{document_id}_{chunk_index}"


def generate_corpus_id(
    name: str,
    timestamp: Optional[datetime] = None,
    namespace: str = "corpus"
) -> str:
    """Generate a corpus ID.
    
    Args:
        name: Corpus name
        timestamp: Creation timestamp (optional)
        namespace: ID namespace prefix
        
    Returns:
        Corpus ID string
    """
    name_normalized = "".join(c for c in name.lower() if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
    timestamp_part = ""
    if timestamp:
        timestamp_part = f"_{int(timestamp.timestamp())}"
    
    return f"{namespace}_{name_normalized}{timestamp_part}"


def generate_index_id(
    corpus_id: str,
    index_type: str,
    timestamp: Optional[datetime] = None
) -> str:
    """Generate an index ID.
    
    Args:
        corpus_id: Parent corpus ID
        index_type: Type of index (private, public, etc.)
        timestamp: Creation timestamp (optional)
        
    Returns:
        Index ID string
    """
    timestamp_part = ""
    if timestamp:
        timestamp_part = f"_{int(timestamp.timestamp())}"
    
    return f"index_{corpus_id}_{index_type}{timestamp_part}"


def parse_document_id(document_id: str) -> Dict[str, Any]:
    """Parse a document ID to extract components.
    
    Args:
        document_id: Document ID to parse
        
    Returns:
        Dictionary with parsed components
    """
    parts = document_id.split('_')
    if len(parts) < 2:
        return {"type": "unknown", "parts": parts}
    
    if parts[0] == "doc":
        if len(parts) >= 3 and len(parts[1]) == 12:  # Content hash format
            return {
                "type": "document",
                "namespace": parts[0],
                "content_hash": parts[1],
                "source_hash": parts[2] if len(parts) > 2 else None
            }
        else:  # Source hash format
            return {
                "type": "document",
                "namespace": parts[0],
                "source_hash": parts[1],
                "timestamp": parts[2] if len(parts) > 2 and parts[2].isdigit() else None
            }
    
    return {"type": "unknown", "parts": parts}


def is_stable_id(identifier: str) -> bool:
    """Check if an ID follows the stable ID format.
    
    Args:
        identifier: ID to check
        
    Returns:
        True if ID follows stable format
    """
    try:
        parsed = parse_document_id(identifier)
        return parsed["type"] in ["document", "chunk", "corpus", "index"]
    except:
        return False


def generate_fingerprint(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate a content fingerprint for deduplication.
    
    Args:
        content: Document content
        metadata: Optional metadata to include in fingerprint
        
    Returns:
        Fingerprint string
    """
    # Normalize content
    normalized_content = content.lower().strip()
    
    # Create fingerprint data
    fingerprint_data = normalized_content
    
    if metadata:
        # Add relevant metadata to fingerprint
        relevant_keys = ['source', 'title', 'author', 'published_date']
        for key in relevant_keys:
            if key in metadata:
                fingerprint_data += str(metadata[key])
    
    # Generate hash
    return hashlib.sha256(fingerprint_data.encode('utf-8')).hexdigest()


# Legacy compatibility functions

def generate_legacy_id(prefix: str = "id") -> str:
    """Generate a legacy-style ID for backward compatibility.
    
    Args:
        prefix: Prefix for the ID
        
    Returns:
        Legacy ID string
    """
    return generate_id(prefix)


def migrate_to_stable_id(legacy_id: str, source: str, content: str) -> str:
    """Migrate a legacy ID to a stable ID.
    
    Args:
        legacy_id: Legacy ID to migrate
        source: Document source
        content: Document content
        
    Returns:
        Stable document ID
    """
    content_hash = generate_content_hash(content)
    return generate_stable_document_id(source, content_hash)
