import re
from pathlib import Path
from typing import List, Tuple, Dict, Any


def validate_text(text: str, min_length: int = 10, max_length: int = 1000000) -> Tuple[bool, str]:
    """Validate text input.
    
    Args:
        text: Text to validate
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text is empty"
    
    text = text.strip()
    
    if len(text) < min_length:
        return False, f"Text too short (minimum {min_length} characters)"
    
    if len(text) > max_length:
        return False, f"Text too long (maximum {max_length} characters)"
    
    # Check for reasonable text content (not just whitespace/special chars)
    if not re.search(r'[a-zA-Z0-9]', text):
        return False, "Text contains no alphanumeric characters"
    
    return True, ""


def validate_file_path(file_path: Path, allowed_extensions: List[str] = None) -> Tuple[bool, str]:
    """Validate file path and basic file properties.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.txt', '.md'])
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"
    
    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check file size (max 10MB)
    try:
        file_size = file_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB
            return False, f"File too large: {file_size / (1024*1024):.1f}MB (max 10MB)"
    except OSError:
        return False, f"Cannot read file: {file_path}"
    
    # Check file extension
    if allowed_extensions:
        if file_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            return False, f"File extension not allowed: {file_path.suffix} (allowed: {allowed_extensions})"
    
    return True, ""


def validate_file_content(file_path: Path, encoding: str = "utf-8") -> Tuple[bool, str, str]:
    """Validate and read file content.
    
    Args:
        file_path: Path to file
        encoding: File encoding to try
        
    Returns:
        Tuple of (is_valid, error_message, content)
    """
    # First validate the file path
    is_valid, error = validate_file_path(file_path)
    if not is_valid:
        return False, error, ""
    
    # Try to read the file
    try:
        content = file_path.read_text(encoding=encoding)
        return True, "", content
    except UnicodeDecodeError:
        # Try with different encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                content = file_path.read_text(encoding=enc)
                return True, "", content
            except UnicodeDecodeError:
                continue
        return False, f"Cannot decode file with any encoding: {file_path}", ""
    except Exception as e:
        return False, f"Error reading file: {e}", ""


def validate_multiple_files(file_paths: List[Path], allowed_extensions: List[str] = None) -> Dict[Path, Tuple[bool, str]]:
    """Validate multiple files.
    
    Args:
        file_paths: List of file paths to validate
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Dictionary mapping file paths to (is_valid, error_message)
    """
    results = {}
    
    for file_path in file_paths:
        is_valid, error = validate_file_path(file_path, allowed_extensions)
        results[file_path] = (is_valid, error)
    
    return results


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get information about a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    try:
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "extension": file_path.suffix.lower(),
            "modified": stat.st_mtime,
            "exists": True
        }
    except OSError:
        return {
            "name": file_path.name,
            "exists": False,
            "error": "Cannot access file"
        }
