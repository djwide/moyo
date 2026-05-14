import json
import csv
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_file(path: Path, encoding: str = "utf-8") -> str:
    """Read a text file with error handling."""
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # Try alternative encodings
        for enc in ["latin-1", "cp1252", "utf-8-sig"]:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode file {path} with any encoding")


def load_lines(text: str) -> Iterator[str]:
    """Yield lines from provided text."""
    for line in text.splitlines():
        line = line.strip()
        if line:  # Skip empty lines
            yield line


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load and parse JSON file."""
    content = load_file(path)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def load_csv_file(path: Path, delimiter: str = ",", has_header: bool = True) -> List[Dict[str, str]]:
    """Load CSV file and return as list of dictionaries."""
    content = load_file(path)
    
    try:
        reader = csv.reader(content.splitlines(), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            return []
        
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows
        
        result = []
        for row in data_rows:
            if len(row) != len(headers):
                # Pad or truncate row to match headers
                row = row[:len(headers)] + [""] * (len(headers) - len(row))
            
            row_dict = dict(zip(headers, row))
            result.append(row_dict)
        
        return result
    except Exception as e:
        raise ValueError(f"Error parsing CSV {path}: {e}")


def load_markdown_file(path: Path) -> str:
    """Load markdown file and return as plain text."""
    content = load_file(path)
    
    # Simple markdown to text conversion
    # Remove markdown formatting
    import re
    
    # Remove headers
    content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
    
    # Remove bold/italic
    content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
    content = re.sub(r'\*(.*?)\*', r'\1', content)
    content = re.sub(r'__(.*?)__', r'\1', content)
    content = re.sub(r'_(.*?)_', r'\1', content)
    
    # Remove code blocks
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content = re.sub(r'`(.*?)`', r'\1', content)
    
    # Remove links
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    
    # Remove images
    content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', content)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content)
    
    return content.strip()


def load_file_by_type(path: Path) -> str:
    """Load file based on its extension and return as text."""
    extension = path.suffix.lower()
    
    if extension in ['.txt', '.text']:
        return load_file(path)
    elif extension == '.json':
        data = load_json_file(path)
        return json.dumps(data, indent=2)
    elif extension == '.csv':
        data = load_csv_file(path)
        # Convert to text representation
        lines = []
        for row in data:
            lines.append(' | '.join(f"{k}: {v}" for k, v in row.items()))
        return '\n'.join(lines)
    elif extension in ['.md', '.markdown']:
        return load_markdown_file(path)
    else:
        # Default to text loading
        return load_file(path)


def extract_text_from_content(content: str, content_type: str = "text") -> str:
    """Extract plain text from different content types."""
    if content_type == "json":
        try:
            data = json.loads(content)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError:
            return content
    elif content_type == "csv":
        try:
            data = load_csv_file_from_string(content)
            lines = []
            for row in data:
                lines.append(' | '.join(f"{k}: {v}" for k, v in row.items()))
            return '\n'.join(lines)
        except Exception:
            return content
    else:
        return content


def load_csv_file_from_string(content: str, delimiter: str = ",") -> List[Dict[str, str]]:
    """Load CSV from string content."""
    try:
        reader = csv.reader(content.splitlines(), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            return []
        
        headers = rows[0]
        data_rows = rows[1:]
        
        result = []
        for row in data_rows:
            if len(row) != len(headers):
                row = row[:len(headers)] + [""] * (len(headers) - len(row))
            
            row_dict = dict(zip(headers, row))
            result.append(row_dict)
        
        return result
    except Exception as e:
        raise ValueError(f"Error parsing CSV content: {e}")


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.txt', '.text', '.json', '.csv', '.md', '.markdown']
