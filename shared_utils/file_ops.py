"""File operations for reading, writing, and processing text files."""

import base64
import codecs
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Tuple

logger = logging.getLogger(__name__)

# Constants
MAX_LINE_LENGTH = 1000
ZERO_WIDTH_MAP = {"0": "\u200b", "1": "\u200c"}


def read_text_file(path: Path, encoding: str = 'utf-8', errors: str = 'ignore') -> str:
    """Read a text file and return its content.
    
    Args:
        path: Path to the text file
        encoding: File encoding (default: utf-8)
        errors: How to handle encoding errors (default: ignore)
        
    Returns:
        File content as string
    """
    try:
        with open(path, 'r', encoding=encoding, errors=errors) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        raise


def read_lines(paths: Iterable[Path], cli_mode: bool = False) -> List[str]:
    """Read lines from multiple files.
    
    Args:
        paths: Iterable of file paths to read
        cli_mode: Whether running in CLI mode
        
    Returns:
        List of lines from all files
    """
    all_lines = []
    
    for path in paths:
        if not path.exists():
            if cli_mode:
                logger.warning(f"File not found: {path}")
            continue
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                all_lines.extend(lines)
                if cli_mode:
                    logger.info(f"Read {len(lines)} lines from {path}")
        except Exception as e:
            if cli_mode:
                logger.error(f"Error reading {path}: {e}")
    
    return all_lines


def iter_text_files(directory: Path) -> Iterable[str]:
    """Iterate over text files in a directory.
    
    Args:
        directory: Directory to search
        
    Yields:
        File paths as strings
    """
    if not directory.exists() or not directory.is_dir():
        return
    
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css']:
            yield str(file_path)


def combine_files(directory: Path, output: Path, cli_mode: bool = False) -> List[str]:
    """Combine all text files in a directory into a single output file.
    
    Args:
        directory: Directory containing files to combine
        output: Output file path
        cli_mode: Whether running in CLI mode
        
    Returns:
        List of all lines from combined files
    """
    all_lines = []
    
    for file_path in iter_text_files(directory):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                all_lines.extend(lines)
                if cli_mode:
                    logger.info(f"Added {len(lines)} lines from {file_path}")
        except Exception as e:
            if cli_mode:
                logger.warning(f"Error reading {file_path}: {e}")
    
    # Write combined output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    
    if cli_mode:
        logger.info(f"Combined {len(all_lines)} lines into {output}")
    
    return all_lines


def clean_text_lines(lines: List[str]) -> List[str]:
    """Clean and process text lines.
    
    Args:
        lines: List of text lines
        
    Returns:
        Cleaned lines
    """
    # Remove empty lines and whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    # Remove comments
    lines = [line for line in lines if not line.startswith('#')]
    
    # Split long lines
    lines = split_long_lines(lines)
    
    return lines


def remove_empty_lines(lines: List[str]) -> List[str]:
    """Remove empty lines from a list of strings.
    
    Args:
        lines: List of text lines
        
    Returns:
        Lines with empty lines removed
    """
    return [line for line in lines if line.strip()]


def split_long_lines(lines: List[str], max_length: int = MAX_LINE_LENGTH) -> List[str]:
    """Split lines that exceed the maximum length.
    
    Args:
        lines: List of text lines
        max_length: Maximum line length
        
    Returns:
        Lines with long lines split
    """
    result = []
    
    for line in lines:
        if len(line) <= max_length:
            result.append(line)
        else:
            # Split on word boundaries
            words = line.split()
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_length:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    if current_line:
                        result.append(current_line)
                    current_line = word
            
            if current_line:
                result.append(current_line)
    
    return result


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text
    """
    latex_chars = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '$': r'\$',
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '~': r'\textasciitilde{}',
    }
    
    for char, replacement in latex_chars.items():
        text = text.replace(char, replacement)
    
    return text


def encode_content(content: bytes) -> Tuple[str, str, str, str, str]:
    """Encode content in multiple formats.
    
    Args:
        content: Binary content to encode
        
    Returns:
        Tuple of (base64, hex, binary, zero_width, latex)
    """
    # Base64 encoding
    base64_encoded = base64.b64encode(content).decode('ascii')
    
    # Hex encoding
    hex_encoded = content.hex()
    
    # Binary encoding
    binary_encoded = ''.join(format(byte, '08b') for byte in content)
    
    # Zero-width encoding
    zero_width_encoded = ''.join(ZERO_WIDTH_MAP.get(bit, bit) for bit in binary_encoded)
    
    # LaTeX encoding
    latex_encoded = escape_latex(content.decode('utf-8', errors='ignore'))
    
    return base64_encoded, hex_encoded, binary_encoded, zero_width_encoded, latex_encoded


def write_outputs(input_path: str) -> Tuple[str, str, str, str, str]:
    """Write encoded outputs for a given input file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Tuple of output file paths
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read content
    with open(input_file, 'rb') as f:
        content = f.read()
    
    # Encode content
    base64_encoded, hex_encoded, binary_encoded, zero_width_encoded, latex_encoded = encode_content(content)
    
    # Write outputs
    base_path = input_file.stem
    
    base64_path = input_file.parent / f"{base_path}_base64.txt"
    hex_path = input_file.parent / f"{base_path}_hex.txt"
    binary_path = input_file.parent / f"{base_path}_binary.txt"
    zero_width_path = input_file.parent / f"{base_path}_zero_width.txt"
    latex_path = input_file.parent / f"{base_path}_latex.txt"
    
    base64_path.write_text(base64_encoded)
    hex_path.write_text(hex_encoded)
    binary_path.write_text(binary_encoded)
    zero_width_path.write_text(zero_width_encoded)
    latex_path.write_text(latex_encoded)
    
    logger.info(f"Wrote encoded outputs for {input_path}")
    
    return str(base64_path), str(hex_path), str(binary_path), str(zero_width_path), str(latex_path)


def encode_file(input_path: Path) -> List[Path]:
    """Encode a file and return paths to encoded outputs.
    
    Args:
        input_path: Path to input file
        
    Returns:
        List of output file paths
    """
    outputs = write_outputs(str(input_path))
    return [Path(output) for output in outputs]


def read_allowlist(allowlist_path: Path) -> List[str]:
    """Read allowlist from file.
    
    Args:
        allowlist_path: Path to allowlist file
        
    Returns:
        List of allowlist entries
    """
    if not allowlist_path.exists():
        return []
    
    try:
        with open(allowlist_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        logger.error(f"Error reading allowlist {allowlist_path}: {e}")
        return []


def update_allowlist_from_path(source: Path, allowlist_path: Path) -> None:
    """Update allowlist from source file.
    
    Args:
        source: Source file path
        allowlist_path: Allowlist file path
    """
    if not source.exists():
        logger.warning(f"Source file not found: {source}")
        return
    
    try:
        # Read source file
        with open(source, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        
        # Clean lines
        lines = [line.strip() for line in lines if line.strip()]
        lines = [line for line in lines if not line.startswith('#')]
        
        # Write allowlist
        allowlist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(allowlist_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        
        logger.info(f"Updated allowlist from {source}: {len(lines)} entries")
        
    except Exception as e:
        logger.error(f"Error updating allowlist: {e}")


def index_filename_for_model(model_key: str) -> str:
    """Generate index filename for a model.
    
    Args:
        model_key: Model key/name
        
    Returns:
        Index filename
    """
    return f"{model_key}_index.faiss"


def safe_index_filename_for_model(model_key: str) -> str:
    """Generate safe index filename for a model.
    
    Args:
        model_key: Model key/name
        
    Returns:
        Safe index filename
    """
    return f"{model_key}_safe_index.faiss"
