from dataclasses import dataclass


@dataclass
class Document:
    """Simple text document representation."""
    id: str
    text: str
    source: str
