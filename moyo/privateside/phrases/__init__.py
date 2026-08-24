"""Local sensitive-phrases corpus: ingest, filter framing, approve labels."""

from moyo.privateside.phrases.ingest import ingest_document, ingest_text, load_document_text
from moyo.privateside.phrases.store import PhraseStore

__all__ = [
    "PhraseStore",
    "ingest_document",
    "ingest_text",
    "load_document_text",
]
