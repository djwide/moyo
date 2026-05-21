"""Map private data into a searchable corpus."""

from .builder import CorpusBuilder, build_corpus_from_files, build_corpus_from_texts, build_corpus_from_gui_bridge
from .schema import CorpusConfig, CorpusBuildResult, CorpusInfo
from .centroids import (
    compute_corpus_centroids,
    extract_topic_tokens,
    tokens_for_corpus,
)

__all__ = [
    "CorpusBuilder",
    "CorpusConfig",
    "CorpusBuildResult",
    "CorpusInfo",
    "build_corpus_from_files",
    "build_corpus_from_texts",
    "build_corpus_from_gui_bridge",
    "compute_corpus_centroids",
    "extract_topic_tokens",
    "tokens_for_corpus",
]
