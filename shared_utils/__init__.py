"""Shared utilities for sente and moyo projects."""

__version__ = "0.1.0"

from .embeddings import (
    embed,
    get_embedding_model,
    get_embedding_dimension,
    resolve_device,
    cuda_available,
    get_device_info,
    clear_embedding_cache,
)
from .chunking import (
    chunk_text,
    chunk_text_simple,
    chunk_lines,
    chunk_text_multi_granularity,
    GranularChunk,
    estimate_token_count,
)
from .faiss_index import FAISSIndex, StringStore, build_index, build_index_from_text
from .storage import ensure_directory, list_files, copy_directory, get_storage, LocalStorage
from .ids import (
    generate_id, 
    generate_timestamp_id,
    generate_stable_document_id,
    generate_content_hash,
    generate_chunk_id,
    generate_corpus_id,
    generate_index_id,
    generate_fingerprint,
    parse_document_id,
    is_stable_id,
    migrate_to_stable_id
)
from .logging import get_logger
from .text_processing import (
    normalize_text,
    deduplicate_texts,
    calculate_text_similarity,
    get_text_statistics,
    TextNormalizationConfig,
    DeduplicationConfig
)
from .document_schema import (
    NormalizedDocument,
    DocumentCollection,
    TextChunk,
    DocumentType,
    DocumentSource
)
from .document_processor import (
    DocumentProcessor,
    DocumentConverter,
    create_document_from_file,
    create_document_from_text,
    batch_process_files
)
from .regex_utils import (
    build_regex_pattern,
    is_literal,
    update_regex_rules,
    update_regex_rules_from_path,
    combine_regex_rules_master,
    filter_comments,
    AhoCorasickMatcher,
    get_aho_corasick_matcher,
    build_aho_corasick_matcher,
    match_text_optimized,
    match_text_simple,
    load_regex_rules
)
from .file_ops import (
    read_lines,
    iter_text_files,
    combine_files,
    clean_text_lines,
    remove_empty_lines,
    split_long_lines,
    escape_latex,
    encode_content,
    write_outputs,
    encode_file,
    read_allowlist,
    update_allowlist_from_path,
    index_filename_for_model,
    safe_index_filename_for_model
)

# Import benchmark module
try:
    from .benchmark import main as benchmark_main
except ImportError:
    benchmark_main = None

__all__ = [
    "embed",
    "get_embedding_model",
    "get_embedding_dimension",
    "resolve_device",
    "cuda_available",
    "get_device_info",
    "clear_embedding_cache", 
    "chunk_text",
    "chunk_text_simple",
    "chunk_lines",
    "chunk_text_multi_granularity",
    "GranularChunk",
    "estimate_token_count",
    "FAISSIndex",
    "StringStore",
    "build_index",
    "build_index_from_text",
    "ensure_directory",
    "list_files", 
    "copy_directory",
    "get_storage",
    "LocalStorage",
    "generate_id",
    "generate_timestamp_id",
    "generate_stable_document_id",
    "generate_content_hash",
    "generate_chunk_id",
    "generate_corpus_id",
    "generate_index_id",
    "generate_fingerprint",
    "parse_document_id",
    "is_stable_id",
    "benchmark_main",
    "migrate_to_stable_id",
    "get_logger",
    "normalize_text",
    "deduplicate_texts",
    "calculate_text_similarity",
    "get_text_statistics",
    "TextNormalizationConfig",
    "DeduplicationConfig",
    "build_regex_pattern",
    "is_literal",
    "update_regex_rules",
    "update_regex_rules_from_path",
    "combine_regex_rules_master",
    "filter_comments",
    "AhoCorasickMatcher",
    "get_aho_corasick_matcher",
    "build_aho_corasick_matcher",
    "match_text_optimized",
    "match_text_simple",
    "load_regex_rules",
    "read_lines",
    "iter_text_files",
    "combine_files",
    "clean_text_lines",
    "remove_empty_lines",
    "split_long_lines",
    "escape_latex",
    "encode_content",
    "write_outputs",
    "encode_file",
    "read_allowlist",
    "update_allowlist_from_path",
    "index_filename_for_model",
    "safe_index_filename_for_model",
    "NormalizedDocument",
    "DocumentCollection",
    "TextChunk",
    "DocumentType",
    "DocumentSource",
    "DocumentProcessor",
    "DocumentConverter",
    "create_document_from_file",
    "create_document_from_text",
    "batch_process_files"
]
