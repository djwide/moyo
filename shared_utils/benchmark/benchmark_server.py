"""Benchmark and profile the SenteGuard linting backend.

This script exercises the :mod:`sentesdk.linter` module and provides
comprehensive performance testing for different models, devices, and configurations.
"""
import argparse
import json
import time
import sys
import pathlib

import torch

# Resolve repo root and add potential package roots to sys.path BEFORE imports
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
# Add top-level 'packages' if present (Cursor monorepo layout)
for parent in [repo_root] + list(repo_root.parents):
    pkg_dir = parent / "packages"
    if pkg_dir.exists():
        sys.path.insert(0, str(pkg_dir))
        break
# Add nested 'sente/packages' if present (repo structure with nested app)
for parent in [repo_root] + list(repo_root.parents):
    nested_pkg_dir = parent / "sente" / "packages"
    if nested_pkg_dir.exists():
        sys.path.insert(0, str(nested_pkg_dir))
        break

# Import linter with robust fallbacks for different layouts
try:
    from sentesdk import linter  # installed or on sys.path via packages/
except ImportError:
    try:
        # Use package aggregator which re-exports linter
        from sente import linter  # relies on sente/__init__.py exporting linter
    except ImportError as e:
        # Direct path fallback to monorepo layout
        try:
            from sente.packages.sentesdk.sentesdk import linter  # type: ignore
        except Exception:
            raise e


EMBEDDING_MODELS = {
    "mini": "all-MiniLM-L6-v2",
    # Free multilingual model (~1GB)
    "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # Option for OpenAI embeddings (requires API access)
    "openai-large": "text-embedding-3-large",
    "openai-small": "text-embedding-3-small",
}

# Hardcoded times (in seconds) for other DLP products when scanning
# a 10k line batch. These are shown after each synthetic benchmark of
# the same size for comparison purposes.
COMPETITOR_BATCH_TIMES = [
    ("Competitor 1 Benchmark", 140.2),
    ("Competitor 2 Benchmark", 70.2),
    ("Competitor 3 Benchmark", 70.3),
]


def benchmark(
    lines, device: str, model_name: str, *, onnx: bool = False, onnx_model: str = "shared_utils/models/miniLM_fp32.onnx", context_window: int = 1, scan_types: str = "all"
) -> float:
    """Time ``lint`` without including model/index loading."""
    
    # For static-only scanning, skip model/index loading and embedding generation
    if scan_types == "static":
        start = time.perf_counter()
        # Run static checks directly without loading model or index
        # Use individual line processing for maximum performance
        static_hits = 0
        for line in lines:
            if linter.static_issue(line):
                static_hits += 1
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start
    
    # For other scan types, load model and index as usual
    if onnx:
        from shared_utils import backend_onnx

        root = linter.DATA_DIR
        index = linter.load_index(root / linter.INDEX_FILENAME, use_gpu=device == "cuda")
        start = time.perf_counter()
        linter.lint(
            lines,
            None,
            index,
            encode_fn=lambda ls: backend_onnx.encode(ls, model_path=onnx_model),
            context_window=context_window,
            scan_types=scan_types,
        )
    else:
        model = linter.load_model(device=device, model_name=model_name)
        index = linter.load_index_for_model(model_name, use_gpu=device == "cuda")
        start = time.perf_counter()
        linter.lint(lines, model, index, context_window=context_window, scan_types=scan_types)

    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def profile(
    lines, device: str, model_name: str, *, onnx: bool = False, onnx_model: str = "shared_utils/models/miniLM_fp32.onnx", context_window: int = 1, scan_types: str = "all"
) -> float:
    """Break down ``lint`` into individual steps for profiling.

    The returned timing includes model/index load, encoding, semantic search,
    static rule checks (competitors) and semantic threshold evaluation.
    """
    
    # For static-only scanning, skip model/index loading and embedding generation
    if scan_types == "static":
        index_t = 0.0
        encode_t = 0.0
        
        start = time.perf_counter()
        # Use individual line processing for maximum performance
        static_hits = 0
        for line in lines:
            if linter.static_issue(line):
                static_hits += 1
        if device == "cuda":
            torch.cuda.synchronize()
        static_t = time.perf_counter() - start
        
        total = index_t + encode_t + static_t
        print(f"index={index_t:.3f}s\nencode={encode_t:.3f}s\nstatic={static_t:.3f}s")
        return total
    
    # For other scan types, load model and index as usual
    if onnx:
        from shared_utils import backend_onnx

        start = time.perf_counter()
        root = linter.DATA_DIR
        index = linter.load_index(root / linter.INDEX_FILENAME, use_gpu=device == "cuda")
        if device == "cuda":
            torch.cuda.synchronize()
        index_t = time.perf_counter() - start
        start = time.perf_counter()
        embeddings = backend_onnx.encode(lines, model_path=onnx_model)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_t = time.perf_counter() - start
    else:
        model = linter.load_model(device=device, model_name=model_name)

        start = time.perf_counter()
        index = linter.load_index_for_model(model_name, use_gpu=device == "cuda")
        if device == "cuda":
            torch.cuda.synchronize()
        index_t = time.perf_counter() - start
        start = time.perf_counter()
        embeddings = model.encode(lines, normalize_embeddings=True)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_t = time.perf_counter() - start

    # Initialize timing variables
    search_t = 0.0
    static_t = 0.0
    semantic_t = 0.0
    
    # Run semantic search if cosine scanning is enabled
    if scan_types in ['cosine', 'all']:
        start = time.perf_counter()
        scores = linter.semantic_scores(embeddings, index)
        if device == "cuda":
            torch.cuda.synchronize()
        search_t = time.perf_counter() - start

    # Run static checks if static scanning is enabled
    if scan_types in ['static', 'all']:
        start = time.perf_counter()
        static_msgs = [linter.static_issue(line) for line in lines]
        if device == "cuda":
            torch.cuda.synchronize()
        static_t = time.perf_counter() - start

    # Run semantic threshold evaluation if cosine scanning is enabled
    if scan_types in ['cosine', 'all']:
        start = time.perf_counter()
        for i, msg in enumerate(static_msgs if 'static' in scan_types or scan_types == 'all' else [None] * len(lines)):
            if msg:
                continue
            if scores[i] > linter.THRESHOLD:
                pass
        semantic_t = time.perf_counter() - start

    total = index_t + encode_t + search_t + static_t + semantic_t
    
    # Print timing breakdown based on enabled scan types
    timing_parts = []
    timing_parts.append(f"index={index_t:.3f}s")
    timing_parts.append(f"encode={encode_t:.3f}s")
    
    if scan_types in ['cosine', 'all']:
        timing_parts.append(f"search={search_t:.3f}s")
    
    if scan_types in ['static', 'all']:
        timing_parts.append(f"static={static_t:.3f}s")
    
    if scan_types in ['cosine', 'all']:
        timing_parts.append(f"semantic={semantic_t:.3f}s")
    
    print("\n".join(timing_parts))
    return total


def main() -> None:
    """Main entry point for the benchmark server."""
    parser = argparse.ArgumentParser(
        description="Benchmark and profile SenteGuard linting performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with default settings
  python -m shared_utils.benchmark

  # Test specific model on CPU
  python -m shared_utils.benchmark --embedding mini --device cpu

  # Compare CPU and GPU performance
  python -m shared_utils.benchmark --device both --embedding mini

  # Test all models
  python -m shared_utils.benchmark --embedding all

  # Profile with detailed timing breakdown
  python -m shared_utils.benchmark --profile --embedding mini

  # Test with custom file
  python -m shared_utils.benchmark --source file --file myfile.txt

  # Test ONNX backend
  python -m shared_utils.benchmark --backend onnx

  # Compare SentenceTransformer vs ONNX backends
  python -m shared_utils.benchmark --backend both

  # Test with different context window sizes
  python -m shared_utils.benchmark --context-window 2

  # Get detailed results for file analysis
  python -m shared_utils.benchmark --source file --file myfile.txt --print-json
        """
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show detailed timing breakdown for index build, encoding, searching and rule checks",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "both"],
        default="auto",
        help="Device to benchmark on (default: auto-detect)",
    )
    parser.add_argument(
        "--embedding",
        nargs="+",
        choices=list(EMBEDDING_MODELS.keys()) + ["all"],
        default=["mini"],
        help="Embedding model(s) to benchmark. Use 'all' for every model (default: mini)",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX runtime encoder instead of SentenceTransformer (deprecated: use --backend onnx)",
    )
    parser.add_argument(
        "--backend",
        choices=["st", "onnx", "both"],
        default=None,
        help="Encoder backend: SentenceTransformer ('st'), ONNX runtime ('onnx') or both",
    )
    parser.add_argument(
        "--onnx-model",
        default="shared_utils/models/miniLM_fp32.onnx",
        help="Path to ONNX model file (used with ONNX backend)",
    )
    parser.add_argument(
        "--file",
        help="Path to a file whose contents will be linted in addition to synthetic test cases",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "file", "both"],
        default="synthetic",
        help="Benchmark synthetic data, a file or both (default: synthetic)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print detailed JSON lint results when scanning a file",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help="Number of lines per context window (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size for embedding requests (useful for OpenAI API optimization)",
    )
    parser.add_argument(
        "--scan-types",
        nargs="+",
        choices=["static", "cosine", "all"],
        default=["all"],
        help="Scan types to benchmark: static (regex), cosine (semantic similarity), or all (default: all)",
    )
    args = parser.parse_args()

    if args.backend is not None:
        if args.backend == "both":
            backend_flags = [False, True]
        else:
            backend_flags = [args.backend == "onnx"]
    else:
        backend_flags = [args.onnx]

    devices = []
    if args.device == "both":
        devices = ["cpu", "cuda"]
    elif args.device == "auto":
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
    else:
        devices = [args.device]

    embeddings = args.embedding
    if "all" in embeddings:
        embeddings = list(EMBEDDING_MODELS.keys())
    
    # Handle scan types
    scan_types_list = args.scan_types
    if "all" in scan_types_list:
        scan_types_list = ["all"]

    include_synth = args.source in ("synthetic", "both")
    include_file = args.source in ("file", "both")
    file_lines = None
    if include_file:
        if not args.file:
            parser.error("--source file or both requires --file")
        with open(args.file, "r", encoding="utf-8") as f:
            file_lines = f.read().splitlines()

    for emb_key in embeddings:
        model_name = EMBEDDING_MODELS[emb_key]
        for scan_type in scan_types_list:
            print(f"\n=== Benchmarking {emb_key} encoder ({model_name}) with {scan_type} scanning ===")
            for onnx_flag in backend_flags:
                backend_name = "ONNX" if onnx_flag else "SentenceTransformer"
                if len(backend_flags) > 1:
                    print(f"-- Backend: {backend_name} --")
                for dev in devices:
                    if dev == "cuda" and not torch.cuda.is_available():
                        print("CUDA not available; skipping GPU benchmark")
                        continue
                if include_synth:
                    for count in [100, 1000, 10000]:
                        lines = ["print('hello')"] * count
                        if args.profile:
                            elapsed = profile(
                                lines,
                                dev,
                                model_name,
                                onnx=onnx_flag,
                                onnx_model=args.onnx_model,
                                context_window=args.context_window,
                                scan_types=scan_type,
                            )
                        else:
                            elapsed = benchmark(
                                lines,
                                dev,
                                model_name,
                                onnx=onnx_flag,
                                onnx_model=args.onnx_model,
                                context_window=args.context_window,
                                scan_types=scan_type,
                            )

                        print(
                            f"{backend_name} {dev}: processed {len(lines)} lines in {elapsed:.3f}s \n"
                        )
                        if count == 10000:
                            for comp, t in COMPETITOR_BATCH_TIMES:
                                print(f"{comp}: processed 10000 lines in {t:.1f}s")
                            print(" ")

                if include_file and file_lines is not None:
                    # Always profile real file input so the timing breakdown
                    # matches the synthetic benchmarks
                    elapsed = profile(
                        file_lines,
                        dev,
                        model_name,
                        onnx=onnx_flag,
                        onnx_model=args.onnx_model,
                        context_window=args.context_window,
                        scan_types=scan_type,
                    )

                    print(
                        f"{backend_name} {dev}: processed {len(file_lines)} lines from {args.file} in {elapsed:.3f}s"
                    )

                    if onnx_flag:
                        from shared_utils import backend_onnx

                        index = linter.load_index_for_model(
                            model_name, use_gpu=dev == "cuda"
                        )
                        results = linter.lint(
                            file_lines,
                            None,
                            index,
                            encode_fn=lambda ls: backend_onnx.encode(
                                ls, model_path=args.onnx_model
                            ),
                            debug=True,
                            return_line_info=True,
                            context_window=args.context_window,
                            scan_types=scan_type,
                        )
                    else:
                        model = linter.load_model(
                            device=dev, model_name=model_name
                        )
                        index = linter.load_index_for_model(
                            model_name, use_gpu=dev == "cuda"
                        )
                        results = linter.lint(
                            file_lines,
                            model,
                            index,
                            debug=True,
                            return_line_info=True,
                            context_window=args.context_window,
                            scan_types=scan_type,
                        )

                    static_hits = sum(bool(r.get("static_hit")) for r in results)
                    semantic_hits = sum(bool(r.get("semantic_hit")) for r in results)
                    total = len(results)
                    print(
                        "Static hits: {} misses: {}\n"
                        "Semantic hits: {} misses: {}".format(
                            static_hits,
                            total - static_hits,
                            semantic_hits,
                            total - semantic_hits,
                        )
                    )

                    if args.print_json:
                        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

