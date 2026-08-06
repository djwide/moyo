import click
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json

from .gui_bridge import (
    GUIBridge,
    ProcessingConfig,
    PRIVATE_INDEX_ROOT,
    process_text_and_build_index,
    process_files_and_build_index,
)

# Root under which all FAISS indexes live (private and public).
INDEX_ROOT = "indexes"


def discover_indexes(root: str = INDEX_ROOT) -> List[Path]:
    """Return all ``*.faiss`` files under the index root, newest first."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    faiss_files = [p for p in root_path.rglob("*.faiss") if p.is_file()]
    return sorted(faiss_files, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_index(index_path: Optional[str], latest: bool) -> Optional[Path]:
    """Resolve which index to use for search/info.

    If ``index_path`` is given, use it. Otherwise fall back to the most
    recently built index, or prompt the user to choose when several exist.
    Returns None if no index could be resolved.
    """
    if index_path:
        return Path(index_path)

    indexes = discover_indexes()
    if not indexes:
        click.echo(f"❌ No indexes found under '{INDEX_ROOT}/'. Build one with 'process' first.")
        return None

    if latest or len(indexes) == 1:
        chosen = indexes[0]
        click.echo(f"Using most recent index: {chosen}")
        return chosen

    click.echo("Available indexes (most recent first):")
    for i, path in enumerate(indexes, start=1):
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        click.echo(f"  {i}. {path}  (built {mtime})")

    choice = click.prompt(
        "Select an index by number", type=click.IntRange(1, len(indexes)), default=1
    )
    return indexes[choice - 1]


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(verbose: bool, debug: bool) -> None:
    """
    moyo Data Input - Process private data and build FAISS indexes.
    
    This tool processes text and files to create searchable vector indexes for private data.
    It supports various input formats and provides flexible configuration options.
    
    \b
    Key Features:
    • Process text directly or from files
    • Support for multiple file formats
    • Configurable chunking and embedding
    • Multiple FAISS index types
    • JSON output for programmatic use
    
    \b
    Examples:
    • Process text: moyo-datainput process "Your text here"
    • Process file: moyo-datainput process --file document.txt
    • Process multiple files: moyo-datainput process --files file1.txt file2.txt
    • Custom chunking: moyo-datainput process --chunk-size 256 --chunk-overlap 25
    """
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.argument('text', required=False)
@click.option('--file', '-f', 'file_path', type=click.Path(exists=True), help='File to process')
@click.option('--files', '-F', 'file_paths', multiple=True, type=click.Path(exists=True), help='Multiple files to process')
@click.option('--chunk-size', default=512, help='Chunk size for text processing')
@click.option('--chunk-overlap', default=50, help='Overlap between chunks')
@click.option('--model', default='all-MiniLM-L6-v2', help='Embedding model to use')
@click.option('--index-type', default='flat', type=click.Choice(['flat', 'ivf', 'hnsw']), help='FAISS index type')
@click.option('--name', '-n', 'index_name', default=None,
              help='Corpus name for the index (defaults to the file/corpus name)')
@click.option('--no-save', is_flag=True, help='Do not save index to disk')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def process(text: str, file_path: str, file_paths: List[str], chunk_size: int, chunk_overlap: int, 
            model: str, index_type: str, index_name: str, no_save: bool, json_output: bool) -> None:
    """Process text or files and build a FAISS index under indexes/private."""
    
    # Create configuration. Indexes are always written under indexes/private,
    # one subdirectory per corpus, with the .faiss file named after the corpus.
    config = ProcessingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=model,
        index_type=index_type,
        save_index=not no_save,
        output_dir=PRIVATE_INDEX_ROOT,
    )
    
    if text:
        # Process text input
        result = process_text_and_build_index(text, config, index_name=index_name)
        
        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                click.echo(f"✅ {result.message}")
                click.echo(f"   Chunks created: {result.chunks_created}")
                click.echo(f"   Vectors created: {result.vectors_created}")
                click.echo(f"   Processing time: {result.processing_time:.2f}s")
                if result.index_path:
                    click.echo(f"   Index saved to: {result.index_path}")
            else:
                click.echo(f"❌ {result.message}")
                for error in result.errors:
                    click.echo(f"   Error: {error}")
    
    elif file_path:
        # Process single file
        result = process_files_and_build_index([file_path], config, index_name=index_name)[0]
        
        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                click.echo(f"✅ {result.message}")
                click.echo(f"   Chunks created: {result.chunks_created}")
                click.echo(f"   Vectors created: {result.vectors_created}")
                click.echo(f"   Processing time: {result.processing_time:.2f}s")
                if result.index_path:
                    click.echo(f"   Index saved to: {result.index_path}")
            else:
                click.echo(f"❌ {result.message}")
                for error in result.errors:
                    click.echo(f"   Error: {error}")
    
    elif file_paths:
        # Process multiple files
        results = process_files_and_build_index(file_paths, config, index_name=index_name)
        
        if json_output:
            click.echo(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            successful = sum(1 for r in results if r.success)
            total_chunks = sum(r.chunks_created for r in results if r.success)
            total_vectors = sum(r.vectors_created for r in results if r.success)
            
            click.echo(f"Processed {len(results)} files: {successful} successful")
            click.echo(f"Total chunks created: {total_chunks}")
            click.echo(f"Total vectors created: {total_vectors}")
            
            for i, result in enumerate(results):
                file_name = Path(file_paths[i]).name
                if result.success:
                    click.echo(f"  ✅ {file_name}: {result.chunks_created} chunks")
                else:
                    click.echo(f"  ❌ {file_name}: {result.message}")
    
    else:
        click.echo("Please provide text input, a file path, or multiple file paths.")
        click.echo("Use --help for more information.")


@cli.command()
@click.argument('index_path', type=click.Path(exists=True), required=False)
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--k', default=10, help='Number of results to return')
@click.option('--latest', is_flag=True, help='Use the most recently built index without prompting')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def search(index_path: str, query: str, k: int, latest: bool, json_output: bool) -> None:
    """Search an index. Defaults to the most recent index or lets you choose."""
    
    resolved = resolve_index(index_path, latest)
    if resolved is None:
        return
    
    bridge = GUIBridge()
    load_result = bridge.load_index(resolved)
    
    if not load_result.success:
        click.echo(f"❌ Failed to load index: {load_result.message}")
        return
    
    search_result = bridge.search_index(query, k)
    
    if json_output:
        click.echo(json.dumps(search_result, indent=2))
    else:
        if search_result["success"]:
            click.echo(f"🔍 Search results for: {query}")
            click.echo(f"Found {len(search_result['results'])} results:")
            
            for result in search_result["results"]:
                click.echo(f"  {result['rank']}. Distance: {result['distance']:.4f}")
                if "metadata" in result and "text_preview" in result["metadata"]:
                    click.echo(f"     Preview: {result['metadata']['text_preview']}")
                click.echo()
        else:
            click.echo(f"❌ Search failed: {search_result['message']}")


@cli.command()
@click.argument('index_path', type=click.Path(exists=True), required=False)
@click.option('--latest', is_flag=True, help='Use the most recently built index without prompting')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def info(index_path: str, latest: bool, json_output: bool) -> None:
    """Show info for an index. Defaults to the most recent index or lets you choose."""
    
    resolved = resolve_index(index_path, latest)
    if resolved is None:
        return
    
    bridge = GUIBridge()
    load_result = bridge.load_index(resolved)
    
    if not load_result.success:
        click.echo(f"❌ Failed to load index: {load_result.message}")
        return
    
    index_info = bridge.get_index_info()
    
    if json_output:
        click.echo(json.dumps(index_info, indent=2))
    else:
        click.echo("📊 Index Information:")
        click.echo(f"  Vector count: {index_info['vector_count']}")
        click.echo(f"  Dimension: {index_info['dimension']}")
        click.echo(f"  Index type: {index_info['index_type']}")
        click.echo(f"  Trained: {index_info['is_trained']}")


@cli.command()
@click.option('--supported', is_flag=True, help='Show supported file extensions')
def list_extensions(supported: bool) -> None:
    """List supported file extensions."""
    from .loaders import get_supported_extensions
    
    extensions = get_supported_extensions()
    
    if supported:
        click.echo("Supported file extensions:")
        for ext in extensions:
            click.echo(f"  {ext}")
    else:
        click.echo(" ".join(extensions))


if __name__ == "__main__":
    cli()
