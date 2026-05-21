import click
from pathlib import Path
from typing import List
import json

from .gui_bridge import GUIBridge, ProcessingConfig, process_text_and_build_index, process_files_and_build_index


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
@click.option('--output-dir', default='indexes/private', help='Output directory for index')
@click.option('--no-save', is_flag=True, help='Do not save index to disk')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def process(text: str, file_path: str, file_paths: List[str], chunk_size: int, chunk_overlap: int, 
            model: str, index_type: str, output_dir: str, no_save: bool, json_output: bool) -> None:
    """Process text or files and build FAISS index."""
    
    # Create configuration
    config = ProcessingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=model,
        index_type=index_type,
        save_index=not no_save,
        output_dir=output_dir
    )
    
    if text:
        # Process text input
        result = process_text_and_build_index(text, config)
        
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
        result = process_files_and_build_index([file_path], config)[0]
        
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
        results = process_files_and_build_index(file_paths, config)
        
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
@click.argument('index_path', type=click.Path(exists=True))
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--k', default=10, help='Number of results to return')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def search(index_path: str, query: str, k: int, json_output: bool) -> None:
    """Search an existing index."""
    
    bridge = GUIBridge()
    load_result = bridge.load_index(index_path)
    
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
@click.argument('index_path', type=click.Path(exists=True))
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def info(index_path: str, json_output: bool) -> None:
    """Get information about an index."""
    
    bridge = GUIBridge()
    load_result = bridge.load_index(index_path)
    
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
