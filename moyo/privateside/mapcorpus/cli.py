import click
from pathlib import Path
from typing import List
import json

from .builder import CorpusBuilder, build_corpus_from_files, build_corpus_from_texts
from .schema import CorpusConfig


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(verbose: bool, debug: bool) -> None:
    """
    Moyo Corpus Builder - Build and manage private knowledge corpora.
    
    This tool builds searchable corpora from various data sources, with support for
    text processing, deduplication, and vector indexing.
    
    \b
    Key Features:
    • Build corpora from files or directories
    • Text chunking and normalization
    • Deduplication and similarity detection
    • Multiple FAISS index types
    • Configurable embedding models
    
    \b
    Examples:
    • Build from directory: moyo-corpus build /path/to/documents
    • Build from text: moyo-corpus build-text "Text 1" "Text 2"
    • Custom configuration: moyo-corpus build --chunk-size 256 --model all-MiniLM-L6-v2
    """
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='indexes/private', help='Output directory for index')
@click.option('--chunk-size', default=512, help='Chunk size for text processing')
@click.option('--chunk-overlap', default=50, help='Overlap between chunks')
@click.option('--model', default='all-MiniLM-L6-v2', help='Embedding model to use')
@click.option('--index-type', default='flat', type=click.Choice(['flat', 'ivf', 'hnsw']), help='FAISS index type')
@click.option('--dedupe/--no-dedupe', default=True, help='Enable/disable deduplication')
@click.option('--normalize/--no-normalize', default=True, help='Enable/disable text normalization')
@click.option('--save-chunks/--no-save-chunks', default=True, help='Save chunk data to disk')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def build(input_path: str, output_dir: str, chunk_size: int, chunk_overlap: int, 
          model: str, index_type: str, dedupe: bool, normalize: bool, 
          save_chunks: bool, json_output: bool) -> None:
    """Build corpus from input path (file or directory)."""
    
    # Create configuration
    config = CorpusConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=model,
        index_type=index_type,
        deduplication_enabled=dedupe,
        normalization_enabled=normalize,
        output_directory=output_dir,
        save_chunks=save_chunks
    )
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        result = build_corpus_from_files([input_path], config)
    elif input_path.is_dir():
        # Directory - find all supported files
        supported_extensions = ['.txt', '.md', '.markdown', '.json', '.csv']
        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f'*{ext}'))
            files.extend(input_path.glob(f'**/*{ext}'))
        
        if not files:
            click.echo(f"No supported files found in {input_path}")
            return
        
        click.echo(f"Found {len(files)} files to process")
        result = build_corpus_from_files(files, config)
    else:
        click.echo(f"Invalid input path: {input_path}")
        return
    
    if json_output:
        click.echo(json.dumps(result.dict(), indent=2))
    else:
        if result.success:
            click.echo(f"✅ {result.message}")
            click.echo(f"   Documents processed: {result.documents_processed}")
            click.echo(f"   Chunks created: {result.chunks_created}")
            click.echo(f"   Vectors created: {result.vectors_created}")
            click.echo(f"   Duplicates removed: {result.duplicates_removed}")
            click.echo(f"   Processing time: {result.processing_time:.2f}s")
            if result.index_path:
                click.echo(f"   Index saved to: {result.index_path}")
        else:
            click.echo(f"❌ {result.message}")
            for error in result.errors:
                click.echo(f"   Error: {error}")


@cli.command()
@click.argument('texts', nargs=-1, required=True)
@click.option('--output-dir', '-o', default='indexes/private', help='Output directory for index')
@click.option('--chunk-size', default=512, help='Chunk size for text processing')
@click.option('--chunk-overlap', default=50, help='Overlap between chunks')
@click.option('--model', default='all-MiniLM-L6-v2', help='Embedding model to use')
@click.option('--index-type', default='flat', type=click.Choice(['flat', 'ivf', 'hnsw']), help='FAISS index type')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def build_text(texts: List[str], output_dir: str, chunk_size: int, chunk_overlap: int, 
               model: str, index_type: str, json_output: bool) -> None:
    """Build corpus from text inputs."""
    
    # Create configuration
    config = CorpusConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=model,
        index_type=index_type,
        output_directory=output_dir
    )
    
    result = build_corpus_from_texts(list(texts), config)
    
    if json_output:
        click.echo(json.dumps(result.dict(), indent=2))
    else:
        if result.success:
            click.echo(f"✅ {result.message}")
            click.echo(f"   Documents processed: {result.documents_processed}")
            click.echo(f"   Chunks created: {result.chunks_created}")
            click.echo(f"   Vectors created: {result.vectors_created}")
            click.echo(f"   Processing time: {result.processing_time:.2f}s")
            if result.index_path:
                click.echo(f"   Index saved to: {result.index_path}")
        else:
            click.echo(f"❌ {result.message}")
            for error in result.errors:
                click.echo(f"   Error: {error}")


@cli.command()
@click.argument('index_path', type=click.Path(exists=True))
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--k', default=10, help='Number of results to return')
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def search(index_path: str, query: str, k: int, json_output: bool) -> None:
    """Search an existing corpus."""
    
    # Load the corpus
    try:
        from shared_utils import FAISSIndex
        index = FAISSIndex.load(Path(index_path))
        
        # Create a temporary builder to use its search method
        builder = CorpusBuilder()
        builder.index = index
        
        search_result = builder.search(query, k)
        
        if json_output:
            click.echo(json.dumps(search_result.dict(), indent=2))
        else:
            if search_result.total_results > 0:
                click.echo(f"🔍 Search results for: '{query}'")
                click.echo(f"Found {search_result.total_results} results in {search_result.search_time:.3f}s:")
                
                for result in search_result.results:
                    click.echo(f"  {result['rank']}. Distance: {result['distance']:.4f}")
                    if "metadata" in result and "text_preview" in result["metadata"]:
                        click.echo(f"     Preview: {result['metadata']['text_preview']}")
                    click.echo()
            else:
                click.echo(f"❌ No results found for: '{query}'")
                
    except Exception as e:
        click.echo(f"❌ Error loading or searching index: {e}")


@cli.command()
@click.argument('index_path', type=click.Path(exists=True))
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def info(index_path: str, json_output: bool) -> None:
    """Get information about a corpus."""
    
    try:
        # Load corpus info
        info_file = Path(index_path) / "corpus_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                corpus_info = json.load(f)
            
            if json_output:
                click.echo(json.dumps(corpus_info, indent=2))
            else:
                click.echo("📊 Corpus Information:")
                click.echo(f"  Corpus ID: {corpus_info.get('corpus_id', 'N/A')}")
                click.echo(f"  Created: {corpus_info.get('created_at', 'N/A')}")
                click.echo(f"  Documents: {corpus_info.get('document_count', 0)}")
                click.echo(f"  Chunks: {corpus_info.get('chunk_count', 0)}")
                click.echo(f"  Vectors: {corpus_info.get('vector_count', 0)}")
                click.echo(f"  Dimension: {corpus_info.get('embedding_dimension', 0)}")
                click.echo(f"  Index Type: {corpus_info.get('index_type', 'N/A')}")
                click.echo(f"  Model: {corpus_info.get('embedding_model', 'N/A')}")
        else:
            # Try to load just the FAISS index
            from shared_utils import FAISSIndex
            index = FAISSIndex.load(Path(index_path))
            
            if json_output:
                info = {
                    "vector_count": index.get_vector_count(),
                    "dimension": index.dimension,
                    "index_type": index.index_type,
                    "is_trained": index.is_trained
                }
                click.echo(json.dumps(info, indent=2))
            else:
                click.echo("📊 Index Information:")
                click.echo(f"  Vector count: {index.get_vector_count()}")
                click.echo(f"  Dimension: {index.dimension}")
                click.echo(f"  Index type: {index.index_type}")
                click.echo(f"  Trained: {index.is_trained}")
                
    except Exception as e:
        click.echo(f"❌ Error loading corpus info: {e}")


@cli.command()
@click.argument('index_path', type=click.Path(exists=True))
@click.option('--json', 'json_output', is_flag=True, help='Output results as JSON')
def analyze(index_path: str, json_output: bool) -> None:
    """Analyze corpus statistics and metadata."""
    
    try:
        # Load corpus info
        info_file = Path(index_path) / "corpus_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                corpus_info = json.load(f)
            
            metadata = corpus_info.get('metadata', {})
            statistics = metadata.get('statistics', {})
            
            if json_output:
                click.echo(json.dumps(statistics, indent=2))
            else:
                click.echo("📈 Corpus Analysis:")
                
                # Text statistics
                text_stats = statistics.get('text_statistics', {})
                click.echo(f"  Text Statistics:")
                click.echo(f"    Total chunks: {text_stats.get('total_chunks', 0)}")
                click.echo(f"    Total characters: {text_stats.get('total_characters', 0)}")
                click.echo(f"    Average length: {text_stats.get('average_length', 0):.1f}")
                click.echo(f"    Length range: {text_stats.get('min_length', 0)} - {text_stats.get('max_length', 0)}")
                
                # Duplicate statistics
                dup_stats = statistics.get('duplicate_statistics', {})
                click.echo(f"  Duplicate Statistics:")
                click.echo(f"    Unique chunks: {dup_stats.get('unique_chunks', 0)}")
                click.echo(f"    Duplicate ratio: {dup_stats.get('duplicate_ratio', 0):.2%}")
                
                # Source distribution
                source_dist = statistics.get('source_distribution', {})
                if source_dist:
                    click.echo(f"  Source Distribution:")
                    for source, count in source_dist.items():
                        click.echo(f"    {source}: {count} chunks")
        else:
            click.echo("❌ No corpus analysis data available")
            
    except Exception as e:
        click.echo(f"❌ Error analyzing corpus: {e}")


if __name__ == "__main__":
    cli()
