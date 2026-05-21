"""CLI interface for LLM-assisted fuzzing and barrier probing."""

import click
import json
import os
from pathlib import Path
from typing import List, Optional

from .llm_fuzzer import LLMFuzzer, LLMFuzzerConfig, fuzz_phrases_for_barrier_analysis
from .barrier_analyzer import BarrierAnalyzer
from .iterative_llm_search import refine_suspicious_pairs
from .schema import BarrierProbeConfig
from ..mapcorpus.builder import CorpusBuilder, CorpusConfig
from shared_utils import FAISSIndex, get_logger

logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(verbose: bool, debug: bool):
    """
    moyo Barrier Probe - LLM-assisted fuzzing and barrier analysis.
    
    This tool uses LLM-assisted techniques to probe information barriers between
    private and public corpora, helping identify potential information leakage.
    
    \b
    Key Features:
    • LLM-assisted phrase fuzzing
    • Semantic similarity analysis
    • Barrier probing between corpora
    • Support for multiple LLM providers
    • Iterative refinement of phrases
    
    \b
    Examples:
    • Fuzz phrases: moyo-probe fuzz -p "data breach" -t "confidential info" -i corpus.index
    • Search corpus: moyo-probe search -c corpus_dir -q "security incident" -k 10
    • Test LLM: moyo-probe test-llm --llm-provider openai --model gpt-4
    """
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.option('--phrases', '-p', multiple=True, help='Phrases to fuzz')
@click.option('--phrases-file', '-f', type=click.Path(exists=True), help='File containing phrases to fuzz')
@click.option('--target-concept', '-t', required=True, help='Target concept to move towards')
@click.option('--corpus-index', '-i', type=click.Path(exists=True), required=True, help='Path to corpus FAISS index')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--llm-provider', default='openai', type=click.Choice(['openai', 'anthropic']), help='LLM provider')
@click.option('--model', default='gpt-4', help='LLM model name')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API key for LLM provider')
@click.option('--max-iterations', default=5, help='Maximum fuzzing iterations')
@click.option('--target-similarity', default=0.95, help='Target similarity to achieve')
@click.option('--search-k', default=10, help='Number of similar phrases to retrieve')
@click.option('--similarity-threshold', default=0.8, help='Minimum similarity threshold')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def fuzz(phrases, phrases_file, target_concept, corpus_index, output, 
         llm_provider, model, api_key, max_iterations, target_similarity, 
         search_k, similarity_threshold, verbose):
    """Fuzz phrases using LLM-assisted semantic transformation."""
    
    # Set up logging
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Load phrases
    all_phrases = list(phrases)
    if phrases_file:
        with open(phrases_file, 'r') as f:
            file_phrases = [line.strip() for line in f if line.strip()]
            all_phrases.extend(file_phrases)
    
    if not all_phrases:
        click.echo("Error: No phrases provided. Use --phrases or --phrases-file.", err=True)
        return
    
    # Load corpus index
    try:
        index = FAISSIndex.load(corpus_index)
        click.echo(f"Loaded corpus index with {index.get_vector_count()} vectors")
    except Exception as e:
        click.echo(f"Error loading corpus index: {e}", err=True)
        return
    
    # Configure fuzzer
    config = LLMFuzzerConfig(
        llm_provider=llm_provider,
        model_name=model,
        api_key=api_key,
        max_iterations=max_iterations,
        target_similarity=target_similarity,
        search_k=search_k,
        similarity_threshold=similarity_threshold
    )
    
    # Run fuzzing
    click.echo(f"Starting LLM-assisted fuzzing for {len(all_phrases)} phrases...")
    click.echo(f"Target concept: {target_concept}")
    click.echo(f"LLM provider: {llm_provider}, Model: {model}")
    
    results = fuzz_phrases_for_barrier_analysis(all_phrases, target_concept, index, config)
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("FUZZING RESULTS")
    click.echo("="*60)
    
    for i, result in enumerate(results):
        click.echo(f"\nPhrase {i+1}:")
        click.echo(f"  Original: {result['original_phrase']}")
        click.echo(f"  Fuzzed:   {result['fuzzed_phrase']}")
        click.echo(f"  Similarity: {result['final_similarity']:.3f}")
        click.echo(f"  Iterations: {result['iterations']}")
        
        if verbose and result['transformation_history']:
            click.echo("  Transformation history:")
            for j, step in enumerate(result['transformation_history']):
                click.echo(f"    {j+1}. {step}")
    
    # Save results
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")
    
    # Summary
    avg_similarity = sum(r['final_similarity'] for r in results) / len(results)
    avg_iterations = sum(r['iterations'] for r in results) / len(results)
    
    click.echo(f"\nSummary:")
    click.echo(f"  Average final similarity: {avg_similarity:.3f}")
    click.echo(f"  Average iterations: {avg_iterations:.1f}")


@cli.command()
@click.option('--corpus-dir', '-c', type=click.Path(exists=True), required=True, help='Corpus directory')
@click.option('--query', '-q', required=True, help='Query phrase to find similar phrases for')
@click.option('--k', default=10, help='Number of similar phrases to retrieve')
@click.option('--similarity-threshold', default=0.8, help='Minimum similarity threshold')
def search(corpus_dir, query, k, similarity_threshold):
    """Search for semantically similar phrases in the corpus."""
    
    # Load corpus index
    try:
        index = FAISSIndex.load(corpus_dir)
        click.echo(f"Loaded corpus index with {index.get_vector_count()} vectors")
    except Exception as e:
        click.echo(f"Error loading corpus index: {e}", err=True)
        return
    
    # Create fuzzer for search functionality
    config = LLMFuzzerConfig(
        search_k=k,
        similarity_threshold=similarity_threshold
    )
    fuzzer = LLMFuzzer(config)
    
    # Search for similar phrases
    click.echo(f"Searching for phrases similar to: '{query}'")
    similar_phrases = fuzzer.find_similar_phrases(query, index, k)
    
    if not similar_phrases:
        click.echo("No similar phrases found.")
        return
    
    # Display results
    click.echo(f"\nFound {len(similar_phrases)} similar phrases:")
    click.echo("-" * 60)
    
    for i, phrase_info in enumerate(similar_phrases):
        click.echo(f"{i+1}. Similarity: {phrase_info['similarity']:.3f}")
        click.echo(f"   Text: {phrase_info['text']}")
        if 'metadata' in phrase_info and phrase_info['metadata']:
            source = phrase_info['metadata'].get('source_document', 'Unknown')
            click.echo(f"   Source: {source}")
        click.echo()


@cli.command()
@click.option('--public-index', '-p', type=click.Path(exists=True), required=True, help='Path to public index')
@click.option('--private-index', '-r', type=click.Path(exists=True), required=True, help='Path to private index')
@click.option('--similarity-threshold', default=0.8, help='Similarity threshold for breach detection')
@click.option('--top-k', default=10, help='Number of top matches to analyze')
@click.option('--llm-top-k', default=5, help='Top results to keep after LLM refinement')
@click.option('--output-json', type=click.Path(), help='File to write JSON report')
@click.option('--output-html', type=click.Path(), help='File to write HTML report')
def analyze(public_index, private_index, similarity_threshold, top_k, llm_top_k, output_json, output_html):
    """Run barrier analysis with optional LLM refinement and export reports."""

    config = BarrierProbeConfig(
        public_index_path=public_index,
        private_index_path=private_index,
        similarity_threshold=similarity_threshold,
    )
    analyzer = BarrierAnalyzer(config)
    result = analyzer.analyze_barriers(top_k=top_k)
    result = refine_suspicious_pairs(result, analyzer, top_k=llm_top_k)

    click.echo(f"Potential breaches found: {result.breach_count}")

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result.dict(), f, indent=2, default=str)
        click.echo(f"JSON report written to {output_json}")

    if output_html:
        lines = ["<html><body>", "<h1>Barrier Probe Report</h1>"]
        lines.append(f"<p>Total breaches: {result.breach_count}</p>")
        lines.append("<table border='1'>")
        lines.append("<tr><th>Rank</th><th>Distance</th><th>Public</th><th>Private</th></tr>")
        for breach in result.potential_breaches:
            lines.append(
                f"<tr><td>{breach.get('rank', '')}</td><td>{breach['distance']:.4f}</td><td>{breach['public_content']}</td><td>{breach['private_content']}</td></tr>"
            )
        lines.append("</table>")
        lines.append("</body></html>")
        Path(output_html).write_text("\n".join(lines), encoding='utf-8')
        click.echo(f"HTML report written to {output_html}")

    return result


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--llm-provider', default='openai', type=click.Choice(['openai', 'anthropic']), help='LLM provider')
@click.option('--model', default='gpt-4', help='LLM model name')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API key for LLM provider')
def test_llm(config, llm_provider, model, api_key):
    """Test LLM connectivity and basic functionality."""
    
    if config:
        with open(config, 'r') as f:
            config_dict = json.load(f)
        fuzzer_config = LLMFuzzerConfig(**config_dict)
    else:
        fuzzer_config = LLMFuzzerConfig(
            llm_provider=llm_provider,
            model_name=model,
            api_key=api_key
        )
    
    fuzzer = LLMFuzzer(fuzzer_config)
    
    if not fuzzer.llm_client:
        click.echo("Error: LLM client not initialized. Check your configuration.", err=True)
        return
    
    # Test with a simple prompt
    test_prompt = "Please respond with 'LLM test successful' if you can read this message."
    
    click.echo(f"Testing LLM connection...")
    click.echo(f"Provider: {fuzzer_config.llm_provider}")
    click.echo(f"Model: {fuzzer_config.model_name}")
    
    response = fuzzer.query_llm(test_prompt)
    
    if response:
        click.echo(f"✅ LLM test successful!")
        click.echo(f"Response: {response}")
    else:
        click.echo("❌ LLM test failed. Check your API key and configuration.", err=True)


@cli.command()
@click.option('--corpus-dir', '-c', type=click.Path(exists=True), required=True, help='Corpus directory')
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze_corpus(corpus_dir, output, verbose):
    """Analyze corpus statistics and structure."""
    
    # Load corpus index
    try:
        index = FAISSIndex.load(corpus_dir)
        click.echo(f"Loaded corpus index with {index.get_vector_count()} vectors")
    except Exception as e:
        click.echo(f"Error loading corpus index: {e}", err=True)
        return
    
    # Analyze corpus
    click.echo("\nCorpus Analysis:")
    click.echo("-" * 40)
    click.echo(f"Total vectors: {index.get_vector_count()}")
    click.echo(f"Index dimension: {index.dimension}")
    click.echo(f"Index type: {index.index_type}")
    
    # Analyze metadata if available
    if index.metadata:
        sources = {}
        for meta in index.metadata:
            source = meta.get('source_document', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        click.echo(f"\nSource distribution:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            click.echo(f"  {source}: {count} phrases")
        
        if verbose:
            click.echo(f"\nSample phrases:")
            for i, meta in enumerate(index.metadata[:5]):
                text = meta.get('text', 'No text available')
                click.echo(f"  {i+1}. {text[:100]}...")
    
    # Save analysis
    if output:
        analysis = {
            "total_vectors": index.get_vector_count(),
            "dimension": index.dimension,
            "index_type": index.index_type,
            "sources": sources if index.metadata else {}
        }
        
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
        click.echo(f"\nAnalysis saved to: {output}")


if __name__ == '__main__':
    cli()
