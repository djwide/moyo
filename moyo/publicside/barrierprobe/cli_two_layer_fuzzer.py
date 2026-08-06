"""CLI interface for the two-layer fuzzing system."""

import click
import json
import logging
from pathlib import Path
from typing import List, Optional

from .two_layer_fuzzer import (
    create_two_layer_fuzzer,
    TwoLayerFuzzer,
    PublicDocumentGraph,
    HypothesisQueryGraph
)
from .llm_hypothesis_generator import HypothesisGenerationConfig

logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Two-layer fuzzing system for barrier analysis."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)


@cli.command()
@click.option('--faiss-index', '-i', required=True, help='Path to FAISS index file')
@click.option('--target-concept', '-t', required=True, help='Target concept to move towards')
@click.option('--initial-phrases', '-p', multiple=True, help='Initial seed phrases')
@click.option('--output', '-o', default='fuzzing_results.json', help='Output file for results')
@click.option('--max-iterations', '-m', default=5, help='Maximum fuzzing iterations')
@click.option('--k-neighbors', '-k', default=10, help='Number of k-NN neighbors for PDG')
@click.option('--embedding-model', '-e', default='all-MiniLM-L6-v2', help='Embedding model to use (MiniLM)')
@click.option('--llm-provider', default=None, help='LLM provider (openai, anthropic)')
@click.option('--llm-model', default='gpt-4', help='LLM model name')
@click.option('--llm-api-key', default=None, help='LLM API key')
@click.option('--llm-temperature', default=0.7, help='LLM temperature')
def run_campaign(faiss_index, target_concept, initial_phrases, output, max_iterations, 
                k_neighbors, embedding_model, llm_provider, llm_model, llm_api_key, llm_temperature):
    """Run a fuzzing campaign."""
    
    # Validate inputs
    if not Path(faiss_index).exists():
        click.echo(f"Error: FAISS index file not found: {faiss_index}")
        return
    
    if not initial_phrases:
        click.echo("Error: At least one initial phrase is required")
        return
    
    # Create LLM config if provider specified
    llm_config = None
    if llm_provider:
        llm_config = HypothesisGenerationConfig(
            llm_provider=llm_provider,
            model_name=llm_model,
            api_key=llm_api_key,
            temperature=llm_temperature
        )
    
    try:
        # Create fuzzer
        click.echo(f"Creating two-layer fuzzer with index: {faiss_index}")
        fuzzer = create_two_layer_fuzzer(
            faiss_index_path=faiss_index,
            embedding_model=embedding_model,
            k_neighbors=k_neighbors,
            llm_config=llm_config
        )
        
        # Update max iterations
        fuzzer.max_iterations = max_iterations
        
        # Run campaign
        click.echo(f"Starting fuzzing campaign for target concept: {target_concept}")
        click.echo(f"Initial phrases: {list(initial_phrases)}")
        
        results = fuzzer.run_fuzzing_campaign(
            target_concept=target_concept,
            initial_phrases=list(initial_phrases)
        )
        
        # Export results
        fuzzer.export_results(results, output)
        
        # Display summary
        click.echo("\n" + "="*50)
        click.echo("FUZZING CAMPAIGN COMPLETED")
        click.echo("="*50)
        click.echo(f"Target Concept: {target_concept}")
        click.echo(f"Total Hypotheses Generated: {results['total_hypotheses']}")
        click.echo(f"Total Iterations: {results['total_iterations']}")
        click.echo(f"PDG Nodes: {results['pdg_stats']['total_nodes']}")
        click.echo(f"PDG Edges: {results['pdg_stats']['total_edges']}")
        click.echo(f"HQG Hypotheses: {results['hqg_stats']['total_hypotheses']}")
        click.echo(f"Retrieval Edges: {results['hqg_stats']['total_retrieval_edges']}")
        click.echo(f"Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error running fuzzing campaign: {e}")
        logger.exception("Fuzzing campaign failed")


@cli.command()
@click.option('--faiss-index', '-i', required=True, help='Path to FAISS index file')
@click.option('--output', '-o', default='graph_stats.json', help='Output file for statistics')
def analyze_graph(faiss_index, output):
    """Analyze the public document graph structure."""
    
    if not Path(faiss_index).exists():
        click.echo(f"Error: FAISS index file not found: {faiss_index}")
        return
    
    try:
        from shared_utils import FAISSIndex, embed
        
        # Load FAISS index
        click.echo(f"Loading FAISS index: {faiss_index}")
        faiss_index_obj = FAISSIndex.load(faiss_index)
        
        # Get index statistics
        index_stats = faiss_index_obj.get_statistics()
        
        # Create PDG for analysis
        pdg = PublicDocumentGraph(faiss_index_obj)
        
        # Get graph statistics
        graph_stats = pdg.get_graph_stats()
        
        # Combine statistics
        stats = {
            "faiss_index": index_stats,
            "public_document_graph": graph_stats,
            "analysis_timestamp": str(pdg.faiss_index.created_at) if hasattr(pdg.faiss_index, 'created_at') else None
        }
        
        # Save statistics
        with open(output, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Display summary
        click.echo("\n" + "="*50)
        click.echo("GRAPH ANALYSIS RESULTS")
        click.echo("="*50)
        click.echo(f"FAISS Index Size: {index_stats.get('total_vectors', 'N/A')}")
        click.echo(f"Embedding Dimension: {index_stats.get('dimension', 'N/A')}")
        click.echo(f"Index Type: {index_stats.get('index_type', 'N/A')}")
        click.echo(f"PDG Nodes: {graph_stats['total_nodes']}")
        click.echo(f"PDG Edges: {graph_stats['total_edges']}")
        click.echo(f"Average Degree: {graph_stats['avg_degree']:.2f}")
        click.echo(f"Statistics saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error analyzing graph: {e}")
        logger.exception("Graph analysis failed")


@cli.command()
@click.option('--results-file', '-r', required=True, help='Path to fuzzing results file')
@click.option('--output', '-o', default='campaign_report.txt', help='Output file for report')
def generate_report(results_file, output):
    """Generate a detailed report from fuzzing campaign results."""
    
    if not Path(results_file).exists():
        click.echo(f"Error: Results file not found: {results_file}")
        return
    
    try:
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Generate report
        report = generate_campaign_report(results)
        
        # Save report
        with open(output, 'w') as f:
            f.write(report)
        
        click.echo(f"Report generated: {output}")
        
    except Exception as e:
        click.echo(f"Error generating report: {e}")
        logger.exception("Report generation failed")


def generate_campaign_report(results: dict) -> str:
    """Generate a detailed campaign report."""
    
    report = []
    report.append("TWO-LAYER FUZZING CAMPAIGN REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Campaign overview
    report.append("CAMPAIGN OVERVIEW")
    report.append("-" * 20)
    report.append(f"Target Concept: {results.get('target_concept', 'N/A')}")
    report.append(f"Total Hypotheses: {results.get('total_hypotheses', 0)}")
    report.append(f"Total Iterations: {results.get('total_iterations', 0)}")
    report.append("")
    
    # Iteration details
    report.append("ITERATION DETAILS")
    report.append("-" * 20)
    for iteration in results.get('iteration_history', []):
        report.append(f"Iteration {iteration.get('iteration', 'N/A')}:")
        report.append(f"  Hypotheses Processed: {iteration.get('hypotheses_processed', 0)}")
        report.append(f"  New Hypotheses Generated: {iteration.get('new_hypotheses_generated', 0)}")
        report.append(f"  Documents Discovered: {iteration.get('documents_discovered', 0)}")
        report.append(f"  Timestamp: {iteration.get('timestamp', 'N/A')}")
        report.append("")
    
    # Graph statistics
    report.append("GRAPH STATISTICS")
    report.append("-" * 20)
    
    pdg_stats = results.get('pdg_stats', {})
    report.append("Public Document Graph (PDG):")
    report.append(f"  Total Nodes: {pdg_stats.get('total_nodes', 0)}")
    report.append(f"  Total Edges: {pdg_stats.get('total_edges', 0)}")
    report.append(f"  Average Degree: {pdg_stats.get('avg_degree', 0):.2f}")
    report.append(f"  Nodes with Embeddings: {pdg_stats.get('nodes_with_embeddings', 0)}")
    report.append("")
    
    hqg_stats = results.get('hqg_stats', {})
    report.append("Hypothesis Query Graph (HQG):")
    report.append(f"  Total Hypotheses: {hqg_stats.get('total_hypotheses', 0)}")
    report.append(f"  Total Retrieval Edges: {hqg_stats.get('total_retrieval_edges', 0)}")
    report.append("")
    
    # Final hypotheses
    report.append("FINAL HYPOTHESES")
    report.append("-" * 20)
    final_hypotheses = results.get('final_hypotheses', [])
    for i, hypothesis_id in enumerate(final_hypotheses, 1):
        report.append(f"{i}. {hypothesis_id}")
    
    return "\n".join(report)


@cli.command()
@click.option('--faiss-index', '-i', required=True, help='Path to FAISS index file')
@click.option('--query', '-q', required=True, help='Query to test')
@click.option('--top-k', '-k', default=10, help='Number of top results to return')
def test_query(faiss_index, query, top_k):
    """Test a single query against the FAISS index."""
    
    if not Path(faiss_index).exists():
        click.echo(f"Error: FAISS index file not found: {faiss_index}")
        return
    
    try:
        from shared_utils import FAISSIndex, embed
        
        # Load FAISS index
        click.echo(f"Loading FAISS index: {faiss_index}")
        faiss_index_obj = FAISSIndex.load(faiss_index)
        
        # Generate embedding for query
        click.echo(f"Generating embedding for query: {query}")
        embeddings = embed([query], "all-MiniLM-L6-v2")
        
        if not embeddings:
            click.echo("Error: Failed to generate embedding for query")
            return
        
        # Search
        click.echo(f"Searching for top {top_k} results...")
        distances, indices, metadata = faiss_index_obj.search(embeddings, top_k)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("SEARCH RESULTS")
        click.echo("="*50)
        
        for i, (distance, idx, meta) in enumerate(zip(distances[0], indices[0], metadata), 1):
            similarity = 1.0 - distance
            click.echo(f"\n{i}. Similarity: {similarity:.3f}")
            click.echo(f"   Index: {idx}")
            if meta:
                click.echo(f"   Metadata: {meta}")
        
    except Exception as e:
        click.echo(f"Error testing query: {e}")
        logger.exception("Query test failed")


if __name__ == '__main__':
    cli()
