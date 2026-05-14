#!/usr/bin/env python3
"""
Demonstration of iterative LLM search functionality.

This script shows how to use the iterative LLM search to find closer
semantic matches by fuzzing the closest matches and using LLM-generated queries.
"""

import sys
import random
from pathlib import Path

# Add the moyo package and shared_utils to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared_utils"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.publicside.barrierprobe.iterative_llm_search import IterativeLLMSearch
from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.schema import BarrierProbeConfig, BarrierProbeResult
from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType
from moyo.privateside.mapcorpus.schema import DocumentChunk
from shared_utils import generate_id
from datetime import datetime


def create_mock_embeddings():
    """Create mock embeddings for demonstration."""
    # Create some mock embeddings with different characteristics
    embeddings = {
        'neural_network': [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 0.5],
        'image_recognition': [0.9, 0.7, 0.8, 0.6, 0.8, 0.7, 0.9, 0.6, 0.8, 0.7],
        'deep_learning': [0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9],
        'financial_data': [0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2],
        'product_roadmap': [0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4],
        'research_paper': [0.6, 0.8, 0.5, 0.7, 0.6, 0.8, 0.5, 0.7, 0.6, 0.8],
        'similar_neural': [0.85, 0.65, 0.88, 0.72, 0.52, 0.82, 0.62, 0.91, 0.68, 0.48],  # Similar to neural_network
        'very_similar': [0.82, 0.58, 0.87, 0.69, 0.51, 0.79, 0.59, 0.88, 0.66, 0.49]  # Very similar to neural_network
    }
    return embeddings


def create_mock_data():
    """Create mock public and private data for demonstration."""
    embeddings = create_mock_embeddings()
    
    # Create public sources
    public_sources = [
        PublicSource(
            id="public_001",
            title="Neural Network Architecture",
            content="This describes a novel neural network architecture for image recognition using convolutional layers with residual connections.",
            source_type=SourceType.PATENT,
            published_date=datetime(2023, 6, 15),
            author="Dr. Jane Smith",
            organization="TechCorp Inc.",
            relevance_score=0.9,
            confidence_score=0.95,
            tags=["neural networks", "image recognition"],
            metadata={'embedding': embeddings['neural_network']}
        ),
        PublicSource(
            id="public_002",
            title="Deep Learning Advances",
            content="Recent advances in deep learning techniques for computer vision applications with attention mechanisms.",
            source_type=SourceType.CONFERENCE_TALK,
            published_date=datetime(2023, 8, 10),
            author="Dr. John Doe",
            organization="AI Research Institute",
            relevance_score=0.85,
            confidence_score=0.88,
            tags=["deep learning", "computer vision"],
            metadata={'embedding': embeddings['deep_learning']}
        ),
        PublicSource(
            id="public_003",
            title="Similar Neural Architecture",
            content="A similar neural network architecture for image recognition that uses convolutional layers with residual connections and attention mechanisms.",
            source_type=SourceType.RESEARCH_PAPER,
            published_date=datetime(2023, 7, 20),
            author="Research Team",
            organization="University Lab",
            relevance_score=0.8,
            confidence_score=0.9,
            tags=["neural networks", "research"],
            metadata={'embedding': embeddings['similar_neural']}
        ),
        PublicSource(
            id="public_004",
            title="Very Similar Architecture",
            content="This describes a very similar neural network architecture for image recognition using convolutional layers with residual connections and advanced optimization.",
            source_type=SourceType.PATENT,
            published_date=datetime(2023, 9, 15),
            author="Dr. Alice Johnson",
            organization="TechCorp Inc.",
            relevance_score=0.95,
            confidence_score=0.98,
            tags=["neural networks", "optimization"],
            metadata={'embedding': embeddings['very_similar']}
        )
    ]
    
    # Create private chunks
    private_chunks = [
        DocumentChunk(
            id="private_001",
            text="Our internal research has developed a neural network architecture for image recognition that uses convolutional layers with residual connections.",
            source_document="private_doc_001",
            chunk_index=0,
            chunk_size=200,
            metadata={
                'title': 'Internal AI Research Report',
                'author': 'Internal Research Team',
                'organization': 'TechCorp Inc.',
                'embedding': embeddings['neural_network']
            }
        ),
        DocumentChunk(
            id="private_002",
            text="Our confidential product roadmap includes plans for a new machine learning platform that will integrate with existing systems.",
            source_document="private_doc_002",
            chunk_index=0,
            chunk_size=200,
            metadata={
                'title': 'Confidential Product Roadmap',
                'author': 'Product Team',
                'organization': 'TechCorp Inc.',
                'embedding': embeddings['product_roadmap']
            }
        ),
        DocumentChunk(
            id="private_003",
            text="Based on current market conditions and our sales pipeline, we project Q4 2023 revenue to increase by 25% compared to Q3.",
            source_document="private_doc_003",
            chunk_index=0,
            chunk_size=200,
            metadata={
                'title': 'Financial Projections Q4 2023',
                'author': 'Finance Team',
                'organization': 'TechCorp Inc.',
                'embedding': embeddings['financial_data']
            }
        )
    ]
    
    return public_sources, private_chunks


def demonstrate_text_fuzzing():
    """Demonstrate text fuzzing functionality."""
    print("=== Text Fuzzing Demonstration ===")
    
    # Create a mock barrier analyzer
    config = BarrierProbeConfig(
        public_index_path="",
        private_index_path="",
        similarity_threshold=0.8
    )
    analyzer = BarrierAnalyzer(config)
    
    # Create iterative search instance (using local LLM fallback)
    searcher = IterativeLLMSearch(analyzer, llm_client=None)
    
    # Test text fuzzing
    original_text = "This describes a novel neural network architecture for image recognition using convolutional layers with residual connections."
    
    print(f"Original text: {original_text}")
    print()
    
    for i in range(3):
        fuzzed_text = searcher.fuzz_text(original_text, fuzz_level=0.15)
        print(f"Fuzzed version {i+1}: {fuzzed_text}")
        print()
    
    # Test different fuzz levels
    print("Fuzz level comparison:")
    for level in [0.05, 0.1, 0.15, 0.2]:
        fuzzed = searcher.fuzz_text(original_text, fuzz_level=level)
        print(f"Level {level}: {fuzzed}")
        print()


def demonstrate_query_generation():
    """Demonstrate LLM query generation."""
    print("=== Query Generation Demonstration ===")
    
    # Create a mock barrier analyzer
    config = BarrierProbeConfig(
        public_index_path="",
        private_index_path="",
        similarity_threshold=0.8
    )
    analyzer = BarrierAnalyzer(config)
    
    # Create iterative search instance (using local LLM fallback)
    searcher = IterativeLLMSearch(analyzer, llm_client=None)
    
    # Test query generation
    test_texts = [
        "This describes a novel neural network architecture for image recognition using convolutional layers with residual connections.",
        "Recent advances in deep learning techniques for computer vision applications with attention mechanisms.",
        "Our confidential product roadmap includes plans for a new machine learning platform that will integrate with existing systems."
    ]
    
    for i, text in enumerate(test_texts, 1):
        query = searcher.generate_llm_query(text, f"Test context {i}")
        print(f"Text {i}: {text}")
        print(f"Generated query: {query}")
        print()


def demonstrate_iterative_search():
    """Demonstrate iterative search functionality."""
    print("=== Iterative Search Demonstration ===")
    
    # Create mock data
    public_sources, private_chunks = create_mock_data()
    
    # Create mock barrier result
    mock_matches = []
    
    # Calculate distances between public and private chunks
    for i, pub_source in enumerate(public_sources):
        for j, priv_chunk in enumerate(private_chunks):
            pub_emb = pub_source.metadata['embedding']
            priv_emb = priv_chunk.metadata['embedding']
            
            # Simple cosine distance calculation
            import numpy as np
            vec1 = np.array(pub_emb)
            vec2 = np.array(priv_emb)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                distance = 1.0
            else:
                vec1_normalized = vec1 / norm1
                vec2_normalized = vec2 / norm2
                cosine_similarity = np.dot(vec1_normalized, vec2_normalized)
                distance = 1.0 - cosine_similarity
            
            mock_matches.append({
                'rank': len(mock_matches) + 1,
                'distance': distance,
                'public_chunk_id': pub_source.id,
                'public_content': pub_source.content,
                'public_source_type': pub_source.source_type.value,
                'public_metadata': pub_source.metadata,
                'private_chunk_id': priv_chunk.id,
                'private_content': priv_chunk.text,
                'private_metadata': priv_chunk.metadata
            })
    
    # Sort by distance
    mock_matches.sort(key=lambda x: x['distance'])
    
    # Create mock barrier result
    barrier_result = BarrierProbeResult(
        probe_id=generate_id("probe"),
        public_index_info={'chunk_count': len(public_sources)},
        private_index_info={'chunk_count': len(private_chunks)},
        similarity_threshold=0.8,
        potential_breaches=[],
        breach_count=0,
        high_risk_breaches=0,
        medium_risk_breaches=0,
        low_risk_breaches=0,
        processing_time=1.0,
        recommendations=[],
        metadata={'closest_matches': mock_matches[:5]}
    )
    
    # Create mock barrier analyzer with mock data
    config = BarrierProbeConfig(
        public_index_path="",
        private_index_path="",
        similarity_threshold=0.8
    )
    analyzer = BarrierAnalyzer(config)
    
    # Manually set the chunks for demonstration
    analyzer.public_builder = type('MockBuilder', (), {'chunks': public_sources})()
    analyzer.private_builder = type('MockBuilder', (), {'chunks': private_chunks})()
    
    # Create iterative search instance (using local LLM fallback)
    searcher = IterativeLLMSearch(analyzer, llm_client=None)
    
    print("Initial closest matches:")
    for i, match in enumerate(mock_matches[:3], 1):
        print(f"  {i}. Distance: {match['distance']:.4f}")
        print(f"     Public: {match['public_content'][:80]}...")
        print(f"     Private: {match['private_content'][:80]}...")
        print()
    
    # Run iterative search
    print("Running iterative search...")
    result = searcher.run_iterative_search(barrier_result, iterations=2, top_k=5)
    
    if result['success']:
        print(f"✅ Iterative search completed!")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Initial avg distance: {result['initial_avg_distance']:.4f}")
        print(f"Final avg distance: {result['final_avg_distance']:.4f}")
        print(f"Improvement: {result['improvement']:.4f} ({result['improvement_percentage']:.1f}%)")
        print(f"Total queries generated: {result['total_queries_generated']}")
        
        print("\nIteration details:")
        for ir in result['iteration_results']:
            print(f"  Iteration {ir['iteration']}: {ir['queries_generated']} queries, {ir['matches_found']} matches")
        
        print("\nBest matches found:")
        for i, match in enumerate(result['best_matches'][:3], 1):
            print(f"  {i}. Distance: {match['distance']:.4f} ({match.get('chunk_type', 'unknown')})")
            content = match.get('content', '')[:80] + "..." if len(match.get('content', '')) > 80 else match.get('content', '')
            print(f"     Content: {content}")
            print()
    else:
        print(f"❌ Iterative search failed: {result.get('message', 'Unknown error')}")


def main():
    """Run the demonstration."""
    print("Iterative LLM Search Demonstration")
    print("=" * 50)
    
    try:
        # Demonstrate text fuzzing
        demonstrate_text_fuzzing()
        
        # Demonstrate query generation
        demonstrate_query_generation()
        
        # Demonstrate iterative search
        demonstrate_iterative_search()
        
        print("=" * 50)
        print("✅ Demonstration completed successfully!")
        print("   The iterative LLM search functionality is working correctly.")
        print("   Key features demonstrated:")
        print("   • Text fuzzing with controlled variations")
        print("   • LLM query generation (with fallback templates)")
        print("   • Iterative search to find closer semantic matches")
        print("   • Improvement tracking across iterations")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
