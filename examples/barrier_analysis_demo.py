#!/usr/bin/env python3
"""
Demonstration of barrier analysis functionality.

This script shows how to compare public and private FAISS indexes
to find closest matches via cosine distance.
"""

import sys
from pathlib import Path

# Add the moyo package and shared_utils to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared_utils"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.publicside.barrierprobe.barrier_analyzer import analyze_barriers, BarrierAnalyzer
from moyo.publicside.barrierprobe.schema import BarrierProbeConfig
from moyo.publicside.barrierprobe.public_index_builder import build_public_index_from_sources
from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType
from moyo.privateside.mapcorpus.builder import CorpusBuilder
from moyo.privateside.mapcorpus.schema import DocumentChunk, CorpusConfig
from shared_utils import embed, generate_id
from datetime import datetime


def create_mock_public_sources():
    """Create mock public sources for demonstration."""
    sources = []
    
    # Mock patent
    sources.append(PublicSource(
        id="patent_001",
        title="Neural Network Architecture for Image Recognition",
        content="This patent describes a novel neural network architecture specifically designed for image recognition tasks. The architecture uses convolutional layers with residual connections and attention mechanisms to achieve state-of-the-art performance on benchmark datasets.",
        source_type=SourceType.PATENT,
        published_date=datetime(2023, 6, 15),
        author="Dr. Jane Smith",
        organization="TechCorp Inc.",
        relevance_score=0.9,
        confidence_score=0.95,
        tags=["neural networks", "image recognition", "deep learning"]
    ))
    
    # Mock press release
    sources.append(PublicSource(
        id="press_001",
        title="TechCorp Announces Breakthrough in AI Research",
        content="TechCorp Inc. today announced a major breakthrough in artificial intelligence research. The company's new neural network architecture has achieved unprecedented accuracy in image recognition tasks, outperforming existing solutions by 15%.",
        source_type=SourceType.PRESS_RELEASE,
        published_date=datetime(2023, 7, 20),
        author="TechCorp Inc.",
        organization="TechCorp Inc.",
        relevance_score=0.8,
        confidence_score=0.9,
        tags=["AI", "breakthrough", "research"]
    ))
    
    # Mock conference talk
    sources.append(PublicSource(
        id="talk_001",
        title="Advances in Deep Learning for Computer Vision",
        content="This presentation covers recent advances in deep learning techniques for computer vision applications. We discuss the evolution of convolutional neural networks and the introduction of attention mechanisms.",
        source_type=SourceType.CONFERENCE_TALK,
        published_date=datetime(2023, 8, 10),
        author="Dr. John Doe",
        organization="AI Research Institute",
        relevance_score=0.85,
        confidence_score=0.88,
        tags=["deep learning", "computer vision", "conference"]
    ))
    
    return sources


def create_mock_private_documents():
    """Create mock private documents for demonstration."""
    documents = []
    
    # Mock private document 1 (similar to public content)
    documents.append({
        'id': 'private_doc_001',
        'title': 'Internal AI Research Report',
        'content': 'Our internal research has developed a neural network architecture for image recognition that uses convolutional layers with residual connections. This approach achieves 15% better performance than existing solutions.',
        'author': 'Internal Research Team',
        'organization': 'TechCorp Inc.'
    })
    
    # Mock private document 2 (different content)
    documents.append({
        'id': 'private_doc_002',
        'title': 'Confidential Product Roadmap',
        'content': 'Our confidential product roadmap includes plans for a new machine learning platform that will integrate with existing systems. The platform will support multiple neural network architectures and provide real-time inference capabilities.',
        'author': 'Product Team',
        'organization': 'TechCorp Inc.'
    })
    
    # Mock private document 3 (very different content)
    documents.append({
        'id': 'private_doc_003',
        'title': 'Financial Projections Q4 2023',
        'content': 'Based on current market conditions and our sales pipeline, we project Q4 2023 revenue to increase by 25% compared to Q3. Key growth drivers include our new AI product line and expansion into European markets.',
        'author': 'Finance Team',
        'organization': 'TechCorp Inc.'
    })
    
    return documents


def build_demo_indexes():
    """Build demo public and private indexes."""
    print("Building demo indexes...")
    
    # Create public index
    print("Creating public index...")
    public_sources = create_mock_public_sources()
    
    from moyo.publicside.barrierprobe.schema import IndexConfig, IndexType
    
    public_config = IndexConfig(
        index_type=IndexType.FLAT,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=30,
        output_directory="examples/demo_indexes",
        deduplication_enabled=True,
        normalization_enabled=True
    )
    
    public_result = build_public_index_from_sources(
        public_sources,
        name="Demo Public Index",
        description="Demo public index for barrier analysis",
        config=public_config
    )
    
    if not public_result.success:
        print(f"❌ Failed to build public index: {public_result.message}")
        return None, None
    
    print(f"✅ Public index built: {public_result.index_path}")
    
    # Create private index
    print("Creating private index...")
    private_docs = create_mock_private_documents()
    
    private_builder = CorpusBuilder()
    
    for doc in private_docs:
        # Create chunks
        from shared_utils import chunk_text
        chunks = chunk_text(doc['content'], chunk_size=200, overlap=30)
        
        for i, chunk_content in enumerate(chunks):
            # Create embedding
            embedding = embed([chunk_content], model_name="all-MiniLM-L6-v2")[0]
            
            # Create document chunk
            chunk = DocumentChunk(
                id=generate_id(f"chunk_{doc['id']}_{i}"),
                text=chunk_content,
                source_document=doc['id'],
                chunk_index=i,
                chunk_size=len(chunk_content),
                embedding=embedding,
                metadata={
                    'title': doc['title'],
                    'author': doc['author'],
                    'organization': doc['organization']
                }
            )
            
            private_builder.add_chunks([chunk])
    
    # Build and save private index
    private_config = CorpusConfig(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=30,
        output_directory="examples/demo_indexes/private_demo_index",
        deduplication_enabled=True,
        normalization_enabled=True
    )
    private_builder.config = private_config
    
    build_result = private_builder.build_index()
    
    if not build_result.success:
        print(f"❌ Failed to build private index: {build_result.message}")
        return None, None
    
    print(f"✅ Private index built: {build_result.index_path}")
    
    return public_result.index_path, build_result.index_path


def demonstrate_barrier_analysis(public_index_path: str, private_index_path: str):
    """Demonstrate barrier analysis functionality."""
    print("\n" + "=" * 50)
    print("Barrier Analysis Demonstration")
    print("=" * 50)
    
    # Perform analysis
    print("Performing barrier analysis...")
    result = analyze_barriers(
        public_index_path=public_index_path,
        private_index_path=private_index_path,
        similarity_threshold=0.8,
        top_k=10
    )
    
    # Display results
    print(f"\n=== Analysis Results ===")
    print(f"Probe ID: {result.probe_id}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
    # Index information
    print(f"\nIndex Information:")
    print(f"  Public chunks: {result.public_index_info.get('chunk_count', 0)}")
    print(f"  Private chunks: {result.private_index_info.get('chunk_count', 0)}")
    
    # Breach analysis
    print(f"\nBreach Analysis:")
    print(f"  Total breaches: {result.breach_count}")
    print(f"  High risk: {result.high_risk_breaches}")
    print(f"  Medium risk: {result.medium_risk_breaches}")
    print(f"  Low risk: {result.low_risk_breaches}")
    
    # Closest matches
    if result.metadata.get('closest_matches'):
        print(f"\nTop 10 Closest Matches (Cosine Distance):")
        for match in result.metadata['closest_matches']:
            print(f"  {match['rank']}. Distance: {match['distance']:.4f}")
            print(f"     Public: {match['public_content']}")
            print(f"     Private: {match['private_content']}")
            print()
    
    # Recommendations
    if result.recommendations:
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")
    
    return result


def demonstrate_individual_analyses(public_index_path: str, private_index_path: str):
    """Demonstrate individual analysis methods."""
    print("\n" + "=" * 50)
    print("Individual Analysis Methods")
    print("=" * 50)
    
    # Create analyzer
    config = BarrierProbeConfig(
        public_index_path=public_index_path,
        private_index_path=private_index_path,
        similarity_threshold=0.8
    )
    
    analyzer = BarrierAnalyzer(config)
    
    if not analyzer.load_indexes():
        print("❌ Failed to load indexes")
        return
    
    # Find closest matches
    print("\nFinding closest matches...")
    closest_matches = analyzer.find_closest_matches(top_k=5)
    
    print(f"Top 5 Closest Matches:")
    for match in closest_matches:
        print(f"  {match['rank']}. Distance: {match['distance']:.4f}")
        print(f"     Public: {match['public_content']}")
        print(f"     Private: {match['private_content']}")
        print()
    
def main():
    """Run the demonstration."""
    print("Barrier Analysis Demonstration")
    print("=" * 50)
    
    try:
        # Build demo indexes
        public_path, private_path = build_demo_indexes()
        
        if not public_path or not private_path:
            print("❌ Failed to build demo indexes")
            return
        
        # Demonstrate barrier analysis
        result = demonstrate_barrier_analysis(public_path, private_path)
        
        # Demonstrate individual analyses
        demonstrate_individual_analyses(public_path, private_path)
        
        print("\n" + "=" * 50)
        print("✅ Demonstration completed successfully!")
        print("   The barrier analysis functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
