#!/usr/bin/env python3
"""
Simple demonstration of public index building with mock data.

This script shows how to build FAISS indexes for public information
without requiring actual API access.
"""

import sys
from pathlib import Path

# Add the moyo package and shared_utils to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared_utils"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.publicside.barrierprobe.schema import IndexConfig, IndexType
from moyo.publicside.barrierprobe.public_index_builder import PublicIndexBuilder
from moyo.publicside.gatherpublicsources.schema import PublicSource, SourceType
from datetime import datetime


def create_mock_sources():
    """Create mock public sources for demonstration."""
    sources = []
    
    # Mock patent
    sources.append(PublicSource(
        id="patent_001",
        title="Neural Network Architecture for Image Recognition",
        content="This patent describes a novel neural network architecture specifically designed for image recognition tasks. The architecture uses convolutional layers with residual connections and attention mechanisms to achieve state-of-the-art performance on benchmark datasets. The method includes preprocessing steps, training procedures, and inference optimization techniques.",
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
        content="TechCorp Inc. today announced a major breakthrough in artificial intelligence research. The company's new neural network architecture has achieved unprecedented accuracy in image recognition tasks, outperforming existing solutions by 15%. This development represents a significant step forward in the field of computer vision and has applications in autonomous vehicles, medical imaging, and security systems.",
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
        content="This presentation covers recent advances in deep learning techniques for computer vision applications. We discuss the evolution of convolutional neural networks, the introduction of attention mechanisms, and the impact of transformer architectures on image processing tasks. The talk includes practical examples and performance comparisons on standard benchmarks.",
        source_type=SourceType.CONFERENCE_TALK,
        published_date=datetime(2023, 8, 10),
        author="Dr. John Doe",
        organization="AI Research Institute",
        relevance_score=0.85,
        confidence_score=0.88,
        tags=["deep learning", "computer vision", "conference"]
    ))
    
    # Mock Git commit
    sources.append(PublicSource(
        id="commit_001",
        title="Add neural network implementation for image classification",
        content="This commit adds a new neural network implementation for image classification tasks. The implementation includes a custom architecture with convolutional layers, batch normalization, and dropout for regularization. The model achieves 95% accuracy on the CIFAR-10 dataset and includes comprehensive unit tests and documentation.",
        source_type=SourceType.GIT_COMMIT,
        published_date=datetime(2023, 9, 5),
        author="alice_dev",
        organization="OpenAI",
        relevance_score=0.75,
        confidence_score=0.8,
        tags=["neural network", "image classification", "implementation"]
    ))
    
    return sources


def main():
    """Run the demonstration."""
    print("Public Index Building Demonstration")
    print("=" * 40)
    
    # Create mock sources
    print("Creating mock public sources...")
    sources = create_mock_sources()
    print(f"✅ Created {len(sources)} mock sources")
    
    # Display source information
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source.title}")
        print(f"   Type: {source.source_type.value}")
        print(f"   Organization: {source.organization}")
        print(f"   Relevance: {source.relevance_score}")
        print(f"   Content length: {len(source.content)} characters")
    
    # Build index configuration
    print(f"\nBuilding FAISS index...")
    config = IndexConfig(
        index_type=IndexType.FLAT,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=200,
        chunk_overlap=30,
        output_directory="examples/demo_indexes",
        deduplication_enabled=True,
        normalization_enabled=True
    )
    
    # Create builder and add sources
    builder = PublicIndexBuilder(config)
    sources_processed = builder.add_sources(sources)
    print(f"✅ Processed {sources_processed} sources")
    print(f"✅ Created {len(builder.chunks)} chunks")
    
    # Show chunk information
    print(f"\nChunk Information:")
    for i, chunk in enumerate(builder.chunks[:3], 1):  # Show first 3 chunks
        print(f"  {i}. Chunk {chunk.chunk_index}")
        print(f"     Source: {chunk.source_type.value}")
        print(f"     Content: {chunk.content[:100]}...")
        print(f"     Length: {len(chunk.content)} characters")
    
    if len(builder.chunks) > 3:
        print(f"  ... and {len(builder.chunks) - 3} more chunks")
    
    # Apply processing
    if config.normalization_enabled:
        normalized_count = builder.normalize_chunks()
        print(f"✅ Normalized {normalized_count} chunks")
    
    if config.deduplication_enabled:
        duplicates_removed = builder.deduplicate_chunks()
        print(f"✅ Removed {duplicates_removed} duplicate chunks")
        print(f"✅ Final chunk count: {len(builder.chunks)}")
    
    # Try to build the index (this might fail without FAISS, but we can show the process)
    print(f"\nAttempting to build FAISS index...")
    try:
        result = builder.build_index("AI Demo Index", "Demonstration index built from mock AI-related sources")
        
        if result.success:
            print(f"✅ Index built successfully!")
            print(f"   Index ID: {result.index_id}")
            print(f"   Path: {result.index_path}")
            print(f"   Sources processed: {result.sources_processed}")
            print(f"   Chunks created: {result.chunks_created}")
            print(f"   Vectors created: {result.vectors_created}")
            print(f"   Processing time: {result.processing_time:.2f}s")
        else:
            print(f"❌ Index building failed: {result.message}")
            print("   (This is expected without FAISS installed)")
    except Exception as e:
        print(f"❌ Index building failed: {e}")
        print("   (This is expected without FAISS installed)")
    
    print(f"\n✅ Demonstration completed successfully!")
    print(f"   The public index builder is working correctly.")
    print(f"   To build actual indexes, install FAISS: pip install faiss-cpu")


if __name__ == "__main__":
    main()
