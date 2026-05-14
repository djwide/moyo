#!/usr/bin/env python3
"""
Example script demonstrating public information index building in barrierprobe.

This script shows how to crawl public sources and build FAISS indexes for
information barrier analysis.
"""

import sys
from pathlib import Path

# Add the moyo package and shared_utils to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared_utils"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.publicside.barrierprobe.schema import IndexConfig, IndexType
from moyo.publicside.barrierprobe.public_index_builder import (
    build_public_index_from_sources,
    load_public_index
)
from moyo.publicside.gatherpublicsources.crawler import crawl_all_sources
from moyo.publicside.gatherpublicsources.schema import SourceType


def example_basic_index_building():
    """Example of basic public index building."""
    print("=== Basic Public Index Building Example ===")
    
    # Crawl public sources for a topic
    topic = "artificial intelligence"
    print(f"Crawling public sources for topic: {topic}")
    
    crawl_result = crawl_all_sources(topic, max_results=50)
    
    if not crawl_result.success:
        print(f"❌ Crawling failed: {crawl_result.message}")
        return None
    
    print(f"✅ Found {crawl_result.sources_found} sources")
    
    # Build index with default configuration
    config = IndexConfig(
        index_type=IndexType.FLAT,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        output_directory="examples/public_indexes"
    )
    
    result = build_public_index_from_sources(
        crawl_result.sources_found,
        name="AI Public Sources",
        description="Public information about artificial intelligence",
        config=config
    )
    
    if result.success:
        print(f"✅ Index built successfully!")
        print(f"   Index ID: {result.index_id}")
        print(f"   Path: {result.index_path}")
        print(f"   Sources processed: {result.sources_processed}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Vectors created: {result.vectors_created}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        return result.index_path
    else:
        print(f"❌ Index building failed: {result.message}")
        return None


def example_filtered_index_building():
    """Example of building an index with source filtering."""
    print("\n=== Filtered Index Building Example ===")
    
    # Crawl public sources
    topic = "machine learning"
    print(f"Crawling public sources for topic: {topic}")
    
    crawl_result = crawl_all_sources(topic, max_results=100)
    
    if not crawl_result.success:
        print(f"❌ Crawling failed: {crawl_result.message}")
        return None
    
    print(f"✅ Found {crawl_result.sources_found} sources")
    
    # Filter to specific source types
    source_types = [SourceType.PATENT, SourceType.CONFERENCE_TALK]
    filtered_sources = [
        s for s in crawl_result.sources_found 
        if s.source_type in source_types
    ]
    
    print(f"Filtered to {len(filtered_sources)} sources of types: {[st.value for st in source_types]}")
    
    # Build index with custom configuration
    config = IndexConfig(
        index_type=IndexType.HNSW,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=256,
        chunk_overlap=25,
        output_directory="examples/public_indexes",
        source_types=source_types,
        deduplication_enabled=True,
        normalization_enabled=True
    )
    
    result = build_public_index_from_sources(
        filtered_sources,
        name="ML Patents and Talks",
        description="Machine learning patents and conference talks",
        config=config
    )
    
    if result.success:
        print(f"✅ Filtered index built successfully!")
        print(f"   Index ID: {result.index_id}")
        print(f"   Sources processed: {result.sources_processed}")
        print(f"   Chunks created: {result.chunks_created}")
        return result.index_path
    else:
        print(f"❌ Index building failed: {result.message}")
        return None


def example_index_searching(index_path: str):
    """Example of searching a built index."""
    print(f"\n=== Index Searching Example ===")
    
    if not index_path:
        print("No index path provided for searching")
        return
    
    # Load the index
    builder = load_public_index(index_path)
    if not builder:
        print(f"❌ Failed to load index from {index_path}")
        return
    
    # Search queries
    queries = [
        "neural networks",
        "deep learning algorithms",
        "computer vision",
        "natural language processing"
    ]
    
    for query in queries:
        print(f"\nSearching for: {query}")
        result = builder.search(query, k=5)
        
        if result.total_results > 0:
            print(f"Found {result.total_results} results in {result.search_time:.3f}s")
            
            for i, res in enumerate(result.results[:3], 1):  # Show top 3
                print(f"  {i}. Distance: {res['distance']:.4f}")
                print(f"     Source: {res['source_type']}")
                print(f"     Content: {res['content'][:150]}...")
                if res['metadata'].get('source_organization'):
                    print(f"     Organization: {res['metadata']['source_organization']}")
        else:
            print("No results found")


def example_index_analysis(index_path: str):
    """Example of analyzing index statistics."""
    print(f"\n=== Index Analysis Example ===")
    
    if not index_path:
        print("No index path provided for analysis")
        return
    
    # Load the index
    builder = load_public_index(index_path)
    if not builder:
        print(f"❌ Failed to load index from {index_path}")
        return
    
    # Analyze chunks
    print(f"Index Analysis:")
    print(f"  Total chunks: {len(builder.chunks)}")
    
    # Source type distribution
    source_types = {}
    for chunk in builder.chunks:
        st = chunk.source_type.value
        source_types[st] = source_types.get(st, 0) + 1
    
    print(f"  Source type distribution:")
    for st, count in source_types.items():
        print(f"    {st}: {count} chunks")
    
    # Organization distribution
    organizations = {}
    for chunk in builder.chunks:
        org = chunk.metadata.get('source_organization')
        if org:
            organizations[org] = organizations.get(org, 0) + 1
    
    if organizations:
        print(f"  Top organizations:")
        sorted_orgs = sorted(organizations.items(), key=lambda x: x[1], reverse=True)
        for org, count in sorted_orgs[:5]:
            print(f"    {org}: {count} chunks")
    
    # Content length statistics
    lengths = [len(chunk.content) for chunk in builder.chunks]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    
    print(f"  Content length statistics:")
    print(f"    Average: {avg_length:.1f} characters")
    print(f"    Range: {min_length} - {max_length} characters")


def main():
    """Run all examples."""
    print("BarrierProbe Public Index Examples")
    print("=" * 50)
    
    try:
        # Run examples
        index_path1 = example_basic_index_building()
        index_path2 = example_filtered_index_building()
        
        # Use the first successful index for searching and analysis
        index_path = index_path1 or index_path2
        
        if index_path:
            example_index_searching(index_path)
            example_index_analysis(index_path)
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
