#!/usr/bin/env python3
"""
Example script demonstrating the corpus builder functionality.

This script shows how to use the CorpusBuilder class to process data from the GUI bridge
and build comprehensive FAISS indexes with normalization and deduplication.
"""

import sys
from pathlib import Path

# Add the moyo package and shared_utils to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared_utils"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.privateside.mapcorpus.builder import (
    CorpusBuilder, 
    build_corpus_from_files, 
    build_corpus_from_texts,
    build_corpus_from_gui_bridge
)
from moyo.privateside.mapcorpus.schema import CorpusConfig
from moyo.privateside.datainput.gui_bridge import GUIBridge, ProcessingConfig


def example_basic_corpus_building():
    """Example of basic corpus building."""
    print("=== Basic Corpus Building Example ===")
    
    # Sample texts
    texts = [
        "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.",
        "Machine Learning is a subset of AI that provides systems the ability to automatically learn.",
        "Deep Learning uses neural networks with multiple layers to model complex patterns in data.",
        "Natural Language Processing focuses on the interaction between computers and human language."
    ]
    
    # Create configuration
    config = CorpusConfig(
        chunk_size=200,
        chunk_overlap=30,
        embedding_model="all-MiniLM-L6-v2",
        index_type="flat",
        deduplication_enabled=True,
        normalization_enabled=True,
        output_directory="examples/corpus_output"
    )
    
    # Build corpus from texts
    result = build_corpus_from_texts(texts, config)
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"   Documents processed: {result.documents_processed}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Vectors created: {result.vectors_created}")
        print(f"   Duplicates removed: {result.duplicates_removed}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        if result.index_path:
            print(f"   Index saved to: {result.index_path}")
    else:
        print(f"❌ {result.message}")
        for error in result.errors:
            print(f"   Error: {error}")
    
    return result


def example_gui_bridge_integration():
    """Example of integrating with GUI bridge data."""
    print("\n=== GUI Bridge Integration Example ===")
    
    # Simulate GUI bridge data
    gui_bridge_data = [
        {
            "text": "This is a document about artificial intelligence and its applications.",
            "source": "gui_input_1"
        },
        {
            "text": "Machine learning algorithms can process large datasets efficiently.",
            "source": "gui_input_2"
        },
        {
            "file_path": "examples/sample_document.txt"
        }
    ]
    
    # Create sample file
    sample_file = Path("examples/sample_document.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("Natural Language Processing enables computers to understand human language.")
    
    # Build corpus from GUI bridge data
    config = CorpusConfig(
        chunk_size=150,
        chunk_overlap=25,
        output_directory="examples/gui_bridge_corpus"
    )
    
    result = build_corpus_from_gui_bridge(gui_bridge_data, config)
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"   Documents processed: {result.documents_processed}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Vectors created: {result.vectors_created}")
        print(f"   Processing time: {result.processing_time:.2f}s")
    else:
        print(f"❌ {result.message}")
    
    return result


def example_advanced_corpus_builder():
    """Example of using the CorpusBuilder class directly."""
    print("\n=== Advanced Corpus Builder Example ===")
    
    # Create builder with custom configuration
    config = CorpusConfig(
        chunk_size=300,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        index_type="flat",
        deduplication_enabled=True,
        normalization_enabled=True,
        min_chunk_length=20,
        max_chunk_length=1000,
        output_directory="examples/advanced_corpus",
        save_chunks=True,
        save_metadata=True
    )
    
    builder = CorpusBuilder(config)
    
    # Add various types of content
    print("Adding text content...")
    builder.add_text(
        "Artificial Intelligence is transforming industries across the globe. "
        "From healthcare to finance, AI applications are becoming increasingly prevalent.",
        "ai_overview"
    )
    
    builder.add_text(
        "Machine Learning algorithms can identify patterns in data that humans might miss. "
        "This capability makes ML valuable for predictive analytics and decision making.",
        "ml_applications"
    )
    
    # Add some duplicate content to test deduplication
    builder.add_text(
        "Artificial Intelligence is transforming industries across the globe. "
        "From healthcare to finance, AI applications are becoming increasingly prevalent.",
        "ai_overview_duplicate"
    )
    
    print(f"Added {len(builder.chunks)} chunks before processing")
    
    # Apply normalization and deduplication
    print("Applying normalization...")
    builder.normalize_corpus()
    
    print("Removing duplicates...")
    duplicates_removed = builder.deduplicate_corpus()
    print(f"Removed {duplicates_removed} duplicates")
    
    # Build the index
    print("Building index...")
    result = builder.build_index()
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"   Final chunks: {len(builder.chunks)}")
        print(f"   Vectors created: {result.vectors_created}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
        # Get corpus information
        corpus_info = builder.get_corpus_info()
        print(f"\n📊 Corpus Info:")
        print(f"   Corpus ID: {corpus_info.corpus_id}")
        print(f"   Document count: {corpus_info.document_count}")
        print(f"   Chunk count: {corpus_info.chunk_count}")
        print(f"   Vector count: {corpus_info.vector_count}")
        
        # Test search
        print(f"\n🔍 Testing search...")
        search_result = builder.search("artificial intelligence", k=3)
        if search_result.total_results > 0:
            print(f"Found {search_result.total_results} results:")
            for result_item in search_result.results:
                print(f"  {result_item['rank']}. Distance: {result_item['distance']:.4f}")
        else:
            print("No search results found")
    else:
        print(f"❌ {result.message}")
    
    return result


def example_file_processing():
    """Example of processing files with the corpus builder."""
    print("\n=== File Processing Example ===")
    
    # Create sample files
    sample_dir = Path("examples/sample_files")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    files = [
        ("ai_intro.txt", "Artificial Intelligence (AI) represents a significant advancement in computer science."),
        ("ml_basics.txt", "Machine Learning is a method of data analysis that automates analytical model building."),
        ("nlp_overview.txt", "Natural Language Processing enables computers to understand and interpret human language."),
        ("deep_learning.md", "# Deep Learning\n\nDeep learning uses neural networks with multiple layers to process data.")
    ]
    
    for filename, content in files:
        file_path = sample_dir / filename
        file_path.write_text(content)
    
    # Build corpus from files
    config = CorpusConfig(
        chunk_size=100,
        chunk_overlap=20,
        output_directory="examples/file_corpus"
    )
    
    result = build_corpus_from_files([str(f) for f in sample_dir.glob("*")], config)
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"   Documents processed: {result.documents_processed}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Vectors created: {result.vectors_created}")
        print(f"   Processing time: {result.processing_time:.2f}s")
    else:
        print(f"❌ {result.message}")
    
    return result


def example_corpus_analysis():
    """Example of analyzing corpus statistics."""
    print("\n=== Corpus Analysis Example ===")
    
    # Create a corpus with various content
    builder = CorpusBuilder()
    
    # Add diverse content
    texts = [
        "Short text.",
        "This is a medium length text with some content about technology and innovation.",
        "This is a much longer text that contains detailed information about artificial intelligence, machine learning, deep learning, neural networks, natural language processing, computer vision, robotics, automation, data science, and various other topics related to modern technology and its applications in different industries.",
        "Another medium text about software development and programming languages.",
        "Short text again."
    ]
    
    for i, text in enumerate(texts):
        builder.add_text(text, f"text_{i}")
    
    # Get statistics before processing
    stats = builder._get_statistics()
    
    print("📈 Corpus Statistics:")
    print(f"  Text Statistics:")
    text_stats = stats["text_statistics"]
    print(f"    Total chunks: {text_stats['total_chunks']}")
    print(f"    Total characters: {text_stats['total_characters']}")
    print(f"    Average length: {text_stats['average_length']:.1f}")
    print(f"    Length range: {text_stats['min_length']} - {text_stats['max_length']}")
    
    print(f"  Duplicate Statistics:")
    dup_stats = stats["duplicate_statistics"]
    print(f"    Unique chunks: {dup_stats['unique_chunks']}")
    print(f"    Duplicate ratio: {dup_stats['duplicate_ratio']:.2%}")
    
    print(f"  Source Distribution:")
    source_dist = stats["source_distribution"]
    for source, count in source_dist.items():
        print(f"    {source}: {count} chunks")
    
    return stats


def main():
    """Run all examples."""
    print("Moyo Corpus Builder Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_corpus_building()
        example_gui_bridge_integration()
        example_advanced_corpus_builder()
        example_file_processing()
        example_corpus_analysis()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
