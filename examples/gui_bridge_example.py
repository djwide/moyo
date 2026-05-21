#!/usr/bin/env python3
"""
Example script demonstrating the GUI bridge functionality.

This script shows how to use the GUIBridge class to process text and files,
build FAISS indexes, and perform searches.
"""

import sys
from pathlib import Path

# Add the moyo package and shared_utils to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared_utils"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.privateside.datainput.gui_bridge import (
    GUIBridge, 
    ProcessingConfig, 
    process_text_and_build_index,
    process_files_and_build_index
)


def example_text_processing():
    """Example of processing text input."""
    print("=== Text Processing Example ===")
    
    # Sample text
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that work and react like humans. Some of the activities 
    computers with artificial intelligence are designed for include speech recognition, 
    learning, planning, and problem solving.
    
    Machine Learning is a subset of AI that provides systems the ability to automatically 
    learn and improve from experience without being explicitly programmed. Machine learning 
    focuses on the development of computer programs that can access data and use it to 
    learn for themselves.
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple 
    layers to model and understand complex patterns in data. It has been particularly 
    successful in areas like image recognition, natural language processing, and speech recognition.
    """
    
    # Create configuration
    config = ProcessingConfig(
        chunk_size=200,
        chunk_overlap=30,
        embedding_model="all-MiniLM-L6-v2",
        index_type="flat",
        save_index=True,
        output_dir="examples/output"
    )
    
    # Process text
    result = process_text_and_build_index(sample_text, config, "ai_documentation")
    
    if result.success:
        print(f"✅ {result.message}")
        print(f"   Chunks created: {result.chunks_created}")
        print(f"   Vectors created: {result.vectors_created}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        if result.index_path:
            print(f"   Index saved to: {result.index_path}")
    else:
        print(f"❌ {result.message}")
        for error in result.errors:
            print(f"   Error: {error}")
    
    return result


def example_file_processing():
    """Example of processing files."""
    print("\n=== File Processing Example ===")
    
    # Create a sample file
    sample_file = Path("examples/sample_document.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    
    sample_content = """
    Natural Language Processing (NLP) is a field of artificial intelligence that focuses 
    on the interaction between computers and human language. It involves the development 
    of algorithms and models that can understand, interpret, and generate human language.
    
    Key applications of NLP include:
    - Text classification and sentiment analysis
    - Machine translation between languages
    - Question answering systems
    - Chatbots and conversational AI
    - Information extraction from text
    """
    
    sample_file.write_text(sample_content)
    
    # Process the file
    config = ProcessingConfig(
        chunk_size=150,
        chunk_overlap=25,
        output_dir="examples/output"
    )
    
    results = process_files_and_build_index([sample_file], config)
    
    for i, result in enumerate(results):
        file_name = sample_file.name
        if result.success:
            print(f"✅ {file_name}: {result.chunks_created} chunks created")
            print(f"   Processing time: {result.processing_time:.2f}s")
        else:
            print(f"❌ {file_name}: {result.message}")
    
    return results


def example_search():
    """Example of searching the index."""
    print("\n=== Search Example ===")
    
    # Create bridge and load index
    bridge = GUIBridge()
    load_result = bridge.load_index("examples/output")
    
    if not load_result.success:
        print(f"❌ Failed to load index: {load_result.message}")
        return
    
    # Perform searches
    queries = [
        "artificial intelligence",
        "machine learning",
        "natural language processing",
        "neural networks"
    ]
    
    for query in queries:
        print(f"\n🔍 Searching for: '{query}'")
        search_result = bridge.search_index(query, k=3)
        
        if search_result["success"]:
            print(f"Found {len(search_result['results'])} results:")
            for result in search_result["results"]:
                print(f"  {result['rank']}. Distance: {result['distance']:.4f}")
                if "metadata" in result and "text_preview" in result["metadata"]:
                    preview = result["metadata"]["text_preview"]
                    print(f"     Preview: {preview[:80]}...")
        else:
            print(f"❌ Search failed: {search_result['message']}")


def example_gui_bridge_usage():
    """Example of using the GUIBridge class directly."""
    print("\n=== GUI Bridge Usage Example ===")
    
    # Create bridge with custom configuration
    config = ProcessingConfig(
        chunk_size=300,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        index_type="flat",
        save_index=False  # Keep in memory only
    )
    
    bridge = GUIBridge(config)
    
    # Process multiple text inputs
    texts = [
        ("AI is transforming industries", "text_1"),
        ("Machine learning algorithms are powerful", "text_2"),
        ("Deep learning requires large datasets", "text_3")
    ]
    
    for text, source in texts:
        result = bridge.process_text(text, source)
        if result.success:
            print(f"✅ Processed '{source}': {result.chunks_created} chunks")
        else:
            print(f"❌ Failed to process '{source}': {result.message}")
    
    # Get index information
    index_info = bridge.get_index_info()
    print(f"\n📊 Index Info:")
    print(f"  Vector count: {index_info['vector_count']}")
    print(f"  Dimension: {index_info['dimension']}")
    print(f"  Index type: {index_info['index_type']}")
    
    # Search the in-memory index
    search_result = bridge.search_index("learning", k=2)
    if search_result["success"]:
        print(f"\n🔍 Search results for 'learning':")
        for result in search_result["results"]:
            print(f"  {result['rank']}. Distance: {result['distance']:.4f}")
    
    # Get processing statistics
    stats = bridge.get_processing_stats()
    print(f"\n📈 Processing Stats:")
    print(f"  Total chunks created: {stats['total_chunks_created']}")
    print(f"  Total vectors created: {stats['total_vectors_created']}")
    print(f"  Total processing time: {stats['total_processing_time']:.2f}s")


def main():
    """Run all examples."""
    print("moyo GUI Bridge Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_text_processing()
        example_file_processing()
        example_search()
        example_gui_bridge_usage()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
