#!/usr/bin/env python3
"""
Simple demonstration of barrier analysis functionality.

This script shows how to compare public and private information
using cosine distance and Sobolev norms without requiring FAISS.
"""

import sys
import numpy as np
from pathlib import Path

# Add the moyo package and shared_utils to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared_utils"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.schema import BarrierProbeConfig
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
        'research_paper': [0.6, 0.8, 0.5, 0.7, 0.6, 0.8, 0.5, 0.7, 0.6, 0.8]
    }
    return embeddings


def create_mock_public_chunks():
    """Create mock public chunks with embeddings."""
    embeddings = create_mock_embeddings()
    
    chunks = []
    
    # Public chunk 1 (neural network related)
    chunks.append(PublicSource(
        id="public_chunk_001",
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
    ))
    
    # Public chunk 2 (deep learning related)
    chunks.append(PublicSource(
        id="public_chunk_002",
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
    ))
    
    # Public chunk 3 (research paper)
    chunks.append(PublicSource(
        id="public_chunk_003",
        title="Research Paper on AI",
        content="A comprehensive research paper discussing various approaches to artificial intelligence and machine learning.",
        source_type=SourceType.RESEARCH_PAPER,
        published_date=datetime(2023, 7, 20),
        author="Research Team",
        organization="University Lab",
        relevance_score=0.8,
        confidence_score=0.9,
        tags=["AI", "research"],
        metadata={'embedding': embeddings['research_paper']}
    ))
    
    return chunks


def create_mock_private_chunks():
    """Create mock private chunks with embeddings."""
    embeddings = create_mock_embeddings()
    
    chunks = []
    
    # Private chunk 1 (similar to public neural network content)
    chunks.append(DocumentChunk(
        id="private_chunk_001",
        text="Our internal research has developed a neural network architecture for image recognition that uses convolutional layers with residual connections.",
        source_document="private_doc_001",
        chunk_index=0,
        chunk_size=200,
        metadata={
            'title': 'Internal AI Research Report',
            'author': 'Internal Research Team',
            'organization': 'TechCorp Inc.',
            'embedding': embeddings['neural_network']  # Similar to public
        }
    ))
    
    # Private chunk 2 (different content)
    chunks.append(DocumentChunk(
        id="private_chunk_002",
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
    ))
    
    # Private chunk 3 (very different content)
    chunks.append(DocumentChunk(
        id="private_chunk_003",
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
    ))
    
    return chunks


def demonstrate_cosine_distance():
    """Demonstrate cosine distance calculations."""
    print("=== Cosine Distance Demonstration ===")
    
    embeddings = create_mock_embeddings()
    
    # Calculate distances between different embeddings
    pairs = [
        ('neural_network', 'neural_network'),  # Same content
        ('neural_network', 'deep_learning'),   # Similar content
        ('neural_network', 'financial_data'),  # Different content
        ('deep_learning', 'research_paper'),   # Somewhat similar
        ('financial_data', 'product_roadmap')  # Different
    ]
    
    for emb1_name, emb2_name in pairs:
        emb1 = np.array(embeddings[emb1_name])
        emb2 = np.array(embeddings[emb2_name])
        
        # Calculate cosine distance
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            distance = 1.0
        else:
            emb1_normalized = emb1 / norm1
            emb2_normalized = emb2 / norm2
            cosine_similarity = np.dot(emb1_normalized, emb2_normalized)
            distance = 1.0 - cosine_similarity
        
        print(f"Distance between '{emb1_name}' and '{emb2_name}': {distance:.4f}")
    
    print()


def demonstrate_sobolev_norms():
    """Demonstrate Sobolev norm calculations."""
    print("=== Sobolev Norm Demonstration ===")
    
    embeddings = create_mock_embeddings()
    
    for name, embedding in embeddings.items():
        vec = np.array(embedding)
        
        # Calculate first-order Sobolev norm
        l2_norm = np.linalg.norm(vec)
        
        if len(vec) > 1:
            gradient = np.diff(vec)
            gradient_norm = np.linalg.norm(gradient)
        else:
            gradient_norm = 0.0
        
        sobolev_norm = np.sqrt(l2_norm**2 + gradient_norm**2)
        
        print(f"Sobolev norm for '{name}': {sobolev_norm:.4f}")
        print(f"  L2 norm: {l2_norm:.4f}")
        print(f"  Gradient norm: {gradient_norm:.4f}")
        print()
    
    print()


def demonstrate_barrier_analysis():
    """Demonstrate barrier analysis with mock data."""
    print("=== Barrier Analysis Demonstration ===")
    
    # Create mock chunks
    public_chunks = create_mock_public_chunks()
    private_chunks = create_mock_private_chunks()
    
    print(f"Created {len(public_chunks)} public chunks and {len(private_chunks)} private chunks")
    
    # Calculate all pairwise distances
    distances = []
    
    for i, pub_chunk in enumerate(public_chunks):
        for j, priv_chunk in enumerate(private_chunks):
            pub_emb = pub_chunk.metadata['embedding']
            priv_emb = priv_chunk.metadata['embedding']
            
            # Calculate cosine distance
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
            
            distances.append({
                'distance': distance,
                'public_chunk': pub_chunk,
                'private_chunk': priv_chunk,
                'public_index': i,
                'private_index': j
            })
    
    # Sort by distance
    distances.sort(key=lambda x: x['distance'])
    
    print(f"\nTop 5 Closest Matches (Cosine Distance):")
    for i, match in enumerate(distances[:5], 1):
        print(f"  {i}. Distance: {match['distance']:.4f}")
        print(f"     Public: {match['public_chunk'].content}")
        print(f"     Private: {match['private_chunk'].text}")
        print()
    
    # Calculate Sobolev norm distances between public and private chunks
    sobolev_distances = []
    
    for i, pub_chunk in enumerate(public_chunks):
        for j, priv_chunk in enumerate(private_chunks):
            pub_emb = np.array(pub_chunk.metadata['embedding'])
            priv_emb = np.array(priv_chunk.metadata['embedding'])
            
            # Calculate Sobolev norm distance between the two embeddings
            # Sobolev norm distance = ||vec1 - vec2||_H1 = sqrt(||vec1 - vec2||^2 + ||grad(vec1 - vec2)||^2)
            diff_vec = pub_emb - priv_emb
            
            # L2 norm of the difference
            l2_norm_diff = np.linalg.norm(diff_vec)
            
            # Gradient norm of the difference (if vectors have more than 1 element)
            if len(diff_vec) > 1:
                gradient_diff = np.diff(diff_vec)
                gradient_norm_diff = np.linalg.norm(gradient_diff)
            else:
                gradient_norm_diff = 0.0
            
            # Sobolev norm distance
            sobolev_distance = np.sqrt(l2_norm_diff**2 + gradient_norm_diff**2)
            
            sobolev_distances.append({
                'distance': sobolev_distance,
                'public_chunk': pub_chunk,
                'private_chunk': priv_chunk,
                'public_index': i,
                'private_index': j,
                'l2_norm_diff': l2_norm_diff,
                'gradient_norm_diff': gradient_norm_diff
            })
    
    # Sort by Sobolev distance (smaller distance = more similar)
    sobolev_distances.sort(key=lambda x: x['distance'])
    
    print(f"\nTop 5 Closest Matches (Sobolev Norm Distance):")
    for i, match in enumerate(sobolev_distances[:5], 1):
        print(f"  {i}. Sobolev Distance: {match['distance']:.4f}")
        print(f"     L2 Norm Diff: {match['l2_norm_diff']:.4f}")
        print(f"     Gradient Norm Diff: {match['gradient_norm_diff']:.4f}")
        print(f"     Public: {match['public_chunk'].content}")
        print(f"     Private: {match['private_chunk'].text}")
        print()
    
    # Identify potential breaches
    similarity_threshold = 0.3
    breaches = [d for d in distances if d['distance'] <= similarity_threshold]
    
    print(f"Potential Breaches (distance <= {similarity_threshold}):")
    if breaches:
        for i, breach in enumerate(breaches, 1):
            print(f"  {i}. Distance: {breach['distance']:.4f}")
            print(f"     Public: {breach['public_chunk'].content}")
            print(f"     Private: {breach['private_chunk'].text}")
            print()
    else:
        print("  No potential breaches detected")
    
    # Compare cosine and Sobolev distances for the same pairs
    print(f"\nDistance Metric Comparison (Top 3 pairs):")
    for i in range(min(5, len(distances))):
        cosine_match = distances[i]
        sobolev_match = sobolev_distances[i]
        
        print(f"  Pair {i+1}:")
        print(f"    Cosine Distance: {cosine_match['distance']:.4f}")
        print(f"    Sobolev Distance: {sobolev_match['distance']:.4f}")
        print(f"    Public: {cosine_match['public_chunk'].content}")
        print(f"    Private: {cosine_match['private_chunk'].text}")
        print()
    
    print()


def main():
    """Run the demonstration."""
    print("Simple Barrier Analysis Demonstration")
    print("=" * 50)
    
    try:
        # Demonstrate cosine distance
        demonstrate_cosine_distance()
        
        # Demonstrate Sobolev norms
        demonstrate_sobolev_norms()
        
        # Demonstrate barrier analysis
        demonstrate_barrier_analysis()
        
        print("=" * 50)
        print("✅ Demonstration completed successfully!")
        print("   The barrier analysis concepts are working correctly.")
        print("   Key features demonstrated:")
        print("   • Cosine distance calculation between embeddings")
        print("   • Sobolev norm distance as alternative semantic similarity metric")
        print("   • Finding closest matches between public and private content")
        print("   • Comparing different distance metrics for semantic analysis")
        print("   • Identifying potential information barrier breaches")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
