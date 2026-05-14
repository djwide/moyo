#!/usr/bin/env python3
"""
Unified Fuzzing Engine Demo

This demo showcases the unified fuzzing engine that combines two-layer index strategy
with text and idea fuzzing capabilities. It takes a corpus of input lines and displays
fuzzed outputs with various scalar fuzzing levels.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the moyo package to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moyo.publicside.barrierprobe.unified_fuzzing_engine import (
    UnifiedFuzzingEngine,
    FuzzingConfig,
    create_unified_fuzzing_engine
)


def create_sample_corpus() -> List[str]:
    """Create a sample corpus for demonstration."""
    return [
        "Data breach incident report from 2023 showing unauthorized access to customer databases",
        "Security vulnerability disclosure policy outlining procedures for reporting security flaws",
        "Confidential information handling procedures for protecting sensitive business data",
        "Network security incident response plan for handling cyber attacks and data breaches",
        "Employee data protection guidelines ensuring privacy of personal information",
        "Cybersecurity threat intelligence report identifying potential security risks",
        "Information security audit findings revealing gaps in current security measures",
        "Data privacy compliance requirements for handling customer information",
        "Incident response team contact information for security emergencies",
        "Security awareness training materials for employees on data protection",
        "Financial data exposure incident involving leaked customer payment information",
        "Proprietary algorithm documentation containing trade secrets and intellectual property",
        "Internal research findings on machine learning models and neural network architectures",
        "Customer database schema and data structure information",
        "Business strategy documents outlining future product development plans",
        "Sensitive customer data processing procedures and privacy controls",
        "Confidential merger and acquisition documents with financial projections",
        "Internal security policies for handling classified information and restricted data",
        "Trade secret protection protocols for proprietary technology and algorithms",
        "Regulatory compliance documentation for data protection and privacy laws"
    ]


def create_test_phrases() -> List[str]:
    """Create test phrases for fuzzing demonstration."""
    return [
        # This phrase should start semantically close to the target concept
        "Unauthorized access to confidential customer data and financial information resulted in privacy violations and data exposure",
        "Our company experienced a significant data breach that exposed sensitive customer information including names, addresses, and payment details",
        "The security incident involved unauthorized access to our internal systems and resulted in the disclosure of confidential business documents",
        "A major vulnerability was discovered in our authentication system that could allow attackers to gain access to restricted employee data",
        "The privacy violation occurred when personal information of over 10,000 customers was accidentally published on our public website",
        "Our internal investigation revealed that proprietary trade secrets and intellectual property were compromised during the cyber attack",
        "The data leak incident exposed confidential financial records and strategic business plans to unauthorized third parties",
        "A serious security flaw in our database system allowed external actors to access classified customer information without authorization",
        "The information security breach resulted in the unauthorized disclosure of sensitive employee records and internal communications",
        "Our systems were compromised by malicious actors who gained access to restricted customer data and confidential business information",
        "The data exposure incident involved the accidental release of private customer details and proprietary technology specifications"
    ]


def display_fuzzing_results(campaign_result, target_concept: str):
    """Display fuzzing results in a formatted way."""
    print("\n" + "="*80)
    print(f"UNIFIED FUZZING ENGINE RESULTS")
    print("="*80)
    print(f"Target Concept: {target_concept}")
    print(f"Total Phrases: {campaign_result.total_phrases}")
    print(f"Total Results: {campaign_result.total_results}")
    print(f"Processing Time: {campaign_result.processing_time:.2f} seconds")
    print(f"Fuzzing Levels: {campaign_result.fuzzing_levels}")
    print()
    print("SIMILARITY SCORE EXPLANATION:")
    print("The similarity score represents the cosine similarity between the FUZZED TEXT")
    print("and the TARGET CONCEPT using MiniLM embeddings. Higher scores (closer to 1.0)")
    print("indicate that the fuzzed text is more semantically similar to the target concept.")
    print("This measures how well the fuzzing process moved the text towards the target.")
    print()
    
    # Display results by fuzzing level
    for level in sorted(campaign_result.fuzzing_levels):
        results = campaign_result.results_by_level[level]
        print(f"FUZZING LEVEL {level:.1f}")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. Original:  {result.original_text}")
            print(f"    Fuzzed:    {result.fuzzed_text}")
            print(f"    Similarity: {result.similarity_score:.3f} (fuzzed text → target concept)")
            print(f"    Method:    {result.transformation_method}")
            print()
        
        print()


def display_similarity_analysis(campaign_result):
    """Display similarity score analysis."""
    print("\n" + "="*80)
    print("SIMILARITY SCORE ANALYSIS")
    print("="*80)
    print("(Similarity = Fuzzed Text → Target Concept)")
    print()
    
    for level in sorted(campaign_result.fuzzing_levels):
        results = campaign_result.results_by_level[level]
        similarities = [r.similarity_score for r in results]
        
        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        min_similarity = min(similarities)
        
        print(f"Level {level:.1f}:")
        print(f"  Average Similarity: {avg_similarity:.3f} (how well fuzzed text matches target)")
        print(f"  Max Similarity:     {max_similarity:.3f}")
        print(f"  Min Similarity:     {min_similarity:.3f}")
        print(f"  Range:              {max_similarity - min_similarity:.3f}")
        print()


def save_results_to_file(campaign_result, target_concept: str, filename: str = "fuzzing_results.json"):
    """Save results to a JSON file."""
    # Convert results to serializable format
    serializable_results = {
        "target_concept": target_concept,
        "total_phrases": campaign_result.total_phrases,
        "total_results": campaign_result.total_results,
        "fuzzing_levels": campaign_result.fuzzing_levels,
        "processing_time": campaign_result.processing_time,
        "metadata": campaign_result.metadata,
        "results_by_level": {}
    }
    
    for level, results in campaign_result.results_by_level.items():
        serializable_results["results_by_level"][str(level)] = [
            {
                "original_text": r.original_text,
                "fuzzed_text": r.fuzzed_text,
                "fuzzing_level": r.fuzzing_level,
                "similarity_score": r.similarity_score,
                "transformation_method": r.transformation_method,
                "target_concept": r.target_concept,
                "metadata": r.metadata,
                "processing_time": r.processing_time
            }
            for r in results
        ]
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✅ Results saved to {filename}")


def demonstrate_custom_fuzzing_levels():
    """Demonstrate custom fuzzing levels."""
    print("\n" + "="*80)
    print("CUSTOM FUZZING LEVELS DEMONSTRATION")
    print("="*80)
    
    # Create engine with custom fuzzing levels
    custom_config = FuzzingConfig(
        fuzzing_levels=[0.2, 0.4, 0.6, 0.8, 1.0],
        embedding_model="all-MiniLM-L6-v2"
    )
    
    engine = create_unified_fuzzing_engine(custom_config)
    
    # Load corpus
    corpus = create_sample_corpus()
    engine.load_corpus(corpus, "custom_demo")
    
    # Test phrases
    test_phrases = [
        "Our organization discovered a major security vulnerability that could allow unauthorized access to confidential customer data and financial records",
        "The incident involved the accidental disclosure of proprietary business information and trade secrets to external parties"
    ]
    
    # Run campaign
    target_concept = "unauthorized access to confidential customer data and financial information resulting in privacy violations and data exposure"
    campaign_result = engine.run_fuzzing_campaign(test_phrases, target_concept)
    
    # Display results
    display_fuzzing_results(campaign_result, target_concept)
    
    return campaign_result


def demonstrate_single_phrase_fuzzing():
    """Demonstrate single phrase fuzzing with detailed analysis."""
    print("\n" + "="*80)
    print("SINGLE PHRASE FUZZING DEMONSTRATION")
    print("="*80)
    
    # Create engine
    config = FuzzingConfig(
        fuzzing_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        embedding_model="all-MiniLM-L6-v2"
    )
    
    engine = create_unified_fuzzing_engine(config)
    
    # Load corpus
    corpus = create_sample_corpus()
    engine.load_corpus(corpus, "single_phrase_demo")
    
    # Single phrase to fuzz
    phrase = "Our company experienced a significant data breach that exposed sensitive customer information including names, addresses, and payment details"
    target_concept = "unauthorized access to confidential customer data and financial information resulting in privacy violations and data exposure"
    
    print(f"Original Phrase: {phrase}")
    print(f"Target Concept:  {target_concept}")
    print()
    
    # Fuzz at different levels
    for level in config.fuzzing_levels:
        result = engine.fuzz_phrase(phrase, target_concept, level)
        print(f"Level {level:.1f}: {result.fuzzed_text}")
        print(f"         Similarity: {result.similarity_score:.3f}")
        print(f"         Processing: {result.processing_time:.3f}s")
        print()


def main():
    """Run the unified fuzzing engine demonstration."""
    print("UNIFIED FUZZING ENGINE DEMONSTRATION")
    print("="*50)
    print("This demo showcases the unified fuzzing engine that combines")
    print("two-layer index strategy with text and idea fuzzing capabilities.")
    print()
    
    try:
        # Create engine with default configuration
        config = FuzzingConfig(
            fuzzing_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            embedding_model="all-MiniLM-L6-v2"
        )
        
        engine = create_unified_fuzzing_engine(config)
        
        # Load corpus
        print("Loading corpus...")
        corpus = create_sample_corpus()
        engine.load_corpus(corpus, "demo_corpus")
        
        # Display corpus stats
        stats = engine.get_corpus_stats()
        print(f"✅ Loaded {stats['total_documents']} documents")
        print(f"✅ FAISS index created: {stats['faiss_index_loaded']}")
        print(f"✅ Embedding model: {stats['embedding_model']}")
        print()
        
        # Test phrases
        test_phrases = create_test_phrases()
        target_concept = "unauthorized access to confidential customer data and financial information resulting in privacy violations and data exposure"
        
        print(f"Running fuzzing campaign on {len(test_phrases)} phrases...")
        print(f"Target concept: {target_concept}")
        print()
        
        # Run fuzzing campaign
        campaign_result = engine.run_fuzzing_campaign(test_phrases, target_concept)
        
        # Display results
        display_fuzzing_results(campaign_result, target_concept)
        display_similarity_analysis(campaign_result)
        
        # Save results
        save_results_to_file(campaign_result, target_concept)
        
        # Additional demonstrations
        demonstrate_single_phrase_fuzzing()
        demonstrate_custom_fuzzing_levels()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Key Features Demonstrated:")
        print("✅ Unified fuzzing engine with two-layer index strategy")
        print("✅ Multiple scalar fuzzing levels (0.1 to 0.9)")
        print("✅ Text and idea transformation capabilities")
        print("✅ Corpus-based context awareness")
        print("✅ Similarity score analysis")
        print("✅ Comprehensive result reporting")
        print("✅ JSON export functionality")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
