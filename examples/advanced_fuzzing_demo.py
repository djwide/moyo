#!/usr/bin/env python3
"""
Advanced Fuzzing Techniques Demo

This script demonstrates the various advanced fuzzing techniques available
in the barrierprobe module.
"""

import sys
import os
from pathlib import Path

# Add the moyo module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from moyo.publicside.barrierprobe.advanced_fuzzing_techniques import (
    AdvancedFuzzingEngine, AdvancedFuzzingConfig
)


def demonstrate_advanced_fuzzing():
    """Demonstrate all advanced fuzzing techniques."""
    
    print("="*80)
    print("ADVANCED FUZZING TECHNIQUES DEMONSTRATION")
    print("="*80)
    
    # Sample text and target concept
    original_text = "The system experienced a major data breach that exposed sensitive customer information"
    target_concept = "unauthorized access to confidential customer data and financial information resulting in privacy violations and data exposure"
    
    print(f"Original Text: {original_text}")
    print(f"Target Concept: {target_concept}")
    print()
    
    # Create configuration
    config = AdvancedFuzzingConfig(
        initial_temperature=1.0,
        cooling_rate=0.95,
        max_iterations=50,  # Reduced for demo
        grammar_mutation_rate=0.3,
        mutation_rate=0.2,
        walk_length=3,  # Reduced for demo
        population_size=10,  # Reduced for demo
        authority_roles=["system administrator", "policy engine", "security officer"]
    )
    
    # Create fuzzing engine
    engine = AdvancedFuzzingEngine(config)
    
    # Demonstrate each technique
    techniques = [
        ("Structure-Aware Grammar Fuzzing", "grammar"),
        ("Mutational Fuzzing", "mutational"),
        ("Random Walk Paraphrasing", "random_walk"),
        ("Differential Random Fuzzing", "differential"),
        ("Role & Authority Hacks", "authority")
    ]
    
    for technique_name, technique_key in techniques:
        print(f"\n{technique_name.upper()}")
        print("-" * len(technique_name))
        
        try:
            result = engine.fuzz_with_technique(original_text, target_concept, technique_key)
            
            if isinstance(result, list):
                print(f"Generated {len(result)} variations:")
                for i, variation in enumerate(result, 1):
                    print(f"  {i}. {variation}")
            else:
                print(f"Result: {result}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Demonstrate simulated annealing optimization
    print(f"\nSIMULATED ANNEALING OPTIMIZATION")
    print("-" * 40)
    
    try:
        # Apply grammar fuzzing with annealing
        result_with_annealing = engine.fuzz_with_technique(
            original_text, target_concept, "grammar", use_annealing=True
        )
        print(f"Grammar fuzzing with annealing: {result_with_annealing}")
        
        # Apply mutational fuzzing with annealing
        result_with_annealing = engine.fuzz_with_technique(
            original_text, target_concept, "mutational", use_annealing=True
        )
        print(f"Mutational fuzzing with annealing: {result_with_annealing}")
        
    except Exception as e:
        print(f"Error with annealing: {e}")
    
    # Demonstrate all techniques at once
    print(f"\nALL TECHNIQUES COMBINED")
    print("-" * 30)
    
    try:
        all_results = engine.fuzz_with_all_techniques(original_text, target_concept)
        
        for technique, result in all_results.items():
            print(f"\n{technique.upper()}:")
            if isinstance(result, list):
                for i, item in enumerate(result, 1):
                    print(f"  {i}. {item}")
            else:
                print(f"  {result}")
                
    except Exception as e:
        print(f"Error with all techniques: {e}")


def demonstrate_configuration():
    """Demonstrate configuration options."""
    
    print("\n" + "="*80)
    print("CONFIGURATION DEMONSTRATION")
    print("="*80)
    
    # Show default configuration
    default_config = AdvancedFuzzingConfig()
    print("Default Configuration:")
    print(f"  Initial Temperature: {default_config.initial_temperature}")
    print(f"  Cooling Rate: {default_config.cooling_rate}")
    print(f"  Max Iterations: {default_config.max_iterations}")
    print(f"  Grammar Mutation Rate: {default_config.grammar_mutation_rate}")
    print(f"  Mutation Rate: {default_config.mutation_rate}")
    print(f"  Walk Length: {default_config.walk_length}")
    print(f"  Population Size: {default_config.population_size}")
    print(f"  Authority Roles: {default_config.authority_roles}")
    print(f"  Embedding Model: {default_config.embedding_model}")
    
    # Show custom configuration
    print("\nCustom Configuration Example:")
    custom_config = AdvancedFuzzingConfig(
        initial_temperature=2.0,
        cooling_rate=0.9,
        max_iterations=200,
        grammar_mutation_rate=0.5,
        mutation_rate=0.3,
        walk_length=10,
        population_size=50,
        authority_roles=["CEO", "CTO", "Chief Security Officer"]
    )
    
    print(f"  Initial Temperature: {custom_config.initial_temperature}")
    print(f"  Cooling Rate: {custom_config.cooling_rate}")
    print(f"  Max Iterations: {custom_config.max_iterations}")
    print(f"  Grammar Mutation Rate: {custom_config.grammar_mutation_rate}")
    print(f"  Mutation Rate: {custom_config.mutation_rate}")
    print(f"  Walk Length: {custom_config.walk_length}")
    print(f"  Population Size: {custom_config.population_size}")
    print(f"  Authority Roles: {custom_config.authority_roles}")


def demonstrate_performance_comparison():
    """Demonstrate performance characteristics of different techniques."""
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    import time
    
    text = "data breach incident"
    target = "confidential information disclosure"
    
    config = AdvancedFuzzingConfig(
        max_iterations=20,  # Reduced for demo
        walk_length=3,
        population_size=10
    )
    
    engine = AdvancedFuzzingEngine(config)
    
    techniques = ["grammar", "mutational", "random_walk", "differential", "authority"]
    
    for technique in techniques:
        try:
            start_time = time.time()
            result = engine.fuzz_with_technique(text, target, technique)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"{technique.upper()}: {duration:.3f}s")
            
        except Exception as e:
            print(f"{technique.upper()}: Error - {e}")


if __name__ == "__main__":
    try:
        demonstrate_advanced_fuzzing()
        demonstrate_configuration()
        demonstrate_performance_comparison()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nFor more information, see the barrierprobe README.md")
        print("CLI usage: moyo-probe advanced-fuzzing --help")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
