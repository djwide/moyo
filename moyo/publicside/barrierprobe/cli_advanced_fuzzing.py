"""
CLI interface for advanced fuzzing techniques.
"""

import click
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from moyo.publicside.barrierprobe.advanced_fuzzing_techniques import (
    AdvancedFuzzingEngine, AdvancedFuzzingConfig
)

logger = logging.getLogger(__name__)


@click.group()
def advanced_fuzzing():
    """Advanced fuzzing techniques for barrier probing."""
    pass


@advanced_fuzzing.command()
@click.option('--text', '-t', required=True, help='Text to fuzz')
@click.option('--target', '-g', required=True, help='Target concept')
@click.option('--technique', '-k', 
              type=click.Choice(['grammar', 'mutational', 'random_walk', 'differential', 'authority']),
              required=True, help='Fuzzing technique to use')
@click.option('--use-annealing', '-a', is_flag=True, help='Use simulated annealing optimization')
@click.option('--output', '-o', help='Output file for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def fuzz(text: str, target: str, technique: str, use_annealing: bool, 
         output: str, verbose: bool):
    """Apply a specific advanced fuzzing technique."""
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = AdvancedFuzzingConfig()
    
    # Create fuzzing engine
    engine = AdvancedFuzzingEngine(config)
    
    try:
        # Apply fuzzing technique
        result = engine.fuzz_with_technique(text, target, technique, use_annealing)
        
        # Prepare output
        output_data = {
            'original_text': text,
            'target_concept': target,
            'technique': technique,
            'use_annealing': use_annealing,
            'result': result
        }
        
        # Display results
        if verbose:
            click.echo(f"Original text: {text}")
            click.echo(f"Target concept: {target}")
            click.echo(f"Technique: {technique}")
            click.echo(f"Annealing: {use_annealing}")
            click.echo(f"Result: {result}")
        
        # Save to file if specified
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            click.echo(f"Results saved to {output}")
        else:
            # Print result to stdout
            if isinstance(result, list):
                for i, item in enumerate(result):
                    click.echo(f"Result {i+1}: {item}")
            else:
                click.echo(result)
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@advanced_fuzzing.command()
@click.option('--text', '-t', required=True, help='Text to fuzz')
@click.option('--target', '-g', required=True, help='Target concept')
@click.option('--use-annealing', '-a', is_flag=True, help='Use simulated annealing optimization')
@click.option('--output', '-o', help='Output file for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def fuzz_all(text: str, target: str, use_annealing: bool, output: str, verbose: bool):
    """Apply all advanced fuzzing techniques."""
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = AdvancedFuzzingConfig()
    
    # Create fuzzing engine
    engine = AdvancedFuzzingEngine(config)
    
    try:
        # Apply all fuzzing techniques
        results = engine.fuzz_with_all_techniques(text, target, use_annealing)
        
        # Prepare output
        output_data = {
            'original_text': text,
            'target_concept': target,
            'use_annealing': use_annealing,
            'results': results
        }
        
        # Display results
        if verbose:
            click.echo(f"Original text: {text}")
            click.echo(f"Target concept: {target}")
            click.echo(f"Annealing: {use_annealing}")
            click.echo("\nResults:")
            for technique, result in results.items():
                click.echo(f"\n{technique.upper()}:")
                if isinstance(result, list):
                    for i, item in enumerate(result):
                        click.echo(f"  {i+1}: {item}")
                else:
                    click.echo(f"  {result}")
        
        # Save to file if specified
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            click.echo(f"Results saved to {output}")
        else:
            # Print results to stdout
            for technique, result in results.items():
                click.echo(f"\n{technique.upper()}:")
                if isinstance(result, list):
                    for i, item in enumerate(result):
                        click.echo(f"  {i+1}: {item}")
                else:
                    click.echo(f"  {result}")
                
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@advanced_fuzzing.command()
@click.option('--input-file', '-i', required=True, help='Input file with phrases to fuzz')
@click.option('--target', '-g', required=True, help='Target concept')
@click.option('--technique', '-k', 
              type=click.Choice(['grammar', 'mutational', 'random_walk', 'differential', 'authority']),
              help='Specific technique (if not specified, all techniques will be used)')
@click.option('--use-annealing', '-a', is_flag=True, help='Use simulated annealing optimization')
@click.option('--output', '-o', required=True, help='Output file for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def batch_fuzz(input_file: str, target: str, technique: str, use_annealing: bool, 
               output: str, verbose: bool):
    """Apply fuzzing techniques to multiple phrases from a file."""
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Read input phrases
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            phrases = [line.strip() for line in f if line.strip()]
    except Exception as e:
        click.echo(f"Error reading input file: {e}", err=True)
        raise click.Abort()
    
    if not phrases:
        click.echo("No phrases found in input file", err=True)
        raise click.Abort()
    
    # Create configuration
    config = AdvancedFuzzingConfig()
    
    # Create fuzzing engine
    engine = AdvancedFuzzingEngine(config)
    
    try:
        results = []
        
        for i, phrase in enumerate(phrases):
            if verbose:
                click.echo(f"Processing phrase {i+1}/{len(phrases)}: {phrase}")
            
            if technique:
                # Apply specific technique
                result = engine.fuzz_with_technique(phrase, target, technique, use_annealing)
                results.append({
                    'original_phrase': phrase,
                    'target_concept': target,
                    'technique': technique,
                    'use_annealing': use_annealing,
                    'result': result
                })
            else:
                # Apply all techniques
                all_results = engine.fuzz_with_all_techniques(phrase, target, use_annealing)
                results.append({
                    'original_phrase': phrase,
                    'target_concept': target,
                    'use_annealing': use_annealing,
                    'results': all_results
                })
        
        # Prepare output data
        output_data = {
            'target_concept': target,
            'technique': technique,
            'use_annealing': use_annealing,
            'total_phrases': len(phrases),
            'results': results
        }
        
        # Save results
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        click.echo(f"Processed {len(phrases)} phrases. Results saved to {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@advanced_fuzzing.command()
@click.option('--config-file', '-c', help='Configuration file path')
@click.option('--output', '-o', help='Output file for configuration')
def config(config_file: str, output: str):
    """Generate or display configuration for advanced fuzzing."""
    
    if config_file:
        # Load and display configuration
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            click.echo("Current configuration:")
            click.echo(json.dumps(config_data, indent=2))
        except Exception as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            raise click.Abort()
    else:
        # Generate default configuration
        config = AdvancedFuzzingConfig()
        config_data = {
            'initial_temperature': config.initial_temperature,
            'cooling_rate': config.cooling_rate,
            'min_temperature': config.min_temperature,
            'max_iterations': config.max_iterations,
            'grammar_mutation_rate': config.grammar_mutation_rate,
            'structure_preservation_weight': config.structure_preservation_weight,
            'mutation_rate': config.mutation_rate,
            'mutation_types': config.mutation_types,
            'walk_length': config.walk_length,
            'walk_step_size': config.walk_step_size,
            'population_size': config.population_size,
            'crossover_rate': config.crossover_rate,
            'mutation_rate_diff': config.mutation_rate_diff,
            'authority_roles': config.authority_roles,
            'nested_instruction_depth': config.nested_instruction_depth,
            'embedding_model': config.embedding_model
        }
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            click.echo(f"Default configuration saved to {output}")
        else:
            click.echo("Default configuration:")
            click.echo(json.dumps(config_data, indent=2))


if __name__ == '__main__':
    advanced_fuzzing()
