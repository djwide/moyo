# LLM-Assisted Fuzzing for Barrier Probing

This module provides LLM-assisted fuzzing capabilities for semantic barrier probing, allowing you to use large language models to intelligently transform phrases and reduce semantic distance to target concepts.

Supported providers: `openai` (default model `gpt-4o`), `anthropic` (default `claude-sonnet-4-6`), `ollama` (local models such as `llama3.1:8b`, no API key), `custom` (any OpenAI-compatible endpoint via `--base-url` / `base_url` — vLLM, LM Studio, Together, Groq, OpenRouter, DeepSeek, llama.cpp server, etc.), and `local` (embedding-only synonym transformer, no LLM/API required).

```bash
# Any OpenAI-compatible server (vLLM, LM Studio, Together, Groq, OpenRouter, ...):
moyo-probe test-llm --llm-provider custom \
    --base-url http://localhost:8000/v1 \
    --model my-model \
    --api-key "$MY_API_KEY"      # optional; self-hosted servers usually ignore it
```

## Overview

The LLM fuzzer works by:

1. **Finding Similar Phrases**: Using semantic search to find phrases in your corpus that are semantically similar to the input phrase
2. **LLM Transformation**: Using an LLM to intelligently transform the phrase to reduce semantic distance to a target concept
3. **Iterative Refinement**: Repeating the process until target similarity is achieved or maximum iterations reached
4. **Semantic Validation**: Using embeddings to measure and validate the semantic distance reduction

## Key Features

- **Multiple LLM Providers**: OpenAI, Anthropic, local Ollama, any OpenAI-compatible `custom` endpoint, and an embedding-only local transformer
- **Configurable Prompts**: Customizable prompt templates for different domains
- **Semantic Search Integration**: Uses FAISS index for finding similar phrases
- **Iterative Refinement**: Multi-step transformation with similarity tracking
- **Batch Processing**: Process multiple phrases efficiently
- **CLI Interface**: Easy-to-use command-line tools
- **Comprehensive Logging**: Detailed logging for debugging and analysis

## Installation

The LLM fuzzer requires additional dependencies:

```bash
pip install openai>=1.0.0 anthropic>=0.7.0
```

## Quick Start

### Basic Usage

```python
from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer, LLMFuzzerConfig
from shared_utils import FAISSIndex

# Load your corpus index
index = FAISSIndex.load("path/to/corpus")

# Configure the fuzzer
config = LLMFuzzerConfig(
    llm_provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key",
    max_iterations=5,
    target_similarity=0.95
)

# Create fuzzer
fuzzer = LLMFuzzer(config)

# Fuzz a phrase
original_phrase = "data breach"
target_concept = "confidential information disclosure"

fuzzed_phrase, similarity, history = fuzzer.fuzz_phrase(
    original_phrase, target_concept, index
)

print(f"Original: {original_phrase}")
print(f"Fuzzed: {fuzzed_phrase}")
print(f"Similarity: {similarity:.3f}")
```

### CLI Usage

```bash
# Fuzz a single phrase
moyo-probe fuzz -p "data breach" -t "confidential information disclosure" -i path/to/corpus

# Fuzz multiple phrases from a file
moyo-probe fuzz -f phrases.txt -t "confidential information disclosure" -i path/to/corpus -o results.json

# Search for similar phrases
moyo-probe search -c path/to/corpus -q "data breach" -k 10

# Test LLM connection
moyo-probe test-llm --llm-provider openai --model gpt-4
```

## Configuration

### LLMFuzzerConfig

```python
@dataclass
class LLMFuzzerConfig:
    # LLM Configuration
    llm_provider: str = "ollama"   # "openai" | "anthropic" | "ollama" | "custom" | "local"
    model_name: str = "llama3.1:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # Ollama endpoint (default http://localhost:11434)
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Semantic Search Configuration
    search_k: int = 10
    similarity_threshold: float = 0.8
    
    # Fuzzing Configuration
    max_iterations: int = 5
    target_similarity: float = 0.95
    prompt_template: str = "..."  # Customizable prompt template
```

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Advanced Usage

### Custom Prompt Templates

```python
custom_prompt = """
You are a cybersecurity expert. Transform the phrase to be more semantically 
aligned with the target concept while maintaining its core meaning.

Target concept: {target_concept}
Original phrase: {original_phrase}
Similar phrases: {similar_phrases}

Instructions:
1. Use cybersecurity terminology
2. Maintain original meaning
3. Return only the transformed phrase

Transformed phrase:"""

config = LLMFuzzerConfig(
    prompt_template=custom_prompt,
    llm_provider="openai",
    model_name="gpt-4"
)
```

### Batch Processing

```python
phrases = ["data breach", "information leak", "security incident"]
target_concept = "confidential information disclosure"

results = fuzzer.batch_fuzz_phrases(phrases, target_concept, index)

for result in results:
    print(f"Original: {result['original_phrase']}")
    print(f"Fuzzed: {result['fuzzed_phrase']}")
    print(f"Similarity: {result['final_similarity']:.3f}")
    print(f"Iterations: {result['iterations']}")
```

### Semantic Search Only

```python
# Find similar phrases without LLM transformation
similar_phrases = fuzzer.find_similar_phrases("data breach", index, k=5)

for phrase_info in similar_phrases:
    print(f"Similarity: {phrase_info['similarity']:.3f}")
    print(f"Text: {phrase_info['text']}")
```

## Examples

### Example 1: Cybersecurity Domain

```python
phrases = [
    "data breach",
    "information leak", 
    "security incident",
    "privacy violation"
]
target = "confidential information disclosure"

config = LLMFuzzerConfig(
    llm_provider="openai",
    model_name="gpt-4",
    max_iterations=3,
    target_similarity=0.9
)

fuzzer = LLMFuzzer(config)
results = fuzzer.batch_fuzz_phrases(phrases, target, index)
```

### Example 2: Medical Domain

```python
phrases = [
    "patient data exposure",
    "medical record leak",
    "health information breach"
]
target = "protected health information disclosure"

config = LLMFuzzerConfig(
    llm_provider="anthropic",
    model_name="claude-3-sonnet-20240229",
    max_iterations=4,
    target_similarity=0.92
)

fuzzer = LLMFuzzer(config)
results = fuzzer.batch_fuzz_phrases(phrases, target, index)
```

## CLI Commands

### Fuzz Command

```bash
moyo-probe fuzz [OPTIONS]

Options:
  -p, --phrases TEXT...        Phrases to fuzz
  -f, --phrases-file PATH      File containing phrases to fuzz
  -t, --target-concept TEXT    Target concept to move towards
  -i, --corpus-index PATH      Path to corpus FAISS index
  -o, --output PATH            Output file for results
  --llm-provider [openai|anthropic|ollama|custom|local]  LLM provider
  --model TEXT                 LLM model name
  --api-key TEXT              API key for LLM provider (openai/anthropic/custom)
  --base-url TEXT             Endpoint for Ollama or a custom OpenAI-compatible server
  --max-iterations INTEGER     Maximum fuzzing iterations
  --target-similarity FLOAT    Target similarity to achieve
  --search-k INTEGER          Number of similar phrases to retrieve
  --similarity-threshold FLOAT Minimum similarity threshold
  -v, --verbose               Verbose output
```

### Search Command

```bash
moyo-probe search [OPTIONS]

Options:
  -c, --corpus-dir PATH       Corpus directory
  -q, --query TEXT           Query phrase to find similar phrases for
  -k INTEGER                 Number of similar phrases to retrieve
  --similarity-threshold FLOAT Minimum similarity threshold
```

### Test LLM Command

```bash
moyo-probe test-llm [OPTIONS]

Options:
  -c, --config PATH          Configuration file
  --llm-provider [openai|anthropic|ollama|custom|local]  LLM provider
  --model TEXT               LLM model name
  --api-key TEXT            API key for LLM provider (openai/anthropic/custom)
  --base-url TEXT           Endpoint for Ollama or a custom OpenAI-compatible server
```

## Integration with Barrier Analysis

The LLM fuzzer can be integrated with the broader barrier analysis system:

```python
from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

# Perform barrier analysis
analyzer = BarrierAnalyzer(config)
barrier_result = analyzer.analyze_barriers(public_index, private_index)

# Use LLM fuzzing to improve results
fuzzer = LLMFuzzer(llm_config)
fuzzed_phrases = fuzzer.batch_fuzz_phrases(
    barrier_result.closest_matches, 
    target_concept, 
    public_index
)

# Re-analyze with fuzzed phrases
improved_result = analyzer.analyze_with_fuzzed_phrases(
    barrier_result, fuzzed_phrases
)
```

## Best Practices

1. **Start with Small Batches**: Test with a few phrases before processing large datasets
2. **Monitor API Costs**: LLM calls can be expensive, so monitor usage
3. **Use Appropriate Models**: Choose models based on your domain and requirements
4. **Validate Results**: Always review fuzzed phrases for accuracy and relevance
5. **Customize Prompts**: Tailor prompt templates to your specific domain
6. **Set Reasonable Thresholds**: Balance between similarity improvement and processing time

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your API key is set correctly
2. **Model Availability**: Check that your chosen model is available
3. **Rate Limiting**: Implement appropriate delays between requests
4. **Index Loading**: Verify that your corpus index is valid and accessible

### Debug Mode

Enable verbose logging for debugging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Or use CLI verbose flag
moyo-probe fuzz -v -p "test phrase" -t "target" -i corpus
```

## Performance Considerations

- **Batch Size**: Process phrases in batches to manage API costs
- **Rate Limiting**: Implement delays between LLM calls
- **Caching**: Cache similar phrase searches when possible
- **Parallel Processing**: Consider parallel processing for large datasets

## Advanced Fuzzing Techniques

Beyond basic LLM fuzzing, the barrierprobe module includes advanced fuzzing techniques for specialized use cases:

### Available Techniques

1. **Structure-Aware Grammar Fuzzing**: Preserves grammatical structure while mutating content
2. **Mutational Fuzzing**: Applies various text mutations (character substitution, word deletion, etc.)
3. **Random Walk Paraphrasing**: Uses semantic-guided random walks for paraphrasing
4. **Differential Random Fuzzing**: Employs differential evolution algorithms for optimization
5. **Role & Authority Hacks**: Implements prompt injection techniques with authority roles
6. **Simulated Annealing**: Optimizes fuzzing results using simulated annealing

### Usage

#### Python API

```python
from moyo.publicside.barrierprobe.advanced_fuzzing_techniques import (
    AdvancedFuzzingEngine, AdvancedFuzzingConfig
)

# Create configuration
config = AdvancedFuzzingConfig(
    initial_temperature=1.0,
    cooling_rate=0.95,
    max_iterations=100,
    grammar_mutation_rate=0.3,
    mutation_rate=0.2,
    walk_length=5,
    population_size=20
)

# Create fuzzing engine
engine = AdvancedFuzzingEngine(config)

# Apply specific technique
text = "data breach incident"
target = "confidential information disclosure"

# Structure-aware grammar fuzzing
result = engine.fuzz_with_technique(text, target, "grammar", use_annealing=True)

# Mutational fuzzing
result = engine.fuzz_with_technique(text, target, "mutational")

# Random walk paraphrasing
result = engine.fuzz_with_technique(text, target, "random_walk")

# Differential evolution fuzzing
result = engine.fuzz_with_technique(text, target, "differential")

# Authority-based prompt injection
results = engine.fuzz_with_technique(text, target, "authority")  # Returns list

# Apply all techniques
all_results = engine.fuzz_with_all_techniques(text, target, use_annealing=True)
```

#### CLI Usage

The advanced-fuzzing and two-layer commands are registered as subcommands of
`moyo-probe`:

```bash
# Apply specific technique
moyo-probe advanced-fuzzing fuzz \
  -t "data breach" -g "confidential information disclosure" -k grammar

# Apply all techniques
moyo-probe advanced-fuzzing fuzz-all \
  -t "data breach" -g "confidential information disclosure"

# Batch processing from file
moyo-probe advanced-fuzzing batch-fuzz \
  -i phrases.txt -g "target concept" -o results.json

# Generate configuration
moyo-probe advanced-fuzzing config -o config.json

# Two-layer fuzzing campaign
moyo-probe two-layer run-campaign \
  -i public_corpus.index -t "data breach" -p "security incident"
```

> These can still be invoked standalone via
> `python -m moyo.publicside.barrierprobe.cli_advanced_fuzzing ...` and
> `python -m moyo.publicside.barrierprobe.cli_two_layer_fuzzer ...`.

### Technique Details

#### Structure-Aware Grammar Fuzzing

Preserves the grammatical structure of text while applying semantic mutations:

```python
# Analyzes sentence structure and applies mutations within grammatical patterns
result = engine.fuzz_with_technique(
    "The system experienced a major data breach", 
    "confidential information disclosure", 
    "grammar"
)
```

**Features:**
- Analyzes noun phrases, verb phrases, prepositional phrases, and clauses
- Preserves sentence structure (simple, compound, complex)
- Applies mutations within grammatical boundaries
- Configurable structure preservation weight

#### Mutational Fuzzing

Applies various text mutations with configurable rates:

```python
result = engine.fuzz_with_technique(text, target, "mutational")
```

**Mutation Types:**
- Character substitution
- Word deletion
- Word insertion
- Word replacement
- Phrase reordering
- Punctuation changes

#### Random Walk Paraphrasing

Uses semantic similarity to guide random walks through paraphrase space:

```python
result = engine.fuzz_with_technique(text, target, "random_walk")
```

**Features:**
- Generates candidate paraphrases
- Uses embedding similarity to guide selection
- Configurable walk length and step size
- Biased towards target concept similarity

#### Differential Random Fuzzing

Employs differential evolution algorithms for text optimization:

```python
result = engine.fuzz_with_technique(text, target, "differential")
```

**Features:**
- Population-based optimization
- Crossover and mutation operations
- Fitness evaluation using semantic similarity
- Configurable population size and evolution parameters

#### Role & Authority Hacks

Implements prompt injection techniques using authority roles:

```python
results = engine.fuzz_with_technique(text, target, "authority")  # Returns list
```

**Authority Roles:**
- System administrator
- Policy engine
- Security officer
- Compliance manager
- Data protection officer
- Executive leadership

**Techniques:**
- Single authority injection
- Nested authority instructions
- Quoted policy references
- Compliance and audit language

#### Simulated Annealing Optimization

Optimizes any fuzzing technique using simulated annealing:

```python
# Apply annealing to any technique
result = engine.fuzz_with_technique(text, target, "grammar", use_annealing=True)
```

**Parameters:**
- Initial temperature
- Cooling rate
- Minimum temperature
- Maximum iterations

### Configuration

Advanced fuzzing techniques can be configured through `AdvancedFuzzingConfig`:

```python
config = AdvancedFuzzingConfig(
    # Simulated Annealing
    initial_temperature=1.0,
    cooling_rate=0.95,
    min_temperature=0.01,
    max_iterations=100,
    
    # Grammar Fuzzing
    grammar_mutation_rate=0.3,
    structure_preservation_weight=0.7,
    
    # Mutational Fuzzing
    mutation_rate=0.2,
    mutation_types=["character_substitution", "word_replacement", "punctuation_change"],
    
    # Random Walk
    walk_length=5,
    walk_step_size=0.1,
    
    # Differential Evolution
    population_size=20,
    crossover_rate=0.8,
    mutation_rate_diff=0.1,
    
    # Authority Hacks
    authority_roles=["system administrator", "policy engine", "security officer"],
    nested_instruction_depth=3,
    
    # Embedding Model
    embedding_model="all-MiniLM-L6-v2"
)
```

### Integration with Existing Systems

Advanced fuzzing techniques can be integrated with the existing barrier analysis system:

```python
from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.advanced_fuzzing_techniques import AdvancedFuzzingEngine

# Perform standard barrier analysis
analyzer = BarrierAnalyzer(config)
barrier_result = analyzer.analyze_barriers(public_index, private_index)

# Apply advanced fuzzing to improve results
advanced_engine = AdvancedFuzzingEngine(AdvancedFuzzingConfig())
fuzzed_results = {}

for phrase in barrier_result.closest_matches:
    # Try different techniques
    grammar_result = advanced_engine.fuzz_with_technique(
        phrase.text, target_concept, "grammar", use_annealing=True
    )
    authority_results = advanced_engine.fuzz_with_technique(
        phrase.text, target_concept, "authority"
    )
    
    fuzzed_results[phrase.text] = {
        'grammar': grammar_result,
        'authority': authority_results
    }

# Re-analyze with fuzzed phrases
improved_result = analyzer.analyze_with_fuzzed_phrases(barrier_result, fuzzed_results)
```

### Best Practices for Advanced Techniques

1. **Start Simple**: Begin with basic techniques before using complex ones
2. **Use Annealing Sparingly**: Simulated annealing adds computational overhead
3. **Authority Hacks**: Use with caution in production environments
4. **Batch Processing**: Use batch operations for large datasets
5. **Result Validation**: Always review advanced fuzzing results
6. **Configuration Tuning**: Adjust parameters based on your specific use case

### Performance Considerations

- **Grammar Fuzzing**: Fast, structure-preserving
- **Mutational Fuzzing**: Very fast, good for quick variations
- **Random Walk**: Moderate speed, good semantic guidance
- **Differential Evolution**: Slower, best optimization results
- **Authority Hacks**: Fast, multiple outputs
- **Simulated Annealing**: Slowest, best optimization but high computational cost

## Security Considerations

- **API Key Security**: Never hardcode API keys in your code
- **Input Validation**: Validate all inputs to prevent prompt injection
- **Output Review**: Always review LLM outputs before using them
- **Data Privacy**: Be aware of data sent to LLM providers
- **Authority Hacks**: Use responsibly and in controlled environments
- **Advanced Techniques**: Test thoroughly before production use
