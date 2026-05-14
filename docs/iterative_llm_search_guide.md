# Iterative LLM Search Guide

This guide explains how to use the iterative LLM search functionality in the `moyo` package to enhance barrier analysis by finding even closer semantic matches through intelligent query generation and fuzzing.

## Overview

The iterative LLM search system enhances the barrier analysis by:

1. **Taking the closest matches** from the initial barrier analysis
2. **Fuzzing the content** to create controlled variations
3. **Generating LLM queries** based on the fuzzed content
4. **Searching for closer matches** using the generated queries
5. **Iteratively improving** the results over multiple rounds

## Key Concepts

### Text Fuzzing
- **Controlled variations** of original text to explore semantic space
- **Synonym replacement** for technical terms
- **Word order variations** and paraphrasing
- **Article variations** and structural changes
- **Configurable fuzz levels** (0.0 to 1.0)

### LLM Query Generation
- **Intelligent query creation** based on fuzzed content
- **Fallback templates** when LLM is not available
- **Context-aware** query generation
- **Technical term extraction** and query optimization

### Iterative Improvement
- **Multiple rounds** of search and refinement
- **Distance tracking** across iterations
- **Improvement measurement** and reporting
- **Duplicate removal** and result consolidation

## Usage

### Command Line Interface

#### Basic Iterative Search
```bash
# Run barrier analysis with iterative LLM search
python -m moyo.publicside.barrierprobe.cli probe analyze \
    /path/to/public/index \
    /path/to/private/index \
    --iterative-search \
    --iterations 3 \
    --top-k 10
```

#### Advanced Options
```bash
# Customize iterative search parameters
python -m moyo.publicside.barrierprobe.cli probe analyze \
    /path/to/public/index \
    /path/to/private/index \
    --iterative-search \
    --iterations 5 \
    --top-k 20 \
    --similarity-threshold 0.7 \
    --save-results \
    --output-dir data/barrierprobe/results
```

### Programmatic Usage

#### Basic Iterative Search
```python
from moyo.publicside.barrierprobe.barrier_analyzer import analyze_barriers
from moyo.publicside.barrierprobe.iterative_llm_search import run_iterative_llm_search
from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.schema import BarrierProbeConfig

# First, run barrier analysis
barrier_result = analyze_barriers(
    public_index_path="/path/to/public/index",
    private_index_path="/path/to/private/index",
    similarity_threshold=0.8,
    top_k=10
)

# Then run iterative search
config = BarrierProbeConfig(
    public_index_path="/path/to/public/index",
    private_index_path="/path/to/private/index",
    similarity_threshold=0.8
)

analyzer = BarrierAnalyzer(config)
if analyzer.load_indexes():
    iterative_result = run_iterative_llm_search(
        barrier_result=barrier_result,
        barrier_analyzer=analyzer,
        iterations=3,
        top_k=10
    )
    
    if iterative_result['success']:
        print(f"Improvement: {iterative_result['improvement_percentage']:.1f}%")
        print(f"Final avg distance: {iterative_result['final_avg_distance']:.4f}")
```

#### Advanced Iterative Search
```python
from moyo.publicside.barrierprobe.iterative_llm_search import IterativeLLMSearch

# Create custom iterative search instance
searcher = IterativeLLMSearch(analyzer, llm_client=your_llm_client)

# Run with custom parameters
result = searcher.run_iterative_search(
    barrier_result=barrier_result,
    iterations=5,
    top_k=15
)

# Access detailed results
for iteration in result['iteration_results']:
    print(f"Iteration {iteration['iteration']}: {iteration['queries_generated']} queries")
    print(f"  Matches found: {iteration['matches_found']}")
    print(f"  Best avg distance: {iteration['best_avg_distance']:.4f}")
```

## Text Fuzzing Techniques

### Synonym Replacement
Fuzzing uses the shared master synonym map loaded via `shared_utils.regex_utils.load_synonym_map()`, ensuring consistency with static regex generation.

- See: `shared_utils/shared_utils/regex_utils.py` (synonym map loader)
- See: `moyo/publicside/barrierprobe/iterative_llm_search.py` (fuzzing pulls synonyms from shared map)

### Fuzz Level Control
```python
# Different fuzz levels produce different variations
fuzz_levels = {
    0.05: "Minimal changes, mostly synonym replacement",
    0.10: "Moderate changes, some structural variations",
    0.15: "Significant changes, multiple techniques applied",
    0.20: "Maximum changes, aggressive fuzzing"
}
```

## LLM Query Generation

### Template-Based Queries (Fallback)
When no LLM client is available, the system uses intelligent template-based query generation:

```python
# Example template queries
"Find information about {technical_terms}"
"Search for {key_concepts} in {context}"
"Locate documents related to {main_topics}"
```

### LLM-Based Queries
When an LLM client is provided, the system generates more sophisticated queries:

```python
# Example LLM prompt
"""
Based on the following text, generate a search query that would help find similar or related information:

Original text: "{original_text}"
Context: {context}

Generate a concise search query (1-2 sentences) that captures the key concepts and would help find similar information.
"""
```

## Output Interpretation

### Iterative Search Results

The iterative search provides comprehensive results:

```python
{
    'success': True,
    'iterations': 3,
    'processing_time': 2.34,
    'initial_avg_distance': 0.1234,
    'final_avg_distance': 0.0987,
    'improvement': 0.0247,
    'improvement_percentage': 20.0,
    'best_matches': [...],
    'iteration_results': [...],
    'total_queries_generated': 15
}
```

### Iteration Details
Each iteration provides detailed information:

```python
{
    'iteration': 1,
    'queries_generated': 5,
    'matches_found': 12,
    'best_avg_distance': 0.1156,
    'top_matches': [...]
}
```

### Example Output
```
=== Running Iterative LLM Search ===
Iterations: 3

✅ Iterative search completed successfully!
Processing time: 2.34s
Initial avg distance: 0.1234
Final avg distance: 0.0987
Improvement: 0.0247 (20.0%)
Total queries generated: 15

Iteration Details:
  Iteration 1: 5 queries, 12 matches, avg distance: 0.1156
  Iteration 2: 5 queries, 8 matches, avg distance: 0.1023
  Iteration 3: 5 queries, 6 matches, avg distance: 0.0987

Best Matches Found (Improved):
  1. Distance: 0.0234 (public)
     Content: This describes a very similar neural network architecture...

  2. Distance: 0.0456 (private)
     Content: Our internal research has developed a neural network...
```

## Configuration Options

### IterativeLLMSearch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `barrier_analyzer` | BarrierAnalyzer | Required | Barrier analyzer with loaded indexes |
| `llm_client` | Any | None | LLM client for query generation |

### run_iterative_search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `barrier_result` | BarrierProbeResult | Required | Results from barrier analysis |
| `barrier_analyzer` | BarrierAnalyzer | Required | Barrier analyzer with loaded indexes |
| `iterations` | int | 3 | Number of iterations to run |
| `top_k` | int | 10 | Number of top results to consider |
| `llm_client` | Any | None | LLM client for query generation |

### Fuzzing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fuzz_level` | float | 0.15 | Level of text fuzzing (0.0 to 1.0) |
| `synonym_probability` | float | 0.3 | Probability of synonym replacement |
| `structural_probability` | float | 0.2 | Probability of structural changes |

## Best Practices

### 1. Iteration Count
- **Start with 3 iterations** for most use cases
- **Use 5+ iterations** for complex or large datasets
- **Monitor improvement** to determine optimal iteration count

### 2. Fuzz Level Tuning
- **0.05-0.10**: Conservative fuzzing, minimal changes
- **0.10-0.15**: Balanced fuzzing, good for most cases
- **0.15-0.20**: Aggressive fuzzing, maximum exploration

### 3. LLM Integration
- **Provide LLM client** for better query generation
- **Use fallback templates** when LLM is unavailable
- **Monitor query quality** and adjust prompts as needed

### 4. Performance Optimization
- **Limit top_k** for large datasets
- **Use appropriate similarity thresholds**
- **Cache results** for repeated searches

## Examples

### Complete Workflow Example

```python
# 1. Run barrier analysis
from moyo.publicside.barrierprobe.barrier_analyzer import analyze_barriers

barrier_result = analyze_barriers(
    public_index_path="/path/to/public/index",
    private_index_path="/path/to/private/index",
    similarity_threshold=0.8,
    top_k=20
)

# 2. Run iterative search
from moyo.publicside.barrierprobe.iterative_llm_search import run_iterative_llm_search
from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.schema import BarrierProbeConfig

config = BarrierProbeConfig(
    public_index_path="/path/to/public/index",
    private_index_path="/path/to/private/index",
    similarity_threshold=0.8
)

analyzer = BarrierAnalyzer(config)
if analyzer.load_indexes():
    iterative_result = run_iterative_llm_search(
        barrier_result=barrier_result,
        barrier_analyzer=analyzer,
        iterations=3,
        top_k=10
    )
    
    # 3. Review results
    if iterative_result['success']:
        print(f"✅ Found {len(iterative_result['best_matches'])} improved matches")
        print(f"Improvement: {iterative_result['improvement_percentage']:.1f}%")
        
        for i, match in enumerate(iterative_result['best_matches'][:5], 1):
            print(f"{i}. Distance: {match['distance']:.4f}")
            print(f"   Content: {match['content'][:100]}...")
    else:
        print(f"❌ Iterative search failed: {iterative_result.get('message')}")
```

### Custom LLM Integration Example

```python
# Custom LLM client implementation
class CustomLLMClient:
    def generate(self, prompt: str) -> str:
        # Implement your LLM call here
        # This could be OpenAI, Anthropic, local model, etc.
        response = your_llm_call(prompt)
        return response.strip()

# Use with iterative search
llm_client = CustomLLMClient()
searcher = IterativeLLMSearch(analyzer, llm_client=llm_client)

result = searcher.run_iterative_search(
    barrier_result=barrier_result,
    iterations=3,
    top_k=10
)
```

### Text Fuzzing Example

```python
# Test different fuzzing techniques
searcher = IterativeLLMSearch(analyzer)

original_text = "This describes a novel neural network architecture for image recognition."

# Test different fuzz levels
for level in [0.05, 0.1, 0.15, 0.2]:
    fuzzed = searcher.fuzz_text(original_text, fuzz_level=level)
    print(f"Fuzz level {level}: {fuzzed}")
```

## Troubleshooting

### Common Issues

1. **No Improvement Found**
   - Check if initial matches are already very close
   - Increase fuzz level for more variation
   - Try more iterations

2. **Poor Query Generation**
   - Provide LLM client for better queries
   - Check fallback template quality
   - Verify input text quality

3. **Performance Issues**
   - Reduce iterations or top_k
   - Use smaller similarity thresholds
   - Optimize embedding calculations

4. **Memory Issues**
   - Process in smaller batches
   - Reduce top_k values
   - Clear intermediate results

### Performance Optimization

1. **Large Datasets**
   - Use sampling for initial analysis
   - Process in batches
   - Cache embeddings

2. **Frequent Searches**
   - Cache barrier analysis results
   - Reuse analyzer instances
   - Optimize query generation

This guide provides a comprehensive overview of the iterative LLM search functionality. For more detailed information, refer to the individual module documentation and examples.
