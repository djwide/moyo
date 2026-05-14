# Barrier Analysis Guide

This guide explains how to use the barrier analysis functionality in the `moyo` package to compare public and private FAISS indexes and identify potential information barrier breaches.

## Overview

The barrier analysis module provides tools to:

1. **Find closest matches** between public and private information using cosine distance
2. **Calculate Sobolev norms** to analyze vector characteristics
3. **Identify potential breaches** based on similarity thresholds
4. **Generate recommendations** for information barrier management

## Key Concepts

### Cosine Distance
- Measures the similarity between two vectors in high-dimensional space
- Range: 0 (identical) to 1 (completely different)
- Lower values indicate more similar content

### Sobolev Norms
- Mathematical measure of vector "smoothness" and complexity
- Higher values indicate more complex or variable content
- Useful for identifying content that might be more sensitive or detailed

### Information Barrier Breaches
- Occur when public and private content are too similar
- Risk levels: High (distance ≤ 0.1), Medium (distance ≤ 0.3), Low (distance ≤ 0.5)
- Require immediate review and potential mitigation

## Usage

### Command Line Interface

#### Basic Analysis
```bash
# Analyze barriers between public and private indexes
python -m moyo.publicside.barrierprobe.cli probe analyze \
    /path/to/public/index \
    /path/to/private/index \
    --similarity-threshold 0.8 \
    --top-k 10
```

#### Advanced Options
```bash
# Save detailed results to file
python -m moyo.publicside.barrierprobe.cli probe analyze \
    /path/to/public/index \
    /path/to/private/index \
    --similarity-threshold 0.7 \
    --top-k 20 \
    --save-results \
    --output-dir data/barrierprobe/results
```

### Programmatic Usage

#### Basic Analysis
```python
from moyo.publicside.barrierprobe.barrier_analyzer import analyze_barriers

# Perform analysis
result = analyze_barriers(
    public_index_path="/path/to/public/index",
    private_index_path="/path/to/private/index",
    similarity_threshold=0.8,
    top_k=10
)

# Access results
print(f"Found {result.breach_count} potential breaches")
print(f"High risk: {result.high_risk_breaches}")
print(f"Medium risk: {result.medium_risk_breaches}")
print(f"Low risk: {result.low_risk_breaches}")
```

#### Advanced Analysis
```python
from moyo.publicside.barrierprobe.barrier_analyzer import BarrierAnalyzer
from moyo.publicside.barrierprobe.schema import BarrierProbeConfig

# Create configuration
config = BarrierProbeConfig(
    public_index_path="/path/to/public/index",
    private_index_path="/path/to/private/index",
    similarity_threshold=0.8,
    max_comparisons=1000
)

# Create analyzer
analyzer = BarrierAnalyzer(config)

# Load indexes
if analyzer.load_indexes():
    # Find closest matches
    closest_matches = analyzer.find_closest_matches(top_k=10)
    
    # Find largest Sobolev norms
    largest_norms = analyzer.find_largest_sobolev_norms(top_k=10, order=1)
    
    # Perform full analysis
    result = analyzer.analyze_barriers(top_k=10)
```

## Output Interpretation

### Analysis Results

The analysis provides several key metrics:

1. **Index Information**
   - Number of chunks in each index
   - Source types and organizations represented

2. **Breach Analysis**
   - Total number of potential breaches
   - Breakdown by risk level (high/medium/low)

3. **Closest Matches**
   - Top K most similar pairs of public/private content
   - Cosine distance for each pair
   - Content previews for review

4. **Sobolev Norms**
   - Top K chunks with largest Sobolev norms
   - Indicates content complexity and potential sensitivity

5. **Recommendations**
   - Actionable advice based on findings
   - Risk assessment and mitigation suggestions

### Example Output
```
=== Barrier Analysis Results ===
Probe ID: probe_20231201_123456
Processing time: 2.34s

Index Information:
  Public chunks: 150
  Private chunks: 75

Breach Analysis:
  Total breaches: 3
  High risk: 1
  Medium risk: 2
  Low risk: 0

Top 10 Closest Matches (Cosine Distance):
  1. Distance: 0.0234
     Public: Neural network architecture for image recognition...
     Private: Internal research on neural networks for computer vision...

  2. Distance: 0.0456
     Public: Deep learning techniques for natural language processing...
     Private: Our NLP implementation using transformer models...

Top 10 Largest Sobolev Norms:
  1. Norm: 2.5100 (public)
     Content: Comprehensive analysis of machine learning algorithms...

  2. Norm: 2.3685 (private)
     Content: Detailed technical specifications for our AI platform...

Recommendations:
  • Found 3 potential information barrier breaches
  • High risk breaches: 1, Medium risk: 2, Low risk: 0
  • Immediate action required: Review high-risk breaches
  • Review medium-risk breaches and consider additional controls
  • Average distance between closest matches: 0.1234
```

## Configuration Options

### BarrierProbeConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `public_index_path` | str | Required | Path to public FAISS index |
| `private_index_path` | str | Required | Path to private FAISS index |
| `similarity_threshold` | float | 0.8 | Threshold for breach detection |
| `max_comparisons` | int | 1000 | Maximum comparisons to perform |
| `output_directory` | str | "data/barrierprobe/results" | Output directory for results |
| `save_detailed_results` | bool | True | Save detailed results to file |
| `include_metadata` | bool | True | Include metadata in results |

### Risk Levels

| Distance Range | Risk Level | Action Required |
|----------------|------------|-----------------|
| 0.0 - 0.1 | High | Immediate review and mitigation |
| 0.1 - 0.3 | Medium | Review and consider additional controls |
| 0.3 - 0.5 | Low | Monitor and document |
| > 0.5 | None | No action required |

## Best Practices

### 1. Regular Analysis
- Perform barrier analysis regularly (weekly/monthly)
- Monitor trends in breach detection
- Update similarity thresholds based on organizational needs

### 2. Threshold Tuning
- Start with conservative thresholds (0.8-0.9)
- Adjust based on false positive/negative rates
- Consider different thresholds for different content types

### 3. Content Review
- Manually review all high-risk breaches
- Investigate medium-risk breaches for context
- Document decisions and mitigation actions

### 4. Integration
- Integrate with existing compliance workflows
- Automate alerts for high-risk breaches
- Maintain audit trails of all analyses

## Troubleshooting

### Common Issues

1. **Index Loading Failures**
   - Verify index paths are correct
   - Ensure indexes were built with compatible versions
   - Check file permissions

2. **Memory Issues**
   - Reduce `max_comparisons` for large indexes
   - Use smaller `top_k` values
   - Consider processing in batches

3. **No Results**
   - Check if indexes contain embeddings
   - Verify similarity threshold is appropriate
   - Ensure indexes contain compatible data

### Performance Optimization

1. **Large Indexes**
   - Use approximate search methods
   - Process in batches
   - Consider sampling for initial analysis

2. **Frequent Analysis**
   - Cache results where appropriate
   - Use incremental analysis
   - Optimize embedding calculations

## Examples

### Complete Workflow Example

```python
# 1. Build public index
from moyo.publicside.barrierprobe.public_index_builder import build_public_index_from_sources

public_result = build_public_index_from_sources(
    sources=public_sources,
    name="Public AI Research",
    description="Public AI research and patents"
)

# 2. Build private index
from moyo.privateside.mapcorpus.builder import CorpusBuilder

private_builder = CorpusBuilder()
# ... add private documents ...
private_builder.save_corpus("/path/to/private/index", "Private Research", "Internal research documents")

# 3. Perform barrier analysis
from moyo.publicside.barrierprobe.barrier_analyzer import analyze_barriers

result = analyze_barriers(
    public_index_path=public_result.index_path,
    private_index_path="/path/to/private/index",
    similarity_threshold=0.8,
    top_k=20
)

# 4. Review results
if result.breach_count > 0:
    print(f"⚠️  Found {result.breach_count} potential breaches")
    for breach in result.potential_breaches:
        print(f"Risk: {breach['risk_level']}, Distance: {breach['distance']:.4f}")
else:
    print("✅ No breaches detected")
```

This guide provides a comprehensive overview of the barrier analysis functionality. For more detailed information, refer to the individual module documentation and examples.
