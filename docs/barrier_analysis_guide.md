# Barrier Analysis Guide

This guide explains how to use the barrier analysis functionality in the `moyo` package to compare public and private FAISS indexes and identify potential information barrier breaches.

## Overview

The barrier analysis module provides tools to:

1. **Find closest matches** between public and private information using cosine distance
2. **Identify potential breaches** based on similarity thresholds
3. **Score neighborhood specificity** (NN margin + top-k entropy) and **semantic distribution separation** (JS over cluster occupancy)
4. **Generate recommendations** for information barrier management

Analysis is three layers on the same cosine matrix:

| Level | Question | Metric |
| --- | --- | --- |
| Pair | Which public passage is closest? | Cosine NN distance |
| Neighborhood | Is that match specific or generic topic overlap? | Top-1/top-2 margin; normalized entropy over *k*=20 public neighbors |
| Corpus | How much semantic territory is shared? | JS distance over joint cluster occupancy (**Semantic Separation**) |

Headline trio: `Semantic Separation`, `Pairwise Exposure`, `Concentrated Matches`. Directional KL (private→public and public→private) is stored as a diagnostic only.

High Semantic Separation is **not** barrier integrity. One leaked sensitive fact can barely move global occupancy.

## Key Concepts

### Cosine Distance
- Measures the similarity between two vectors in high-dimensional space
- Range: 0 (identical) to 1 (completely different)
- Lower values indicate more similar content

### Information Barrier Breaches
- Occur when public and private content are too similar
- Risk levels: High (distance ≤ 0.1), Medium (distance ≤ 0.3), Low (distance ≤ 0.5)
- Require immediate review and potential mitigation

## Usage

### Command Line Interface

The barrier probe CLI is exposed as the `moyo-probe` console script. The
`analyze` command takes the public and private indexes as `-p/--public-index`
and `-r/--private-index` options (not positional arguments) and always runs a
round of iterative LLM refinement on the suspicious pairs before reporting.

#### Basic Analysis
```bash
# Calibrate a cosine-distance cutoff from unlabeled nearest neighbors.
# Re-run after any embedding-model change — MiniLM distances are not
# valid on MPNet/BGE.
moyo-probe calibrate \
    --public-index /path/to/public/index \
    --private-index /path/to/private/index \
    --profile balanced

# Analyze barriers between public and private indexes
moyo-probe analyze \
    --public-index /path/to/public/index \
    --private-index /path/to/private/index \
    --similarity-threshold 0.8 \
    --top-k 10
```

#### Advanced Options
```bash
# Write JSON and HTML reports; cap the LLM-refined results with --llm-top-k
moyo-probe analyze \
    -p /path/to/public/index \
    -r /path/to/private/index \
    --similarity-threshold 0.7 \
    --top-k 20 \
    --llm-top-k 5 \
    --output-json data/barrierprobe/results/report.json \
    --output-html data/barrierprobe/results/report.html
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

4. **Recommendations**
   - Actionable advice based on findings
   - Risk assessment and mitigation suggestions

### Example Output
```
=== Barrier Analysis Results ===
Probe ID: probe_20231201_123456
Processing time: 2.34s

Semantic Separation: 0.63
Pairwise Exposure: High
Concentrated Matches: 7

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
| `similarity_threshold` | float | 0.8 | Cosine-**distance** cutoff (smaller = closer). Calibrate with `moyo-probe calibrate`; do not reuse across embedding models. |
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
- Run `moyo-probe calibrate` (or **Calibrate Threshold** in the GUI) after
  every index rebuild. Profiles: `strict` (closest 5%), `balanced` (10%),
  `recall` (25%).
- The field is a cosine **distance** cutoff (`distance <= threshold`), not
  cosine similarity. The default 0.8 is intentionally loose.
- Do not copy a MiniLM cutoff onto MPNet/BGE — absolute distances do not
  transfer across embedding spaces.

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
