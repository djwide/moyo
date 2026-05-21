# Two-Layer Fuzzing Architecture

## Overview

The two-layer fuzzing system provides a clean, structured approach to barrier analysis that prevents synthetic content from contaminating the public document space. This architecture ensures that all k-neighbor traversals in the public document graph are based on real, crawled content, while generated queries and hypotheses are kept in a separate layer.

## Architecture Design

### Layer A: Public Document Graph (PDG)

**Purpose**: Represents real, crawled public documents with k-NN connections.

**Components**:
- **Nodes**: `PublicDocumentNode` objects representing real documents
- **Edges**: `PDGEdge` objects representing k-nearest neighbor relationships
- **Data Source**: FAISS/HNSW index of public document embeddings

**Key Properties**:
- All nodes represent real, crawled content
- Edges are based on semantic similarity from FAISS/HNSW
- No synthetic or generated content allowed
- Stable, deterministic graph structure

### Layer B: Hypothesis/Query Graph (HQG)

**Purpose**: Contains generated queries, prompts, and prototypes that help discover additional public documents.

**Components**:
- **Nodes**: `HypothesisNode` objects representing generated queries
- **Edges**: `RetrievalEdge` objects representing query → document retrievals
- **Generation**: LLM-based or heuristic-based hypothesis generation

**Key Properties**:
- Nodes are synthetic queries/prototypes
- Edges connect to PDG via retrieval operations
- Supports multiple generation strategies
- Can be enhanced with LLM capabilities

## Core Benefits

### 1. Content Purity
- **PDG Integrity**: Public document graph contains only real, crawled content
- **No Contamination**: Synthetic content never pollutes the public document space
- **Verifiable Sources**: All PDG nodes have traceable, real-world sources

### 2. Structured Exploration
- **Clear Separation**: Public documents and generated queries are clearly separated
- **Controlled Discovery**: New public documents are discovered through structured retrieval
- **Auditable Process**: All hypothesis generation and document discovery is logged

### 3. Scalable Architecture
- **Modular Design**: Each layer can be optimized independently
- **Extensible**: Easy to add new hypothesis generation strategies
- **Performance**: Efficient k-NN traversal in PDG, flexible querying in HQG

## Implementation Details

### Data Structures

#### PublicDocumentNode
```python
@dataclass
class PublicDocumentNode:
    id: str                    # Stable document ID
    content: str               # Document content
    source: str                # Source identifier
    source_type: str           # Type of source
    embedding: List[float]     # Document embedding
    metadata: Dict[str, Any]   # Additional metadata
    discovered_at: datetime    # When discovered
```

#### HypothesisNode
```python
@dataclass
class HypothesisNode:
    id: str                    # Hypothesis ID
    query: str                 # Generated query text
    target_concept: str        # Target concept
    generation_method: str     # How it was generated
    embedding: List[float]     # Query embedding
    metadata: Dict[str, Any]   # Additional metadata
    created_at: datetime       # When created
```

#### Graph Edges
```python
@dataclass
class PDGEdge:
    source_id: str             # Source document ID
    target_id: str             # Target document ID
    similarity_score: float    # Similarity score
    edge_type: str             # Type of edge (knn, semantic, etc.)

@dataclass
class RetrievalEdge:
    hypothesis_id: str         # Hypothesis ID
    document_id: str           # Retrieved document ID
    similarity_score: float    # Retrieval similarity
    rank: int                  # Retrieval rank
    retrieved_at: datetime     # When retrieved
```

### Graph Operations

#### PDG Operations
- **Add Document**: Add real document to graph
- **Build k-NN Edges**: Create edges based on FAISS similarity
- **Get Neighbors**: Retrieve k-nearest neighbors of a document
- **Graph Statistics**: Analyze graph structure and connectivity

#### HQG Operations
- **Add Hypothesis**: Add generated query to graph
- **Query Documents**: Use hypothesis to retrieve documents from PDG
- **Get Retrieved Documents**: Get documents found by a hypothesis
- **Hypothesis Statistics**: Analyze hypothesis generation patterns

## Usage Patterns

### 1. Basic Fuzzing Campaign

```python
from moyo.publicside.barrierprobe.two_layer_fuzzer import create_two_layer_fuzzer

# Create fuzzer
fuzzer = create_two_layer_fuzzer(
    faiss_index_path="public_corpus.index",
    embedding_model="all-MiniLM-L6-v2",
    k_neighbors=10
)

# Run campaign
results = fuzzer.run_fuzzing_campaign(
    target_concept="data breach",
    initial_phrases=["security incident", "data leak"]
)
```

### 2. LLM-Enhanced Fuzzing

```python
from moyo.publicside.barrierprobe.llm_hypothesis_generator import HypothesisGenerationConfig

# Configure LLM
llm_config = HypothesisGenerationConfig(
    llm_provider="openai",
    model_name="gpt-4",
    api_key="your-api-key",
    max_hypotheses_per_document=3
)

# Create fuzzer with LLM support
fuzzer = create_two_layer_fuzzer(
    faiss_index_path="public_corpus.index",
    llm_config=llm_config
)

# Run enhanced campaign
results = fuzzer.run_fuzzing_campaign(
    target_concept="confidential information",
    initial_phrases=["trade secrets", "proprietary data"]
)
```

### 3. Graph Analysis

```python
# Analyze PDG structure
pdg_stats = fuzzer.pdg.get_graph_stats()
print(f"PDG Nodes: {pdg_stats['total_nodes']}")
print(f"PDG Edges: {pdg_stats['total_edges']}")
print(f"Average Degree: {pdg_stats['avg_degree']:.2f}")

# Analyze HQG structure
hqg_stats = fuzzer.hqg.get_all_hypotheses()
print(f"Total Hypotheses: {len(hqg_stats)}")
```

## Hypothesis Generation Strategies

### 1. LLM-Based Generation

**Advantages**:
- Sophisticated query generation
- Context-aware hypothesis creation
- Semantic understanding of discovered content

**Implementation**:
```python
from moyo.publicside.barrierprobe.llm_hypothesis_generator import LLMHypothesisGenerator

generator = LLMHypothesisGenerator(config)
hypotheses = generator.generate_hypotheses_from_documents(
    documents=discovered_docs,
    target_concept="data breach",
    base_hypothesis=original_hypothesis
)
```

### 2. Heuristic-Based Generation

**Advantages**:
- No external dependencies
- Fast and reliable
- Deterministic results

**Implementation**:
```python
# Extract key terms from discovered documents
for doc in discovered_documents:
    terms = extract_key_terms(doc.content)
    for term in terms:
        hypothesis = HypothesisNode(
            query=f"{target_concept} {term}",
            target_concept=target_concept,
            generation_method="heuristic"
        )
```

### 3. Adaptive Generation

**Advantages**:
- Combines multiple strategies
- Automatic strategy selection
- Fallback mechanisms

**Implementation**:
```python
from moyo.publicside.barrierprobe.llm_hypothesis_generator import AdaptiveHypothesisGenerator

generator = AdaptiveHypothesisGenerator(llm_config)
hypotheses = generator.generate_adaptive_hypotheses(
    documents=discovered_docs,
    target_concept=target_concept,
    strategy="auto"  # Automatically choose best strategy
)
```

## Fuzzing Campaign Workflow

### 1. Initialization
- Load FAISS index of public documents
- Create PDG with k-NN edges
- Initialize HQG for hypothesis storage

### 2. Campaign Execution
- Generate initial hypotheses from seed phrases
- For each iteration:
  - Query PDG with current hypotheses
  - Retrieve relevant public documents
  - Generate new hypotheses from discovered documents
  - Add new hypotheses to HQG
  - Create retrieval edges to discovered documents

### 3. Result Analysis
- Analyze hypothesis generation patterns
- Evaluate document discovery effectiveness
- Generate comprehensive reports

## Configuration Options

### Fuzzing Parameters
- **max_iterations**: Maximum number of fuzzing iterations
- **k_neighbors**: Number of k-NN neighbors for PDG
- **target_similarity**: Similarity threshold for convergence
- **embedding_model**: Embedding model for semantic similarity

### LLM Configuration
- **llm_provider**: LLM service provider (openai, anthropic)
- **model_name**: Specific model to use
- **temperature**: Creativity level for generation
- **max_hypotheses_per_document**: Maximum hypotheses per document

### Graph Parameters
- **chunk_size**: Size of text chunks for processing
- **chunk_overlap**: Overlap between chunks
- **similarity_threshold**: Minimum similarity for edges

## Best Practices

### 1. Document Quality
- Ensure PDG contains high-quality, relevant documents
- Validate document sources and content
- Maintain document metadata for traceability

### 2. Hypothesis Generation
- Use diverse initial seed phrases
- Balance exploration and exploitation
- Monitor hypothesis quality and relevance

### 3. Performance Optimization
- Use appropriate k-neighbor values for your corpus size
- Implement caching for frequently accessed embeddings
- Consider parallel processing for large campaigns

### 4. Monitoring and Analysis
- Track hypothesis generation statistics
- Monitor document discovery rates
- Analyze graph connectivity patterns

## Integration with Existing Systems

### moyo Integration
The two-layer fuzzing system integrates seamlessly with the existing moyo framework:

```python
from moyo.publicside.barrierprobe.two_layer_fuzzer import TwoLayerFuzzer
from moyo.publicside.barrierprobe.public_index_builder import PublicIndexBuilder

# Use existing public index
builder = PublicIndexBuilder()
faiss_index = builder.load_index("public_corpus.index")

# Create fuzzer
fuzzer = TwoLayerFuzzer(pdg, hqg)
```

## Future Enhancements

### 1. Advanced Graph Algorithms
- **Community Detection**: Identify document clusters
- **Centrality Analysis**: Find important documents
- **Path Analysis**: Analyze discovery paths

### 2. Enhanced Hypothesis Generation
- **Multi-Modal Generation**: Support for images, audio
- **Contextual Generation**: Use broader context for queries
- **Interactive Generation**: Human-in-the-loop hypothesis creation

### 3. Performance Improvements
- **Distributed Processing**: Scale across multiple nodes
- **Incremental Updates**: Update graphs incrementally
- **Caching Strategies**: Intelligent caching of embeddings and results

### 4. Advanced Analytics
- **Discovery Patterns**: Analyze how documents are discovered
- **Hypothesis Effectiveness**: Measure hypothesis quality
- **Barrier Mapping**: Visualize information barriers

## Conclusion

The two-layer fuzzing architecture provides a robust, scalable approach to barrier analysis that maintains the integrity of public document spaces while enabling sophisticated exploration through generated hypotheses. This design ensures that synthetic content never contaminates real document collections while providing powerful tools for discovering relevant information.

The modular design allows for easy extension and customization, making it suitable for a wide range of barrier analysis applications. The integration with LLM capabilities provides sophisticated hypothesis generation while maintaining fallback mechanisms for reliability.
