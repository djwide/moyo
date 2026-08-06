"""Two-layer fuzzing system for barrier analysis.

This module implements a clean two-layer architecture:
- Layer A: Public Document Graph (PDG) - real crawled documents
- Layer B: Hypothesis/Query Graph (HQG) - generated queries/prototypes

This ensures synthetic content never contaminates the public document space.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import json

from shared_utils import (
    embed,
    FAISSIndex, 
    NormalizedDocument,
    DocumentCollection,
    generate_stable_document_id,
    generate_content_hash
)

# NOTE: `llm_hypothesis_generator` imports the node dataclasses defined below,
# so importing it at module top would create a circular import. It is imported
# lazily inside `create_two_layer_fuzzer`; annotations reference it only as
# forward references (safe thanks to `from __future__ import annotations`).
if TYPE_CHECKING:
    from .llm_hypothesis_generator import (
        AdaptiveHypothesisGenerator,
        HypothesisGenerationConfig,
    )

logger = logging.getLogger(__name__)


@dataclass
class PublicDocumentNode:
    """A node in the Public Document Graph (PDG)."""
    id: str
    content: str
    source: str
    source_type: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.id is None:
            self.id = generate_stable_document_id(self.source, generate_content_hash(self.content))


@dataclass
class HypothesisNode:
    """A node in the Hypothesis/Query Graph (HQG)."""
    id: str
    query: str
    target_concept: str
    generation_method: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"hypothesis_{generate_content_hash(self.query)[:12]}"


@dataclass
class RetrievalEdge:
    """An edge from HQG to PDG representing a query → document retrieval."""
    hypothesis_id: str
    document_id: str
    similarity_score: float
    rank: int
    retrieved_at: datetime = field(default_factory=datetime.now)


@dataclass
class PDGEdge:
    """An edge within the Public Document Graph (PDG)."""
    source_id: str
    target_id: str
    similarity_score: float
    edge_type: str = "knn"  # "knn", "semantic", etc.


class PublicDocumentGraph:
    """Layer A: Graph of real public documents with k-NN edges."""
    
    def __init__(self, faiss_index: FAISSIndex, k_neighbors: int = 10):
        """Initialize the public document graph.
        
        Args:
            faiss_index: FAISS index containing public document embeddings
            k_neighbors: Number of nearest neighbors to connect
        """
        self.faiss_index = faiss_index
        self.k_neighbors = k_neighbors
        self.nodes: Dict[str, PublicDocumentNode] = {}
        self.edges: List[PDGEdge] = []
        self.node_embeddings: Dict[str, List[float]] = {}
        
    def add_document(self, document: PublicDocumentNode) -> None:
        """Add a document to the PDG."""
        self.nodes[document.id] = document
        if document.embedding:
            self.node_embeddings[document.id] = document.embedding
            
    def build_knn_edges(self) -> None:
        """Build k-NN edges between all documents in the graph."""
        logger.info(f"Building k-NN edges for {len(self.nodes)} documents")
        
        # Get all embeddings
        node_ids = list(self.nodes.keys())
        embeddings = [self.node_embeddings.get(node_id) for node_id in node_ids]
        
        # Filter out nodes without embeddings
        valid_pairs = [(node_id, emb) for node_id, emb in zip(node_ids, embeddings) if emb is not None]
        if not valid_pairs:
            logger.warning("No valid embeddings found for building k-NN edges")
            return
            
        valid_ids, valid_embeddings = zip(*valid_pairs)
        
        # Build edges for each node
        for i, (node_id, embedding) in enumerate(zip(valid_ids, valid_embeddings)):
            # Find k nearest neighbors (excluding self)
            distances, indices = self.faiss_index.search([embedding], self.k_neighbors + 1)
            
            # Add edges to nearest neighbors (skip first if it's self)
            for j, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if j == 0 and idx == i:  # Skip self
                    continue
                    
                if idx < len(valid_ids):
                    neighbor_id = valid_ids[idx]
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    edge = PDGEdge(
                        source_id=node_id,
                        target_id=neighbor_id,
                        similarity_score=similarity,
                        edge_type="knn"
                    )
                    self.edges.append(edge)
                    
    def get_neighbors(self, node_id: str, max_neighbors: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get k-nearest neighbors of a document node.
        
        Args:
            node_id: ID of the source node
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of (neighbor_id, similarity_score) tuples
        """
        neighbors = []
        
        # Find edges where this node is the source
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbors.append((edge.target_id, edge.similarity_score))
                
        # Sort by similarity and limit
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if max_neighbors:
            neighbors = neighbors[:max_neighbors]
            
        return neighbors
    
    def get_document_by_id(self, node_id: str) -> Optional[PublicDocumentNode]:
        """Get a document node by ID."""
        return self.nodes.get(node_id)
    
    def get_all_documents(self) -> List[PublicDocumentNode]:
        """Get all documents in the graph."""
        return list(self.nodes.values())
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "avg_degree": len(self.edges) / len(self.nodes) if self.nodes else 0,
            "nodes_with_embeddings": len(self.node_embeddings)
        }


class HypothesisQueryGraph:
    """Layer B: Graph of generated hypotheses/queries that connect to PDG."""
    
    def __init__(self, pdg: PublicDocumentGraph):
        """Initialize the hypothesis query graph.
        
        Args:
            pdg: Reference to the public document graph
        """
        self.pdg = pdg
        self.nodes: Dict[str, HypothesisNode] = {}
        self.retrieval_edges: List[RetrievalEdge] = {}
        
    def add_hypothesis(self, hypothesis: HypothesisNode) -> None:
        """Add a hypothesis to the HQG."""
        self.nodes[hypothesis.id] = hypothesis
        
    def query_public_documents(self, hypothesis_id: str, top_k: int = 10) -> List[RetrievalEdge]:
        """Query the public document graph with a hypothesis.
        
        Args:
            hypothesis_id: ID of the hypothesis to use for querying
            top_k: Number of top documents to retrieve
            
        Returns:
            List of retrieval edges
        """
        hypothesis = self.nodes.get(hypothesis_id)
        if not hypothesis or not hypothesis.embedding:
            logger.warning(f"No embedding found for hypothesis {hypothesis_id}")
            return []
            
        # Query the FAISS index
        distances, indices, metadata = self.pdg.faiss_index.search(
            [hypothesis.embedding], 
            top_k
        )
        
        # Create retrieval edges
        edges = []
        for i, (distance, idx, meta) in enumerate(zip(distances[0], indices[0], metadata)):
            if idx < len(self.pdg.nodes):
                # Get document ID from metadata or index
                doc_id = meta.get('id') if meta else f"doc_{idx}"
                similarity = 1.0 - distance
                
                edge = RetrievalEdge(
                    hypothesis_id=hypothesis_id,
                    document_id=doc_id,
                    similarity_score=similarity,
                    rank=i + 1
                )
                edges.append(edge)
                
        # Store edges
        if hypothesis_id not in self.retrieval_edges:
            self.retrieval_edges[hypothesis_id] = []
        self.retrieval_edges[hypothesis_id].extend(edges)
        
        return edges
    
    def get_retrieved_documents(self, hypothesis_id: str) -> List[PublicDocumentNode]:
        """Get documents retrieved by a hypothesis."""
        edges = self.retrieval_edges.get(hypothesis_id, [])
        documents = []
        
        for edge in edges:
            doc = self.pdg.get_document_by_id(edge.document_id)
            if doc:
                documents.append(doc)
                
        return documents
    
    def get_hypothesis_by_id(self, hypothesis_id: str) -> Optional[HypothesisNode]:
        """Get a hypothesis by ID."""
        return self.nodes.get(hypothesis_id)
    
    def get_all_hypotheses(self) -> List[HypothesisNode]:
        """Get all hypotheses in the graph."""
        return list(self.nodes.values())


class TwoLayerFuzzer:
    """Main fuzzing system that coordinates the two-layer architecture."""
    
    def __init__(self, 
                 pdg: PublicDocumentGraph,
                 hqg: HypothesisQueryGraph,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_iterations: int = 5,
                 target_similarity: float = 0.95,
                 hypothesis_generator: Optional[AdaptiveHypothesisGenerator] = None):
        """Initialize the two-layer fuzzer.
        
        Args:
            pdg: Public document graph
            hqg: Hypothesis query graph
            embedding_model: Embedding model to use
            max_iterations: Maximum fuzzing iterations
            target_similarity: Target similarity threshold
            hypothesis_generator: Optional hypothesis generator for LLM-based fuzzing
        """
        self.pdg = pdg
        self.hqg = hqg
        self.embedding_model = embedding_model
        self.max_iterations = max_iterations
        self.target_similarity = target_similarity
        self.hypothesis_generator = hypothesis_generator
        self.iteration_history: List[Dict[str, Any]] = []
        
    def generate_initial_hypotheses(self, 
                                  target_concept: str,
                                  initial_phrases: List[str]) -> List[HypothesisNode]:
        """Generate initial hypotheses from seed phrases.
        
        Args:
            target_concept: Target concept to move towards
            initial_phrases: Initial seed phrases
            
        Returns:
            List of generated hypothesis nodes
        """
        hypotheses = []
        
        for phrase in initial_phrases:
            # Create hypothesis node
            hypothesis = HypothesisNode(
                id=None,  # Will be auto-generated
                query=phrase,
                target_concept=target_concept,
                generation_method="seed"
            )
            
            # Generate embedding
            embeddings = embed([phrase], self.embedding_model)
            if embeddings:
                hypothesis.embedding = embeddings[0]
                
            hypotheses.append(hypothesis)
            self.hqg.add_hypothesis(hypothesis)
            
        logger.info(f"Generated {len(hypotheses)} initial hypotheses")
        return hypotheses
    
    def fuzz_hypothesis(self, 
                       hypothesis: HypothesisNode,
                       discovered_documents: List[PublicDocumentNode]) -> List[HypothesisNode]:
        """Fuzz a hypothesis based on discovered documents.
        
        Args:
            hypothesis: Hypothesis to fuzz
            discovered_documents: Documents discovered by this hypothesis
            
        Returns:
            List of new hypothesis nodes
        """
        new_hypotheses = []
        
        if self.hypothesis_generator:
            # Use LLM-based hypothesis generation
            llm_hypotheses = self.hypothesis_generator.generate_adaptive_hypotheses(
                discovered_documents, 
                hypothesis.target_concept, 
                hypothesis
            )
            
            # Generate embeddings for LLM hypotheses
            for llm_hypothesis in llm_hypotheses:
                embeddings = embed([llm_hypothesis.query], self.embedding_model)
                if embeddings:
                    llm_hypothesis.embedding = embeddings[0]
                    new_hypotheses.append(llm_hypothesis)
                    self.hqg.add_hypothesis(llm_hypothesis)
        else:
            # Fallback to simple heuristic-based generation
            for doc in discovered_documents[:5]:  # Limit to top 5
                sentences = doc.content.split('.')
                for sentence in sentences[:3]:  # Take first 3 sentences
                    if len(sentence.strip()) > 20:  # Minimum length
                        new_query = sentence.strip()
                        
                        # Create new hypothesis
                        new_hypothesis = HypothesisNode(
                            id=None,
                            query=new_query,
                            target_concept=hypothesis.target_concept,
                            generation_method=f"fuzzed_from_{hypothesis.id}"
                        )
                        
                        # Generate embedding
                        embeddings = embed([new_query], self.embedding_model)
                        if embeddings:
                            new_hypothesis.embedding = embeddings[0]
                            
                        new_hypotheses.append(new_hypothesis)
                        self.hqg.add_hypothesis(new_hypothesis)
                    
        logger.info(f"Generated {len(new_hypotheses)} new hypotheses from {hypothesis.id}")
        return new_hypotheses
    
    def run_fuzzing_iteration(self, 
                            target_concept: str,
                            current_hypotheses: List[HypothesisNode]) -> List[HypothesisNode]:
        """Run one iteration of the fuzzing process.
        
        Args:
            target_concept: Target concept to move towards
            current_hypotheses: Current hypotheses to expand
            
        Returns:
            List of new hypotheses generated
        """
        new_hypotheses = []
        discovered_documents = set()
        
        # Query public documents with each hypothesis
        for hypothesis in current_hypotheses:
            # Query PDG
            retrieval_edges = self.hqg.query_public_documents(hypothesis.id, top_k=10)
            
            # Get discovered documents
            docs = self.hqg.get_retrieved_documents(hypothesis.id)
            discovered_documents.update(docs)
            
            # Fuzz hypothesis based on discovered documents
            fuzzed_hypotheses = self.fuzz_hypothesis(hypothesis, docs)
            new_hypotheses.extend(fuzzed_hypotheses)
            
        # Record iteration
        iteration_record = {
            "iteration": len(self.iteration_history) + 1,
            "timestamp": datetime.now(),
            "hypotheses_processed": len(current_hypotheses),
            "new_hypotheses_generated": len(new_hypotheses),
            "documents_discovered": len(discovered_documents),
            "target_concept": target_concept
        }
        self.iteration_history.append(iteration_record)
        
        logger.info(f"Iteration {iteration_record['iteration']}: "
                   f"{iteration_record['new_hypotheses_generated']} new hypotheses, "
                   f"{iteration_record['documents_discovered']} documents discovered")
        
        return new_hypotheses
    
    def run_fuzzing_campaign(self, 
                           target_concept: str,
                           initial_phrases: List[str]) -> Dict[str, Any]:
        """Run a complete fuzzing campaign.
        
        Args:
            target_concept: Target concept to move towards
            initial_phrases: Initial seed phrases
            
        Returns:
            Campaign results
        """
        logger.info(f"Starting fuzzing campaign for target concept: {target_concept}")
        
        # Generate initial hypotheses
        current_hypotheses = self.generate_initial_hypotheses(target_concept, initial_phrases)
        all_hypotheses = current_hypotheses.copy()
        
        # Run iterations
        for iteration in range(self.max_iterations):
            if not current_hypotheses:
                logger.info("No more hypotheses to process, stopping")
                break
                
            # Run one iteration
            new_hypotheses = self.run_fuzzing_iteration(target_concept, current_hypotheses)
            
            # Update for next iteration
            current_hypotheses = new_hypotheses
            all_hypotheses.extend(new_hypotheses)
            
            # Check if we've reached target similarity
            # (This would need to be implemented based on your similarity metrics)
            
        # Compile results
        results = {
            "target_concept": target_concept,
            "total_hypotheses": len(all_hypotheses),
            "total_iterations": len(self.iteration_history),
            "iteration_history": self.iteration_history,
            "final_hypotheses": [h.id for h in all_hypotheses],
            "pdg_stats": self.pdg.get_graph_stats(),
            "hqg_stats": {
                "total_hypotheses": len(self.hqg.nodes),
                "total_retrieval_edges": sum(len(edges) for edges in self.hqg.retrieval_edges.values())
            }
        }
        
        logger.info(f"Fuzzing campaign completed: {results['total_hypotheses']} hypotheses generated")
        return results
    
    def export_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Export fuzzing results to JSON.
        
        Args:
            results: Results from fuzzing campaign
            output_path: Path to save results
        """
        # Convert datetime objects to strings for JSON serialization
        exportable_results = json.loads(
            json.dumps(results, default=str)
        )
        
        with open(output_path, 'w') as f:
            json.dump(exportable_results, f, indent=2)
            
        logger.info(f"Results exported to {output_path}")


def create_two_layer_fuzzer(faiss_index_path: str,
                          embedding_model: str = "all-MiniLM-L6-v2",
                          k_neighbors: int = 10,
                          llm_config: Optional[HypothesisGenerationConfig] = None) -> TwoLayerFuzzer:
    """Create a two-layer fuzzer from a FAISS index.
    
    Args:
        faiss_index_path: Path to FAISS index file
        embedding_model: Embedding model to use
        k_neighbors: Number of k-NN neighbors for PDG
        llm_config: Optional LLM configuration for hypothesis generation
        
    Returns:
        Configured TwoLayerFuzzer instance
    """
    # Load FAISS index
    faiss_index = FAISSIndex.load(faiss_index_path)
    
    # Create PDG
    pdg = PublicDocumentGraph(faiss_index, k_neighbors)
    
    # Create HQG
    hqg = HypothesisQueryGraph(pdg)
    
    # Create hypothesis generator if LLM config provided
    hypothesis_generator = None
    if llm_config:
        from .llm_hypothesis_generator import AdaptiveHypothesisGenerator
        hypothesis_generator = AdaptiveHypothesisGenerator(llm_config)
    
    # Create fuzzer
    fuzzer = TwoLayerFuzzer(pdg, hqg, embedding_model, hypothesis_generator=hypothesis_generator)
    
    return fuzzer


def run_fuzzing_example():
    """Example usage of the two-layer fuzzer."""
    # This would be used with a real FAISS index
    # fuzzer = create_two_layer_fuzzer("path/to/public_corpus.index")
    
    # Example campaign
    # results = fuzzer.run_fuzzing_campaign(
    #     target_concept="confidential information",
    #     initial_phrases=["data breach", "security incident", "leaked documents"]
    # )
    
    # fuzzer.export_results(results, "fuzzing_results.json")
    pass
