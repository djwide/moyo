"""Barrier analyzer for comparing public and private FAISS indexes.

This module implements distance-based analysis between private and public corpora
to detect potential information leakage or barrier breaches.

Key principle: For each private phrase, search its nearest public neighbor.
This yields the closest "leak candidates" - private information that may be
too similar to publicly available information.

Important: Public and private corpora MUST be embedded at the same granularity
(same chunk_size, chunk_overlap, and embedding_model) for valid distance comparisons.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from pathlib import Path
from dataclasses import dataclass, field
import json

from .schema import BarrierProbeConfig, BarrierProbeResult
from .distribution import (
    DEFAULT_NEIGHBORHOOD_K,
    DistributionLayer,
    attach_neighborhood,
    build_distribution_layer,
)
from .public_index_builder import load_public_index
from moyo.privateside.mapcorpus.builder import CorpusBuilder
from shared_utils import generate_id, embed
from shared_utils.index_spec import (
    GRANULARITY_MULTI,
    GRANULARITY_PHRASES,
    compare_index_specs,
    load_index_spec,
    spec_from_config,
)

logger = logging.getLogger(__name__)


@dataclass
class NearestNeighborResult:
    """Result of finding nearest public neighbor for a private phrase."""
    private_index: int
    private_chunk_id: str
    private_content: str
    public_index: int
    public_chunk_id: str
    public_content: str
    distance: float
    private_metadata: Dict[str, Any] = field(default_factory=dict)
    public_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistanceDistribution:
    """Statistics about the distribution of nearest-neighbor distances."""
    min_distance: float
    max_distance: float
    mean_distance: float
    median_distance: float
    std_distance: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    percentile_99: float
    total_private_phrases: int


class BarrierAnalyzer:
    """Analyzer for comparing public and private information barriers.
    
    This analyzer implements the core distance calculation strategy:
    - For each private phrase p_i, search its nearest public neighbor
    - Track the global minimum pair (closest "leak candidate")
    - Compute top N closest pairs and per-private nearest distances distribution
    
    Important: Both corpora must use the same embedding granularity (chunk_size,
    chunk_overlap, embedding_model) for valid distance comparisons.
    """
    
    def __init__(self, config: BarrierProbeConfig):
        """Initialize the barrier analyzer.
        
        Args:
            config: Barrier probe configuration
        """
        self.config = config
        self.public_builder: Optional[Any] = None
        self.private_builder: Optional[CorpusBuilder] = None
        # Cache for nearest neighbor results
        self._nn_cache: Optional[List[NearestNeighborResult]] = None
        self._spec_errors: List[str] = []
        self._distance_matrix: Optional[np.ndarray] = None
        self._private_embeddings: Optional[np.ndarray] = None
        self._public_embeddings: Optional[np.ndarray] = None
        self._private_chunks: Optional[List[Any]] = None
        self._public_chunks: Optional[List[Any]] = None

    def _filter_results(self, matches: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Filter matches based on similarity threshold and return top results.

        Args:
            matches: List of match dictionaries including a ``distance`` field.
            top_k: Maximum number of results to return after filtering.

        Returns:
            Filtered list of matches sorted by distance.
        """
        if not matches:
            return []

        filtered = [m for m in matches if m.get("distance") is not None and m["distance"] <= self.config.similarity_threshold]
        filtered.sort(key=lambda x: x["distance"])
        return filtered[:top_k]
        
    def load_indexes(self) -> bool:
        """Load both public and private indexes.
        
        Returns:
            True if both indexes loaded successfully
        """
        try:
            # Load public index
            logger.info(f"Loading public index from {self.config.public_index_path}")
            self.public_builder = load_public_index(self.config.public_index_path)
            if not self.public_builder:
                logger.error(f"Failed to load public index from {self.config.public_index_path}")
                return False
            
            # Load private index
            logger.info(f"Loading private index from {self.config.private_index_path}")
            self.private_builder = CorpusBuilder.load_from_path(self.config.private_index_path)
            if not self.private_builder:
                logger.error(f"Failed to load private index from {self.config.private_index_path}")
                return False
            
            # Verify embedding granularity consistency
            errors = self._verify_granularity_consistency()
            if errors:
                self._spec_errors = errors
                for msg in errors:
                    logger.error(msg)
                return False
            
            logger.info("Both indexes loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            return False
    
    def _verify_granularity_consistency(self) -> List[str]:
        """Require matching embedding model / normalization (and chunk packing).

        Returns a list of hard-error messages. Empty means the pair is comparable.
        """
        public_spec = None
        private_spec = None
        if self.config.public_index_path:
            public_spec = load_index_spec(self.config.public_index_path)
        if self.config.private_index_path:
            private_spec = load_index_spec(self.config.private_index_path)

        if public_spec is None and self.public_builder is not None:
            public_spec = spec_from_config(
                getattr(self.public_builder, "config", None) or {},
                granularity=GRANULARITY_MULTI,
            )
        if private_spec is None and self.private_builder is not None:
            cfg = getattr(self.private_builder, "config", None)
            granularity = GRANULARITY_MULTI
            if cfg is not None:
                granularity = getattr(cfg, "granularity", GRANULARITY_MULTI) or GRANULARITY_MULTI
            private_spec = spec_from_config(cfg or {}, granularity=granularity)

        if public_spec is None or private_spec is None:
            logger.warning("Could not verify granularity consistency: spec not available")
            return []

        errors = compare_index_specs(private_spec, public_spec)
        if (
            private_spec.granularity == GRANULARITY_PHRASES
            and public_spec.granularity == GRANULARITY_MULTI
        ):
            logger.info(
                "Private index is phrase-level; public multi-granularity sentence/item "
                "chunks are the matching unit. Embedding model and L2-normalization "
                "must still agree."
            )
        return errors
    
    def calculate_cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine distance (1 - cosine similarity)
        """
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        vec1_normalized = vec1 / norm1
        vec2_normalized = vec2 / norm2
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(vec1_normalized, vec2_normalized)
        
        # Return cosine distance (1 - similarity)
        return 1.0 - cosine_similarity

    def _ensure_matrices(self) -> bool:
        """Load embeddings and the private×public cosine-distance matrix once."""
        if self._distance_matrix is not None:
            return True
        if not self.public_builder or not self.private_builder:
            logger.error("Indexes not loaded")
            return False

        public_embeddings, public_chunks = self._vectors_and_chunks(
            self.public_builder, side="public"
        )
        private_embeddings, private_chunks = self._vectors_and_chunks(
            self.private_builder, side="private"
        )
        if not public_embeddings or not private_embeddings:
            logger.warning("No embeddings found in one or both indexes")
            return False

        public_matrix = np.vstack(public_embeddings)
        private_matrix = np.vstack(private_embeddings)
        public_norms = np.linalg.norm(public_matrix, axis=1, keepdims=True)
        private_norms = np.linalg.norm(private_matrix, axis=1, keepdims=True)
        public_normalized = public_matrix / np.where(public_norms == 0, 1, public_norms)
        private_normalized = private_matrix / np.where(private_norms == 0, 1, private_norms)
        similarity_matrix = np.dot(private_normalized, public_normalized.T)

        self._public_embeddings = public_normalized
        self._private_embeddings = private_normalized
        self._public_chunks = public_chunks
        self._private_chunks = private_chunks
        self._distance_matrix = 1.0 - similarity_matrix
        return True
    
    def find_nearest_public_neighbors(self, top_k: int = 1) -> List[NearestNeighborResult]:
        """For each private phrase, find its nearest public neighbor(s).
        
        This implements the core distance calculation:
        - For each private vector p_i, search top k in Public
        - Keep the closest (i, j*, dist) for each private phrase
        
        Args:
            top_k: Number of nearest public neighbors to find for each private phrase
            
        Returns:
            List of NearestNeighborResult, one per private phrase (with top_k=1)
            or multiple per private phrase (with top_k>1)
        """
        if not self._ensure_matrices():
            return []

        logger.info(f"Finding nearest public neighbors for each private phrase (top_k={top_k})...")
        public_chunks = self._public_chunks or []
        private_chunks = self._private_chunks or []
        distance_matrix = self._distance_matrix
        if distance_matrix is None:
            return []

        results = []
        
        # For each private phrase, find its top_k nearest public neighbors
        for i, priv_chunk in enumerate(private_chunks):
            distances_to_public = distance_matrix[i]
            
            # Get indices of top_k nearest (smallest distance)
            if top_k >= len(public_chunks):
                nearest_indices = np.argsort(distances_to_public)
            else:
                nearest_indices = np.argpartition(distances_to_public, top_k)[:top_k]
                nearest_indices = nearest_indices[np.argsort(distances_to_public[nearest_indices])]
            
            for j in nearest_indices[:top_k]:
                pub_chunk = public_chunks[j]
                
                # Get content (handle different attribute names)
                private_content = getattr(priv_chunk, 'text', '') or getattr(priv_chunk, 'content', '')
                public_content = getattr(pub_chunk, 'content', '') or getattr(pub_chunk, 'text', '')
                
                result = NearestNeighborResult(
                    private_index=i,
                    private_chunk_id=priv_chunk.id,
                    private_content=private_content,
                    public_index=int(j),
                    public_chunk_id=pub_chunk.id,
                    public_content=public_content,
                    distance=float(distances_to_public[j]),
                    private_metadata=getattr(priv_chunk, 'metadata', {}),
                    public_metadata=getattr(pub_chunk, 'metadata', {})
                )
                results.append(result)
        
        # Cache results for the primary (top_k=1) case
        if top_k == 1:
            self._nn_cache = results
        
        logger.info(f"Found {len(results)} nearest neighbor pairs")
        return results

    def _vectors_and_chunks(self, builder: Any, side: str):
        """Load embeddings from chunk records, or reconstruct from FAISS."""
        embeddings = []
        chunks = []
        stored = getattr(builder, "chunks", None) or []
        for chunk in stored:
            vector = getattr(chunk, "embedding", None)
            if vector:
                embeddings.append(np.array(vector, dtype=np.float32))
                chunks.append(chunk)

        if embeddings:
            return embeddings, chunks

        index = getattr(builder, "faiss_index", None) or getattr(builder, "index", None)
        if index is None or getattr(index, "index", None) is None:
            logger.warning("No embeddings found for %s index", side)
            return [], []
        ntotal = index.index.ntotal
        if ntotal <= 0:
            return [], []
        try:
            matrix = index.index.reconstruct_n(0, ntotal)
        except Exception as exc:
            logger.warning("Could not reconstruct %s FAISS vectors: %s", side, exc)
            return [], []

        metadata = getattr(index, "metadata", None) or []
        string_store = getattr(index, "string_store", None)
        reconstructed = []
        for i in range(ntotal):
            meta = metadata[i] if i < len(metadata) else {}
            text = meta.get("text") or meta.get("text_preview") or ""
            if string_store is not None and meta.get("text_id") is not None:
                stored_text = string_store.get(meta["text_id"])
                if stored_text:
                    text = stored_text
            reconstructed.append(
                type(
                    "Chunk",
                    (),
                    {
                        "id": meta.get("id") or meta.get("chunk_id") or f"{side}_{i}",
                        "text": text,
                        "content": text,
                        "metadata": meta,
                        "source_type": meta.get("source_type"),
                        "source_document": meta.get("source")
                        or meta.get("source_document")
                        or "",
                    },
                )()
            )
        return [np.asarray(row, dtype=np.float32) for row in matrix], reconstructed

    def calibrate_threshold(self, profile: str = "balanced"):
        """Recommend a cosine-distance cutoff from unlabeled nearest neighbors."""
        from .calibrate import calibrate_from_distances

        if not self.public_builder or not self.private_builder:
            if not self.load_indexes():
                raise RuntimeError(
                    "Failed to load indexes"
                    + (": " + "; ".join(self._spec_errors) if self._spec_errors else "")
                )
        nn = self.find_nearest_public_neighbors(top_k=1)
        if not nn:
            raise RuntimeError("No nearest-neighbor distances available to calibrate")
        model = None
        pub_cfg = getattr(self.public_builder, "config", None)
        if pub_cfg is not None:
            model = getattr(pub_cfg, "embedding_model", None)
        return calibrate_from_distances(
            [r.distance for r in nn],
            profile=profile,
            embedding_model=model,
        )
    
    def find_closest_matches(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find the closest matches between private and public chunks.
        
        For each private phrase, finds its nearest public neighbor, then returns
        the top_k pairs with smallest distances (the closest "leak candidates").
        
        This implements:
        - For each private vector p_i: search top 1 in Public, keep (i, j*, dist)
        - Update global minimum (or keep top N pairs)
        
        Args:
            top_k: Number of closest matches to return (global top N pairs)
            
        Returns:
            List of closest matches with distances, sorted by distance
        """
        # Get all nearest neighbor pairs (one per private phrase)
        if self._nn_cache is not None:
            nn_results = self._nn_cache
        else:
            nn_results = self.find_nearest_public_neighbors(top_k=1)
        
        if not nn_results:
            return []
        
        # Sort by distance to get global top N closest pairs
        sorted_results = sorted(nn_results, key=lambda x: x.distance)
        
        results = []
        for i, match in enumerate(sorted_results[:top_k]):
            # Get source_type if available
            public_source_type = "unknown"
            if hasattr(self.public_builder, 'chunks'):
                for chunk in self.public_builder.chunks:
                    if chunk.id == match.public_chunk_id:
                        public_source_type = getattr(chunk.source_type, 'value', str(chunk.source_type)) if hasattr(chunk, 'source_type') else "unknown"
                        break
            
            result = {
                'rank': i + 1,
                'distance': match.distance,
                'public_chunk_id': match.public_chunk_id,
                'public_content': match.public_content[:200] + "..." if len(match.public_content) > 200 else match.public_content,
                'public_source_type': public_source_type,
                'public_metadata': match.public_metadata,
                'private_chunk_id': match.private_chunk_id,
                'private_content': match.private_content[:200] + "..." if len(match.private_content) > 200 else match.private_content,
                'private_metadata': match.private_metadata,
                'private_index': match.private_index,
                'public_index': match.public_index
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} closest matches (leak candidates)")
        return results
    
    def get_distance_distribution(self) -> Optional[DistanceDistribution]:
        """Get the distribution of per-private nearest distances.
        
        This computes statistics about how close each private phrase is to its
        nearest public neighbor, useful for understanding overall barrier integrity.
        
        Returns:
            DistanceDistribution with statistics, or None if no data
        """
        # Get all nearest neighbor pairs
        if self._nn_cache is not None:
            nn_results = self._nn_cache
        else:
            nn_results = self.find_nearest_public_neighbors(top_k=1)
        
        if not nn_results:
            return None
        
        distances = np.array([r.distance for r in nn_results])
        
        return DistanceDistribution(
            min_distance=float(np.min(distances)),
            max_distance=float(np.max(distances)),
            mean_distance=float(np.mean(distances)),
            median_distance=float(np.median(distances)),
            std_distance=float(np.std(distances)),
            percentile_25=float(np.percentile(distances, 25)),
            percentile_75=float(np.percentile(distances, 75)),
            percentile_90=float(np.percentile(distances, 90)),
            percentile_95=float(np.percentile(distances, 95)),
            percentile_99=float(np.percentile(distances, 99)),
            total_private_phrases=len(nn_results)
        )
    
    def get_global_minimum_pair(self) -> Optional[NearestNeighborResult]:
        """Get the single closest private-public pair (the closest "leak candidate").
        
        Returns:
            The NearestNeighborResult with the smallest distance, or None
        """
        if self._nn_cache is not None:
            nn_results = self._nn_cache
        else:
            nn_results = self.find_nearest_public_neighbors(top_k=1)
        
        if not nn_results:
            return None
        
        return min(nn_results, key=lambda x: x.distance)
    
    def get_top_n_closest_pairs(self, n: int = 10) -> Tuple[List[NearestNeighborResult], DistanceDistribution]:
        """Get top N closest pairs along with the full distance distribution.
        
        This is the main analysis method that yields:
        - Top N closest pairs (the closest "leak candidates")
        - Per-private nearest distances distribution
        
        Args:
            n: Number of top pairs to return
            
        Returns:
            Tuple of (top_n_pairs, distance_distribution)
        """
        # Get all nearest neighbor pairs
        nn_results = self.find_nearest_public_neighbors(top_k=1)
        
        if not nn_results:
            return [], None
        
        # Get top N
        sorted_results = sorted(nn_results, key=lambda x: x.distance)
        top_n = sorted_results[:n]
        
        # Get distribution
        distribution = self.get_distance_distribution()
        
        return top_n, distribution

    def search_phrase(self, phrase: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Search indexes for chunks similar to the provided phrase.

        Args:
            phrase: Text to embed and compare against indexes.
            top_k: Number of results to return for each index.

        Returns:
            Dictionary with ``public`` and ``private`` match lists.
        """
        if not self.public_builder or not self.private_builder:
            logger.error("Indexes not loaded")
            return {}

        try:
            query_emb = embed([phrase])[0]
        except Exception as e:
            logger.error(f"Failed to embed phrase: {e}")
            return {}

        results = {"public": [], "private": []}

        for chunk in self.public_builder.chunks:
            if chunk.embedding:
                dist = self.calculate_cosine_distance(query_emb, chunk.embedding)
                results["public"].append({
                    "distance": dist,
                    "chunk_id": chunk.id,
                    "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "source_type": chunk.source_type.value,
                    "metadata": chunk.metadata,
                })

        for chunk in self.private_builder.chunks:
            if chunk.embedding:
                dist = self.calculate_cosine_distance(query_emb, chunk.embedding)
                results["private"].append({
                    "distance": dist,
                    "chunk_id": chunk.id,
                    "content": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "metadata": chunk.metadata,
                })

        results["public"].sort(key=lambda x: x["distance"])
        results["private"].sort(key=lambda x: x["distance"])

        results["public"] = results["public"][:top_k]
        results["private"] = results["private"][:top_k]

        return results

    def score_distribution_layer(self) -> DistributionLayer:
        """JS occupancy, directional KL, and per-private margin / top-k entropy."""
        if not self._ensure_matrices():
            return DistributionLayer(
                semantic_separation=None,
                pairwise_exposure="None",
                concentrated_matches=0,
            )
        k = getattr(self.config, "neighborhood_k", None) or DEFAULT_NEIGHBORHOOD_K
        temperature = getattr(self.config, "softmax_temperature", None)
        if temperature is None:
            temperature = 0.10
        return build_distribution_layer(
            self._distance_matrix,
            self._private_embeddings,
            self._public_embeddings,
            neighborhood_k=int(k),
            temperature=float(temperature),
            n_clusters=getattr(self.config, "cluster_count", None),
        )

    def analyze_barriers(self, top_k: int = 10) -> BarrierProbeResult:
        """Perform comprehensive barrier analysis.
        
        This analysis:
        1. For each private phrase, finds its nearest public neighbor
        2. Identifies the global minimum pair (closest "leak candidate")
        3. Returns top N closest pairs + per-private nearest distances distribution
        
        Args:
            top_k: Number of top results to return for each analysis
            
        Returns:
            BarrierProbeResult with analysis results
        """
        start_time = time.time()
        
        # Load indexes
        if not self.load_indexes():
            detail = "; ".join(self._spec_errors) if self._spec_errors else "Failed to load indexes"
            if self._spec_errors:
                raise RuntimeError(detail)
            return BarrierProbeResult(
                probe_id=generate_id("probe"),
                public_index_info={},
                private_index_info={},
                similarity_threshold=self.config.similarity_threshold,
                recommendations=[detail],
                metadata={"error": detail},
            )
        
        # Get index information with granularity details
        public_config = getattr(self.public_builder, 'config', None)
        private_config = getattr(self.private_builder, 'config', None)
        
        public_index_info = {
            'chunk_count': len(self.public_builder.chunks),
            'source_types': list(set(
                getattr(chunk.source_type, 'value', str(chunk.source_type)) 
                for chunk in self.public_builder.chunks 
                if hasattr(chunk, 'source_type')
            )),
            'organizations': list(set(
                chunk.metadata.get('source_organization') 
                for chunk in self.public_builder.chunks 
                if chunk.metadata.get('source_organization')
            )),
            'granularity': {
                'chunk_size': getattr(public_config, 'chunk_size', None),
                'chunk_overlap': getattr(public_config, 'chunk_overlap', None),
                'embedding_model': getattr(public_config, 'embedding_model', None)
            } if public_config else {}
        }
        
        private_index_info = {
            'chunk_count': len(self.private_builder.chunks),
            'document_count': len(set(chunk.source_document for chunk in self.private_builder.chunks)),
            'granularity': {
                'chunk_size': getattr(private_config, 'chunk_size', None),
                'chunk_overlap': getattr(private_config, 'chunk_overlap', None),
                'embedding_model': getattr(private_config, 'embedding_model', None)
            } if private_config else {}
        }
        
        # Get top N closest pairs and distance distribution
        # This searches: for each private phrase → find nearest public neighbor
        top_pairs, distance_dist = self.get_top_n_closest_pairs(top_k * 5)
        
        # Get global minimum pair
        global_min = self.get_global_minimum_pair()
        
        # Filter to matches within threshold
        raw_matches = self.find_closest_matches(top_k * 5)
        filtered_matches = self._filter_results(raw_matches, top_k)
        layer = self.score_distribution_layer()
        by_private = layer.by_private_index()

        def _neighborhood_for(match: Dict[str, Any]):
            idx = match.get("private_index")
            if idx is None:
                return None
            return by_private.get(int(idx))

        raw_matches = [attach_neighborhood(m, _neighborhood_for(m)) for m in raw_matches]
        filtered_matches = [
            attach_neighborhood(m, _neighborhood_for(m)) for m in filtered_matches
        ]

        # Identify potential breaches
        potential_breaches = []
        breach_count = len(filtered_matches)
        high_risk = 0
        medium_risk = 0
        low_risk = 0

        for match in filtered_matches:
            # Determine risk level based on distance
            if match['distance'] <= 0.1:
                risk_level = "high"
                high_risk += 1
            elif match['distance'] <= 0.3:
                risk_level = "medium"
                medium_risk += 1
            else:
                risk_level = "low"
                low_risk += 1

            breach_info = attach_neighborhood(
                {
                    'rank': match.get('rank'),
                    'type': 'similarity_breach',
                    'risk_level': risk_level,
                    'distance': match['distance'],
                    'public_chunk_id': match['public_chunk_id'],
                    'private_chunk_id': match['private_chunk_id'],
                    'public_content': match['public_content'],
                    'private_content': match['private_content'],
                    'public_metadata': match.get('public_metadata', {}),
                    'private_metadata': match.get('private_metadata', {}),
                    'public_source_type': match.get('public_source_type'),
                    'private_index': match.get('private_index'),
                },
                _neighborhood_for(match),
            )
            potential_breaches.append(breach_info)
        
        # Generate recommendations based on distance distribution
        recommendations = list(layer.headline_lines())
        
        if breach_count > 0:
            recommendations.append(f"Found {breach_count} potential information barrier breaches")
            recommendations.append(f"High risk breaches: {high_risk}, Medium risk: {medium_risk}, Low risk: {low_risk}")
            
            if high_risk > 0:
                recommendations.append("Immediate action required: Review high-risk breaches")
            
            if medium_risk > 0:
                recommendations.append("Review medium-risk breaches and consider additional controls")
        else:
            recommendations.append("No potential breaches detected above threshold")
        
        # Add distribution-based insights
        if distance_dist:
            recommendations.append(f"Distance distribution: min={distance_dist.min_distance:.4f}, "
                                 f"median={distance_dist.median_distance:.4f}, "
                                 f"max={distance_dist.max_distance:.4f}")
            recommendations.append(f"95th percentile distance: {distance_dist.percentile_95:.4f}")
            
            # Warn about overall barrier strength
            if distance_dist.median_distance < 0.3:
                recommendations.append("WARNING: Median distance is low - private corpus may have significant overlap with public sources")
            elif distance_dist.median_distance < 0.5:
                recommendations.append("CAUTION: Moderate overlap detected between private and public corpora")
        
        if global_min:
            recommendations.append(f"Closest leak candidate: distance={global_min.distance:.4f}")
        
        processing_time = time.time() - start_time
        
        # Prepare distance distribution for metadata
        dist_dict = None
        if distance_dist:
            dist_dict = {
                'min_distance': distance_dist.min_distance,
                'max_distance': distance_dist.max_distance,
                'mean_distance': distance_dist.mean_distance,
                'median_distance': distance_dist.median_distance,
                'std_distance': distance_dist.std_distance,
                'percentile_25': distance_dist.percentile_25,
                'percentile_75': distance_dist.percentile_75,
                'percentile_90': distance_dist.percentile_90,
                'percentile_95': distance_dist.percentile_95,
                'percentile_99': distance_dist.percentile_99,
                'total_private_phrases': distance_dist.total_private_phrases
            }
        
        return BarrierProbeResult(
            probe_id=generate_id("probe"),
            public_index_info=public_index_info,
            private_index_info=private_index_info,
            similarity_threshold=self.config.similarity_threshold,
            potential_breaches=potential_breaches,
            breach_count=breach_count,
            high_risk_breaches=high_risk,
            medium_risk_breaches=medium_risk,
            low_risk_breaches=low_risk,
            processing_time=processing_time,
            recommendations=recommendations,
            semantic_separation=layer.semantic_separation,
            pairwise_exposure=layer.pairwise_exposure,
            concentrated_matches=layer.concentrated_matches,
            distribution_diagnostics=layer.diagnostics(),
            metadata={
                'closest_matches': raw_matches,
                'top_breaches': filtered_matches,
                'total_comparisons': len(self._nn_cache) if self._nn_cache else 0,
                'global_minimum_pair': {
                    'private_chunk_id': global_min.private_chunk_id,
                    'public_chunk_id': global_min.public_chunk_id,
                    'distance': global_min.distance,
                    'private_content': global_min.private_content[:200] + "..." if len(global_min.private_content) > 200 else global_min.private_content,
                    'public_content': global_min.public_content[:200] + "..." if len(global_min.public_content) > 200 else global_min.public_content
                } if global_min else None,
                'distance_distribution': dist_dict,
                'distribution_layer': layer.to_dict(),
            }
        )


def analyze_barriers(
    public_index_path: str,
    private_index_path: str,
    similarity_threshold: float = 0.8,
    top_k: int = 10
) -> BarrierProbeResult:
    """Convenience function to analyze information barriers.
    
    Args:
        public_index_path: Path to public index
        private_index_path: Path to private index
        similarity_threshold: Threshold for breach detection
        top_k: Number of top results to return
        
    Returns:
        BarrierProbeResult with analysis results
    """
    config = BarrierProbeConfig(
        public_index_path=public_index_path,
        private_index_path=private_index_path,
        similarity_threshold=similarity_threshold
    )
    
    analyzer = BarrierAnalyzer(config)
    return analyzer.analyze_barriers(top_k)
