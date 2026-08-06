"""Unified Fuzzing Engine for Barrier Analysis.

This module combines the two-layer index strategy with text and idea fuzzing capabilities
into a single, comprehensive fuzzing system for barrier analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import time

from shared_utils import (
    embed,
    FAISSIndex, 
    normalize_text,
    TextNormalizationConfig,
    get_logger
)

logger = get_logger(__name__)


@dataclass
class FuzzingConfig:
    """Unified configuration for the fuzzing engine."""
    
    # Core Configuration — local Ollama LLM for generation + vectors
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "ollama"  # "ollama", "openai", "anthropic", "local"
    model_name: str = "llama3.1:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = "http://localhost:11434"
    
    # Two-Layer Configuration
    k_neighbors: int = 10
    max_iterations: int = 5
    
    # Fuzzing Configuration
    similarity_threshold: float = 0.8
    target_similarity: float = 0.95
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Scalar Fuzzing Levels
    fuzzing_levels: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Transformation Settings
    max_hypotheses_per_document: int = 3
    min_query_length: int = 10
    max_query_length: int = 200


@dataclass
class DocumentNode:
    """A document node in the unified system."""
    id: str
    content: str
    source: str
    source_type: str = "public"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
            self.id = f"doc_{content_hash}"


@dataclass
class HypothesisNode:
    """A hypothesis node for fuzzing."""
    id: str
    query: str
    target_concept: str
    generation_method: str
    fuzzing_level: float = 0.5
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            import hashlib
            query_hash = hashlib.md5(self.query.encode()).hexdigest()[:12]
            self.id = f"hyp_{query_hash}"


@dataclass
class FuzzingResult:
    """Result of a fuzzing operation."""
    original_text: str
    fuzzed_text: str
    fuzzing_level: float
    similarity_score: float
    transformation_method: str
    target_concept: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


@dataclass
class FuzzingCampaignResult:
    """Result of a complete fuzzing campaign."""
    target_concept: str
    total_phrases: int
    total_results: int
    fuzzing_levels: List[float]
    results_by_level: Dict[float, List[FuzzingResult]]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedTransformationEngine:
    """Unified transformation engine combining all fuzzing capabilities."""
    
    def __init__(self, config: FuzzingConfig):
        """Initialize the transformation engine.
        
        Args:
            config: Fuzzing configuration
        """
        self.config = config
        self.normalization_config = TextNormalizationConfig(
            lowercase=True,
            normalize_unicode=True,
            normalize_whitespace=True,
            remove_urls=True,
            remove_emails=True,
            normalize_punctuation=True
        )
        self.transformation_patterns = self._load_transformation_patterns()
    
    def _load_transformation_patterns(self) -> Dict[str, List[str]]:
        """Load comprehensive transformation patterns."""
        # Load synonym map directly from data directory
        synonyms = {}
        try:
            import json
            from pathlib import Path
            synonym_file = Path(__file__).resolve().parents[3] / "data" / "synonym_map.json"
            if synonym_file.exists():
                with open(synonym_file, 'r', encoding='utf-8') as f:
                    synonyms = json.load(f)
                logger.info(f"Loaded {len(synonyms)} synonym groups from {synonym_file}")
            else:
                logger.warning(f"Synonym map file not found at {synonym_file}")
        except Exception as e:
            logger.warning(f"Failed to load synonym map: {e}")
            synonyms = {}
        
        # Merge with our built-in patterns
        built_in_synonyms = {
            "sensitive": ["confidential", "private", "restricted", "classified", "proprietary"],
            "data": ["information", "content", "material", "details", "records"],
            "exposure": ["disclosure", "leak", "release", "publication", "reveal"],
            "breach": ["violation", "compromise", "incident", "failure", "infraction"],
            "security": ["protection", "safety", "defense", "safeguard", "shield"],
            "system": ["platform", "infrastructure", "framework", "architecture", "network"],
            "analysis": ["examination", "evaluation", "assessment", "review", "study"],
            "research": ["investigation", "study", "exploration", "inquiry", "analysis"],
            "development": ["creation", "construction", "building", "establishment", "formation"],
            "implementation": ["deployment", "execution", "application", "integration", "rollout"],
            "confidential": ["private", "restricted", "classified", "sensitive", "proprietary"],
            "information": ["data", "content", "material", "details", "intelligence"],
            "disclosure": ["exposure", "leak", "release", "publication", "revelation"],
            "incident": ["event", "occurrence", "situation", "case", "episode"],
            "vulnerability": ["weakness", "flaw", "gap", "exposure", "risk"],
            "threat": ["danger", "risk", "hazard", "peril", "menace"],
            "attack": ["assault", "strike", "offensive", "raid", "intrusion"],
            "defense": ["protection", "safeguard", "shield", "barrier", "fortification"]
        }
        
        # Merge shared synonyms with built-in patterns
        merged_synonyms = {**synonyms, **built_in_synonyms}
        
        built_in_patterns = {
            "synonyms": merged_synonyms,
            "intensifiers": {
                "major": ["significant", "substantial", "considerable", "notable", "important"],
                "critical": ["essential", "vital", "crucial", "important", "key"],
                "serious": ["severe", "grave", "major", "significant", "substantial"],
                "important": ["critical", "essential", "vital", "crucial", "key"],
                "significant": ["substantial", "major", "considerable", "notable", "important"],
                "severe": ["serious", "grave", "critical", "acute", "intense"],
                "extensive": ["comprehensive", "thorough", "complete", "detailed", "wide-ranging"],
                "comprehensive": ["extensive", "thorough", "complete", "detailed", "all-encompassing"]
            },
            "technical_terms": {
                "neural network": ["deep learning model", "artificial neural network", "ANN", "neural architecture"],
                "machine learning": ["ML", "artificial intelligence", "AI", "automated learning"],
                "algorithm": ["method", "technique", "procedure", "approach", "process"],
                "optimization": ["improvement", "enhancement", "refinement", "tuning", "adjustment"],
                "performance": ["efficiency", "effectiveness", "capability", "functionality", "output"],
                "database": ["data repository", "information store", "data warehouse", "data system"],
                "encryption": ["cryptographic protection", "data encoding", "security encoding", "cipher"],
                "authentication": ["identity verification", "access control", "user validation", "login security"],
                "authorization": ["access permission", "user rights", "access control", "privilege management"]
            },
            "context_modifiers": {
                "internal": ["private", "confidential", "restricted", "proprietary", "sensitive"],
                "external": ["public", "open", "accessible", "available", "shared"],
                "unauthorized": ["illegal", "prohibited", "forbidden", "restricted", "blocked"],
                "authorized": ["permitted", "allowed", "approved", "sanctioned", "legitimate"],
                "accidental": ["unintentional", "inadvertent", "unplanned", "unexpected", "mistaken"],
                "intentional": ["deliberate", "purposeful", "planned", "calculated", "conscious"]
            }
        }
    
    def transform_text(self, text: str, target_concept: str, fuzzing_level: float, 
                      context_documents: Optional[List[DocumentNode]] = None) -> str:
        """Transform text towards target concept with specified fuzzing level.
        
        Args:
            text: Original text to transform
            target_concept: Target concept to move towards
            fuzzing_level: Level of fuzzing (0.0 to 1.0)
            context_documents: Optional context documents for transformation
            
        Returns:
            Transformed text
        """
        start_time = time.time()
        
        try:
            # Normalize input text
            normalized_text = normalize_text(text, self.normalization_config)
            
            # Get embeddings for similarity calculation
            original_emb = embed([normalized_text], self.config.embedding_model)[0]
            target_emb = embed([target_concept], self.config.embedding_model)[0]
            
            # Calculate current similarity
            similarity = sum(a * b for a, b in zip(original_emb, target_emb))
            
            # Apply transformations based on fuzzing level
            transformed_text = normalized_text
            
            # Level 1: Basic synonym replacement (0.1-0.3)
            if fuzzing_level >= 0.1:
                transformed_text = self._apply_synonym_replacement(transformed_text, fuzzing_level)
            
            # Level 2: Intensifier adjustment (0.3-0.5)
            if fuzzing_level >= 0.3:
                transformed_text = self._apply_intensifier_adjustment(transformed_text, target_concept, fuzzing_level)
            
            # Level 3: Technical term alignment (0.5-0.7)
            if fuzzing_level >= 0.5:
                transformed_text = self._apply_technical_term_alignment(transformed_text, target_concept, fuzzing_level)
            
            # Level 4: Context-aware transformation (0.7-0.9)
            if fuzzing_level >= 0.7 and context_documents:
                transformed_text = self._apply_context_aware_transformation(transformed_text, target_concept, context_documents, fuzzing_level)
            
            # Level 5: Advanced semantic transformation (0.9+)
            if fuzzing_level >= 0.9:
                transformed_text = self._apply_advanced_semantic_transformation(transformed_text, target_concept, fuzzing_level)
            
            return transformed_text
            
        except Exception as e:
            logger.warning(f"Transformation failed: {e}")
            return self._apply_fallback_transformation(text, target_concept, fuzzing_level)
    
    def _apply_synonym_replacement(self, text: str, fuzzing_level: float) -> str:
        """Apply synonym replacement based on fuzzing level."""
        words = text.split()
        replacement_probability = min(fuzzing_level * 0.5, 0.3)  # Max 30% replacement
        
        # Check if synonyms are available
        synonyms_dict = self.transformation_patterns.get("synonyms")
        if not synonyms_dict:
            logger.warning("No synonyms available for replacement")
            return text
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            for category, synonyms in synonyms_dict.items():
                if word_lower == category and random.random() < replacement_probability:
                    replacement = random.choice(synonyms)
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    break
        
        return ' '.join(words)
    
    def _apply_intensifier_adjustment(self, text: str, target_concept: str, fuzzing_level: float) -> str:
        """Apply intensifier adjustments based on target concept and fuzzing level."""
        # Check if target concept suggests need for intensifiers
        if any(word in target_concept.lower() for word in ["critical", "important", "significant", "major", "severe"]):
            words = text.split()
            replacement_probability = min((fuzzing_level - 0.3) * 0.5, 0.4)  # Max 40% replacement
            
            # Check if intensifiers are available
            intensifiers_dict = self.transformation_patterns.get("intensifiers")
            if not intensifiers_dict:
                logger.warning("No intensifiers available for adjustment")
                return text
            
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?;:')
                
                for category, intensifiers in intensifiers_dict.items():
                    if word_lower == category and random.random() < replacement_probability:
                        replacement = random.choice(intensifiers)
                        if word[0].isupper():
                            replacement = replacement.capitalize()
                        words[i] = replacement
                        break
            
            return ' '.join(words)
        
        return text
    
    def _apply_technical_term_alignment(self, text: str, target_concept: str, fuzzing_level: float) -> str:
        """Align technical terms with target concept."""
        words = text.split()
        replacement_probability = min((fuzzing_level - 0.5) * 0.4, 0.3)  # Max 30% replacement
        
        # Check if technical terms are available
        technical_terms_dict = self.transformation_patterns.get("technical_terms")
        if not technical_terms_dict:
            logger.warning("No technical terms available for alignment")
            return text
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            for term, variations in technical_terms_dict.items():
                if word_lower in term.split() and random.random() < replacement_probability:
                    # Find a variation that appears in target concept
                    target_words = target_concept.lower().split()
                    for variation in variations:
                        if variation.lower() in target_words:
                            if word[0].isupper():
                                variation = variation.capitalize()
                            words[i] = variation
                            break
                    break
        
        return ' '.join(words)
    
    def _apply_context_aware_transformation(self, text: str, target_concept: str, 
                                          context_documents: List[DocumentNode], fuzzing_level: float) -> str:
        """Apply context-aware transformation using discovered documents."""
        # Extract common terms from context documents
        context_terms = set()
        for doc in context_documents[:3]:  # Use top 3 context documents
            words = doc.content.lower().split()
            context_terms.update(words)
        
        # Apply context-based transformations
        words = text.split()
        replacement_probability = min((fuzzing_level - 0.7) * 0.3, 0.2)  # Max 20% replacement
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Check if word appears in context and can be replaced
            if word_lower in context_terms and random.random() < replacement_probability:
                # Find a synonym that also appears in context
                for category, synonyms in self.transformation_patterns["synonyms"].items():
                    if word_lower == category:
                        for synonym in synonyms:
                            if synonym.lower() in context_terms:
                                if word[0].isupper():
                                    synonym = synonym.capitalize()
                                words[i] = synonym
                                break
                        break
        
        return ' '.join(words)
    
    def _apply_advanced_semantic_transformation(self, text: str, target_concept: str, fuzzing_level: float) -> str:
        """Apply advanced semantic transformations for high fuzzing levels."""
        words = text.split()
        replacement_probability = min((fuzzing_level - 0.9) * 0.5, 0.15)  # Max 15% replacement
        
        # Apply context modifier transformations
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            for category, modifiers in self.transformation_patterns["context_modifiers"].items():
                if word_lower == category and random.random() < replacement_probability:
                    replacement = random.choice(modifiers)
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    break
        
        return ' '.join(words)
    
    def _apply_fallback_transformation(self, text: str, target_concept: str, fuzzing_level: float) -> str:
        """Apply fallback transformation when embedding fails."""
        words = text.split()
        target_words = target_concept.lower().split()
        replacement_probability = min(fuzzing_level * 0.2, 0.1)  # Max 10% replacement
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Simple replacement logic
            if word_lower == "data" and "information" in target_words and random.random() < replacement_probability:
                words[i] = "information" if word[0].islower() else "Information"
            elif word_lower == "system" and "platform" in target_words and random.random() < replacement_probability:
                words[i] = "platform" if word[0].islower() else "Platform"
            elif word_lower == "analysis" and "examination" in target_words and random.random() < replacement_probability:
                words[i] = "examination" if word[0].islower() else "Examination"
        
        return ' '.join(words)


class UnifiedFuzzingEngine:
    """Unified fuzzing engine combining two-layer strategy with text fuzzing capabilities."""
    
    def __init__(self, config: Optional[FuzzingConfig] = None):
        """Initialize the unified fuzzing engine.
        
        Args:
            config: Fuzzing configuration
        """
        self.config = config or FuzzingConfig()
        self.transformation_engine = UnifiedTransformationEngine(self.config)
        self.document_graph: Dict[str, DocumentNode] = {}
        self.hypothesis_graph: Dict[str, HypothesisNode] = {}
        self.faiss_index: Optional[FAISSIndex] = None
        
    def load_corpus(self, corpus_lines: List[str], source_name: str = "corpus") -> None:
        """Load a corpus of text lines into the fuzzing engine.
        
        Args:
            corpus_lines: List of text lines to load
            source_name: Name of the corpus source
        """
        logger.info(f"Loading corpus with {len(corpus_lines)} lines")
        
        # Create document nodes
        documents = []
        for i, line in enumerate(corpus_lines):
            if line.strip():  # Skip empty lines
                doc = DocumentNode(
                    id=f"{source_name}_{i}",
                    content=line.strip(),
                    source=source_name,
                    source_type="public"
                )
                documents.append(doc)
                self.document_graph[doc.id] = doc
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = embed(texts, self.config.embedding_model)
        
        # Assign embeddings to documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        # Create FAISS index
        self.faiss_index = FAISSIndex(dimension=len(embeddings[0]), index_type="flat")
        
        # Add vectors and metadata to the index
        metadata = [{"id": doc.id, "content": doc.content, "source": doc.source} for doc in documents]
        self.faiss_index.add_vectors(embeddings, metadata)
        
        logger.info(f"✅ Loaded {len(documents)} documents and created FAISS index")
    
    def fuzz_phrase(self, phrase: str, target_concept: str, fuzzing_level: float) -> FuzzingResult:
        """Fuzz a single phrase with specified level.
        
        Args:
            phrase: Phrase to fuzz
            target_concept: Target concept to move towards
            fuzzing_level: Level of fuzzing (0.0 to 1.0)
            
        Returns:
            Fuzzing result
        """
        start_time = time.time()
        
        # Find similar documents for context
        context_documents = []
        if self.faiss_index:
            try:
                phrase_emb = embed([phrase], self.config.embedding_model)[0]
                distances, indices, metadata = self.faiss_index.search(phrase_emb, k=5)
                
                for i, (distance, idx, meta) in enumerate(zip(distances, indices, metadata)):
                    if idx < len(self.document_graph):
                        doc_id = meta.get('id', f"doc_{idx}")
                        if doc_id in self.document_graph:
                            context_documents.append(self.document_graph[doc_id])
            except Exception as e:
                logger.warning(f"Failed to find context documents: {e}")
        
        # Transform the phrase
        fuzzed_text = self.transformation_engine.transform_text(
            phrase, target_concept, fuzzing_level, context_documents
        )
        
        # Calculate similarity score
        try:
            original_emb = embed([phrase], self.config.embedding_model)[0]
            fuzzed_emb = embed([fuzzed_text], self.config.embedding_model)[0]
            target_emb = embed([target_concept], self.config.embedding_model)[0]
            
            original_similarity = sum(a * b for a, b in zip(original_emb, target_emb))
            fuzzed_similarity = sum(a * b for a, b in zip(fuzzed_emb, target_emb))
            
            similarity_score = fuzzed_similarity
        except Exception as e:
            logger.warning(f"Failed to calculate similarity: {e}")
            similarity_score = 0.0
        
        processing_time = time.time() - start_time
        
        return FuzzingResult(
            original_text=phrase,
            fuzzed_text=fuzzed_text,
            fuzzing_level=fuzzing_level,
            similarity_score=similarity_score,
            transformation_method="unified_transformation",
            target_concept=target_concept,
            metadata={
                "context_documents": len(context_documents),
                "embedding_model": self.config.embedding_model
            },
            processing_time=processing_time
        )
    
    def run_fuzzing_campaign(self, phrases: List[str], target_concept: str, 
                           fuzzing_levels: Optional[List[float]] = None) -> FuzzingCampaignResult:
        """Run a complete fuzzing campaign on multiple phrases.
        
        Args:
            phrases: List of phrases to fuzz
            target_concept: Target concept to move towards
            fuzzing_levels: List of fuzzing levels to test (defaults to config levels)
            
        Returns:
            Campaign results
        """
        start_time = time.time()
        
        if fuzzing_levels is None:
            fuzzing_levels = self.config.fuzzing_levels
        
        logger.info(f"Starting fuzzing campaign with {len(phrases)} phrases and {len(fuzzing_levels)} levels")
        
        results_by_level = {}
        total_results = 0
        
        for level in fuzzing_levels:
            level_results = []
            
            for phrase in phrases:
                result = self.fuzz_phrase(phrase, target_concept, level)
                level_results.append(result)
                total_results += 1
            
            results_by_level[level] = level_results
            logger.info(f"Completed fuzzing level {level}: {len(level_results)} results")
        
        processing_time = time.time() - start_time
        
        return FuzzingCampaignResult(
            target_concept=target_concept,
            total_phrases=len(phrases),
            total_results=total_results,
            fuzzing_levels=fuzzing_levels,
            results_by_level=results_by_level,
            processing_time=processing_time,
            metadata={
                "embedding_model": self.config.embedding_model,
                "corpus_size": len(self.document_graph),
                "transformation_engine": "unified"
            }
        )
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded corpus."""
        return {
            "total_documents": len(self.document_graph),
            "faiss_index_loaded": self.faiss_index is not None,
            "embedding_model": self.config.embedding_model,
            "fuzzing_levels": self.config.fuzzing_levels
        }


def create_unified_fuzzing_engine(config: Optional[FuzzingConfig] = None) -> UnifiedFuzzingEngine:
    """Create a unified fuzzing engine with the given configuration.
    
    Args:
        config: Optional fuzzing configuration
        
    Returns:
        Configured unified fuzzing engine
    """
    return UnifiedFuzzingEngine(config)
