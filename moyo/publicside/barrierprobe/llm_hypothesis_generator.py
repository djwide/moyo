"""LLM-based hypothesis generator for two-layer fuzzing system.

This module provides sophisticated hypothesis generation using LLMs
to create better queries for discovering public documents.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from .two_layer_fuzzer import HypothesisNode, PublicDocumentNode

logger = logging.getLogger(__name__)


class LocalHypothesisGenerator:
    """Local hypothesis generator using embedding-based text transformation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the local hypothesis generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.transformation_patterns = self._load_transformation_patterns()
    
    def _load_transformation_patterns(self) -> Dict[str, List[str]]:
        """Load common text transformation patterns for hypothesis generation."""
        merged_synonyms = {
            "data": ["information", "content", "material", "details"],
            "breach": ["violation", "compromise", "incident", "failure"],
            "security": ["protection", "safety", "defense", "safeguard"],
            "system": ["platform", "infrastructure", "framework", "architecture"],
            "analysis": ["examination", "evaluation", "assessment", "review"],
            "research": ["investigation", "study", "exploration", "inquiry"],
            "confidential": ["private", "restricted", "classified", "sensitive"],
            "information": ["data", "content", "material", "details"],
            "disclosure": ["exposure", "leak", "release", "publication"],
            "incident": ["event", "occurrence", "situation", "case"]
        }

        return {
            "synonyms": merged_synonyms,
            "intensifiers": {
                "major": ["significant", "substantial", "considerable", "notable"],
                "critical": ["essential", "vital", "crucial", "important"],
                "serious": ["severe", "grave", "major", "significant"],
                "important": ["critical", "essential", "vital", "crucial"],
                "significant": ["substantial", "major", "considerable", "notable"]
            },
            "technical_terms": {
                "neural network": ["deep learning model", "artificial neural network", "ANN"],
                "machine learning": ["ML", "artificial intelligence", "AI"],
                "algorithm": ["method", "technique", "procedure", "approach"],
                "optimization": ["improvement", "enhancement", "refinement", "tuning"],
                "performance": ["efficiency", "effectiveness", "capability", "functionality"]
            }
        }
    
    def generate_hypotheses_from_documents(self,
                                         documents: List[PublicDocumentNode],
                                         target_concept: str,
                                         base_hypothesis: Optional[HypothesisNode] = None) -> List[HypothesisNode]:
        """Generate hypotheses from discovered documents using local transformation.
        
        Args:
            documents: List of discovered public documents
            target_concept: Target concept to move towards
            base_hypothesis: Original hypothesis that led to these documents
            
        Returns:
            List of generated hypothesis nodes
        """
        hypotheses = []
        
        for doc in documents[:5]:  # Limit to top 5 documents
            doc_hypotheses = self._generate_hypotheses_for_document(
                doc, target_concept, base_hypothesis
            )
            hypotheses.extend(doc_hypotheses)
            
        logger.info(f"Generated {len(hypotheses)} hypotheses from {len(documents)} documents")
        return hypotheses
    
    def _generate_hypotheses_for_document(self,
                                        document: PublicDocumentNode,
                                        target_concept: str,
                                        base_hypothesis: Optional[HypothesisNode] = None) -> List[HypothesisNode]:
        """Generate hypotheses for a single document using local transformation.
        
        Args:
            document: Document to generate hypotheses from
            target_concept: Target concept to move towards
            base_hypothesis: Original hypothesis that led to this document
            
        Returns:
            List of generated hypothesis nodes
        """
        hypotheses = []
        
        # Extract sentences from document content
        sentences = [s.strip() for s in document.content.split('.') if len(s.strip()) > 20]
        
        for sentence in sentences[:3]:  # Take first 3 sentences
            # Apply local transformation to move towards target concept
            transformed_sentence = self._transform_towards_target(sentence, target_concept)
            
            # Create hypothesis node
            hypothesis = HypothesisNode(
                id=None,  # Will be auto-generated
                query=transformed_sentence,
                target_concept=target_concept,
                generation_method=f"local_transform_from_{document.id}",
                metadata={
                    "source_document": document.id,
                    "original_sentence": sentence,
                    "transformation_method": "local_embedding_based"
                }
            )
            
            # Generate embedding
            from shared_utils import embed
            embeddings = embed([transformed_sentence], self.model_name)
            if embeddings:
                hypothesis.embedding = embeddings[0]
                
            hypotheses.append(hypothesis)
            
        return hypotheses
    
    def _transform_towards_target(self, text: str, target_concept: str) -> str:
        """Transform text towards target concept using local patterns.
        
        Args:
            text: Original text to transform
            target_concept: Target concept to move towards
            
        Returns:
            Transformed text
        """
        import random
        
        # Get embeddings for similarity calculation
        try:
            from shared_utils import embed
            original_emb = embed([text], self.model_name)[0]
            target_emb = embed([target_concept], self.model_name)[0]
            
            # Calculate similarity
            similarity = sum(a * b for a, b in zip(original_emb, target_emb))
            
            # If already very similar, make minor adjustments
            if similarity > 0.9:
                return self._apply_minor_adjustments(text)
            
            # Apply transformation patterns
            transformed = self._apply_synonym_replacement(text)
            transformed = self._apply_intensifier_adjustment(transformed, target_concept)
            transformed = self._apply_technical_term_alignment(transformed, target_concept)
            
            return transformed
            
        except Exception as e:
            logger.warning(f"Embedding-based transformation failed: {e}")
            return self._apply_fallback_transformation(text, target_concept)
    
    def _apply_minor_adjustments(self, text: str) -> str:
        """Apply minor adjustments to already similar text."""
        adjustments = [
            ("This describes", "This explains"),
            ("This shows", "This demonstrates"),
            ("This includes", "This contains"),
            ("This provides", "This offers"),
            ("This enables", "This allows"),
            ("This improves", "This enhances")
        ]
        
        for original, replacement in adjustments:
            if original in text:
                return text.replace(original, replacement)
        
        return text
    
    def _apply_synonym_replacement(self, text: str) -> str:
        """Apply synonym replacement based on patterns."""
        import random
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Check synonyms
            for category, synonyms in self.transformation_patterns["synonyms"].items():
                if word_lower == category and random.random() < 0.3:
                    replacement = random.choice(synonyms)
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = replacement
                    break
        
        return ' '.join(words)
    
    def _apply_intensifier_adjustment(self, text: str, target_concept: str) -> str:
        """Apply intensifier adjustments based on target concept."""
        import random
        
        # Check if target concept suggests need for intensifiers
        if any(word in target_concept.lower() for word in ["critical", "important", "significant", "major"]):
            words = text.split()
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?;:')
                
                for category, intensifiers in self.transformation_patterns["intensifiers"].items():
                    if word_lower == category and random.random() < 0.4:
                        replacement = random.choice(intensifiers)
                        if word[0].isupper():
                            replacement = replacement.capitalize()
                        words[i] = replacement
                        break
            
            return ' '.join(words)
        
        return text
    
    def _apply_technical_term_alignment(self, text: str, target_concept: str) -> str:
        """Align technical terms with target concept."""
        import random
        
        # Extract common technical terms from target concept
        target_words = target_concept.lower().split()
        
        # Apply technical term alignment
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            for term, variations in self.transformation_patterns["technical_terms"].items():
                if word_lower in term.split() and random.random() < 0.3:
                    # Find a variation that appears in target words
                    for variation in variations:
                        if variation.lower() in target_words:
                            if word[0].isupper():
                                variation = variation.capitalize()
                            words[i] = variation
                            break
                    break
        
        return ' '.join(words)
    
    def _apply_fallback_transformation(self, text: str, target_concept: str) -> str:
        """Apply fallback transformation when embedding fails."""
        # Simple word replacement based on target concept
        target_words = target_concept.lower().split()
        
        # Find words in text that could be replaced with target words
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Simple replacement logic
            if word_lower == "data" and "information" in target_words:
                words[i] = "information" if word[0].islower() else "Information"
            elif word_lower == "system" and "platform" in target_words:
                words[i] = "platform" if word[0].islower() else "Platform"
            elif word_lower == "analysis" and "examination" in target_words:
                words[i] = "examination" if word[0].islower() else "Examination"
        
        return ' '.join(words)


@dataclass
class HypothesisGenerationConfig:
    """Configuration for hypothesis generation."""
    llm_provider: str = "ollama"  # "openai", "anthropic", "ollama", "local"
    model_name: str = "llama3.1:8b"
    api_key: Optional[str] = None
    base_url: Optional[str] = "http://localhost:11434"
    max_hypotheses_per_document: int = 3
    min_query_length: int = 10
    max_query_length: int = 200
    temperature: float = 0.7
    max_tokens: int = 150


class LLMHypothesisGenerator:
    """LLM-based hypothesis generator for sophisticated query creation."""
    
    def __init__(self, config: HypothesisGenerationConfig):
        """Initialize the LLM hypothesis generator.
        
        Args:
            config: Configuration for hypothesis generation
        """
        self.config = config
        self.llm_client = self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        """Initialize the LLM client based on provider."""
        try:
            from moyo.llm.testing import FakeDeterministicLLM, is_test_mode
            if is_test_mode() or self.config.llm_provider in ("test", "echo"):
                return FakeDeterministicLLM(
                    model_name=self.config.model_name or "echo-test"
                )
        except Exception:
            if self.config.llm_provider in ("test", "echo"):
                return None

        if self.config.llm_provider == "local":
            # Offline synonym fallback (not the default).
            return LocalHypothesisGenerator(self.config.model_name)
        elif self.config.llm_provider == "ollama":
            from moyo.publicside.barrierprobe.llm_fuzzer import OllamaClient

            client = OllamaClient(
                self.config.model_name,
                base_url=self.config.base_url,
            )
            if not client.is_available():
                logger.error(
                    "Ollama not reachable for hypothesis generation at %s",
                    client.base_url,
                )
                return None
            return client
        elif self.config.llm_provider == "openai":
            try:
                from openai import OpenAI
                kwargs = {}
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                return OpenAI(**kwargs)
            except ImportError:
                logger.error("OpenAI library not installed")
                return None
        elif self.config.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                kwargs = {}
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                return Anthropic(**kwargs)
            except ImportError:
                logger.error("Anthropic library not installed")
                return None
        else:
            logger.error(f"Unsupported LLM provider: {self.config.llm_provider}")
            return None
    
    def generate_hypotheses_from_documents(self,
                                         documents: List[PublicDocumentNode],
                                         target_concept: str,
                                         base_hypothesis: Optional[HypothesisNode] = None) -> List[HypothesisNode]:
        """Generate hypotheses from discovered documents.
        
        Args:
            documents: List of discovered public documents
            target_concept: Target concept to move towards
            base_hypothesis: Original hypothesis that led to these documents
            
        Returns:
            List of generated hypothesis nodes
        """
        hypotheses = []

        if isinstance(self.llm_client, LocalHypothesisGenerator):
            return self.llm_client.generate_hypotheses_from_documents(
                documents, target_concept, base_hypothesis
            )
        
        for doc in documents[:5]:  # Limit to top 5 documents
            doc_hypotheses = self._generate_hypotheses_for_document(
                doc, target_concept, base_hypothesis
            )
            hypotheses.extend(doc_hypotheses)
            
        logger.info(f"Generated {len(hypotheses)} hypotheses from {len(documents)} documents")
        return hypotheses
    
    def _generate_hypotheses_for_document(self,
                                        document: PublicDocumentNode,
                                        target_concept: str,
                                        base_hypothesis: Optional[HypothesisNode] = None) -> List[HypothesisNode]:
        """Generate hypotheses for a single document."""
        if not self.llm_client:
            return self._fallback_hypothesis_generation(document, target_concept)
            
        try:
            # Create prompt for hypothesis generation
            prompt = self._create_hypothesis_prompt(document, target_concept, base_hypothesis)
            
            # Generate hypotheses using LLM
            response = self._call_llm(prompt)
            
            # Parse response
            hypotheses = self._parse_llm_response(response, target_concept, document.id)
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Error generating hypotheses for document {document.id}: {e}")
            return self._fallback_hypothesis_generation(document, target_concept)
    
    def _create_hypothesis_prompt(self,
                                document: PublicDocumentNode,
                                target_concept: str,
                                base_hypothesis: Optional[HypothesisNode] = None) -> str:
        """Create a prompt for hypothesis generation."""
        
        prompt = f"""You are an expert at generating search queries to discover information about specific concepts.

TARGET CONCEPT: {target_concept}

DISCOVERED DOCUMENT:
Source: {document.source}
Content: {document.content[:500]}...

ORIGINAL QUERY: {base_hypothesis.query if base_hypothesis else "N/A"}

TASK: Generate {self.config.max_hypotheses_per_document} new search queries that could help discover more information about the target concept. These queries should be:
1. Related to the discovered document content
2. Focused on finding more information about the target concept
3. Specific and actionable
4. Between {self.config.min_query_length} and {self.config.max_query_length} characters

Generate queries in this format:
QUERY 1: [query text]
QUERY 2: [query text]
QUERY 3: [query text]

Focus on:
- Key terms and phrases from the document
- Related concepts and synonyms
- Specific aspects of the target concept
- Technical terminology that might appear in similar documents
"""

        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        from moyo.llm.testing import FakeDeterministicLLM

        system = (
            "You are a research assistant that proposes retrieval queries. "
            "Follow the requested QUERY N: format exactly."
        )
        if isinstance(self.llm_client, FakeDeterministicLLM) or self.config.llm_provider in (
            "test",
            "echo",
        ):
            return self.llm_client.generate(
                prompt,
                system=system,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        if self.config.llm_provider == "ollama":
            return self.llm_client.generate(
                prompt,
                system=system,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        if self.config.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""
        elif self.config.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def _parse_llm_response(self, response: str, target_concept: str, document_id: str) -> List[HypothesisNode]:
        """Parse LLM response into hypothesis nodes."""
        hypotheses = []
        
        # Extract queries from response
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('QUERY'):
                # Extract query text
                query_text = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                
                # Validate query
                if (self.config.min_query_length <= len(query_text) <= self.config.max_query_length and
                    query_text and not query_text.startswith('QUERY')):
                    
                    hypothesis = HypothesisNode(
                        id=None,  # Will be auto-generated
                        query=query_text,
                        target_concept=target_concept,
                        generation_method=f"llm_from_doc_{document_id}"
                    )
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _fallback_hypothesis_generation(self,
                                      document: PublicDocumentNode,
                                      target_concept: str) -> List[HypothesisNode]:
        """Fallback hypothesis generation when LLM is not available."""
        hypotheses = []
        
        # Extract key phrases from document content
        content = document.content
        
        # Simple phrase extraction
        sentences = content.split('.')
        for sentence in sentences[:3]:  # Take first 3 sentences
            sentence = sentence.strip()
            if len(sentence) >= self.config.min_query_length:
                # Create hypothesis from sentence
                hypothesis = HypothesisNode(
                    id=None,
                    query=sentence,
                    target_concept=target_concept,
                    generation_method=f"fallback_from_doc_{document.id}"
                )
                hypotheses.append(hypothesis)
                
                if len(hypotheses) >= self.config.max_hypotheses_per_document:
                    break
        
        return hypotheses
    
    def generate_semantic_variations(self,
                                   hypothesis: HypothesisNode,
                                   target_concept: str) -> List[HypothesisNode]:
        """Generate semantic variations of an existing hypothesis."""
        if not self.llm_client:
            return []
            
        try:
            prompt = f"""Generate semantic variations of this search query to help discover more information about the target concept.

ORIGINAL QUERY: {hypothesis.query}
TARGET CONCEPT: {target_concept}

Generate 3 different variations that:
1. Use different but related terminology
2. Focus on different aspects of the same concept
3. Use synonyms and alternative phrasings
4. Maintain the same semantic intent

Format:
VARIATION 1: [query]
VARIATION 2: [query]
VARIATION 3: [query]
"""

            response = self._call_llm(prompt)
            variations = self._parse_llm_response(response, target_concept, f"variation_of_{hypothesis.id}")
            
            return variations
            
        except Exception as e:
            logger.error(f"Error generating semantic variations: {e}")
            return []
    
    def generate_concept_expansions(self,
                                  target_concept: str,
                                  discovered_terms: List[str]) -> List[HypothesisNode]:
        """Generate hypotheses that expand on the target concept using discovered terms."""
        if not self.llm_client:
            return []
            
        try:
            prompt = f"""Based on the discovered terms and target concept, generate search queries that expand the search space.

TARGET CONCEPT: {target_concept}
DISCOVERED TERMS: {', '.join(discovered_terms[:10])}

Generate 5 search queries that:
1. Combine the target concept with discovered terms
2. Explore related concepts and domains
3. Use broader and narrower search terms
4. Focus on different aspects of the target concept
5. Include technical and non-technical variations

Format:
EXPANSION 1: [query]
EXPANSION 2: [query]
EXPANSION 3: [query]
EXPANSION 4: [query]
EXPANSION 5: [query]
"""

            response = self._call_llm(prompt)
            expansions = self._parse_llm_response(response, target_concept, "concept_expansion")
            
            return expansions
            
        except Exception as e:
            logger.error(f"Error generating concept expansions: {e}")
            return []


class AdaptiveHypothesisGenerator:
    """Adaptive hypothesis generator that combines multiple strategies."""
    
    def __init__(self, llm_config: HypothesisGenerationConfig):
        """Initialize the adaptive hypothesis generator.
        
        Args:
            llm_config: Configuration for LLM-based generation
        """
        self.llm_generator = LLMHypothesisGenerator(llm_config)
        self.generation_history: List[Dict[str, Any]] = []
        
    def generate_adaptive_hypotheses(self,
                                   documents: List[PublicDocumentNode],
                                   target_concept: str,
                                   base_hypothesis: Optional[HypothesisNode] = None,
                                   strategy: str = "auto") -> List[HypothesisNode]:
        """Generate hypotheses using adaptive strategies.
        
        Args:
            documents: Discovered documents
            target_concept: Target concept
            base_hypothesis: Original hypothesis
            strategy: Generation strategy ("llm", "fallback", "auto")
            
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        # Determine strategy
        if strategy == "auto":
            strategy = self._determine_strategy(documents, base_hypothesis)
        
        # Generate hypotheses based on strategy
        if strategy == "llm" and self.llm_generator.llm_client:
            hypotheses = self.llm_generator.generate_hypotheses_from_documents(
                documents, target_concept, base_hypothesis
            )
        else:
            # Fallback strategy
            hypotheses = self._fallback_generation(documents, target_concept, base_hypothesis)
        
        # Record generation
        self._record_generation(strategy, len(hypotheses), documents, base_hypothesis)
        
        return hypotheses
    
    def _determine_strategy(self,
                          documents: List[PublicDocumentNode],
                          base_hypothesis: Optional[HypothesisNode] = None) -> str:
        """Determine the best generation strategy based on context."""
        
        # Use LLM if available and we have good documents
        if (self.llm_generator.llm_client and 
            documents and 
            any(len(doc.content) > 100 for doc in documents)):
            return "llm"
        
        return "fallback"
    
    def _fallback_generation(self,
                           documents: List[PublicDocumentNode],
                           target_concept: str,
                           base_hypothesis: Optional[HypothesisNode] = None) -> List[HypothesisNode]:
        """Fallback hypothesis generation."""
        hypotheses = []
        
        # Extract key terms from documents
        all_terms = set()
        for doc in documents:
            words = doc.content.lower().split()
            # Filter for meaningful terms
            meaningful_terms = [w for w in words if len(w) > 4 and w.isalpha()]
            all_terms.update(meaningful_terms[:10])  # Top 10 terms per document
        
        # Generate hypotheses combining target concept with discovered terms
        for term in list(all_terms)[:5]:  # Use top 5 terms
            query = f"{target_concept} {term}"
            hypothesis = HypothesisNode(
                id=None,
                query=query,
                target_concept=target_concept,
                generation_method="adaptive_fallback"
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _record_generation(self,
                         strategy: str,
                         hypothesis_count: int,
                         documents: List[PublicDocumentNode],
                         base_hypothesis: Optional[HypothesisNode] = None):
        """Record hypothesis generation for analysis."""
        record = {
            "timestamp": datetime.now(),
            "strategy": strategy,
            "hypothesis_count": hypothesis_count,
            "document_count": len(documents),
            "base_hypothesis_id": base_hypothesis.id if base_hypothesis else None,
            "document_sources": [doc.source for doc in documents]
        }
        self.generation_history.append(record)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about hypothesis generation."""
        if not self.generation_history:
            return {"total_generations": 0}
        
        strategies = [r["strategy"] for r in self.generation_history]
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_generations": len(self.generation_history),
            "strategy_distribution": strategy_counts,
            "avg_hypotheses_per_generation": sum(r["hypothesis_count"] for r in self.generation_history) / len(self.generation_history)
        }
