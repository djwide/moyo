"""LLM-assisted fuzzing for barrier probing.

This module provides functionality to use LLMs (defaulting to OpenAI) to assist in fuzzing
by finding semantically close phrases and using the LLM to reduce semantic distance.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time

from shared_utils import (
    embed,
    FAISSIndex,
    normalize_text,
    TextNormalizationConfig,
    get_logger
)

logger = get_logger(__name__)


class LocalLLMClient:
    """Local LLM client using embedding-based text transformation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the local LLM client.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.transformation_patterns = self._load_transformation_patterns()
    
    def _load_transformation_patterns(self) -> Dict[str, List[str]]:
        """Load common text transformation patterns."""
        # Load synonym map directly from data directory
        shared_synonyms = {}
        try:
            import json
            from pathlib import Path
            synonym_file = Path(__file__).resolve().parents[3] / "data" / "synonym_map.json"
            if synonym_file.exists():
                with open(synonym_file, 'r', encoding='utf-8') as f:
                    shared_synonyms = json.load(f)
                logger.info(f"Loaded {len(shared_synonyms)} synonym groups from {synonym_file}")
            else:
                logger.warning(f"Synonym map file not found at {synonym_file}")
        except Exception as e:
            logger.warning(f"Failed to load synonym map: {e}")
            shared_synonyms = {}
        
        # Merge with built-in patterns
        built_in_synonyms = {
            "sensitive": ["confidential", "private", "restricted", "classified"],
            "data": ["information", "content", "material", "details"],
            "exposure": ["disclosure", "leak", "release", "publication"],
            "breach": ["violation", "compromise", "incident", "failure"],
            "security": ["protection", "safety", "defense", "safeguard"],
            "system": ["platform", "infrastructure", "framework", "architecture"],
            "analysis": ["examination", "evaluation", "assessment", "review"],
            "research": ["investigation", "study", "exploration", "inquiry"],
            "development": ["creation", "construction", "building", "establishment"],
            "implementation": ["deployment", "execution", "application", "integration"]
        }
        
        # Merge shared synonyms with built-in patterns
        merged_synonyms = {**shared_synonyms, **built_in_synonyms}
        
        return {
            "synonyms": merged_synonyms,
            "intensifiers": {
                "novel": ["innovative", "groundbreaking", "revolutionary", "cutting-edge"],
                "advanced": ["sophisticated", "state-of-the-art", "high-tech", "modern"],
                "comprehensive": ["extensive", "thorough", "complete", "detailed"],
                "critical": ["essential", "vital", "crucial", "important"],
                "significant": ["substantial", "major", "considerable", "notable"]
            },
            "technical_terms": {
                "neural network": ["deep learning model", "artificial neural network", "ANN", "neural architecture"],
                "machine learning": ["ML", "artificial intelligence", "AI", "automated learning"],
                "algorithm": ["method", "technique", "procedure", "approach"],
                "optimization": ["improvement", "enhancement", "refinement", "tuning"],
                "performance": ["efficiency", "effectiveness", "capability", "functionality"]
            }
        }
    
    def transform_text(self, original_text: str, target_concept: str, similar_phrases: List[str]) -> str:
        """Transform text using embedding-based similarity and pattern matching.
        
        Args:
            original_text: Original text to transform
            target_concept: Target concept to move towards
            similar_phrases: List of similar phrases for context
            
        Returns:
            Transformed text
        """
        import random
        
        # Get embeddings for similarity calculation
        try:
            original_emb = embed([original_text], self.model_name)[0]
            target_emb = embed([target_concept], self.model_name)[0]
            
            # Calculate similarity
            similarity = sum(a * b for a, b in zip(original_emb, target_emb))
            
            # If already very similar, make minor adjustments
            if similarity > 0.9:
                return self._apply_minor_adjustments(original_text)
            
            # Apply transformation patterns
            transformed = self._apply_synonym_replacement(original_text)
            transformed = self._apply_intensifier_adjustment(transformed, target_concept)
            transformed = self._apply_technical_term_alignment(transformed, similar_phrases)
            
            return transformed
            
        except Exception as e:
            logger.warning(f"Embedding-based transformation failed: {e}")
            return self._apply_fallback_transformation(original_text, target_concept)
    
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
    
    def _apply_technical_term_alignment(self, text: str, similar_phrases: List[str]) -> str:
        """Align technical terms with similar phrases."""
        import random
        
        # Extract common technical terms from similar phrases
        common_terms = set()
        for phrase in similar_phrases[:3]:  # Use top 3 similar phrases
            words = phrase.lower().split()
            for term, variations in self.transformation_patterns["technical_terms"].items():
                if term in phrase.lower():
                    common_terms.update(variations)
        
        # Apply technical term alignment
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            for term, variations in self.transformation_patterns["technical_terms"].items():
                if word_lower in term.split() and random.random() < 0.3:
                    # Find a variation that appears in common terms
                    for variation in variations:
                        if variation.lower() in common_terms:
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


class OllamaClient:
    """Minimal client for a locally running Ollama server.

    Talks to the native Ollama HTTP API (``/api/generate``) using only the
    Python standard library, so it adds no new dependencies. Ollama runs the
    model fully locally and offloads to the GPU automatically when one is
    available (e.g. an NVIDIA RTX card via CUDA).
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, model_name: str, base_url: Optional[str] = None,
                 timeout: int = 180):
        self.model_name = model_name
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Return True if the Ollama server responds and lists models."""
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning(f"Ollama not reachable at {self.base_url}: {exc}")
            return False

    def list_models(self) -> List[str]:
        """Return the model tags currently installed on the server."""
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception as exc:
            logger.warning(f"Could not list Ollama models: {exc}")
            return []

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate a completion from the local model (non-streaming)."""
        import urllib.request
        import urllib.error

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return (body.get("response") or "").strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Ollama request failed ({exc.code}): {detail}. "
                f"Is the model '{self.model_name}' pulled? Try: ollama pull {self.model_name}"
            ) from exc


@dataclass
class LLMFuzzerConfig:
    """Configuration for LLM-assisted fuzzing."""
    
    # LLM Configuration
    # Supported providers:
    #   "openai"    – OpenAI hosted API
    #   "anthropic" – Anthropic hosted API
    #   "ollama"    – local Ollama server (native API)
    #   "custom"    – any OpenAI-compatible endpoint (vLLM, LM Studio, Together,
    #                 Groq, OpenRouter, DeepSeek, llama.cpp server, etc.) via base_url
    #   "local"     – embedding-only synonym transformer (no LLM/API)
    llm_provider: str = "local"
    model_name: str = "all-MiniLM-L6-v2"  # Default to local MiniLM model
    api_key: Optional[str] = None
    # Endpoint for self-hosted / OpenAI-compatible providers (Ollama, custom).
    base_url: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Semantic Search Configuration
    search_k: int = 10  # Number of closest phrases to retrieve
    similarity_threshold: float = 0.8  # Minimum similarity to consider
    
    # Fuzzing Configuration
    max_iterations: int = 5  # Maximum fuzzing iterations
    target_similarity: float = 0.95  # Target similarity to achieve
    prompt_template: str = """
You are an expert at semantic analysis and text transformation. Your task is to modify the given phrase to make it semantically closer to the target concept while maintaining its core meaning.

Target concept: {target_concept}

Original phrase: {original_phrase}

Similar phrases from the corpus:
{similar_phrases}

Instructions:
1. Analyze the semantic relationship between the original phrase and target concept
2. Study the similar phrases to understand the semantic patterns
3. Transform the original phrase to reduce semantic distance to the target
4. Maintain the original meaning while making it more semantically aligned
5. Return only the transformed phrase, no explanations

Transformed phrase:"""


class LLMFuzzer:
    """LLM-assisted fuzzer for semantic barrier probing."""
    
    def __init__(self, config: Optional[LLMFuzzerConfig] = None):
        """Initialize the LLM fuzzer.
        
        Args:
            config: Configuration for the fuzzer
        """
        self.config = config or LLMFuzzerConfig()
        self.normalization_config = TextNormalizationConfig(
            lowercase=True,
            normalize_unicode=True,
            normalize_whitespace=True,
            remove_urls=True,
            remove_emails=True,
            normalize_punctuation=True
        )
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm_client()
        self.interaction_log: List[Dict[str, str]] = []
        
    def _initialize_llm_client(self):
        """Initialize the LLM client based on configuration."""
        if self.config.llm_provider == "local":
            # Use local embedding-based text transformation
            return LocalLLMClient(self.config.model_name)
        elif self.config.llm_provider == "ollama":
            client = OllamaClient(self.config.model_name, base_url=self.config.base_url)
            if not client.is_available():
                logger.error(
                    f"Ollama server not reachable at {client.base_url}. "
                    "Start it with 'ollama serve' and pull a model "
                    f"(e.g. 'ollama pull {self.config.model_name}')."
                )
                return None
            installed = client.list_models()
            if installed and not any(
                m == self.config.model_name or m.startswith(self.config.model_name + ":")
                for m in installed
            ):
                logger.warning(
                    f"Model '{self.config.model_name}' is not installed in Ollama. "
                    f"Available: {', '.join(installed) or 'none'}. "
                    f"Pull it with: ollama pull {self.config.model_name}"
                )
            return client
        elif self.config.llm_provider == "openai":
            try:
                from openai import OpenAI
                kwargs = {}
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                return OpenAI(**kwargs)
            except ImportError:
                logger.error("OpenAI library not installed. Install with: pip install openai")
                return None
        elif self.config.llm_provider == "custom":
            # Any OpenAI-compatible endpoint (vLLM, LM Studio, Together, Groq,
            # OpenRouter, DeepSeek, llama.cpp server, ...). Requires base_url.
            if not self.config.base_url:
                logger.error(
                    "Provider 'custom' requires a base_url pointing at an "
                    "OpenAI-compatible endpoint (e.g. http://localhost:8000/v1)."
                )
                return None
            try:
                from openai import OpenAI
                # Many self-hosted servers ignore the key but the SDK requires
                # a non-empty value, so fall back to a placeholder.
                return OpenAI(
                    api_key=self.config.api_key or "not-needed",
                    base_url=self.config.base_url,
                )
            except ImportError:
                logger.error("OpenAI library not installed. Install with: pip install openai")
                return None
        elif self.config.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                kwargs = {}
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                return Anthropic(**kwargs)
            except ImportError:
                logger.error("Anthropic library not installed. Install with: pip install anthropic")
                return None
        else:
            logger.error(f"Unsupported LLM provider: {self.config.llm_provider}")
            return None
    
    def query_llm(self, prompt: str) -> Optional[str]:
        """Query the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response or None if failed
        """
        if not self.llm_client:
            logger.error("LLM client not initialized")
            return None
            
        try:
            if self.config.llm_provider == "local":
                # Extract information from the prompt for local transformation
                original_phrase, target_concept, similar_phrases = self._parse_fuzzing_prompt(prompt)
                text = self.llm_client.transform_text(original_phrase, target_concept, similar_phrases)
            elif self.config.llm_provider == "ollama":
                text = self.llm_client.generate(
                    prompt,
                    system=(
                        "You are a helpful assistant for semantic text transformation. "
                        "Return only the transformed phrase, with no preamble or explanation."
                    ),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            elif self.config.llm_provider in ("openai", "custom"):
                response = self.llm_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for semantic text transformation."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                text = (response.choices[0].message.content or "").strip()
            elif self.config.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system="You are a helpful assistant for semantic text transformation.",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = (response.content[0].text if response.content else "").strip()
            else:
                return None

            self.interaction_log.append({"prompt": prompt, "response": text})
            return text

        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return None
    
    def _parse_fuzzing_prompt(self, prompt: str) -> Tuple[str, str, List[str]]:
        """Parse the fuzzing prompt to extract original phrase, target concept, and similar phrases.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Tuple of (original_phrase, target_concept, similar_phrases)
        """
        lines = prompt.split('\n')
        original_phrase = ""
        target_concept = ""
        similar_phrases = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Target concept:"):
                target_concept = line.replace("Target concept:", "").strip()
            elif line.startswith("Original phrase:"):
                original_phrase = line.replace("Original phrase:", "").strip()
            elif line.startswith("Similar phrases from the corpus:"):
                continue
            elif line and line[0].isdigit() and '.' in line:
                # Extract phrase from numbered list
                phrase = line.split('.', 1)[1].strip()
                if '(' in phrase and 'similarity:' in phrase:
                    phrase = phrase.split('(')[0].strip()
                similar_phrases.append(phrase)
        
        return original_phrase, target_concept, similar_phrases
    
    def find_similar_phrases(self, 
                           query: str, 
                           index: FAISSIndex, 
                           k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find semantically similar phrases in the index.
        
        Args:
            query: Query phrase to find similar phrases for
            index: FAISS index containing the corpus
            k: Number of results to return (defaults to config.search_k)
            
        Returns:
            List of similar phrases with metadata
        """
        if k is None:
            k = self.config.search_k
            
        # Normalize query
        normalized_query = normalize_text(query, self.normalization_config)
        
        # Generate query embedding
        query_embeddings = embed([normalized_query])
        if not query_embeddings:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search index
        distances, indices, metadata = index.search(query_embeddings[0], k=k)
        
        # Filter by similarity threshold
        results = []
        for i, (distance, idx, meta) in enumerate(zip(distances, indices, metadata)):
            if distance >= self.config.similarity_threshold:
                results.append({
                    "rank": i + 1,
                    "similarity": distance,
                    "index": idx,
                    "metadata": meta,
                    "text": meta.get("text", "")
                })
        
        return results
    
    def create_fuzzing_prompt(self, 
                            target_concept: str,
                            original_phrase: str,
                            similar_phrases: List[Dict[str, Any]]) -> str:
        """Create a prompt for LLM-assisted fuzzing.
        
        Args:
            target_concept: The target concept to move towards
            original_phrase: The original phrase to transform
            similar_phrases: List of similar phrases from the corpus
            
        Returns:
            Formatted prompt for the LLM
        """
        # Format similar phrases
        phrases_text = ""
        for i, phrase_info in enumerate(similar_phrases[:5]):  # Top 5 phrases
            phrases_text += f"{i+1}. {phrase_info['text']} (similarity: {phrase_info['similarity']:.3f})\n"
        
        # Create prompt using template
        prompt = self.config.prompt_template.format(
            target_concept=target_concept,
            original_phrase=original_phrase,
            similar_phrases=phrases_text
        )
        
        return prompt
    
    def fuzz_phrase(self,
                   original_phrase: str,
                   target_concept: str,
                   index: FAISSIndex) -> Tuple[str, float, List[str], List[Dict[str, str]]]:
        """Fuzz a phrase to reduce semantic distance to target concept.
        
        Args:
            original_phrase: The original phrase to fuzz
            target_concept: The target concept to move towards
            index: FAISS index for semantic search
            
        Returns:
            Tuple of (fuzzed_phrase, final_similarity, transformation_history)
        """
        current_phrase = original_phrase
        transformation_history = [original_phrase]
        interactions: List[Dict[str, str]] = []
        
        logger.info(f"Starting fuzzing for phrase: '{original_phrase}'")
        logger.info(f"Target concept: '{target_concept}'")
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"Fuzzing iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Find similar phrases
            similar_phrases = self.find_similar_phrases(current_phrase, index)
            
            if not similar_phrases:
                logger.warning("No similar phrases found, stopping fuzzing")
                break
            
            # Create fuzzing prompt
            prompt = self.create_fuzzing_prompt(target_concept, current_phrase, similar_phrases)

            # Query LLM
            fuzzed_phrase = self.query_llm(prompt)
            
            if not fuzzed_phrase:
                logger.warning("LLM query failed, stopping fuzzing")
                break

            interactions.append({"prompt": prompt, "response": fuzzed_phrase})
            
            # Normalize the fuzzed phrase
            fuzzed_phrase = normalize_text(fuzzed_phrase, self.normalization_config)
            
            # Check if we've achieved target similarity
            fuzzed_embeddings = embed([fuzzed_phrase])
            if fuzzed_embeddings:
                # Calculate similarity to target concept
                target_embeddings = embed([target_concept])
                if target_embeddings:
                    # Simple cosine similarity (assuming normalized embeddings)
                    similarity = sum(a * b for a, b in zip(fuzzed_embeddings[0], target_embeddings[0]))
                    
                    logger.info(f"Iteration {iteration + 1}: similarity = {similarity:.3f}")
                    
                    if similarity >= self.config.target_similarity:
                        logger.info(f"Target similarity achieved: {similarity:.3f}")
                        transformation_history.append(fuzzed_phrase)
                        return fuzzed_phrase, similarity, transformation_history, interactions
                    
                    current_phrase = fuzzed_phrase
                    transformation_history.append(fuzzed_phrase)
                else:
                    logger.warning("Failed to generate target embedding")
                    break
            else:
                logger.warning("Failed to generate fuzzed phrase embedding")
                break
            
            # Rate limiting
            time.sleep(1)
        
        # Return the best result we achieved
        if transformation_history:
            final_phrase = transformation_history[-1]
            final_embeddings = embed([final_phrase])
            target_embeddings = embed([target_concept])
            
            if final_embeddings and target_embeddings:
                final_similarity = sum(a * b for a, b in zip(final_embeddings[0], target_embeddings[0]))
                return final_phrase, final_similarity, transformation_history, interactions

        return original_phrase, 0.0, transformation_history, interactions
    
    def batch_fuzz_phrases(self,
                          phrases: List[str],
                          target_concept: str,
                          index: FAISSIndex,
                          analyzer: Optional[Any] = None,
                          analysis_top_k: int = 5) -> List[Dict[str, Any]]:
        """Fuzz multiple phrases in batch.
        
        Args:
            phrases: List of phrases to fuzz
            target_concept: The target concept to move towards
            index: FAISS index for semantic search
            
        Returns:
            List of fuzzing results for each phrase
        """
        results = []
        
        for i, phrase in enumerate(phrases):
            logger.info(f"Fuzzing phrase {i+1}/{len(phrases)}: '{phrase}'")
            
            fuzzed_phrase, similarity, history, interactions = self.fuzz_phrase(phrase, target_concept, index)

            analysis = None
            if analyzer is not None:
                try:
                    analysis = analyzer.search_phrase(fuzzed_phrase, top_k=analysis_top_k)
                except Exception as e:
                    logger.warning(f"Analysis failed for phrase '{fuzzed_phrase}': {e}")

            results.append({
                "original_phrase": phrase,
                "fuzzed_phrase": fuzzed_phrase,
                "final_similarity": similarity,
                "transformation_history": history,
                "iterations": len(history) - 1,
                "target_concept": target_concept,
                "prompt_response_log": interactions,
                "analysis": analysis,
            })
            
            # Rate limiting between phrases
            time.sleep(2)
        
        return results


def create_fuzzer_from_config(config_dict: Dict[str, Any]) -> LLMFuzzer:
    """Create an LLM fuzzer from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured LLMFuzzer instance
    """
    config = LLMFuzzerConfig(**config_dict)
    return LLMFuzzer(config)


def fuzz_phrases_for_barrier_analysis(phrases: List[str],
                                    target_concept: str,
                                    index: FAISSIndex,
                                    config: Optional[LLMFuzzerConfig] = None,
                                    analyzer: Optional[Any] = None,
                                    analysis_top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for fuzzing phrases for barrier analysis.
    
    Args:
        phrases: List of phrases to fuzz
        target_concept: The target concept to move towards
        index: FAISS index for semantic search
        config: Optional fuzzer configuration
        
    Returns:
        List of fuzzing results
    """
    fuzzer = LLMFuzzer(config)
    return fuzzer.batch_fuzz_phrases(phrases, target_concept, index, analyzer=analyzer, analysis_top_k=analysis_top_k)
