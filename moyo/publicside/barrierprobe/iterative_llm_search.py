"""Iterative LLM-based semantic search enhancement."""

import time
import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from .schema import BarrierProbeResult
from .barrier_analyzer import BarrierAnalyzer
from shared_utils import generate_id
from shared_utils import embed
from moyo.publicside.barrierprobe.llm_fuzzer import OllamaClient, DEFAULT_OLLAMA_MODEL

logger = logging.getLogger(__name__)


class IterativeLLMSearch:
    """Iterative LLM-based semantic search enhancement."""
    
    def __init__(self, barrier_analyzer: BarrierAnalyzer, llm_client=None):
        """Initialize the iterative LLM search.
        
        Args:
            barrier_analyzer: Barrier analyzer with loaded indexes
            llm_client: LLM client for generating responses. Defaults to local
                Ollama ``llama3.1:8b``.
        """
        self.barrier_analyzer = barrier_analyzer
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                from moyo.llm.testing import FakeDeterministicLLM, is_test_mode
                if is_test_mode():
                    self.llm_client = FakeDeterministicLLM(model_name="echo-test")
                else:
                    self.llm_client = OllamaClient(DEFAULT_OLLAMA_MODEL)
            except Exception:
                self.llm_client = OllamaClient(DEFAULT_OLLAMA_MODEL)
        self.iteration_results = []
        
    def fuzz_text(self, text: str, fuzz_level: float = 0.1) -> str:
        """Apply controlled fuzzing to text to create variations.
        
        Args:
            text: Original text to fuzz
            fuzz_level: Level of fuzzing (0.0 to 1.0)
            
        Returns:
            Fuzzed text
        """
        if fuzz_level <= 0:
            return text
            
        # Split into sentences
        sentences = text.split('. ')
        fuzzed_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Apply different fuzzing techniques based on fuzz_level
            if random.random() < fuzz_level * 0.3:
                # Synonym replacement
                sentence = self._apply_synonym_replacement(sentence)
                
            if random.random() < fuzz_level * 0.2:
                # Word order variation
                sentence = self._apply_word_order_variation(sentence)
                
            if random.random() < fuzz_level * 0.2:
                # Add/remove articles
                sentence = self._apply_article_variation(sentence)
                
            if random.random() < fuzz_level * 0.3:
                # Paraphrase structure
                sentence = self._apply_paraphrase_variation(sentence)
                
            fuzzed_sentences.append(sentence)
        
        return '. '.join(fuzzed_sentences)
    
    def _apply_synonym_replacement(self, text: str) -> str:
        """Apply synonym replacement to text using shared master synonym map."""
        try:
            from shared_utils.regex_utils import load_synonym_map
        except Exception:
            load_synonym_map = None  # type: ignore

        # Load external synonym map if available; fallback to no-op
        synonym_map: Dict[str, List[str]] = {}
        if load_synonym_map is not None:
            try:
                synonym_map = load_synonym_map()
            except Exception:
                synonym_map = {}

        if not synonym_map:
            return text

        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            # Find any key whose token appears in the current word/phrase
            for key, syns in synonym_map.items():
                if not syns:
                    continue
                if key in word_lower:
                    if random.random() < 0.3:
                        replacement = random.choice(syns)
                        if word and word[0].isupper():
                            replacement = replacement.capitalize()
                        words[i] = replacement
                    break

        return ' '.join(words)
    
    def _apply_word_order_variation(self, text: str) -> str:
        """Apply word order variations to text."""
        # Simple passive/active voice variation
        if random.random() < 0.5:
            # Convert to passive voice where possible
            if 'is' in text.lower() or 'are' in text.lower():
                return text  # Already passive-like
            else:
                # Add passive construction
                words = text.split()
                if len(words) > 3:
                    # Simple passive transformation
                    return f"This {words[0].lower()} is {words[1]} by {words[2]}"
        
        return text
    
    def _apply_article_variation(self, text: str) -> str:
        """Apply article variations to text."""
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ['a', 'an', 'the']:
                if random.random() < 0.3:
                    # Remove article
                    words[i] = ''
                elif random.random() < 0.2:
                    # Change article
                    if word.lower() == 'a':
                        words[i] = 'an' if i + 1 < len(words) and words[i + 1][0].lower() in 'aeiou' else 'a'
                    elif word.lower() == 'an':
                        words[i] = 'a'
        
        return ' '.join(word for word in words if word)
    
    def _apply_paraphrase_variation(self, text: str) -> str:
        """Apply paraphrase variations to text."""
        # Simple paraphrase patterns
        paraphrases = [
            ("This describes", "This explains"),
            ("This shows", "This demonstrates"),
            ("This includes", "This contains"),
            ("This provides", "This offers"),
            ("This enables", "This allows"),
            ("This improves", "This enhances"),
            ("This supports", "This facilitates"),
            ("This uses", "This employs"),
            ("This implements", "This applies"),
            ("This analyzes", "This examines")
        ]
        
        for original, replacement in paraphrases:
            if original.lower() in text.lower():
                if random.random() < 0.4:
                    text = text.replace(original, replacement)
                    break
        
        return text
    
    def generate_llm_query(self, original_text: str, context: str = "") -> str:
        """Generate an LLM query based on original text and context.
        
        Args:
            original_text: Original text to base query on
            context: Additional context for the query
            
        Returns:
            Generated LLM query
        """
        if self.llm_client is None:
            # Fallback to template-based query generation
            return self._generate_template_query(original_text, context)
        
        # Use LLM to generate query
        prompt = f"""
        Based on the following text, generate a search query that would help find similar or related information:
        
        Original text: "{original_text}"
        Context: {context}
        
        Generate a concise search query (1-2 sentences) that captures the key concepts and would help find similar information.
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}")
            return self._generate_template_query(original_text, context)
    
    def _generate_template_query(self, original_text: str, context: str = "") -> str:
        """Generate a template-based query when LLM is not available."""
        # Extract key terms and create a query
        words = original_text.split()
        
        # Find technical terms (words with more than 6 characters or common tech terms)
        tech_terms = []
        for word in words:
            word_clean = word.lower().strip('.,!?;:')
            if len(word_clean) > 6 or word_clean in ['neural', 'network', 'learning', 'algorithm', 'system', 'model', 'data', 'analysis']:
                tech_terms.append(word_clean)
        
        # Create query from key terms
        if tech_terms:
            key_terms = list(set(tech_terms))[:5]  # Top 5 unique terms
            query = f"Find information about {' '.join(key_terms)}"
        else:
            # Fallback to first few words
            query = f"Find information about {' '.join(words[:5])}"
        
        return query
    
    def search_with_llm_response(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content using LLM-generated query.
        
        Args:
            query: LLM-generated query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Generate embedding for the query
            query_embedding = embed([query], model_name="all-MiniLM-L6-v2")[0]
            
            # Search in both public and private indexes
            results = []
            
            # Search public index
            if self.barrier_analyzer.public_builder and self.barrier_analyzer.public_builder.chunks:
                public_results = self._search_in_chunks(
                    query_embedding, 
                    self.barrier_analyzer.public_builder.chunks,
                    'public',
                    top_k
                )
                results.extend(public_results)
            
            # Search private index
            if self.barrier_analyzer.private_builder and self.barrier_analyzer.private_builder.chunks:
                private_results = self._search_in_chunks(
                    query_embedding,
                    self.barrier_analyzer.private_builder.chunks,
                    'private',
                    top_k
                )
                results.extend(private_results)
            
            # Sort by distance and return top_k
            results.sort(key=lambda x: x['distance'])
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching with LLM response: {e}")
            return []
    
    def _search_in_chunks(self, query_embedding: List[float], chunks: List, chunk_type: str, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar chunks using query embedding."""
        results = []
        
        for chunk in chunks:
            if hasattr(chunk, 'embedding') and chunk.embedding:
                distance = self.barrier_analyzer.calculate_cosine_distance(query_embedding, chunk.embedding)
                
                result = {
                    'distance': distance,
                    'chunk_type': chunk_type,
                    'chunk_id': chunk.id,
                    'content': chunk.content if hasattr(chunk, 'content') else chunk.text,
                    'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
                
                if chunk_type == 'public' and hasattr(chunk, 'source_type'):
                    result['source_type'] = chunk.source_type.value
                
                results.append(result)
        
        # Sort by distance and return top_k
        results.sort(key=lambda x: x['distance'])
        return results[:top_k]
    
    def run_iterative_search(self, barrier_result: BarrierProbeResult, iterations: int = 3, top_k: int = 10) -> Dict[str, Any]:
        """Run iterative LLM-based search to find closer matches.
        
        Args:
            barrier_result: Results from barrier analysis
            iterations: Number of iterations to run
            top_k: Number of top results to consider in each iteration
            
        Returns:
            Results of iterative search
        """
        start_time = time.time()
        logger.info(f"Starting iterative LLM search with {iterations} iterations")
        
        # Get initial closest matches
        initial_matches = barrier_result.metadata.get('closest_matches', [])
        if not initial_matches:
            logger.warning("No initial matches found for iterative search")
            return {
                'success': False,
                'message': "No initial matches found",
                'iterations': [],
                'best_matches': [],
                'improvement': 0.0
            }
        
        # Track best matches across iterations
        best_matches = initial_matches.copy()
        best_avg_distance = sum(match['distance'] for match in best_matches) / len(best_matches)
        
        iteration_results = []
        
        for iteration in range(iterations):
            logger.info(f"Starting iteration {iteration + 1}/{iterations}")
            
            # Get current best matches
            current_matches = best_matches[:top_k]
            
            # Generate fuzzed queries for each match
            fuzzed_queries = []
            for match in current_matches:
                # Fuzz the content
                original_content = match.get('public_content', '') or match.get('private_content', '')
                fuzzed_content = self.fuzz_text(original_content, fuzz_level=0.15)
                
                # Generate LLM query
                context = f"Iteration {iteration + 1}, looking for similar information to: {original_content[:100]}..."
                query = self.generate_llm_query(fuzzed_content, context)
                
                fuzzed_queries.append({
                    'original_match': match,
                    'fuzzed_content': fuzzed_content,
                    'generated_query': query
                })
            
            # Search with each fuzzed query
            iteration_matches = []
            for query_info in fuzzed_queries:
                query_results = self.search_with_llm_response(
                    query_info['generated_query'], 
                    top_k=top_k
                )
                
                # Add query info to results
                for result in query_results:
                    result['query_info'] = query_info
                    result['iteration'] = iteration + 1
                
                iteration_matches.extend(query_results)
            
            # Remove duplicates and sort by distance
            unique_matches = self._remove_duplicate_matches(iteration_matches)
            unique_matches.sort(key=lambda x: x['distance'])
            
            # Check if we found better matches
            if unique_matches:
                current_avg_distance = sum(match['distance'] for match in unique_matches[:top_k]) / min(len(unique_matches), top_k)
                
                if current_avg_distance < best_avg_distance:
                    best_matches = unique_matches[:top_k]
                    best_avg_distance = current_avg_distance
                    logger.info(f"Iteration {iteration + 1}: Found better matches (avg distance: {current_avg_distance:.4f})")
                else:
                    logger.info(f"Iteration {iteration + 1}: No improvement (avg distance: {current_avg_distance:.4f})")
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'queries_generated': len(fuzzed_queries),
                'matches_found': len(unique_matches),
                'best_avg_distance': best_avg_distance,
                'top_matches': unique_matches[:5] if unique_matches else []
            }
            iteration_results.append(iteration_result)
        
        # Calculate overall improvement
        initial_avg_distance = sum(match['distance'] for match in initial_matches[:top_k]) / min(len(initial_matches), top_k)
        improvement = initial_avg_distance - best_avg_distance
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'iterations': iterations,
            'processing_time': processing_time,
            'initial_avg_distance': initial_avg_distance,
            'final_avg_distance': best_avg_distance,
            'improvement': improvement,
            'improvement_percentage': (improvement / initial_avg_distance * 100) if initial_avg_distance > 0 else 0,
            'best_matches': best_matches[:top_k],
            'iteration_results': iteration_results,
            'total_queries_generated': sum(ir['queries_generated'] for ir in iteration_results)
        }

    def refine_barrier_result(self, barrier_result: BarrierProbeResult, top_k: int = 5) -> BarrierProbeResult:
        """Rerun LLM queries for suspicious pairs and update scores in result."""
        if not barrier_result.potential_breaches:
            logger.info("No potential breaches to refine")
            return barrier_result

        for breach in barrier_result.potential_breaches:
            base_text = breach.get('public_content') or breach.get('private_content') or ''
            query = self.generate_llm_query(base_text)
            results = self.search_with_llm_response(query, top_k=top_k)
            breach['llm_query'] = query
            breach['llm_results'] = results
            if results:
                breach['llm_refined_distance'] = results[0]['distance']

        barrier_result.metadata['iterative_llm_search'] = {
            'breaches_refined': len(barrier_result.potential_breaches)
        }
        return barrier_result
    
    def _remove_duplicate_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate matches based on chunk ID."""
        seen_ids = set()
        unique_matches = []
        
        for match in matches:
            chunk_id = match.get('chunk_id')
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_matches.append(match)
        
        return unique_matches


def run_iterative_llm_search(
    barrier_result: BarrierProbeResult,
    barrier_analyzer: BarrierAnalyzer,
    iterations: int = 3,
    top_k: int = 10,
    llm_client=None
) -> Dict[str, Any]:
    """Convenience function to run iterative LLM search.
    
    Args:
        barrier_result: Results from barrier analysis
        barrier_analyzer: Barrier analyzer with loaded indexes
        iterations: Number of iterations to run
        top_k: Number of top results to consider
        llm_client: LLM client for generating responses
        
    Returns:
        Results of iterative search
    """
    searcher = IterativeLLMSearch(barrier_analyzer, llm_client)
    return searcher.run_iterative_search(barrier_result, iterations, top_k)


def refine_suspicious_pairs(
    barrier_result: BarrierProbeResult,
    barrier_analyzer: BarrierAnalyzer,
    top_k: int = 5,
    llm_client=None,
) -> BarrierProbeResult:
    """Convenience wrapper to refine suspicious pairs in a barrier result."""
    searcher = IterativeLLMSearch(barrier_analyzer, llm_client)
    return searcher.refine_barrier_result(barrier_result, top_k)
