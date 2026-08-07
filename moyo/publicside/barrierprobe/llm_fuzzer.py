"""LLM-assisted fuzzing for barrier probing.

Defaults to a locally running Ollama model (``llama3.1:8b``) for *generation*.
Semantic similarity / FAISS neighbour lookup uses MiniLM embeddings
(``all-MiniLM-L6-v2``) via ``shared_utils.embed``.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
import random

from shared_utils import (
    embed,
    FAISSIndex,
    normalize_text,
    TextNormalizationConfig,
    get_logger
)

logger = get_logger(__name__)

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# basic = paraphrase only; full = English transforms (no translation);
# full-multilingual = full transforms plus translation into target languages.
BASIC_FUZZ_STRATEGIES = ("paraphrase",)
FULL_FUZZ_STRATEGIES = ("abstract", "summarize", "typo")
MULTILINGUAL_FUZZ_STRATEGIES = FULL_FUZZ_STRATEGIES + ("translate",)
FUZZ_STRATEGIES = BASIC_FUZZ_STRATEGIES + MULTILINGUAL_FUZZ_STRATEGIES
FUZZ_MODES = ("basic", "full", "full-multilingual")

# Default target languages for ``full-multilingual`` mode: Spanish, French,
# and Mainland (Simplified) Chinese. Callers may append additional languages.
DEFAULT_MULTILINGUAL_LANGUAGES = ("Spanish", "French", "Chinese (Simplified)")


def strategies_for_fuzz_mode(mode: str) -> List[str]:
    """Resolve fuzz strategies for a fuzz mode.

    - ``basic`` -> paraphrase only
    - ``full`` -> abstract / summarize / typo (English only, no translation)
    - ``full-multilingual`` -> the ``full`` transforms plus translation
    """
    key = (mode or "basic").strip().lower()
    if key == "basic":
        return list(BASIC_FUZZ_STRATEGIES)
    if key == "full":
        return list(FULL_FUZZ_STRATEGIES)
    if key == "full-multilingual":
        return list(MULTILINGUAL_FUZZ_STRATEGIES)
    raise ValueError(f"Unknown fuzz mode {mode!r}; expected one of {FUZZ_MODES}")


@dataclass
class LocalizedText:
    """English report text with optional foreign-language provenance."""

    english: str
    original_language: Optional[str] = None
    original_text: Optional[str] = None

    @property
    def was_translated(self) -> bool:
        lang = (self.original_language or "").strip().lower()
        return bool(lang) and lang not in {"english", "en", "eng"}

    def for_report(self) -> str:
        """Markdown suitable for exploration / fuzz reports."""
        body = (self.english or "").strip()
        if not self.was_translated:
            return body
        lang = self.original_language.strip()
        original = (self.original_text or "").strip()
        parts = [f"_Translated from {lang}_", "", body]
        if original and original != body:
            parts.extend(
                [
                    "",
                    f"<details>",
                    f"<summary>Original ({lang})</summary>",
                    "",
                    original,
                    "",
                    "</details>",
                ]
            )
        return "\n".join(parts)


_ENGLISH_HINT_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "with",
        "that",
        "this",
        "from",
        "as",
        "by",
        "it",
        "not",
        "at",
        "have",
        "has",
        "had",
        "you",
        "your",
        "we",
        "they",
        "their",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "can",
        "will",
        "would",
        "should",
        "about",
        "into",
        "than",
        "then",
        "also",
        "more",
        "most",
        "some",
        "any",
        "all",
        "if",
        "but",
        "so",
        "do",
        "does",
        "did",
        "been",
        "being",
        "there",
        "these",
        "those",
        "such",
        "only",
        "other",
        "over",
        "after",
        "before",
        "between",
        "through",
        "during",
        "without",
        "within",
        "under",
        "again",
        "further",
        "once",
        "here",
        "each",
        "few",
        "own",
        "same",
        "too",
        "very",
        "just",
        "because",
        "while",
        "where",
        "although",
        "however",
        "therefore",
        "information",
        "recipe",
        "public",
        "known",
        "source",
    }
)


def _looks_like_english(text: str) -> bool:
    """Heuristic: mostly Latin letters plus common English function words."""
    sample = (text or "").strip()
    if not sample:
        return True
    # Non-Latin scripts (Cyrillic, CJK, Arabic, etc.) → not English.
    non_latin = sum(
        1
        for ch in sample
        if ch.isalpha() and ord(ch) > 0x024F and not (0x1E00 <= ord(ch) <= 0x1EFF)
    )
    letters = sum(1 for ch in sample if ch.isalpha())
    if letters and non_latin / letters > 0.08:
        return False
    tokens = [t.lower() for t in sample.replace("\n", " ").split() if t.isalpha()]
    if len(tokens) < 4:
        # Short snippets: treat as English if overwhelmingly ASCII letters.
        ascii_letters = sum(1 for ch in sample if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        return (not letters) or (ascii_letters / max(letters, 1) > 0.92)
    hits = sum(1 for t in tokens if t in _ENGLISH_HINT_WORDS)
    return hits / len(tokens) >= 0.12


def _parse_localization_json(reply: str) -> Optional[Tuple[str, str]]:
    """Parse ``{"language":..., "english":...}`` from an LLM reply."""
    raw = (reply or "").strip()
    if not raw:
        return None
    # Strip optional markdown fences.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(raw[start : end + 1])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    language = data.get("language") or data.get("lang") or ""
    english = data.get("english") or data.get("translation") or data.get("text") or ""
    if not isinstance(language, str) or not isinstance(english, str):
        return None
    return language.strip(), english.strip()


class LocalLLMClient:
    """Offline synonym / pattern transformer (fallback when no LLM is available)."""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize the local transformer.
        
        Args:
            model_name: MiniLM embedding model used for similarity gating
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
    #   "local"     – embedding-only synonym transformer (offline fallback; not default)
    llm_provider: str = "ollama"
    model_name: str = DEFAULT_OLLAMA_MODEL
    api_key: Optional[str] = None
    # Endpoint for self-hosted / OpenAI-compatible providers (Ollama, custom).
    base_url: Optional[str] = DEFAULT_OLLAMA_BASE_URL
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Semantic Search Configuration
    search_k: int = 10  # Number of closest phrases to retrieve
    similarity_threshold: float = 0.8  # Minimum similarity to consider
    
    # Fuzzing Configuration
    max_iterations: int = 5  # Maximum fuzzing iterations
    target_similarity: float = 0.95  # Target similarity to achieve
    # MiniLM for similarity / FAISS neighbour lookup (generation stays on Ollama).
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # ``basic`` = paraphrase only; ``full`` = abstract/summarize/typo (English);
    # ``full-multilingual`` = full plus translation into ``multilingual_languages``.
    fuzz_mode: str = "basic"
    # Explicit override; when empty, derived from ``fuzz_mode``.
    fuzz_strategies: List[str] = field(default_factory=list)
    # Target languages for ``full-multilingual``; empty -> defaults (Spanish,
    # French, Mainland/Simplified Chinese).
    multilingual_languages: List[str] = field(default_factory=list)
    prompt_template: str = """
You are an expert at semantic text transformation for information retrieval and barrier probing.

Target concept: {target_concept}
Fuzz strategy for this step: {strategy}
Strategy instructions: {strategy_instructions}

Original phrase: {original_phrase}

Similar phrases from the corpus (optional context):
{similar_phrases}

Instructions:
1. Apply ONLY the named fuzz strategy to the original phrase.
2. Prefer keeping the phrase useful for retrieving information about the target concept.
3. Return only the transformed phrase, no explanations or quotes.

Transformed phrase:"""

    def __post_init__(self) -> None:
        mode = (self.fuzz_mode or "basic").strip().lower()
        if mode not in FUZZ_MODES:
            mode = "basic"
        self.fuzz_mode = mode
        if mode == "full-multilingual" and not self.multilingual_languages:
            self.multilingual_languages = list(DEFAULT_MULTILINGUAL_LANGUAGES)
        if not self.fuzz_strategies:
            self.fuzz_strategies = strategies_for_fuzz_mode(self.fuzz_mode)


STRATEGY_INSTRUCTIONS = {
    "paraphrase": (
        "Reword the phrase with different wording and syntax while preserving meaning. "
        "Keep the result in English."
    ),
    "translate": (
        "Translate the phrase into another natural language (prefer Spanish, French, "
        "or Mainland/Simplified Chinese). Return the foreign-language phrase only — do "
        "not translate it back to English in this step."
    ),
    "abstract": (
        "Raise the level of abstraction: replace concrete specifics with more general "
        "categories while keeping the core topic recognizable. Keep the result in English."
    ),
    "summarize": (
        "Compress the phrase into a shorter English summary that retains the key claim or ask."
    ),
    "typo": (
        "Introduce realistic intentional typos / keyboard-adjacent errors / mild "
        "misspellings while keeping the phrase readable to a human. Keep the language "
        "of the original phrase."
    ),
}

REWORD_SYSTEM = (
    "You are a research assistant that turns a user's information request into "
    "effective retrieval queries. You return only the queries, no commentary."
)


class LLMFuzzer:
    """LLM-assisted fuzzer for semantic barrier probing.

    White-box paths (``fuzz_phrase``) steer toward a known target concept.
    Black-box helpers such as :meth:`reword_for_retrieval` do not take a target
    concept — they only diversify a naive prompt for downstream probing.
    """
    
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

    @classmethod
    def local_ollama(
        cls,
        model_name: str = "llama3.1:8b",
        base_url: Optional[str] = None,
        max_tokens: int = 800,
        temperature: float = 0.7,
    ) -> "LLMFuzzer":
        """Build a fuzzer backed by a locally running Ollama model.

        Defaults to ``llama3.1:8b`` — the same local model used for black-box
        explore rewording.
        """
        return cls(
            LLMFuzzerConfig(
                llm_provider="ollama",
                model_name=model_name,
                base_url=base_url or OllamaClient.DEFAULT_BASE_URL,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    def _initialize_llm_client(self):
        """Initialize the LLM client based on configuration."""
        if self.config.llm_provider == "local":
            # Offline synonym transformer; still uses MiniLM for similarity gating.
            return LocalLLMClient(
                getattr(self.config, "embedding_model", DEFAULT_EMBEDDING_MODEL)
            )
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
    
    def query_llm(self, prompt: str, system: Optional[str] = None) -> Optional[str]:
        """Query the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            system: Optional system message (ignored by the embedding-only
                ``local`` provider). Defaults to a short transformation system
                prompt for chat providers.
            
        Returns:
            The LLM response or None if failed
        """
        if not self.llm_client:
            logger.error("LLM client not initialized")
            return None

        default_system = (
            "You are a helpful assistant for semantic text transformation. "
            "Return only the transformed phrase, with no preamble or explanation."
        )
        system_msg = system if system is not None else default_system
            
        try:
            if self.config.llm_provider == "local":
                # Extract information from the prompt for local transformation
                original_phrase, target_concept, similar_phrases = self._parse_fuzzing_prompt(prompt)
                text = self.llm_client.transform_text(original_phrase, target_concept, similar_phrases)
            elif self.config.llm_provider == "ollama":
                text = self.llm_client.generate(
                    prompt,
                    system=system_msg,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            elif self.config.llm_provider in ("openai", "custom"):
                response = self.llm_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
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
                    system=system_msg,
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

    def localize_to_english(self, text: str) -> LocalizedText:
        """Translate non-English text to English for report writing.

        Returns the original text unchanged (language unset) when the content
        is already English or localization fails.
        """
        raw = (text or "").strip()
        if not raw:
            return LocalizedText(english="")

        if _looks_like_english(raw):
            return LocalizedText(english=raw)

        ask = (
            "Detect the language of the text below. If it is not English, translate "
            "it into clear English. Reply with ONLY a JSON object on one line:\n"
            '{"language":"<English name of the source language>",'
            '"english":"<English translation>"}\n'
            'If the text is already English, use language "English" and put the '
            "original text in english unchanged.\n\n"
            f"TEXT:\n{raw[:6000]}"
        )
        try:
            reply = self.query_llm(
                ask,
                system=(
                    "You are a precise translation assistant. Output JSON only, "
                    "no markdown fences."
                ),
            ) or ""
        except Exception as exc:
            logger.warning("localize_to_english failed (%s); keeping original", exc)
            return LocalizedText(english=raw)

        parsed = _parse_localization_json(reply)
        if not parsed:
            return LocalizedText(english=raw)

        language, english = parsed
        if not english.strip():
            return LocalizedText(english=raw)
        if (language or "").strip().lower() in {"english", "en", "eng"}:
            return LocalizedText(english=english.strip())
        return LocalizedText(
            english=english.strip(),
            original_language=language.strip(),
            original_text=raw,
        )

    def text_for_report(self, text: str) -> str:
        """English report body with original-language annotation when needed."""
        return self.localize_to_english(text).for_report()

    @staticmethod
    def _parse_numbered_lines(text: str) -> List[str]:
        """Extract list items from a numbered/bulleted LLM response."""
        import re

        items: List[str] = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line)
            line = line.strip().strip('"').strip("'").strip()
            if line:
                items.append(line)
        return items

    @staticmethod
    def _augment_reword_seeds(prompt: str, seeds: List[str], n: int) -> List[str]:
        """Deterministically pad seeds when the LLM under-delivers."""
        base = prompt.strip().rstrip("?.")
        fallbacks = [
            base,
            f"{base} overview and key facts",
            f"detailed explanation of {base}",
            f"history and background of {base}",
            f"technical details and specifics of {base}",
            f"common questions and authoritative sources about {base}",
        ]
        out = list(seeds)
        seen = {s.lower() for s in out}
        for cand in fallbacks:
            if len(out) >= n:
                break
            if cand.lower() not in seen:
                out.append(cand)
                seen.add(cand.lower())
        return out

    def reword_for_retrieval(
        self,
        prompt: str,
        n: int = 5,
        fuzz_mode: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> List[str]:
        """Black-box reword a naive prompt into ``n`` retrieval query seeds.

        No target concept is used — explore / public-side probing is black-box
        and only needs diverse retrieval phrasings of the user's request.

        ``fuzz_mode``:
        - ``basic`` — paraphrase-only rewording (default)
        - ``full`` — rotate abstract / summarize / typo (English only)
        - ``full-multilingual`` — the ``full`` transforms plus one translated
          seed per target language (``languages`` or the configured/default
          Spanish, French, Mainland Chinese)
        """
        mode = (fuzz_mode or self.config.fuzz_mode or "basic").strip().lower()
        if mode not in FUZZ_MODES:
            mode = "basic"
        if mode == "full-multilingual":
            langs = [
                l.strip()
                for l in (languages or self.config.multilingual_languages
                          or DEFAULT_MULTILINGUAL_LANGUAGES)
                if l and l.strip()
            ]
            return self._reword_multilingual(prompt, n, langs)
        if mode == "full":
            return self._reword_with_strategies(
                prompt, n, strategies_for_fuzz_mode("full")
            )
        return self._reword_paraphrase_batch(prompt, n)

    def _reword_multilingual(
        self, prompt: str, n: int, languages: List[str]
    ) -> List[str]:
        """Seeds for ``full-multilingual``: one translation per language + English.

        Guarantees a translated seed for every requested language, then fills the
        remaining slots with English ``full`` transforms. The effective seed count
        is ``max(n, len(languages))`` so all requested languages are always covered.
        """
        langs = languages or list(DEFAULT_MULTILINGUAL_LANGUAGES)
        seeds: List[str] = []
        for lang in langs:
            translated = self._translate_prompt_to(prompt, lang)
            if translated:
                seeds.append(translated)

        target = max(max(1, n), len(seeds))
        remaining = target - len(seeds)
        if remaining > 0:
            seeds.extend(
                self._reword_with_strategies(
                    prompt, remaining, list(FULL_FUZZ_STRATEGIES)
                )
            )
        return self._dedupe_and_pad_seeds(prompt, seeds, target)

    def _translate_prompt_to(self, prompt: str, language: str) -> Optional[str]:
        """Translate the request into ``language`` as a retrieval query."""
        ask = (
            f"Translate the following information request into {language}. "
            f"Return only the {language} translation phrased as a search query, "
            "with no explanation, notes, or quotes.\n\n"
            f"Request: {prompt.strip()}"
        )
        try:
            text = self.query_llm(ask, system=REWORD_SYSTEM) or ""
        except Exception as exc:
            logger.warning("Translation to %s failed (%s)", language, exc)
            return None
        for raw in text.splitlines():
            cleaned = raw.strip().lstrip("0123456789.-)•* ").strip().strip('"').strip("'")
            if cleaned:
                return cleaned
        return None

    def _reword_paraphrase_batch(self, prompt: str, n: int) -> List[str]:
        """Generate ``n`` paraphrase-style retrieval queries in one LLM call."""
        ask = (
            f'A non-technical user asked: "{prompt}".\n'
            f"How can I reword this request to most effectively retrieve information? "
            f"Give me {n} different answers. Each should approach the topic from a "
            f"different angle. Return each reworded query on its own line, numbered "
            f"1 to {n}, with no extra commentary."
        )
        text = ""
        try:
            text = self.query_llm(ask, system=REWORD_SYSTEM) or ""
        except Exception as exc:
            logger.warning("Black-box rewording failed (%s); using deterministic seeds", exc)

        seeds = [s for s in self._parse_numbered_lines(text) if s]
        return self._dedupe_and_pad_seeds(prompt, seeds, n)

    def _reword_with_strategies(
        self, prompt: str, n: int, strategies: List[str]
    ) -> List[str]:
        """Build seeds by rotating named fuzz strategies (full explore mode)."""
        if not strategies:
            strategies = list(BASIC_FUZZ_STRATEGIES)
        seeds: List[str] = []
        for i in range(max(1, n)):
            strategy = strategies[i % len(strategies)]
            transformed = self._apply_blackbox_strategy(prompt, strategy)
            if transformed:
                seeds.append(transformed)
        return self._dedupe_and_pad_seeds(prompt, seeds, n)

    def _apply_blackbox_strategy(self, phrase: str, strategy: str) -> Optional[str]:
        """Apply one fuzz strategy to a prompt without a target concept."""
        strategy_key = (strategy or "paraphrase").lower()
        if strategy_key == "typo":
            # Prefer a light deterministic typo pass; fall back to LLM if empty.
            local = self._apply_intentional_typos(phrase.strip())
            if local and local.strip().lower() != phrase.strip().lower():
                return local.strip()

        instructions = STRATEGY_INSTRUCTIONS.get(
            strategy_key, STRATEGY_INSTRUCTIONS["paraphrase"]
        )
        ask = (
            "Transform the user's information request into one effective retrieval "
            f"query using ONLY the '{strategy_key}' strategy.\n"
            f"Strategy instructions: {instructions}\n\n"
            f"Original request: {phrase.strip()}\n\n"
            "Return only the transformed query, with no quotes or explanation."
        )
        try:
            text = self.query_llm(ask, system=REWORD_SYSTEM) or ""
        except Exception as exc:
            logger.warning(
                "Black-box strategy %s failed (%s)", strategy_key, exc
            )
            return None
        line = (text or "").strip().strip('"').strip("'")
        if not line:
            return None
        # If the model returned multiple lines, keep the first non-empty.
        for raw in line.splitlines():
            cleaned = raw.strip()
            cleaned = cleaned.lstrip("0123456789.-)•* ").strip().strip('"').strip("'")
            if cleaned:
                return cleaned
        return None

    def _dedupe_and_pad_seeds(
        self, prompt: str, seeds: List[str], n: int
    ) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for s in seeds:
            key = s.lower()
            if key not in seen:
                deduped.append(s)
                seen.add(key)
        if len(deduped) < n:
            deduped = self._augment_reword_seeds(prompt, deduped, n)
        return deduped[:n]

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
                           k: Optional[int] = None,
                           enforce_threshold: bool = True) -> List[Dict[str, Any]]:
        """Find semantically similar phrases in the index.
        
        Args:
            query: Query phrase to find similar phrases for
            index: FAISS index containing the corpus
            k: Number of results to return (defaults to config.search_k)
            enforce_threshold: When True (default), drop results below the
                configured similarity threshold. When False, return the top-k
                results regardless, each tagged with ``above_threshold`` so the
                caller can decide what to do with weaker matches.
            
        Returns:
            List of similar phrases with metadata, best first. Each entry
            includes ``similarity`` and an ``above_threshold`` flag.
        """
        if k is None:
            k = self.config.search_k
        
        # Normalize query
        normalized_query = normalize_text(query, self.normalization_config)
        
        # MiniLM embeddings for FAISS neighbour lookup.
        emb_model = getattr(self.config, "embedding_model", DEFAULT_EMBEDDING_MODEL)
        query_embeddings = embed([normalized_query], emb_model)
        if not query_embeddings:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search index
        distances, indices, metadata = index.search(query_embeddings[0], k=k)
        
        results = []
        for i, (distance, idx, meta) in enumerate(zip(distances, indices, metadata)):
            above_threshold = distance >= self.config.similarity_threshold
            if enforce_threshold and not above_threshold:
                continue
            results.append({
                "rank": i + 1,
                "similarity": distance,
                "above_threshold": above_threshold,
                "index": idx,
                "metadata": meta,
                "text": self._resolve_result_text(meta, index)
            })
        
        return results

    @staticmethod
    def _resolve_result_text(meta: Dict[str, Any], index: FAISSIndex) -> str:
        """Best-effort text for a hit: full text, then preview, then string store.

        Different builders populate metadata differently (``text`` from the
        public index builder, ``text_preview`` from the datainput builder), so
        fall back through the available fields and finally the string store.
        """
        text = meta.get("text") or meta.get("text_preview")
        if not text and meta.get("text_id") is not None:
            text = index.get_text_by_id(meta["text_id"])
        return text or ""
    
    def create_fuzzing_prompt(
        self,
        target_concept: str,
        original_phrase: str,
        similar_phrases: List[Dict[str, Any]],
        strategy: str = "paraphrase",
    ) -> str:
        """Create a prompt for LLM-assisted fuzzing with a named strategy."""
        phrases_text = ""
        for i, phrase_info in enumerate(similar_phrases[:5]):
            phrases_text += (
                f"{i+1}. {phrase_info['text']} "
                f"(similarity: {phrase_info['similarity']:.3f})\n"
            )
        strategy_key = (strategy or "paraphrase").lower()
        instructions = STRATEGY_INSTRUCTIONS.get(
            strategy_key, STRATEGY_INSTRUCTIONS["paraphrase"]
        )
        return self.config.prompt_template.format(
            target_concept=target_concept,
            original_phrase=original_phrase,
            similar_phrases=phrases_text or "(none)",
            strategy=strategy_key,
            strategy_instructions=instructions,
        )

    @staticmethod
    def _apply_intentional_typos(text: str, rate: float = 0.18) -> str:
        """Introduce light keyboard-adjacent typos without destroying readability."""
        if not text or len(text) < 3:
            return text
        neighbors = {
            "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr",
            "f": "dg", "g": "fh", "h": "gj", "i": "uo", "j": "hk",
            "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip",
            "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry",
            "u": "yi", "v": "cb", "w": "qe", "x": "zc", "y": "tu",
            "z": "x",
        }
        chars = list(text)
        # Typo ~rate of alphabetic characters (at least one when possible).
        alpha_idxs = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if not alpha_idxs:
            return text
        n_typos = max(1, int(len(alpha_idxs) * rate))
        for idx in random.sample(alpha_idxs, min(n_typos, len(alpha_idxs))):
            ch = chars[idx]
            opts = neighbors.get(ch.lower(), "")
            if not opts:
                continue
            repl = random.choice(opts)
            chars[idx] = repl.upper() if ch.isupper() else repl
        return "".join(chars)

    def _cosine(self, a: List[float], b: List[float]) -> float:
        denom = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        if not denom:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / denom

    def fuzz_phrase(
        self,
        original_phrase: str,
        target_concept: str,
        index: FAISSIndex,
    ) -> Tuple[str, float, List[str], List[Dict[str, str]]]:
        """Fuzz a phrase toward ``target_concept`` via rotating strategies.

        Mode ``basic`` uses paraphrase only; ``full`` rotates translate,
        abstract, summarize, and typo. Generation uses the configured LLM
        (Ollama by default); similarity is scored with MiniLM embeddings.
        """
        current_phrase = original_phrase
        transformation_history = [original_phrase]
        interactions: List[Dict[str, str]] = []
        strategies = list(self.config.fuzz_strategies or strategies_for_fuzz_mode(self.config.fuzz_mode))
        emb_model = getattr(self.config, "embedding_model", DEFAULT_EMBEDDING_MODEL)

        logger.info("Starting fuzzing for phrase: %r", original_phrase)
        logger.info("Target concept: %r", target_concept)
        logger.info("Fuzz mode=%s strategies=%s", self.config.fuzz_mode, strategies)

        for iteration in range(self.config.max_iterations):
            strategy = strategies[iteration % len(strategies)]
            logger.info(
                "Fuzzing iteration %d/%d [%s]",
                iteration + 1,
                self.config.max_iterations,
                strategy,
            )

            similar_phrases = self.find_similar_phrases(current_phrase, index)
            prompt = self.create_fuzzing_prompt(
                target_concept, current_phrase, similar_phrases or [], strategy=strategy
            )

            if strategy == "typo" and self.config.llm_provider == "local":
                fuzzed_phrase = self._apply_intentional_typos(current_phrase)
            else:
                fuzzed_phrase = self.query_llm(prompt)
                if strategy == "typo" and fuzzed_phrase:
                    # Reinforce with a light local typo pass if LLM stayed too clean.
                    if fuzzed_phrase.strip().lower() == current_phrase.strip().lower():
                        fuzzed_phrase = self._apply_intentional_typos(current_phrase)

            if not fuzzed_phrase:
                logger.warning("LLM query failed, stopping fuzzing")
                break

            interactions.append(
                {"prompt": prompt, "response": fuzzed_phrase, "strategy": strategy}
            )
            fuzzed_phrase = normalize_text(fuzzed_phrase, self.normalization_config)

            fuzzed_embeddings = embed([fuzzed_phrase], emb_model)
            target_embeddings = embed([target_concept], emb_model)
            if not fuzzed_embeddings or not target_embeddings:
                logger.warning("Failed to generate embeddings for similarity check")
                break

            similarity = self._cosine(fuzzed_embeddings[0], target_embeddings[0])
            logger.info(
                "Iteration %d [%s]: similarity = %.3f",
                iteration + 1,
                strategy,
                similarity,
            )

            transformation_history.append(fuzzed_phrase)
            current_phrase = fuzzed_phrase

            if similarity >= self.config.target_similarity:
                logger.info("Target similarity achieved: %.3f", similarity)
                return fuzzed_phrase, similarity, transformation_history, interactions

            time.sleep(0.2)

        if transformation_history:
            final_phrase = transformation_history[-1]
            final_embeddings = embed([final_phrase], emb_model)
            target_embeddings = embed([target_concept], emb_model)
            if final_embeddings and target_embeddings:
                final_similarity = self._cosine(final_embeddings[0], target_embeddings[0])
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

            localized = self.localize_to_english(fuzzed_phrase)
            history_report = [self.text_for_report(step) for step in history]

            results.append({
                "original_phrase": phrase,
                "fuzzed_phrase": fuzzed_phrase,
                "fuzzed_phrase_english": localized.english,
                "fuzzed_phrase_original_language": localized.original_language,
                "fuzzed_phrase_for_report": localized.for_report(),
                "final_similarity": similarity,
                "transformation_history": history,
                "transformation_history_for_report": history_report,
                "iterations": len(history) - 1,
                "target_concept": target_concept,
                "fuzz_mode": self.config.fuzz_mode,
                "fuzz_strategies": list(self.config.fuzz_strategies),
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
