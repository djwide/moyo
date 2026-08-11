"""Regex utilities for building and managing regex patterns with Aho-Corasick optimization."""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import Aho-Corasick, fallback gracefully if not available
try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    ahocorasick = None
    AHOCORASICK_AVAILABLE = False
    logger.warning("pyahocorasick not available, falling back to regex-only matching")

def load_synonym_map(synonym_file: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load an optional synonym map for regex expansion.

    The shared ``data/synonym_map.json`` file has been removed. Callers that
    need synonyms should supply them inline or pass ``synonym_file`` explicitly.
    With no file, returns an empty map (token regexes fall back to the token
    itself).
    """
    if synonym_file is None:
        return {}
    try:
        if synonym_file.exists():
            with open(synonym_file, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning("Synonym map file not found: %s", synonym_file)
        return {}
    except Exception as e:
        logger.error("Error loading synonym map from %s: %s", synonym_file, e)
        return {}

# Load synonym map lazily to avoid import issues
SYNONYM_MAP = None

def _get_synonym_map():
    """Get synonym map, loading it if not already loaded."""
    global SYNONYM_MAP
    if SYNONYM_MAP is None:
        SYNONYM_MAP = load_synonym_map()
    return SYNONYM_MAP


def build_regex_pattern(text: str, impact: int = 2) -> str:
    """Return a regex that matches ``text`` with minor variations.
    
    Args:
        text: Text to convert to regex pattern
        impact: Impact level (not used in current implementation)
        
    Returns:
        Regex pattern string
    """
    # Import text processing functions
    from .text_processing import normalize_text, TextNormalizationConfig
    
    # Normalize text for consistent pattern matching
    config = TextNormalizationConfig(
        lowercase=True,
        normalize_unicode=True,
        normalize_whitespace=True,
        remove_urls=False,  # Keep URLs for pattern matching
        remove_emails=False,  # Keep emails for pattern matching
        normalize_punctuation=True,
        keep_punctuation=True,
        remove_special_chars=False
    )
    
    normalized_text = normalize_text(text, config)
    
    # Check for literal patterns (base64, hex, etc.)
    if re.fullmatch(r"[A-Za-z0-9+/=]+", normalized_text) or re.fullmatch(r"[0-9A-Fa-f]+", normalized_text):
        return re.escape(normalized_text)
    
    # Extract tokens using improved tokenization
    tokens = re.findall(r"\w+", normalized_text)
    if not tokens:
        return re.escape(normalized_text)
    
    groups = []
    synonym_map = _get_synonym_map()
    for tok in tokens:
        # Look for exact match first, then try singular/plural forms
        synonyms = synonym_map.get(tok) or synonym_map.get(tok.rstrip("s"), [tok])
        group = "(?:" + "|".join(map(re.escape, synonyms)) + ")"
        groups.append(group)
    
    return r"\b" + r"\W*".join(groups) + r"\b"


def is_literal(text: str) -> bool:
    """Check if a text pattern is a pure literal (no special regex tokens).
    
    Args:
        text: Text to check
        
    Returns:
        True if the text is a literal pattern
    """
    return not re.search(r"[.\^$*+?{}\[\]\\|()]", text)


def update_regex_rules(combined: Path, regex_path: Path) -> None:
    """Update regex rules from combined text file.
    
    Args:
        combined: Path to combined text file
        regex_path: Path to regex rules JSON file
    """
    # Import text processing functions
    from .text_processing import deduplicate_texts, DeduplicationConfig, normalize_text, TextNormalizationConfig
    
    if regex_path.exists():
        try:
            rules = json.loads(regex_path.read_text())
        except Exception:
            rules = {}
    else:
        rules = {}

    file_lines = combined.read_text().splitlines()
    new_lines = [line for line in filter_comments(file_lines) if line.strip()]
    
    # Normalize and deduplicate lines before processing
    config = TextNormalizationConfig(
        lowercase=True,
        normalize_unicode=True,
        normalize_whitespace=True,
        remove_urls=False,
        remove_emails=False,
        normalize_punctuation=True,
        keep_punctuation=True,
        remove_special_chars=False
    )
    
    # Normalize all lines
    normalized_lines = [normalize_text(line.strip(), config) for line in new_lines if line.strip()]
    
    # Deduplicate lines
    dedup_config = DeduplicationConfig(
        exact_duplicates=True,
        similar_duplicates=True,
        similarity_threshold=0.95,
        case_sensitive=False
    )
    
    unique_lines, duplicates_removed = deduplicate_texts(normalized_lines, dedup_config)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate lines during regex rule generation")
    
    for line in unique_lines:
        if line.strip():
            pattern = build_regex_pattern(line.strip())
            rules[pattern] = {
                "msg": "User-defined pattern",
                "literal": is_literal(pattern),
                "impact": 2,  # Default to moderate impact
            }
            exact = re.escape(line.strip())
            rules[exact] = {
                "msg": "User-defined pattern",
                "literal": is_literal(exact),
                "impact": 2,  # Default to moderate impact
            }

    regex_path.write_text(json.dumps(rules, indent=4))
    logger.info(f"Updated regex rules from {combined} (processed {len(unique_lines)} unique patterns)")


def update_regex_rules_from_path(source: Path, regex_path: Path, impact: int = 2) -> None:
    """Update regex rules from a source file or directory.
    
    Args:
        source: Source file or directory path
        regex_path: Path to regex rules JSON file
        impact: Impact level for patterns
    """
    if regex_path.exists():
        try:
            rules = json.loads(regex_path.read_text())
        except Exception:
            rules = {}
    else:
        rules = {}
    
    if source.is_file():
        paths = [source]
    else:
        paths = [p for p in source.rglob("*") if p.is_file()]

    for file_path in paths:
        try:
            lines = file_path.read_text().splitlines()
            for line in lines:
                if line.strip():
                    pattern = build_regex_pattern(line.strip(), impact)
                    rules[pattern] = {
                        "msg": "User-defined pattern",
                        "literal": is_literal(pattern),
                        "impact": impact,
                    }
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")

    regex_path.write_text(json.dumps(rules, indent=4))
    logger.info(f"Updated regex rules from {source}")


def filter_comments(lines: List[str]) -> List[str]:
    """Filter out lines that start with '#' (comments).
    
    Args:
        lines: List of text lines
        
    Returns:
        Lines with comments filtered out
    """
    return [line for line in lines if not line.strip().startswith("#")]


class AhoCorasickMatcher:
    """Aho-Corasick automaton for efficient multi-pattern string matching."""
    
    def __init__(self):
        """Initialize the Aho-Corasick matcher."""
        self.automaton = None
        self.literal_patterns: Dict[str, Dict] = {}
        self.regex_patterns: List[Tuple[re.Pattern, Dict]] = []
        self._compiled = False
    
    def add_patterns(self, patterns: Dict[str, Dict]) -> None:
        """Add patterns to the matcher.
        
        Args:
            patterns: Dictionary of pattern strings to metadata
        """
        if not AHOCORASICK_AVAILABLE:
            # Fallback to regex-only mode
            for pattern, metadata in patterns.items():
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    self.regex_patterns.append((compiled_pattern, metadata))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            return
        
        # Separate literal and regex patterns
        literal_patterns = {}
        regex_patterns = []
        
        for pattern, metadata in patterns.items():
            if is_literal(pattern):
                literal_patterns[pattern] = metadata
            else:
                try:
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    regex_patterns.append((compiled_pattern, metadata))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        # Build Aho-Corasick automaton for literal patterns
        if literal_patterns:
            self.automaton = ahocorasick.Automaton()
            for pattern, metadata in literal_patterns.items():
                self.automaton.add_word(pattern.lower(), (pattern, metadata))
            self.automaton.make_automaton()
            self.literal_patterns = literal_patterns
        
        self.regex_patterns = regex_patterns
        self._compiled = True
    
    def find_matches(self, text: str) -> List[Tuple[str, Dict, int, int]]:
        """Find all matches in the given text.
        
        Args:
            text: Text to search for patterns
            
        Returns:
            List of tuples: (pattern, metadata, start_pos, end_pos)
        """
        if not self._compiled:
            return []
        
        matches = []
        
        # Use Aho-Corasick for literal patterns
        if self.automaton and self.literal_patterns:
            for end_index, (pattern, metadata) in self.automaton.iter(text.lower()):
                start_index = end_index - len(pattern) + 1
                matches.append((pattern, metadata, start_index, end_index + 1))
        
        # Use regex for complex patterns
        for pattern, metadata in self.regex_patterns:
            for match in pattern.finditer(text):
                matches.append((match.group(), metadata, match.start(), match.end()))
        
        # Sort by start position
        matches.sort(key=lambda x: x[2])
        return matches
    
    def has_match(self, text: str) -> bool:
        """Check if any pattern matches in the given text.
        
        Args:
            text: Text to check
            
        Returns:
            True if any pattern matches
        """
        if not self._compiled:
            return False
        
        # Check literal patterns with Aho-Corasick
        if self.automaton and self.literal_patterns:
            for _ in self.automaton.iter(text.lower()):
                return True
        
        # Check regex patterns
        for pattern, _ in self.regex_patterns:
            if pattern.search(text):
                return True
        
        return False


@lru_cache(maxsize=1)
def get_aho_corasick_matcher() -> AhoCorasickMatcher:
    """Get a singleton Aho-Corasick matcher instance.
    
    Returns:
        AhoCorasickMatcher instance
    """
    return AhoCorasickMatcher()


def load_regex_rules(rules_path: Path) -> Dict[str, Dict]:
    """Load regex rules from a JSON file.
    
    Args:
        rules_path: Path to regex rules JSON file
        
    Returns:
        Dictionary of pattern strings to metadata
    """
    if not rules_path.exists():
        return {}
    
    try:
        with open(rules_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading regex rules from {rules_path}: {e}")
        return {}


def build_aho_corasick_matcher(rules_path: Path) -> AhoCorasickMatcher:
    """Build an Aho-Corasick matcher from regex rules file.
    
    Args:
        rules_path: Path to regex rules JSON file
        
    Returns:
        Configured AhoCorasickMatcher instance
    """
    patterns = load_regex_rules(rules_path)
    matcher = AhoCorasickMatcher()
    matcher.add_patterns(patterns)
    return matcher


def match_text_optimized(text: str, rules_path: Path) -> List[Tuple[str, Dict, int, int]]:
    """Match text against patterns using optimized Aho-Corasick + regex approach.
    
    Args:
        text: Text to match against
        rules_path: Path to regex rules JSON file
        
    Returns:
        List of matches: (pattern, metadata, start_pos, end_pos)
    """
    matcher = build_aho_corasick_matcher(rules_path)
    return matcher.find_matches(text)


def match_text_simple(text: str, patterns: Dict[str, Dict]) -> List[Tuple[str, Dict, int, int]]:
    """Match text against patterns using simple regex approach (fallback).
    
    Args:
        text: Text to match against
        patterns: Dictionary of pattern strings to metadata
        
    Returns:
        List of matches: (pattern, metadata, start_pos, end_pos)
    """
    matches = []
    
    for pattern_str, metadata in patterns.items():
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                matches.append((match.group(), metadata, match.start(), match.end()))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern_str}': {e}")
    
    # Sort by start position
    matches.sort(key=lambda x: x[2])
    return matches


def combine_regex_rules_master(static_hits_dir: Path, master_file: Path) -> None:
    """Combine all regex rule files in static_hits_dir into master_file.
    
    - Overwrites master_file completely
    - Assigns deterministic rule ids (rule_1, rule_2, ...)
    - Preserves/uses each rule's msg field if present
    """
    combined_rules = {
        "_meta": {
            "version": "0.1.0",
            "spdx_license": "Apache-2.0",
            "notes": "Auto-generated master regex rules from component files and internal patterns. If a rule provided an id, it may be preserved in value as 'id'; top-level keys remain the regex patterns.",
            "provenance": []
        }
    }
    
    # Fresh overwrite: remove existing file first
    try:
        if master_file.exists():
            master_file.unlink()
    except Exception:
        pass
    
    # Get all JSON files in the directory, excluding the master file
    json_files = [f for f in static_hits_dir.glob("*.json") if f.name != master_file.name]
    
    aggregated: dict[str, dict] = {}
    for json_file in sorted(json_files, key=lambda p: p.name):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_rules = json.load(f)
            
            # Merge rules; skip metadata
            for pattern, rule_data in file_rules.items():
                if pattern == "_meta":
                    continue
                if isinstance(rule_data, dict):
                    aggregated[pattern] = {
                        "msg": rule_data.get("msg", "User-defined pattern"),
                        "literal": bool(rule_data.get("literal", False)),
                        "impact": rule_data.get("impact", 2),
                        # keep any provided id for later reference (not used as key)
                        **({"id": rule_data.get("id")} if rule_data.get("id") else {})
                    }
                else:
                    aggregated[pattern] = {"msg": str(rule_data), "literal": False, "impact": 2}
            
            combined_rules["_meta"]["provenance"].append(f"Combined from: {json_file.name}")
            logger.info(f"Added rules from {json_file.name}")
        except Exception as e:
            logger.warning(f"Failed to process {json_file}: {e}")
    
    # Assign deterministic rule numbers (used by linter if values include id)
    for idx, (pattern, data) in enumerate(aggregated.items(), start=1):
        # Set id inside the value so linter can prefer it
        data.setdefault("id", f"rule_{idx}")
        combined_rules[pattern] = data
    
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(combined_rules, f, indent=2)
    
    logger.info(f"Combined {len(json_files)} files into master regex rules: {master_file}")
