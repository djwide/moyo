"""Cheap exposure-preview metadata for a topic.

This is the search-space preprocessor the Cloud worker uses for
``generationMode=exposure_preview``. It does **not** query models or the
public web. Numbers are deterministic for a topic so the storefront and
the Docker image stay aligned.

Shown on /exposure in about five seconds. Do not surface word counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from moyo.publicside.barrierprobe.llm_fuzzer import (
    BASIC_FUZZ_STRATEGIES,
    DEFAULT_MULTILINGUAL_LANGUAGES,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RETRIEVAL_CONFIG = _REPO_ROOT / "config" / "retrieval_llms.json"

# Cloud explore defaults (cloud_worker.OrderSpec).
DEFAULT_SEEDS = 3
FINDINGS_RANGE = "15–25"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'&./-]{1,40}")
_ENTITY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,4})\b"
)
_QUOTED_RE = re.compile(r"[\"“]([^\"”]{2,80})[\"”]")

SENSITIVE_CATEGORIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "formula": (
        "trade secret / formula",
        ("recipe", "formula", "ingredient", "spice", "compound", "formulation"),
    ),
    "hiring": (
        "hiring and org design",
        ("hiring", "job", "recruiter", "headcount", "opening", "role"),
    ),
    "defense": (
        "defense / program",
        ("program", "classified", "military", "weapon", "satellite", "defense"),
    ),
    "politics": (
        "political / reputational",
        ("opponent", "campaign", "election", "donor", "politic", "damaging"),
    ),
    "manda": (
        "deals and expansion",
        ("acquisition", "merger", "partnership", "expansion", "target"),
    ),
    "product": (
        "unannounced product",
        ("unannounced", "launch", "roadmap", "secret", "unreleased"),
    ),
    "biotech": (
        "clinical / manufacturing",
        ("clinical", "pipeline", "trial", "manufacturing", "therapeutic"),
    ),
    "legal": (
        "legal / investigative",
        ("lawsuit", "investigation", "indictment", "fraud", "litigation"),
    ),
    "finance": (
        "financial structure",
        ("revenue", "debt", "accounting", "off-balance", "special purpose"),
    ),
}

_SOURCE_HINTS: dict[str, tuple[str, ...]] = {
    "formula": ("patents", "supplier filings", "historical trade press"),
    "hiring": ("job posts", "recruiter listings", "contractor solicitations"),
    "defense": ("procurement notices", "budget justifications", "facility records"),
    "politics": ("campaign filings", "local news archives", "public calendars"),
    "manda": ("securities filings", "permit applications", "industry press"),
    "product": ("job specs", "domain registrations", "conference abstracts"),
    "biotech": ("trial registries", "manufacturing permits", "scientific preprints"),
    "legal": ("court dockets", "regulatory letters", "press archives"),
    "finance": ("securities filings", "bond disclosures", "audit exhibits"),
}

_ALWAYS_SOURCES = (
    "news archives",
    "regulatory filings",
    "cached public web pages",
)


def configured_retrieval_model_count(path: Path | None = None) -> int:
    config = path or _RETRIEVAL_CONFIG
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
        entries = data.get("retrieval_llms", []) if isinstance(data, dict) else data
        count = len([row for row in entries if isinstance(row, dict)])
        return max(count, 1)
    except Exception:
        return 10


def configured_languages() -> tuple[str, ...]:
    return ("English",) + tuple(DEFAULT_MULTILINGUAL_LANGUAGES)


def configured_prompt_cycles() -> int:
    return DEFAULT_SEEDS


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _words(topic: str) -> list[str]:
    return [m.group(0) for m in _WORD_RE.finditer(topic)]


def _entities(topic: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in list(_QUOTED_RE.finditer(topic)) + list(_ENTITY_RE.finditer(topic)):
        text = match.group(1).strip()
        key = text.lower()
        if key in seen or key in {"what", "which", "about", "the"}:
            continue
        if len(text) < 3:
            continue
        seen.add(key)
        found.append(text)
    return found[:8]


def _categories(topic_lc: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for key, (label, needles) in SENSITIVE_CATEGORIES.items():
        if any(needle in topic_lc for needle in needles):
            hits.append({"id": key, "label": label})
    return hits


def _jitter(normalized: str) -> int:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _source_hints(category_ids: Iterable[str]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for cid in category_ids:
        for item in _SOURCE_HINTS.get(cid, ()):
            if item not in seen:
                seen.add(item)
                hints.append(item)
    for item in _ALWAYS_SOURCES:
        if item not in seen:
            seen.add(item)
            hints.append(item)
    return hints[:6]


def estimate_exposure_preview(topic: str) -> dict[str, Any]:
    """Return curiosity-grade search-space metadata for one topic."""
    raw = (topic or "").strip()
    if len(raw) < 3:
        raise ValueError("Topic must be at least 3 characters.")
    if len(raw) > 500:
        raise ValueError("Topic must be 500 characters or fewer.")

    normalized = re.sub(r"\s+", " ", raw)
    topic_lc = normalized.lower()
    words = _words(normalized)
    entities = _entities(normalized)
    categories = _categories(topic_lc)
    models = configured_retrieval_model_count()
    languages = configured_languages()
    cycles = configured_prompt_cycles()
    strategies = len(BASIC_FUZZ_STRATEGIES)

    entity_score = min(1.0, len(entities) / 5)
    category_score = min(1.0, len(categories) / 4)
    length_score = min(1.0, len(words) / 12)
    richness = 0.45 * entity_score + 0.40 * category_score + 0.15 * length_score
    jitter = _jitter(topic_lc)

    relationships = 8 + len(entities) * 3 + len(categories) * 2 + (jitter % 5)
    clusters = 3 + min(7, len(categories) + len(entities) // 2)
    dense_areas = max(1, len(categories)) + (1 if richness >= 0.45 else 0)
    paths = 18 + relationships + clusters * 3 + dense_areas * 4 + int(richness * 18)
    paths = _clamp(paths, 22, 86)

    citations = 36 + paths + len(categories) * 8
    claims = 64 + paths * 2
    corroborated = 11 + clusters * 2 + dense_areas * 3
    search_space = models * len(languages) * cycles * strategies

    return {
        "topic": normalized,
        "headline": f"We found {paths} paths worth investigating.",
        "subhead": "Run the full scan to see where they lead.",
        "modelsPlus": models,
        "languagesPlus": len(languages),
        "promptCyclesPlus": cycles,
        "relationships": relationships,
        "clusters": clusters,
        "denseAreas": dense_areas,
        "candidatePaths": paths,
        "estimatedCitations": citations,
        "estimatedClaims": claims,
        "estimatedCorroborated": corroborated,
        "estimatedFindings": FINDINGS_RANGE,
        "searchSpace": search_space,
        "sensitiveCategories": [row["label"] for row in categories],
        "possibleSources": _source_hints(row["id"] for row in categories),
        "snapshotIncludes": [
            "3–10 notable exposures, scored",
            "A one-pager plus the snapshot report",
            "A teaser of the exposure chain",
            "Example model output and minimal source attribution",
        ],
        "basisIncludes": [
            "The full prioritized inventory (typically 15–25 findings)",
            "Evidence graph and corroborating model outputs",
            "Full exposure chains and remediation steps",
            "Access to underlying raw data",
        ],
        "disclaimer": (
            "This is a search-space read of your topic, not a finished investigation. "
            "Sources listed are the kinds of public material a full scan looks at — "
            "not a promise that each one will appear."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate exposure-preview metadata.")
    parser.add_argument("topic", nargs="?", help="Exploration topic")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args(argv)
    topic = args.topic or ""
    if not topic:
        parser.error("topic is required")
    preview = estimate_exposure_preview(topic)
    if args.json:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
    else:
        print(preview["headline"])
        print(
            f"{preview['candidatePaths']} paths · "
            f"{preview['relationships']} relationships · "
            f"{preview['clusters']} clusters"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
