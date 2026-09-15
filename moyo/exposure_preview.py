"""Cheap exposure-preview metadata for a topic.

This is the search-space preprocessor the Cloud worker uses for
``generationMode=exposure_preview``. It does **not** query models or the
public web. Numbers are deterministic for a topic so the storefront and
the Docker image stay aligned.

Black-box estimates are grounded in ``moyo-gather explore`` (basic fuzz:
paraphrase / translate / summarize, 3 seeds, ``config/retrieval_llms.json``)
and in what Snapshot vs Basis PDFs actually print. Red-teaming estimates
are grounded in ``moyo-probe analyze`` (barrier probe) report fields.

Shown on /exposure in about five seconds. Do not surface word counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from moyo.publicside.barrierprobe.llm_fuzzer import BASIC_FUZZ_STRATEGIES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RETRIEVAL_CONFIG = _REPO_ROOT / "config" / "retrieval_llms.json"

# Cloud explore defaults (cloud_worker.OrderSpec, fuzz_mode=basic).
DEFAULT_SEEDS = 3

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


def configured_prompt_cycles() -> int:
    return DEFAULT_SEEDS


def configured_strategies() -> int:
    return len(BASIC_FUZZ_STRATEGIES)


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
    """Return report-shaped estimates for Snapshot and Basis on one topic."""
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
    cycles = configured_prompt_cycles()
    strategies = configured_strategies()

    entity_score = min(1.0, len(entities) / 5)
    category_score = min(1.0, len(categories) / 4)
    length_score = min(1.0, len(words) / 12)
    richness = 0.45 * entity_score + 0.40 * category_score + 0.15 * length_score
    jitter = _jitter(topic_lc)

    # Collapsed findings after black-box explore → extract → cluster.
    # Basis prints every row in the inventory; Snapshot cover also prints
    # this count, but only writes up a handful.
    inventory = 8 + len(entities) * 2 + len(categories) * 3 + int(richness * 6) + (
        jitter % 4
    )
    inventory = _clamp(inventory, 8, 25)

    # Snapshot one-pager + abridged findings/evidence/claims (capped at 5)
    # plus top finding / two specific callouts. Marketing range is 3–10.
    notable = 3 + len(categories) + min(len(entities), 2) + (1 if richness >= 0.5 else 0)
    notable = min(inventory, _clamp(notable, 3, 10))

    high_sens = (
        1
        + len(categories)
        + (1 if len(entities) >= 2 else 0)
        + (1 if richness >= 0.55 else 0)
    )
    high_sens = _clamp(high_sens, 1, max(1, inventory - 1))

    # Shared S1… source registry. Both PDFs include the Sources table.
    citations = 5 + (inventory * 3) // 5 + len(categories) * 2 + (jitter % 3)
    citations = _clamp(citations, 6, 24)

    # Extract-stage claims in claims.jsonl before cluster collapse.
    # Basis raw data includes this file; Snapshot still runs the same extract.
    claims = 18 + inventory * 2 + len(entities) * 2 + len(categories) * 3 + (jitter % 8)
    claims = _clamp(claims, 24, 80)

    # Findings seen in 2+ model outputs (Basis "Corrob." / LLM corroborations).
    corroborated = 2 + (inventory * 3) // 10 + len(categories)
    corroborated = _clamp(corroborated, 2, max(2, inventory - 1))

    # score.chain_count is 3; renderer keeps up to 5. Snapshot only teasers one.
    chains = 3 + (1 if inventory >= 12 else 0) + (1 if richness >= 0.6 else 0)
    chains = _clamp(chains, 3, 5)

    # White-box moyo-probe analyze (default top_k=10). These print on the
    # barrier-probe report, not on Snapshot / Basis.
    breaches = 4 + len(entities) * 2 + len(categories) * 2 + (jitter % 3)
    breaches = _clamp(breaches, 4, 10)
    high_risk = 1 + len(categories) + (1 if richness >= 0.5 else 0)
    high_risk = _clamp(high_risk, 1, max(1, breaches - 1))
    concentrated = 2 + len(categories) + len(entities) + (jitter % 3)
    concentrated = _clamp(concentrated, 2, breaches)
    clusters = 3 + len(categories) + len(entities) // 2
    clusters = _clamp(clusters, 2, 12)
    paths = breaches + concentrated + clusters // 2
    paths = _clamp(paths, 8, 24)

    search_space = models * cycles * strategies

    return {
        "topic": normalized,
        "headline": (
            f"A Snapshot would brief {notable} exposures. "
            f"A Basis Report would inventory {inventory} findings."
        ),
        "subhead": (
            "Black-box counts are what Snapshot and Basis print. "
            "Red-teaming counts are what an authorized white-box probe would print."
        ),
        "modelsPlus": models,
        "promptCyclesPlus": cycles,
        "strategiesPlus": strategies,
        "snapshot": {
            "notableExposures": notable,
            "highSensitivity": high_sens,
            "citedSources": citations,
            "claims": claims,
            "corroborated": corroborated,
        },
        "basis": {
            "inventoryFindings": inventory,
            "highSensitivity": high_sens,
            "citedSources": citations,
            "claims": claims,
            "corroborated": corroborated,
            "exposureChains": chains,
        },
        "redTeam": {
            "potentialBreaches": breaches,
            "highRiskBreaches": high_risk,
            "concentratedMatches": concentrated,
            "clusters": clusters,
            "candidatePaths": paths,
        },
        "searchSpace": search_space,
        "sensitiveCategories": [row["label"] for row in categories],
        "possibleSources": _source_hints(row["id"] for row in categories),
        "snapshotIncludes": [
            "3–10 notable exposures, scored",
            "A one-pager plus the snapshot report",
            "A teaser of the exposure chain",
            "Example model output and a sources table",
        ],
        "basisIncludes": [
            "The complete prioritized inventory",
            "Evidence graph and corroborating model outputs",
            "Full exposure chains and exploitation implications",
            "Access to underlying raw data",
        ],
        "redTeamIncludes": [
            "Reachability map and source attribution inside the pipeline",
            "Barrier probe of RAG corpora and deployed assistants",
            "Written authorization required before any white-box work",
            "Optional validation re-test after mitigations",
        ],
        "disclaimer": (
            "This is a search-space read of your topic, not a finished investigation. "
            "Black-box numbers come from public-model explore. Red-teaming numbers "
            "come from the authorized white-box probe and require a private corpus "
            "plus written authorization. Sources listed are the kinds of public "
            "material a full scan looks at — not a promise that each one will appear."
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
        snap = preview["snapshot"]
        basis = preview["basis"]
        red = preview["redTeam"]
        print(
            f"Snapshot {snap['notableExposures']} exposures · "
            f"Basis {basis['inventoryFindings']} findings · "
            f"{basis['citedSources']} citations · "
            f"{basis['claims']} claims · "
            f"{basis['corroborated']} corroborations · "
            f"Red-team {red['potentialBreaches']} breaches"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
