"""Naive-prompt topic exploration via multiple LLMs.

Takes a plain-language request from a non-technical user (e.g. "give me all the
info you can on the recipe for Coca-Cola"), rewords it into several effective
retrieval queries via the runtime utility fuzzer (local Ollama on the desktop;
a hosted cheap model on Cloud Run — black-box, no target concept),
then explores each reworded query against every configured retrieval LLM
(closed API, open API) in parallel.

Pipeline:

1. Parallel raw retrieval (completion order ignored).
2. :func:`compile_raw_responses` — sort, organise, label, and translate foreign
   answers into a :class:`CompiledCorpus` (before any analysis).
3. Summary / claims analysis over the compiled corpus only.
4. Render ``exploration.md`` with raw compiled findings first, analysis last.

Public entry points:

- :func:`reword_prompt` -- naive prompt -> N reworded query "seeds" (local fuzzer).
- :func:`explore_topic` -- run the full fan-out and build the markdown report.
- :func:`explore_and_save` -- as above, then persist the markdown to disk.
- :func:`explore_and_save_many` -- run :func:`explore_and_save` for each prompt.
"""

from __future__ import annotations

import asyncio
import logging
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from moyo.llm.client import LLMClient
from moyo.llm.registry import get_default_llm, get_retrieval_llms

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

# Provider-aware retrieval scheduler. The stock config has ~7 default models;
# a generic N-worker queue either serializes a host or stampedes it. Caps are
# global (all hosts) and per API host, with a circuit breaker so a failing
# provider stops receiving work for the rest of the process.
RETRIEVAL_GLOBAL_CONCURRENCY = 12  # ~10–14
RETRIEVAL_PER_PROVIDER_CONCURRENCY = 2  # 1–2
PROVIDER_CIRCUIT_FAILURE_THRESHOLD = 2

_CIRCUIT_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "rate limit",
    "rate_limit",
    "overloaded",
    "connection",
    "429",
    "502",
    "503",
    "529",
    "unauthorized",
    "invalid_api_key",
    "invalid api key",
    "insufficient",
    "quota",
    "circuit open",
)

RETRIEVAL_SYSTEM = (
    "You are a knowledgeable research assistant. Answer with factual, specific "
    "information about the query. Use short paragraphs or bullet points. If you "
    "are uncertain or lack reliable information, say so briefly rather than "
    "inventing details. When you rely on identifiable sources, include "
    "citations: prefer URLs when known, otherwise named documents, reports, "
    "papers, datasets, or archival references. Put them inline next to the "
    "relevant fact and/or in a short trailing `Sources:` / `References:` list. "
    "Do not invent URLs or document titles."
)
SUMMARY_SYSTEM = (
    "You are a research analyst. You synthesise multiple sources into one clear, "
    "well-organised markdown summary, attributing claims to their source when "
    "sources disagree or when a claim is notable. Prefer points of precision "
    "(exact numbers, quantities, dates, ratios, named ingredients/chemicals, "
    "identifiers, measurements) over vague generalities. You explicitly "
    "highlight disagreements between sources — conflicting facts, dates, "
    "quantities, or conclusions — and how different models diverge (what one "
    "family emphasises, omits, or uniquely asserts). When retrieval answers "
    "include citations (URLs, document titles, report names, Sources/References "
    "entries), you preserve them and attach them to the claims they support — "
    "never invent citations that do not appear in the source text."
)
CLAIMS_SUMMARY_SYSTEM = (
    "You are a research analyst who extracts and ranks factual claims from "
    "multi-source retrieval. You favour precision over generality: prefer "
    "exact values, quantities, dates, ratios, named entities, chemical/"
    "ingredient lists, and identifiers when the sources supply them. You are "
    "sceptical of single-source colour, explicit about corroboration, and "
    "careful to surface disagreements between sources (conflicting facts, "
    "dates, quantities, named entities, or conclusions) as well as "
    "distinctions between models (unique claims, omissions, conflicting "
    "frames). You preserve citations supplied by retrieval answers "
    "(URLs, document titles, report names, Sources/References entries) and "
    "attach them to the claims they support. Never invent citations."
)

# Local Ollama model used for explore/summarize synthesis (not retrieval fan-out).
DEFAULT_SUMMARY_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_SUMMARY_NUM_CTX = 32768
DEFAULT_SUMMARY_PER_SOURCE_CHARS = 700

# How "high impact" is defined for claims summaries (also embedded in the prompt).
# Domain-agnostic: classified, proprietary, and personal/sensitive information.
DEFAULT_HIGH_IMPACT_DEFINITION = (
    "A claim is high-impact when being right or wrong about it would "
    "substantially change how someone understands exposure, risk, or next "
    "actions regarding classified, proprietary, or personal/sensitive "
    "information on the topic. Prefer claims that: "
    "(1) reveal or specifically describe protected content itself "
    "(secrets, formulas, credentials, personal data, internal plans, "
    "source code, keys, medical or financial identifiers, and similar); "
    "(2) give load-bearing specifics (exact values, identifiers, dates, "
    "locations, quantities, named entities) that unlock or verify that content; "
    "(3) state who has custody or access, how it is protected, or its "
    "legal, classification, or regulatory status; "
    "(4) describe exfiltration, leakage, bypass, or reconstruction paths; "
    "(5) contradict official denials or widely repeated public narratives "
    "with a concrete alternative; or "
    "(6) change what a reader would do next to verify, contain, or use "
    "the information. "
    "Deprioritise atmospheric background, generic public restatements "
    "without new specificity, and stylistic colour."
)

# Back-compat alias.
HIGH_IMPACT_DEFINITION = DEFAULT_HIGH_IMPACT_DEFINITION


def build_impact_definition(
    extra: Optional[str] = None,
    extra_files: Optional[List[str]] = None,
) -> str:
    """Compose the base high-impact definition with optional user extras.

    Precedence of extras (all appended after the base, in order):
    ``MOYO_IMPACT_DEFINITION`` env, ``MOYO_IMPACT_DEFINITION_FILE`` env,
    ``extra_files`` paths, then ``extra`` text.
    """
    import os
    from pathlib import Path

    parts: List[str] = [DEFAULT_HIGH_IMPACT_DEFINITION]
    extras: List[str] = []

    env_text = (os.environ.get("MOYO_IMPACT_DEFINITION") or "").strip()
    if env_text:
        extras.append(env_text)

    env_file = (os.environ.get("MOYO_IMPACT_DEFINITION_FILE") or "").strip()
    file_paths = list(extra_files or [])
    if env_file:
        file_paths.insert(0, env_file)
    for raw_path in file_paths:
        path = Path(raw_path)
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read impact-definition file %s: %s", path, exc)
            continue
        if text:
            extras.append(text)

    if extra and extra.strip():
        extras.append(extra.strip())

    if not extras:
        return parts[0]
    joined = "\n\n".join(extras)
    return (
        f"{parts[0]}\n\n"
        "Additional user-specified impact criteria (treat as additive; "
        f"do not drop the base criteria):\n{joined}"
    )


@dataclass
class RetrievalResult:
    """One (seed x LLM) retrieval outcome."""

    seed: str
    llm_label: str
    provider: str
    model: str
    kind: str  # closed | open | local
    text: str = ""
    error: Optional[str] = None
    # Language the model was prompted in (the seed's language). ``None`` means
    # English / never translated. Non-English implies the response was translated
    # back to English for the report.
    language: Optional[str] = None
    # Fuzz strategy that produced the seed (original / paraphrase / typo / shuffle).
    strategy: Optional[str] = None
    # Original (untranslated) response body, kept when ``text`` was translated.
    original_text: Optional[str] = None
    # Redacted provider payload (no auth headers); written to provider_responses.jsonl.
    provider_record: Optional[Dict[str, Any]] = None
    # Stable indices from the job grid — used to compile the report in a
    # deterministic order regardless of parallel completion order.
    seed_index: int = 0
    llm_index: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())

    @property
    def source_label(self) -> str:
        """Attribution label: ``Kimi (Mandarin Chinese)`` when non-English, else bare label."""
        if _is_foreign_language(self.language):
            return f"{self.llm_label} ({self.language})"
        return self.llm_label


@dataclass
class CompiledQuery:
    """One reworded query and its per-LLM answers, after compile/localize."""

    seed_index: int
    text: str
    language: Optional[str]
    strategy: Optional[str]
    language_group: str
    results: List[RetrievalResult] = field(default_factory=list)


@dataclass
class CompiledCorpus:
    """Organized, labeled, English-localized raw retrieval responses.

    Built before any summary / claims analysis so the report's detailed body is
    independent of synthesis order or parallel completion order.
    """

    prompt: str
    fuzz_mode: str
    languages: Optional[List[str]]
    queries: List[CompiledQuery]
    llm_labels: List[str]

    @property
    def results(self) -> List[RetrievalResult]:
        out: List[RetrievalResult] = []
        for q in self.queries:
            out.extend(q.results)
        return out

    @property
    def seeds(self) -> List[str]:
        return [q.text for q in self.queries]


@dataclass
class ExploreResult:
    """Full outcome of a topic exploration."""

    prompt: str
    seeds: List[str]
    results: List[RetrievalResult]
    markdown: str
    summary: Optional[str] = None
    claims_summary: Optional[str] = None
    output_path: Optional[str] = None
    summary_path: Optional[str] = None
    llm_labels: List[str] = field(default_factory=list)
    llm_statuses: List[LLMStatus] = field(default_factory=list)
    retrieval_check_path: Optional[str] = None


# --- Prompt rewording -------------------------------------------------------
def _augment_seeds(prompt: str, n: int) -> List[str]:
    """Deterministic seed list when the local fuzzer is unavailable."""
    from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

    return LLMFuzzer._augment_reword_seeds(prompt, [], n)[:n]


def resolve_multilingual_languages(extra_languages: Optional[List[str]]) -> List[str]:
    """Combine the default multilingual targets with any user-requested extras.

    Defaults are Spanish, French and Mainland (Simplified) Chinese; ``extra_languages``
    are appended (de-duplicated, case-insensitive).
    """
    from moyo.publicside.barrierprobe.llm_fuzzer import DEFAULT_MULTILINGUAL_LANGUAGES

    languages = list(DEFAULT_MULTILINGUAL_LANGUAGES)
    seen = {l.lower() for l in languages}
    for lang in extra_languages or []:
        cleaned = (lang or "").strip()
        if cleaned and cleaned.lower() not in seen:
            languages.append(cleaned)
            seen.add(cleaned.lower())
    return languages


def resolve_explore_languages(
    extra_languages: Optional[List[str]],
    *,
    language_selection_explicit: bool = False,
) -> List[str]:
    """Languages for ``multilingual`` explore besides English.

    When ``language_selection_explicit`` is true (customer chose languages on
    the storefront), use only ``extra_languages`` — an empty list means
    English-only. Otherwise keep legacy GUI defaults via
    :func:`resolve_multilingual_languages`.
    """
    if language_selection_explicit:
        out: List[str] = []
        seen: set[str] = set()
        for lang in extra_languages or []:
            cleaned = (lang or "").strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                out.append(cleaned)
                seen.add(key)
        return out
    return resolve_multilingual_languages(extra_languages)


def reword_prompt(
    prompt: str,
    llm: Optional[LLMClient] = None,
    n: int = 3,
    fuzzer: Optional[Any] = None,
    fuzz_mode: str = "basic",
    languages: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
) -> List[str]:
    """Reword a naive ``prompt`` into distinct retrieval queries.

    Black-box: uses :class:`~moyo.publicside.barrierprobe.llm_fuzzer.LLMFuzzer`
    with the runtime utility LLM (local Ollama ``llama3.1:8b`` on the desktop;
    Vertex Gemini Flash on Cloud Run). No target concept is
    supplied — explore only diversifies the user's request for retrieval.

    ``fuzz_mode`` ``basic`` emits ``n`` English seeds rotating original /
    paraphrase (no translate); ``multilingual`` emits ``n`` seeds
    per language (English plus each extra language in ``languages``) rotating
    original / paraphrase. Pass ``strategies`` to override the mode default rotation
    a la carte (include ``abstract``, ``typo`` or ``shuffle`` explicitly). ``llm`` is ignored (kept for
    call-site compatibility).
    Pass ``fuzzer`` to inject a preconfigured :class:`LLMFuzzer`.
    """
    return [
        s.text
        for s in reword_prompt_seeds(
            prompt,
            llm=llm,
            n=n,
            fuzzer=fuzzer,
            fuzz_mode=fuzz_mode,
            languages=languages,
            strategies=strategies,
        )
    ]


def reword_prompt_tagged(
    prompt: str,
    llm: Optional[LLMClient] = None,
    n: int = 3,
    fuzzer: Optional[Any] = None,
    fuzz_mode: str = "basic",
    languages: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
) -> List[tuple]:
    """Back-compat: ``(seed, language)`` pairs from :func:`reword_prompt_seeds`."""
    return [
        (s.text, s.language)
        for s in reword_prompt_seeds(
            prompt,
            llm=llm,
            n=n,
            fuzzer=fuzzer,
            fuzz_mode=fuzz_mode,
            languages=languages,
            strategies=strategies,
        )
    ]


def reword_prompt_seeds(
    prompt: str,
    llm: Optional[LLMClient] = None,
    n: int = 3,
    fuzzer: Optional[Any] = None,
    fuzz_mode: str = "basic",
    languages: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
) -> List[Any]:
    """Return :class:`~moyo.publicside.barrierprobe.llm_fuzzer.QuerySeed` objects."""
    from moyo.publicside.barrierprobe.llm_fuzzer import (
        LLMFuzzer,
        QuerySeed,
        normalize_fuzz_mode,
    )

    del llm  # explore rewording uses the runtime utility fuzzer
    mode = normalize_fuzz_mode(fuzz_mode)
    if fuzzer is None:
        fuzzer = LLMFuzzer.for_runtime(
            fuzz_mode=mode,
            multilingual_languages=list(languages or []),
        )
    try:
        return list(
            fuzzer.reword_for_retrieval_seeds(
                prompt,
                n=n,
                fuzz_mode=mode,
                languages=languages,
                strategies=strategies,
            )
        )
    except Exception as exc:
        logger.warning("Local fuzzer rewording failed (%s); using deterministic seeds", exc)
        return [
            QuerySeed(text=s, language=None, strategy="paraphrase")
            for s in _augment_seeds(prompt, n)
        ]


def _is_foreign_language(language: Optional[str]) -> bool:
    lang = (language or "").strip().lower()
    return bool(lang) and lang not in {"english", "en", "eng"}


def _language_group_name(language: Optional[str]) -> str:
    """Display name for a language group in the exploration report."""
    if _is_foreign_language(language):
        return language.strip()  # type: ignore[union-attr]
    return "English"


# --- Report localization ----------------------------------------------------
def localize_text_for_report(text: str, fuzzer: Optional[Any] = None) -> str:
    """Translate non-English text to English and annotate the source language."""
    body = (text or "").strip()
    if not body:
        return ""
    if fuzzer is None:
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

        fuzzer = LLMFuzzer.for_runtime()
    try:
        return fuzzer.text_for_report(body)
    except Exception as exc:
        logger.warning("Report localization failed (%s); keeping original text", exc)
        return body


def _localize_one_result(result: RetrievalResult, fuzzer: Any) -> None:
    """Translate one foreign-language result to English in place."""
    if not result.ok or not _is_foreign_language(result.language):
        return
    try:
        english = (fuzzer.localize_to_english(result.text).english or "").strip()
    except Exception as exc:
        logger.warning("Report localization failed (%s); keeping original text", exc)
        return
    if english and english != result.text.strip():
        result.original_text = result.text
        result.text = english


def _parallel_localize_results(
    pending: List[RetrievalResult],
    fuzzer: Any,
    workers: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
) -> None:
    """Translate foreign-language results in parallel (capped by ``workers``)."""
    total = len(pending)
    if not total or fuzzer is None:
        return

    max_workers = total if workers is None else max(1, int(workers))
    max_workers = min(max_workers, total)
    done = 0
    lock = threading.Lock()

    def _run(result: RetrievalResult) -> None:
        nonlocal done
        _localize_one_result(result, fuzzer)
        with lock:
            done += 1
            if progress:
                progress(
                    f"Translating [{done}/{total}] "
                    f"{result.source_label} / seed {result.seed_index + 1} ..."
                )

    if max_workers == 1:
        for result in pending:
            _run(result)
        return

    if progress:
        progress(
            f"Translating {total} foreign-language answer(s) "
            f"with {max_workers} worker(s) ..."
        )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run, result) for result in pending]
        for fut in as_completed(futures):
            fut.result()


def localize_results_for_report(
    results: List[RetrievalResult],
    fuzzer: Optional[Any] = None,
    progress: Optional[ProgressFn] = None,
    workers: Optional[int] = None,
) -> None:
    """In-place: translate foreign-language retrieval bodies back to English.

    Prefer :func:`compile_raw_responses`, which organises and labels results
    while translating. This helper remains for callers that only need localization.
    ``workers`` caps concurrent translation calls (default: one per pending
    result). Pass ``workers=1`` for sequential translation.
    """
    from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

    if fuzzer is None:
        fuzzer = LLMFuzzer.for_runtime()

    pending = [r for r in results if r.ok and _is_foreign_language(r.language)]
    _parallel_localize_results(
        pending, fuzzer, workers=workers, progress=progress
    )


def compile_raw_responses(
    prompt: str,
    query_seeds: List[Any],
    raw_results: List[RetrievalResult],
    retrieval_llms: List[LLMClient],
    fuzz_mode: str = "basic",
    languages: Optional[List[str]] = None,
    fuzzer: Optional[Any] = None,
    progress: Optional[ProgressFn] = None,
    workers: Optional[int] = None,
) -> CompiledCorpus:
    """Compile, organise, label, and localize raw retrieval responses.

    This runs *before* any summary / claims analysis. Parallel fetch order is
    discarded: results are sorted by ``(language group, seed_index, llm_index)``,
    foreign answers are translated to English here (concurrently, capped by
    ``workers``), and each query is labeled with its strategy and language group.
    """
    def _report(msg: str) -> None:
        if progress:
            progress(msg)

    lang_order = _language_group_order(languages)
    lang_rank = {name: i for i, name in enumerate(lang_order)}

    # Stable sort — never rely on ThreadPool completion order.
    sorted_results = sorted(
        raw_results,
        key=lambda r: (
            lang_rank.get(_language_group_name(r.language), 10_000),
            r.seed_index,
            r.llm_index,
        ),
    )

    foreign = [r for r in sorted_results if r.ok and _is_foreign_language(r.language)]
    total_foreign = len(foreign)
    if total_foreign and fuzzer is None:
        _report(
            f"Compiling {len(sorted_results)} raw response(s); "
            f"skipping translation of {total_foreign} foreign-language answer(s) "
            "(no local fuzzer)"
        )
    elif total_foreign:
        translate_workers = (
            total_foreign if workers is None else max(1, int(workers))
        )
        translate_workers = min(translate_workers, total_foreign)
        _report(
            f"Compiling {len(sorted_results)} raw response(s); "
            f"translating {total_foreign} foreign-language answer(s) "
            f"with {translate_workers} worker(s) ..."
        )
        _parallel_localize_results(
            foreign, fuzzer, workers=workers, progress=progress
        )
    else:
        _report(
            f"Compiling {len(sorted_results)} raw response(s); "
            "no foreign-language answers to translate"
        )

    # Group by seed_index into labeled queries, preserving seed definition order
    # within each language group via the sort above.
    by_seed: Dict[int, CompiledQuery] = {}
    for i, qs in enumerate(query_seeds):
        by_seed[i] = CompiledQuery(
            seed_index=i,
            text=qs.text,
            language=qs.language,
            strategy=qs.strategy,
            language_group=_language_group_name(qs.language),
            results=[],
        )
    for result in sorted_results:
        query = by_seed.get(result.seed_index)
        if query is None:
            query = CompiledQuery(
                seed_index=result.seed_index,
                text=result.seed,
                language=result.language,
                strategy=result.strategy,
                language_group=_language_group_name(result.language),
                results=[],
            )
            by_seed[result.seed_index] = query
        query.results.append(result)

    # Emit queries in language-group order, then seed_index.
    queries = sorted(
        by_seed.values(),
        key=lambda q: (
            lang_rank.get(q.language_group, 10_000),
            q.seed_index,
        ),
    )
    _report(
        f"Compiled corpus: {len(queries)} quer(ies) × "
        f"{len(retrieval_llms)} LLM(s), labeled and localized"
    )
    return CompiledCorpus(
        prompt=prompt,
        fuzz_mode=(fuzz_mode or "basic").strip().lower(),
        languages=list(languages) if languages else None,
        queries=queries,
        llm_labels=[llm.label for llm in retrieval_llms],
    )


# --- Retrieval fan-out ------------------------------------------------------
def _provider_record_for_result(
    result: RetrievalResult,
    llm: LLMClient,
    prompt: str,
    record: Optional[Dict[str, Any]],
    *,
    error: Optional[str],
) -> Dict[str, Any]:
    """Attach scan metadata to a redacted provider payload (no auth headers)."""
    from moyo.llm.content_filter import redact_secrets

    row: Dict[str, Any] = dict(record or {})
    row.update(
        {
            "seed": result.seed,
            "seed_index": result.seed_index,
            "llm_index": result.llm_index,
            "strategy": result.strategy,
            "language": result.language,
            "llm_label": result.llm_label,
            "provider": result.provider or row.get("provider") or llm.spec.provider,
            "model": result.model or row.get("model") or llm.spec.model,
            "error": error,
            "content": result.text or row.get("content") or "",
        }
    )
    # Never persist the retrieval prompt's system key or Authorization-like fields.
    row.pop("prompt", None)
    row.pop("messages", None)
    return redact_secrets(row)


def retrieve(
    seed: str,
    llm: LLMClient,
    max_tokens: Optional[int] = None,
    language: Optional[str] = None,
    strategy: Optional[str] = None,
    seed_index: int = 0,
    llm_index: int = 0,
) -> RetrievalResult:
    """Query a single LLM with a single reworded seed.

    ``language`` is the seed's language. For a non-English seed the model is
    prompted entirely in that language (and is expected to answer in it); the
    response is translated back to English during :func:`compile_raw_responses`.
    """
    if _is_foreign_language(language):
        prompt = (
            f"Please answer entirely in {language}. "
            "Provide all the factual information you can about the following query. "
            "Be specific and comprehensive; prefer concrete facts, names, dates and "
            "figures. Use short paragraphs or bullet points.\n\n"
            f"Query ({language}): {seed}"
        )
    else:
        prompt = (
            "Provide all the factual information you can about the following query. "
            "Be specific and comprehensive; prefer concrete facts, names, dates and "
            "figures. Use short paragraphs or bullet points.\n\n"
            f"Query: {seed}"
        )
    result = RetrievalResult(
        seed=seed,
        llm_label=llm.label,
        provider=llm.spec.provider,
        model=llm.spec.model,
        kind=llm.kind,
        language=language,
        strategy=strategy,
        seed_index=seed_index,
        llm_index=llm_index,
    )
    try:
        from moyo.llm.content_filter import strip_reasoning_spill

        if hasattr(llm, "complete_result"):
            outcome = llm.complete_result(
                prompt, system=RETRIEVAL_SYSTEM, max_tokens=max_tokens
            )
            raw_text = outcome.text or ""
            record = outcome.provider_record
        else:
            raw_text = llm.complete(prompt, system=RETRIEVAL_SYSTEM, max_tokens=max_tokens) or ""
            record = llm.last_provider_record() if hasattr(llm, "last_provider_record") else None
        result.text = strip_reasoning_spill(raw_text)
        result.provider_record = _provider_record_for_result(
            result, llm, prompt, record, error=None
        )
    except Exception as exc:
        from moyo.llm.client import format_llm_error

        result.error = format_llm_error(exc)
        record = None
        if hasattr(llm, "last_provider_record"):
            record = llm.last_provider_record()
        result.provider_record = _provider_record_for_result(
            result, llm, prompt, record, error=result.error
        )
        logger.warning("Retrieval failed for %s via %s: %s", seed[:60], llm.label, result.error)
    return result


def get_summary_llm(override: Optional[LLMClient] = None) -> LLMClient:
    """LLM used for exploration narrative + claims summaries.

    Defaults to local Ollama (``llama3.1:8b``) on the desktop, or Vertex
    Gemini Flash on Cloud Run, with an enlarged ``num_ctx`` so
    multi-source corpora fit. Override with a configured client (e.g. CLI
    ``--provider``), or via ``MOYO_SUMMARY_MODEL`` / ``MOYO_SUMMARY_NUM_CTX`` /
    ``MOYO_SUMMARY_BASE_URL``. Under ``--test`` / ``MOYO_TEST_MODE``, returns
    an offline echo client.
    """
    if override is not None:
        return override

    import os

    from moyo.llm.client import LLMSpec, ensure_env_loaded

    try:
        from moyo.llm.testing import is_test_mode, test_llm_spec
        if is_test_mode():
            return LLMClient(test_llm_spec())
    except Exception:
        pass

    from moyo.llm.utility import running_in_cloud, utility_llm_spec

    ensure_env_loaded()

    if running_in_cloud():
        spec = utility_llm_spec()
        spec.max_tokens = int(os.environ.get("MOYO_SUMMARY_MAX_TOKENS") or "2500")
        spec.timeout = int(os.environ.get("MOYO_SUMMARY_TIMEOUT") or "600")
        spec.temperature = 0.3
        return LLMClient(spec)

    model = (os.environ.get("MOYO_SUMMARY_MODEL") or DEFAULT_SUMMARY_OLLAMA_MODEL).strip()
    base_url = (
        os.environ.get("MOYO_SUMMARY_BASE_URL")
        or os.environ.get("MOYO_OLLAMA_BASE_URL")
        or "http://127.0.0.1:11434"
    ).strip()
    try:
        num_ctx = int(os.environ.get("MOYO_SUMMARY_NUM_CTX") or DEFAULT_SUMMARY_NUM_CTX)
    except ValueError:
        num_ctx = DEFAULT_SUMMARY_NUM_CTX
    try:
        timeout = int(os.environ.get("MOYO_SUMMARY_TIMEOUT") or "600")
    except ValueError:
        timeout = 600
    try:
        max_tokens = int(os.environ.get("MOYO_SUMMARY_MAX_TOKENS") or "2500")
    except ValueError:
        max_tokens = 2500

    return LLMClient(
        LLMSpec(
            provider="ollama",
            model=model,
            base_url=base_url,
            label=f"Local Ollama {model} (summary)",
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=timeout,
            num_ctx=num_ctx,
        )
    )


def _summary_per_source_chars(llm: LLMClient) -> int:
    """Tighter per-source budget for local Ollama context windows."""
    import os

    raw = (os.environ.get("MOYO_SUMMARY_PER_SOURCE_CHARS") or "").strip()
    if raw:
        try:
            return max(200, int(raw))
        except ValueError:
            pass
    if (llm.spec.provider or "").lower() == "ollama":
        return DEFAULT_SUMMARY_PER_SOURCE_CHARS
    return 1500


def _synthesize_summary(
    prompt: str, results: List[RetrievalResult], llm: LLMClient
) -> Optional[str]:
    """Ask the summary LLM to consolidate all findings into one summary."""
    usable = [r for r in results if r.ok]
    if not usable:
        return None

    corpus = _corpus_blocks(usable, per_source_chars=_summary_per_source_chars(llm))
    ask = (
        f'The user originally asked: "{prompt}".\n\n'
        "Below are answers gathered from several different LLMs for several "
        "reworded versions of the request. Synthesise them into one cohesive "
        "markdown summary of everything learned about the subject. Organise by "
        "theme, prefer precise specifics (numbers, quantities, dates, named "
        "ingredients/chemicals, identifiers) over vague generalities, note "
        "where sources agree or disagree, and flag anything that seems "
        "uncertain or speculative. Include a short `## Points of precision` "
        "section listing the most concrete specifics found. Include a dedicated "
        "`## Points of disagreement` section that lists concrete conflicts "
        "between sources (e.g. different dates, quantities, named entities, "
        "or conclusions), naming which Source labels take each side. In "
        "addition to the well-corroborated themes, explicitly call out any "
        "outlier or unusual responses — claims made by only one source, "
        "surprising or contradictory answers, or notably divergent takes "
        "(including from sources queried in another language). In that outlier "
        "discussion, highlight when some models give higher specificity or "
        "exact numbers / dates / quantities / identifiers while others speak "
        "only in generalities.\n\n"
        "Include a dedicated section titled `## Distinctions between models` that "
        "highlights how models differ: unique specifics one model alone offered, "
        "claims one model refused or hedged while others answered, conflicting "
        "frames or emphasis, and any language-specific divergence. Name each "
        "model with its Source label.\n\n"
        "When naming sources that attest to a claim, use the Source label exactly "
        "as given. If a source is tagged with a non-English language "
        "(e.g. `Kimi (Mandarin Chinese)`), keep that language annotation in the "
        "attribution so readers can see which attestations came from a "
        "non-English query. If a claim has 3 or more attesting sources, put the "
        "label list in an HTML `<details><summary>Attesting sources (N)</summary>` "
        "block so the main line stays short.\n\n"
        "If any attesting response cites supporting material (URLs, document / "
        "report titles, or entries under Sources/References/Citations — also "
        "surfaced as `[Citations from this source]` blocks below), attach those "
        "citations to the claim as `Citations: ...`. Deduplicate. Do not invent "
        "citations that are not present in the source text.\n\n"
        f"{corpus}"
    )
    try:
        text = llm.complete(ask, system=SUMMARY_SYSTEM, max_tokens=1500)
        return _collapse_source_lists(text) if text else None
    except Exception as exc:
        logger.warning("Summary synthesis failed (%s)", exc)
        return None


_URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+", re.IGNORECASE)
_CITATION_SECTION_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?(?:\*{0,2})?(?:key\s+)?"
    r"(?:sources?|references?|citations?|bibliography|works cited)"
    # Allow both `**Sources:**` (colon inside bold) and `**Sources**:`.
    r"(?:\*{0,2})?\s*:?\s*(?:\*{0,2})?\s*$"
)
_CITATION_BULLET_RE = re.compile(
    r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(?P<item>.+?)\s*$"
)


def _clean_citation(raw: str) -> str:
    text = (raw or "").strip().rstrip(".,;")
    text = re.sub(r"\*+", "", text).strip()
    return text


def _extract_citations(text: str, limit: int = 24) -> List[str]:
    """Pull URLs and Sources/References entries from a retrieval answer."""
    body = (text or "").strip()
    if not body:
        return []

    found: List[str] = []
    seen: set[str] = set()

    def _add(item: str) -> None:
        cleaned = _clean_citation(item)
        if not cleaned or len(cleaned) < 3:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        found.append(cleaned)

    for match in _URL_RE.finditer(body):
        _add(match.group(0))

    lines = body.splitlines()
    i = 0
    while i < len(lines):
        if not _CITATION_SECTION_RE.match(lines[i].strip()):
            i += 1
            continue
        i += 1
        blank_streak = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                blank_streak += 1
                i += 1
                # Two blank lines ends the citation block.
                if blank_streak >= 2:
                    break
                continue
            blank_streak = 0
            if line.startswith("#") or _CITATION_SECTION_RE.match(line):
                break
            bullet = _CITATION_BULLET_RE.match(lines[i])
            if not bullet:
                # Only list items under Sources/References count as citations;
                # free prose belongs to the answer body.
                break
            item = _clean_citation(bullet.group("item"))
            # Skip category labels like "Official Reports:" and keep leaf citations.
            if item and not item.endswith(":"):
                _add(item)
            i += 1

    return found[:limit]


def _truncate_preserving_citations(text: str, limit: int) -> str:
    """Truncate answer text while keeping a trailing Sources/References section."""
    body = (text or "").strip()
    if len(body) <= limit:
        return body

    section_start = None
    for match in _CITATION_SECTION_RE.finditer(body):
        section_start = match.start()
    cite_tail = ""
    if section_start is not None and section_start >= limit // 3:
        cite_tail = body[section_start:].strip()
        # Cap the preserved citation section so it cannot dominate the budget.
        cite_budget = max(280, min(len(cite_tail), limit // 3))
        cite_tail = cite_tail[:cite_budget].rstrip()
        head_budget = max(400, limit - len(cite_tail) - 20)
        return body[:head_budget].rstrip() + " ...\n\n" + cite_tail
    return body[:limit].rstrip() + " ..."


def _corpus_blocks(results: List[RetrievalResult], per_source_chars: int = 1500) -> str:
    blocks = []
    for r in results:
        if not r.ok:
            continue
        full = r.text.strip()
        citations = _extract_citations(full)
        snippet = _truncate_preserving_citations(full, per_source_chars)
        # Source label already embeds non-English language (e.g. "Kimi (Mandarin Chinese)").
        strategy_note = f" | strategy: {r.strategy}" if r.strategy else ""
        lang_note = (
            " | response translated to English for this report"
            if _is_foreign_language(r.language)
            else ""
        )
        block = (
            f"[Source: {r.source_label} | query: {r.seed}{strategy_note}{lang_note}]\n"
            f"{snippet}"
        )
        # Always surface extracted citations after truncation so the summariser
        # can attach them to claims even when the body was shortened.
        if citations:
            cite_lines = "\n".join(f"- {c}" for c in citations)
            block = f"{block}\n\n[Citations from this source]\n{cite_lines}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def _synthesize_claims_summary(
    prompt: str,
    results: List[RetrievalResult],
    llm: LLMClient,
    impact_definition: Optional[str] = None,
) -> Optional[str]:
    """Build a concise ranked claims brief emphasising corroboration × impact."""
    usable = [r for r in results if r.ok]
    if not usable:
        return None

    impact_definition = impact_definition or build_impact_definition()
    # Use language-annotated labels so the summariser can attribute non-English
    # attestations (e.g. "Kimi (Mandarin Chinese)") distinctly from English ones.
    source_labels = sorted({r.source_label for r in usable})
    corpus = _corpus_blocks(usable, per_source_chars=_summary_per_source_chars(llm))

    ask = (
        f'The user originally asked: "{prompt}".\n\n'
        "Below are answers from several retrieval LLMs (each distinct "
        f"`Source:` label is one source). Successful sources: "
        f"{', '.join(source_labels)}.\n\n"
        "Write a *concise* markdown claims brief for a busy reader. Rules:\n"
        "1. Extract atomic factual claims. Prefer *points of precision* — "
        "exact numbers, quantities, ratios, dates, named ingredients/"
        "chemicals, measurements, identifiers, and concrete lists — over "
        "vague restatements (\"closely guarded secret\", \"many ingredients\"). "
        "When merging paraphrases, keep the most specific wording present in "
        "the sources.\n"
        "2. Corroboration count = number of *distinct Source labels* that "
        "assert the claim. The same LLM queried in different languages counts "
        "as distinct sources (e.g. `Kimi` and `Kimi (Mandarin Chinese)` are "
        "two sources). The same LLM across different English queries counts once.\n"
        "3. Only treat a claim as corroborated if at least 2 distinct sources "
        "support it (except in `## Points of precision`, where a precise "
        "single-source value may be listed if clearly attributed).\n"
        "4. Rank primarily by corroboration (higher first), then by impact, "
        "then by specificity (more precise before more general).\n"
        f"5. High-impact definition: {impact_definition}\n"
        "6. An *outlier* is a claim that stands out from the rest: asserted by "
        "only one source while others are silent or disagree, surprising or "
        "counter-intuitive relative to the consensus, or uniquely specific / "
        "extreme. Note when an outlier came from a non-English source. "
        "Especially call out specificity gaps: when some model(s) give exact "
        "numbers, dates, quantities, names, or other concrete identifiers while "
        "the rest speak only in generalities (name the specific model(s) and "
        "what concrete detail they alone supplied).\n"
        "7. Keep the whole brief short: aim for roughly 15–30 bullets max. "
        "Omit low-impact filler and pure generalities even if corroborated.\n"
        "8. When listing attesting sources, copy the Source label *exactly*, "
        "including any parenthetical language "
        "(e.g. `Kimi (Mandarin Chinese)`, `Grok (French)`). This tells the "
        "reader which attestations came from a non-English query.\n"
        "9. Attribution format (keep the claim line short — do NOT inline long "
        "source lists):\n"
        "   **claim** — *N sources*\n"
        "   <details>\n"
        "   <summary>Attesting sources (N)</summary>\n\n"
        "   `LabelA`, `LabelB (Spanish)`, ...\n\n"
        "   </details>\n"
        "   Citations: `Document or URL`; `Another citation`\n"
        "   One short clause on why it matters.\n"
        "   For 1–2 sources you may keep labels inline; for 3+ always use "
        "<details>.\n"
        "10. Citations: when attesting responses include URLs, document / report "
        "titles, or entries under Sources/References/Citations (also listed in "
        "`[Citations from this source]` blocks below), attach those citations "
        "to the claim they support using a `Citations:` line. Deduplicate "
        "across attesting sources. Prefer the most specific form present "
        "(URL over bare title when both appear). Never invent citations. "
        "Omit the `Citations:` line only when none of the attesting sources "
        "provided any.\n"
        "11. Use this structure exactly:\n"
        f"# Claims summary: {prompt}\n\n"
        "_Ranked by corroboration, then impact, then specificity. A source is "
        "a distinct retrieval LLM label (language-annotated when non-English). "
        "Citations attached to a claim come from the retrieval answers, not "
        "from the LLM labels._\n\n"
        "## High-impact corroborated claims\n"
        "Numbered list. Each item uses the attribution format above "
        "(including Citations when available), then one short clause on why "
        "it is high-impact. Prefer precise wording.\n\n"
        "## Points of precision\n"
        "Bullets for the most concrete specifics found (exact values, "
        "quantities, dates, ratios, named chemicals/ingredients, identifiers, "
        "measurements). Same attribution format. Include even if only one "
        "source stated the precise figure, but mark single-source clearly. "
        "Skip vague claims here.\n\n"
        "## Points of disagreement\n"
        "Bullets for concrete conflicts between sources — different facts, "
        "dates, quantities, named entities, causal claims, or conclusions. "
        "Each bullet: state the disputed point, then which Source labels "
        "assert side A vs side B (and any other camps). Prefer precise "
        "conflicting values over vague 'sources differ' wording. Skip only "
        "if sources do not materially disagree.\n\n"
        "## Other corroborated claims\n"
        "Bullets for corroborated but lower-impact facts. Same attribution "
        "format (keep language annotations and citations). Skip this section "
        "if empty.\n\n"
        "## Outlier or unusual responses\n"
        "Bullets for claims that diverge from the consensus: single-source "
        "surprises, contradictions of the majority, unusually specific / "
        "extreme assertions, or cases where one model gives exact numbers / "
        "dates / identifiers while others stay vague. Each item: **claim** — "
        "*source(s)*, Citations when present, why it is an outlier, and when "
        "relevant which model(s) were more specific vs. general. Skip only if "
        "nothing qualifies.\n\n"
        "## Distinctions between models\n"
        "Short bullets on how models differ: unique claims, refusals/hedges, "
        "conflicting emphasis, precision gaps, or language-specific "
        "divergence. Each bullet names the model(s) with exact Source labels. "
        "Skip only if models are indistinguishable.\n\n"
        "## Contested or single-source (notable only)\n"
        "Only include if high-impact *and* either contested across sources "
        "or supported by a single source. Mark as contested or single-source. "
        "Include Citations when present. Disagreements that belong in "
        "`## Points of disagreement` should go there instead of here. "
        "Skip if nothing qualifies.\n\n"
        f"{corpus}"
    )
    try:
        text = llm.complete(
            ask,
            system=CLAIMS_SUMMARY_SYSTEM,
            max_tokens=max(1800, llm.spec.max_tokens),
        )
        return _collapse_source_lists(text) if text else None
    except Exception as exc:
        logger.warning("Claims summary synthesis failed (%s)", exc)
        return None


_SOURCE_LIST_RE = re.compile(
    r"(?P<head>\*(?P<n>\d+)\s+sources?\*)\s*\((?P<body>(?:`[^`]+`\s*,\s*)*`[^`]+`)\)"
    r"(?P<dot>\.)?",
    re.IGNORECASE,
)


def _collapse_source_lists(md: str, min_sources: int = 3) -> str:
    """Wrap inline ``*N sources* (`A`, `B`, ...)`` lists in <details> blocks."""

    def repl(match: re.Match) -> str:
        n = int(match.group("n"))
        body = match.group("body").strip()
        label_count = body.count("`") // 2
        if n < min_sources and label_count < min_sources:
            return match.group(0)
        # Drop the trailing period that closed the parenthetical; keep following
        # prose on the next line via the original surrounding whitespace.
        return (
            f"{match.group('head')}\n"
            f"<details>\n"
            f"<summary>Attesting sources ({n})</summary>\n\n"
            f"{body}\n\n"
            f"</details>"
        )

    return _SOURCE_LIST_RE.sub(repl, md)


# --- CLI / GUI LLM status ---------------------------------------------------
@dataclass
class LLMStatus:
    """Preflight health check for one retrieval LLM."""

    name: str
    status: str  # ok | fail | partial
    reason: str = ""


# Process-wide provider health. Cloud report builds skip the connectivity
# probe and reuse whatever was last recorded here.
_PROVIDER_HEALTH: Dict[str, LLMStatus] = {}
_PROVIDER_HEALTH_LOCK = threading.Lock()


def _provider_health_key(llm: LLMClient) -> str:
    spec = llm.spec
    return f"{spec.provider}|{spec.model}|{spec.base_url or ''}"


def get_cached_provider_health(llm: LLMClient) -> Optional[LLMStatus]:
    with _PROVIDER_HEALTH_LOCK:
        cached = _PROVIDER_HEALTH.get(_provider_health_key(llm))
    if cached is None:
        return None
    return LLMStatus(name=llm.label, status=cached.status, reason=cached.reason)


def set_cached_provider_health(llm: LLMClient, status: LLMStatus) -> LLMStatus:
    stored = LLMStatus(name=llm.label, status=status.status, reason=status.reason)
    with _PROVIDER_HEALTH_LOCK:
        _PROVIDER_HEALTH[_provider_health_key(llm)] = stored
    return stored


def clear_provider_health_cache() -> None:
    with _PROVIDER_HEALTH_LOCK:
        _PROVIDER_HEALTH.clear()
    _PROVIDER_CIRCUITS.reset()


def record_retrieval_health(
    llms: Sequence[LLMClient],
    results: Sequence[RetrievalResult],
) -> None:
    """Update the global health cache from actual retrieval outcomes."""
    by_label: Dict[str, List[RetrievalResult]] = {}
    for result in results:
        by_label.setdefault(result.llm_label, []).append(result)
    for llm in llms:
        items = by_label.get(llm.label) or []
        if not items:
            continue
        n_ok = sum(1 for row in items if row.ok)
        if n_ok:
            status = LLMStatus(name=llm.label, status="ok")
        else:
            sample = next((row for row in items if row.error), None)
            status = LLMStatus(
                name=llm.label,
                status="fail",
                reason=_short_error(sample.error if sample else "") or "retrieval failed",
            )
        set_cached_provider_health(llm, status)


class ProviderCircuitBreaker:
    """Skip further calls to a host after consecutive provider-level failures."""

    def __init__(self, threshold: int = PROVIDER_CIRCUIT_FAILURE_THRESHOLD):
        self.threshold = max(1, int(threshold))
        self._fail_counts: Dict[str, int] = {}
        self._open: Dict[str, str] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._fail_counts.clear()
            self._open.clear()

    def skip_reason(self, key: str) -> Optional[str]:
        with self._lock:
            reason = self._open.get(key)
        if not reason:
            return None
        return f"circuit open for {key}: {reason}"

    def record_success(self, key: str) -> None:
        with self._lock:
            self._fail_counts[key] = 0
            self._open.pop(key, None)

    def record_failure(self, key: str, reason: str) -> bool:
        """Count a failure. True if this call opened the circuit."""
        cleaned = " ".join((reason or "provider error").split())[:120]
        with self._lock:
            if key in self._open:
                return False
            n = self._fail_counts.get(key, 0) + 1
            self._fail_counts[key] = n
            if n >= self.threshold:
                self._open[key] = cleaned
                return True
            return False


_PROVIDER_CIRCUITS = ProviderCircuitBreaker()


def scheduler_provider_key(llm: LLMClient) -> str:
    """Concurrency / circuit key: API host, not the generic 'custom' provider."""
    spec = llm.spec
    provider = (spec.provider or "unknown").strip().lower() or "unknown"
    if provider in {"openai", "anthropic", "ollama", "echo"}:
        return provider
    host = urlparse(spec.base_url or "").netloc.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host or provider


def _int_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def retrieval_global_concurrency(override: Optional[int] = None) -> int:
    if override is not None:
        return max(1, int(override))
    return _int_setting(
        "MOYO_RETRIEVAL_GLOBAL_CONCURRENCY",
        RETRIEVAL_GLOBAL_CONCURRENCY,
        1,
        32,
    )


def retrieval_per_provider_concurrency() -> int:
    return _int_setting(
        "MOYO_RETRIEVAL_PER_PROVIDER_CONCURRENCY",
        RETRIEVAL_PER_PROVIDER_CONCURRENCY,
        1,
        4,
    )


def _circuit_failure_reason(result: RetrievalResult) -> Optional[str]:
    err = (result.error or "").strip()
    if not err:
        return None
    text = err.lower()
    if "circuit open" in text:
        return None
    if any(marker in text for marker in _CIRCUIT_ERROR_MARKERS):
        return err
    return None


def skip_llm_preflight() -> bool:
    """Cloud report builds must not spend a round-trip probing every model."""
    flag = os.environ.get("MOYO_SKIP_LLM_PREFLIGHT", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    try:
        from moyo.llm.utility import running_in_cloud

        return running_in_cloud()
    except Exception:
        return False


def _short_error(error: Optional[str], empty: bool = False) -> str:
    """Compress a provider exception into one CLI-friendly reason line."""
    if empty:
        return "no content returned"
    if not error:
        return ""
    text = " ".join(error.split())
    # Prefer the nested provider message / error string when present.
    # Providers mix quote styles: {'message': '...'}, {"error": "..."}, {'error': "..."}.
    for pattern in (
        r"'message'\s*:\s*'([^']*)'",
        r'"message"\s*:\s*"([^"]*)"',
        r"'message'\s*:\s*\"([^\"]*)\"",
        r'"message"\s*:\s*\'([^\']*)\'',
        r"'error'\s*:\s*\"([^\"]*)\"",
        r'"error"\s*:\s*"([^"]*)"',
        r"'error'\s*:\s*'([^']*)'",
        r'"error"\s*:\s*\'([^\']*)\'',
    ):
        match = re.search(pattern, text)
        if match:
            text = match.group(1).replace("\\'", "'").replace('\\"', '"')
            break
    # Drop trailing remediation URLs / long metric dumps.
    text = re.split(r"\s+For (?:details|more information)\b", text, maxsplit=1)[0]
    text = re.split(r"\s+Add credits\b", text, maxsplit=1)[0]
    text = re.split(r"\s+You can purchase\b", text, maxsplit=1)[0]
    text = re.split(r"\s+\* Quota exceeded\b", text, maxsplit=1)[0]
    text = text.strip().rstrip(".")
    if len(text) > 90:
        text = text[:87] + "..."
    return text


# Preflight completion budget. Some providers reject tiny caps (Perplexity
# requires >= 16) and Gemini thinking models often need headroom so reasoning
# tokens do not exhaust the budget before visible content is written.
PROBE_MAX_TOKENS = 256


def probe_llm(llm: LLMClient, *, use_cache: bool = True) -> LLMStatus:
    """Send a tiny completion to see whether ``llm`` is reachable and authorized."""
    if use_cache:
        cached = get_cached_provider_health(llm)
        if cached is not None:
            return cached
    try:
        text = llm.complete(
            "Reply with the single word: ok",
            system="You are a connectivity probe. Reply with only: ok",
            max_tokens=PROBE_MAX_TOKENS,
            retries=0,  # preflight should fail fast
        )
        if text and str(text).strip():
            status = LLMStatus(name=llm.label, status="ok")
        else:
            status = LLMStatus(name=llm.label, status="fail", reason="no content returned")
    except Exception as exc:
        status = LLMStatus(
            name=llm.label, status="fail", reason=_short_error(str(exc))
        )
    return set_cached_provider_health(llm, status)


def check_retrieval_llms(
    llms: List[LLMClient],
    progress: Optional[ProgressFn] = None,
    workers: Optional[int] = None,
    *,
    use_cache: bool = True,
) -> List[LLMStatus]:
    """Probe each retrieval LLM and return name/status/reason rows."""
    if not llms:
        return []

    def _report(msg: str) -> None:
        if progress:
            progress(msg)

    _report(f"Checking {len(llms)} retrieval LLM(s) ...")
    statuses: List[Optional[LLMStatus]] = [None] * len(llms)
    max_workers = len(llms) if workers is None else max(1, int(workers))
    max_workers = min(max_workers, len(llms))

    def _run(index: int, llm: LLMClient) -> None:
        statuses[index] = probe_llm(llm, use_cache=use_cache)

    if max_workers == 1:
        for i, llm in enumerate(llms):
            _run(i, llm)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run, i, llm) for i, llm in enumerate(llms)]
            for fut in as_completed(futures):
                fut.result()

    return [s for s in statuses if s is not None]


def format_llm_status_table(statuses: List[LLMStatus]) -> str:
    """Render a compact name / status / reason table for preflight output."""
    rows = [(s.name, s.status, s.reason) for s in statuses]
    headers = ("name", "status", "reason")
    widths = [len(h) for h in headers]
    for name, status, reason in rows:
        widths[0] = max(widths[0], len(name))
        widths[1] = max(widths[1], len(status))
        widths[2] = max(widths[2], len(reason))

    def fmt(cols: tuple) -> str:
        return (
            f"{cols[0]:<{widths[0]}}  "
            f"{cols[1]:<{widths[1]}}  "
            f"{cols[2]}"
        ).rstrip()

    lines = [fmt(headers), fmt(tuple("-" * w for w in widths))]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


RETRIEVAL_CHECK_BASENAME = "llm-retrieval-check"


def llm_status_rows(statuses: List[LLMStatus]) -> List[Dict[str, str]]:
    return [
        {"name": s.name, "status": s.status, "reason": s.reason}
        for s in statuses
    ]


def retrieval_statuses_from_result(result: ExploreResult) -> List[LLMStatus]:
    """Post-run ok/fail/partial per LLM from actual retrievals."""
    from collections import OrderedDict

    by_label: OrderedDict[str, List[RetrievalResult]] = OrderedDict()
    for label in result.llm_labels:
        by_label[label] = []
    for r in result.results:
        by_label.setdefault(r.llm_label, []).append(r)

    statuses: List[LLMStatus] = []
    for label, items in by_label.items():
        n_ok = sum(1 for r in items if r.ok)
        n = len(items)
        if n_ok == n and n > 0:
            statuses.append(LLMStatus(name=label, status="ok"))
        elif n_ok == 0:
            sample = next((r for r in items if r.error), None)
            empty = sample is None and any(not r.text.strip() for r in items)
            statuses.append(
                LLMStatus(
                    name=label,
                    status="fail",
                    reason=_short_error(sample.error if sample else None, empty=empty),
                )
            )
        else:
            sample = next((r for r in items if not r.ok), None)
            empty = bool(sample and not sample.error and not sample.text.strip())
            statuses.append(
                LLMStatus(
                    name=label,
                    status="partial",
                    reason=(
                        f"{n_ok}/{n} ok; "
                        + _short_error(sample.error if sample else None, empty=empty)
                    ),
                )
            )
    return statuses


def format_retrieval_table(result: ExploreResult) -> str:
    """Back-compat post-run table; prefer :func:`format_llm_status_table`."""
    return format_llm_status_table(retrieval_statuses_from_result(result))


def render_llm_retrieval_check(result: ExploreResult) -> tuple[str, dict]:
    """Markdown + JSON payload for the LLM Retrieval Check document."""
    preflight = list(result.llm_statuses or [])
    retrieval = retrieval_statuses_from_result(result)
    n_ok = sum(1 for s in preflight if s.status == "ok")
    n_ret_ok = sum(1 for s in retrieval if s.status == "ok")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "prompt": result.prompt,
        "generated_at": generated,
        "preflight": llm_status_rows(preflight),
        "preflight_ok": n_ok,
        "preflight_total": len(preflight),
        "retrieval": llm_status_rows(retrieval),
        "retrieval_ok": n_ret_ok,
        "retrieval_total": len(retrieval),
    }
    lines = [
        "# LLM Retrieval Check",
        "",
        f"_Generated {generated}_",
        "",
        f"**Prompt:** {result.prompt}",
        "",
        f"**Preflight:** {n_ok}/{len(preflight)} working",
        f"**Retrieval this run:** {n_ret_ok}/{len(retrieval)} fully successful",
        "",
        "## Preflight probe",
        "",
        "```",
        format_llm_status_table(preflight) if preflight else "(no preflight statuses)",
        "```",
        "",
        "## Retrieval results",
        "",
        "```",
        format_llm_status_table(retrieval) if retrieval else "(no retrievals)",
        "```",
        "",
    ]
    return "\n".join(lines), payload


def write_llm_retrieval_check(result: ExploreResult, directory: Path) -> Path:
    """Write ``llm-retrieval-check.md`` and ``.json`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    markdown, payload = render_llm_retrieval_check(result)
    md_path = directory / f"{RETRIEVAL_CHECK_BASENAME}.md"
    json_path = directory / f"{RETRIEVAL_CHECK_BASENAME}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result.retrieval_check_path = str(md_path)
    logger.info("Wrote LLM retrieval check to %s", md_path)
    return md_path


# --- Markdown rendering -----------------------------------------------------
def _kind_label(kind: str) -> str:
    return {"closed": "Closed API", "open": "Open API", "local": "Local"}.get(kind, kind)


def render_markdown(
    corpus: CompiledCorpus,
    summary: Optional[str],
    llms: List[LLMClient],
) -> str:
    """Render the exploration report from a compiled corpus.

    Raw compiled findings (language → query → model) are written first; any
    synthesised summary is appended afterward so analysis never shapes how
    the raw responses are organised or labeled.
    """
    lines: List[str] = []
    lines.append(f"# Topic exploration: {corpus.prompt}")
    lines.append("")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    mode = (corpus.fuzz_mode or "basic").strip().lower()
    lines.append(f"_Fuzz mode: `{mode}`_")
    # Annotate actual fuzz techniques used (from seeds, else mode defaults).
    from moyo.publicside.barrierprobe.llm_fuzzer import strategies_for_fuzz_mode

    techniques = list(
        dict.fromkeys(
            (q.strategy or "").strip().lower()
            for q in corpus.queries
            if (q.strategy or "").strip()
        )
    ) or strategies_for_fuzz_mode(mode)
    if techniques:
        tech = ", ".join(f"`{t}`" for t in techniques)
        lines.append(f"_Techniques ({mode}): {tech}_")
    if corpus.languages:
        lines.append(
            f"_Languages: English + {', '.join(corpus.languages)} "
            f"(full strategy set per language)_"
        )
    lines.append("")

    # Configured sources, grouped by kind.
    lines.append("## Retrieval sources")
    lines.append("")
    for kind in ("closed", "open", "local"):
        group = [llm for llm in llms if llm.kind == kind]
        if not group:
            continue
        names = ", ".join(f"`{llm.label}`" for llm in group)
        lines.append(f"- **{_kind_label(kind)}:** {names}")
    lines.append("")

    # Seed catalogue from the compiled queries (already language-ordered).
    lines.append("## Reworded query seeds")
    lines.append("")
    lang_order = _language_group_order(corpus.languages)
    for lang_name in lang_order:
        lang_queries = [q for q in corpus.queries if q.language_group == lang_name]
        if not lang_queries:
            continue
        lines.append(f"### {lang_name}")
        lines.append("")
        for i, q in enumerate(lang_queries, 1):
            strat = f" `{q.strategy}`" if q.strategy else ""
            lines.append(f"{i}.{strat} {q.text}")
        lines.append("")

    # Raw compiled findings first — no analysis.
    lines.append("## Detailed findings by language, query, and source")
    lines.append("")
    for lang_name in lang_order:
        lang_queries = [q for q in corpus.queries if q.language_group == lang_name]
        if not lang_queries:
            continue
        lines.append(f"### {lang_name}")
        lines.append("")
        for i, q in enumerate(lang_queries, 1):
            strat_note = f" [{q.strategy}]" if q.strategy else ""
            lines.append(f"#### Query {i}{strat_note}: {q.text}")
            lines.append("")
            # Results are already llm_index-ordered within the compile step.
            for r in q.results:
                header = f"##### {r.source_label}  _({_kind_label(r.kind)})_"
                lines.append(header)
                lines.append("")
                if r.error:
                    lines.append(f"> Retrieval failed: {r.error}")
                elif not r.text.strip():
                    lines.append("> (no content returned)")
                else:
                    lines.append(r.text.strip())
                    if r.original_text and r.original_text.strip():
                        lines.append("")
                        lines.append("<details>")
                        lines.append(
                            f"<summary>Original response ({r.language})</summary>"
                        )
                        lines.append("")
                        lines.append(r.original_text.strip())
                        lines.append("")
                        lines.append("</details>")
                lines.append("")

    # Analysis last — produced only after the corpus was compiled.
    if summary:
        lines.append("## Summary of findings")
        lines.append("")
        lines.append(summary.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _language_group_order(languages: Optional[List[str]] = None) -> List[str]:
    """English first, then configured target languages (stable label order)."""
    ordered: List[str] = ["English"]
    for lang in languages or []:
        name = _language_group_name(lang)
        if name not in ordered:
            ordered.append(name)
    return ordered


def _ordered_language_groups(
    results: List[RetrievalResult],
    languages: Optional[List[str]] = None,
) -> List[str]:
    """English first, then configured target languages, then any leftovers."""
    ordered = _language_group_order(languages)
    for r in results:
        name = _language_group_name(r.language)
        if name not in ordered:
            ordered.append(name)
    present = {_language_group_name(r.language) for r in results}
    if not present:
        return ordered
    return [name for name in ordered if name in present]


def _run_coro(coro):
    """Run ``coro`` from sync explore code, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def _call_timeout_seconds(llm: LLMClient) -> float:
    return max(1.0, float(getattr(llm.spec, "timeout", None) or 120))


def _timeout_retrieval_result(
    seed: str,
    llm: LLMClient,
    *,
    language: Optional[str],
    strategy: Optional[str],
    seed_index: int,
    llm_index: int,
    timeout: float,
    exc: BaseException,
) -> RetrievalResult:
    from moyo.llm.client import format_llm_error

    result = RetrievalResult(
        seed=seed,
        llm_label=llm.label,
        provider=llm.spec.provider,
        model=llm.spec.model,
        kind=llm.kind,
        language=language,
        strategy=strategy,
        seed_index=seed_index,
        llm_index=llm_index,
    )
    if isinstance(exc, asyncio.TimeoutError):
        result.error = f"timed out after {int(timeout)}s"
    else:
        result.error = format_llm_error(exc)
    return result


def _skipped_retrieval_result(
    seed: str,
    llm: LLMClient,
    *,
    language: Optional[str],
    strategy: Optional[str],
    seed_index: int,
    llm_index: int,
    reason: str,
) -> RetrievalResult:
    return RetrievalResult(
        seed=seed,
        llm_label=llm.label,
        provider=llm.spec.provider,
        model=llm.spec.model,
        kind=llm.kind,
        language=language,
        strategy=strategy,
        seed_index=seed_index,
        llm_index=llm_index,
        error=reason,
    )


async def _retrieve_jobs_async(
    jobs: Sequence[Tuple[int, Any, int, LLMClient]],
    *,
    max_tokens: Optional[int],
    max_workers: int,
    progress: Optional[ProgressFn],
    per_provider_concurrency: Optional[int] = None,
    circuit: Optional[ProviderCircuitBreaker] = None,
) -> List[RetrievalResult]:
    """One bounded batch: global cap, per-host cap, and a provider circuit breaker."""
    total = len(jobs)
    if not total:
        return []
    global_limit = max(1, int(max_workers))
    per_provider = max(1, int(per_provider_concurrency or retrieval_per_provider_concurrency()))
    per_provider = min(per_provider, global_limit)
    breaker = circuit or _PROVIDER_CIRCUITS
    global_sem = asyncio.Semaphore(global_limit)
    provider_sems: Dict[str, asyncio.Semaphore] = {}
    done = 0
    lock = threading.Lock()
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=global_limit)

    def _provider_sem(key: str) -> asyncio.Semaphore:
        sem = provider_sems.get(key)
        if sem is None:
            sem = asyncio.Semaphore(per_provider)
            provider_sems[key] = sem
        return sem

    async def _one(
        job_index: int,
        seed_index: int,
        qs: Any,
        llm_index: int,
        llm: LLMClient,
    ) -> Tuple[int, RetrievalResult]:
        nonlocal done
        key = scheduler_provider_key(llm)
        skip = breaker.skip_reason(key)
        if skip:
            result = _skipped_retrieval_result(
                qs.text,
                llm,
                language=qs.language,
                strategy=qs.strategy,
                seed_index=seed_index,
                llm_index=llm_index,
                reason=skip,
            )
        else:
            async with global_sem:
                async with _provider_sem(key):
                    skip = breaker.skip_reason(key)
                    if skip:
                        result = _skipped_retrieval_result(
                            qs.text,
                            llm,
                            language=qs.language,
                            strategy=qs.strategy,
                            seed_index=seed_index,
                            llm_index=llm_index,
                            reason=skip,
                        )
                    else:
                        # Start the per-call clock only when this POST actually
                        # runs — after paraphrasing and after a worker slot.
                        timeout = _call_timeout_seconds(llm)
                        fut = loop.run_in_executor(
                            executor,
                            partial(
                                retrieve,
                                qs.text,
                                llm,
                                max_tokens,
                                qs.language,
                                qs.strategy,
                                seed_index,
                                llm_index,
                            ),
                        )
                        try:
                            result = await asyncio.wait_for(
                                asyncio.shield(fut), timeout=timeout
                            )
                        except Exception as exc:
                            result = _timeout_retrieval_result(
                                qs.text,
                                llm,
                                language=qs.language,
                                strategy=qs.strategy,
                                seed_index=seed_index,
                                llm_index=llm_index,
                                timeout=timeout,
                                exc=exc,
                            )
                            logger.warning(
                                "Retrieval failed for %s via %s: %s",
                                qs.text[:60],
                                llm.label,
                                result.error,
                            )
                        if result.ok:
                            breaker.record_success(key)
                        else:
                            trip = _circuit_failure_reason(result)
                            if trip and breaker.record_failure(key, trip):
                                logger.warning(
                                    "Provider circuit open for %s after failures: %s",
                                    key,
                                    trip,
                                )
        with lock:
            done += 1
            n = done
        lang_tag = f" [{qs.language}]" if _is_foreign_language(qs.language) else ""
        strat_tag = f"/{qs.strategy}" if qs.strategy else ""
        if progress:
            progress(
                f"[{n}/{total}] {llm.label}{lang_tag}{strat_tag}: {qs.text[:70]}"
            )
        return job_index, result

    gathered: List[Any] = []
    try:
        gathered = await asyncio.gather(
            *[
                _one(job_index, seed_index, qs, llm_index, llm)
                for job_index, (seed_index, qs, llm_index, llm) in enumerate(jobs)
            ],
            return_exceptions=True,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    ordered: List[Optional[RetrievalResult]] = [None] * total
    for item in gathered:
        if isinstance(item, BaseException):
            logger.warning("Retrieval batch item failed: %s", item)
            continue
        job_index, result = item
        ordered[job_index] = result
    return [row for row in ordered if row is not None]


# --- Orchestration ----------------------------------------------------------
def explore_topic(
    prompt: str,
    default_llm: Optional[LLMClient] = None,
    retrieval_llms: Optional[List[LLMClient]] = None,
    num_seeds: int = 3,
    summarize: bool = False,
    max_tokens: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
    workers: Optional[int] = None,
    impact_definition: Optional[str] = None,
    impact_definition_files: Optional[List[str]] = None,
    fuzz_mode: str = "basic",
    extra_languages: Optional[List[str]] = None,
    language_selection_explicit: bool = False,
    strategies: Optional[List[str]] = None,
) -> ExploreResult:
    """Run the full naive-prompt exploration and return an :class:`ExploreResult`.

    Retrieval calls (seed × LLM) run in one bounded concurrent batch
    (``asyncio.gather`` + global/per-provider semaphores + per-call
    ``wait_for``). Default global concurrency is 12 (~10–14); each API host
    is capped at 2 concurrent calls. A provider circuit breaker stops further
    work to a host after consecutive timeouts / 429s / connection failures.
    ``workers`` overrides the global cap (``1`` is sequential). Translations
    use the per-provider cap because they hit a single utility LLM.

    ``fuzz_mode`` ``basic`` (default) emits ``num_seeds`` English seeds
    rotating original / paraphrase (no translate);
    ``multilingual`` emits ``num_seeds`` of original / paraphrase
    per extra language (English plus each selected language).
    Pass ``strategies`` to override that rotation a la carte (include
    ``translate``, ``summarize``, ``typo`` or ``shuffle`` explicitly);
    mode still controls language fan-out.
    Every seed is sent to every retrieval LLM.

    ``impact_definition`` / ``impact_definition_files`` add user-specific
    high-impact criteria on top of :data:`DEFAULT_HIGH_IMPACT_DEFINITION`.
    """

    def _report(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    default_llm = default_llm or get_default_llm()
    retrieval_llms = retrieval_llms if retrieval_llms is not None else get_retrieval_llms()
    if not retrieval_llms:
        retrieval_llms = [default_llm]

    # Cloud report builds skip connectivity probes; provider health is cached
    # globally from probes (local/CLI) and from actual retrieval outcomes.
    if skip_llm_preflight():
        llm_statuses = [
            status
            for status in (get_cached_provider_health(llm) for llm in retrieval_llms)
            if status is not None
        ]
        _report(
            "Skipping LLM preflight (cloud report build); "
            f"{len(llm_statuses)} cached provider health row(s)"
        )
        if llm_statuses:
            _report(format_llm_status_table(llm_statuses))
    else:
        llm_statuses = check_retrieval_llms(
            retrieval_llms, progress=_report, workers=workers
        )
        _report(format_llm_status_table(llm_statuses))
        n_ok = sum(1 for s in llm_statuses if s.status == "ok")
        _report(f"LLM preflight: {n_ok}/{len(llm_statuses)} working")

    resolved_impact = build_impact_definition(
        extra=impact_definition, extra_files=impact_definition_files
    )

    from moyo.publicside.barrierprobe.llm_fuzzer import (
        normalize_fuzz_mode,
        normalize_fuzz_strategies,
    )

    mode = normalize_fuzz_mode(fuzz_mode)
    extras = [str(x).strip() for x in (extra_languages or []) if str(x).strip()]
    if mode == "multilingual" and language_selection_explicit and not extras:
        mode = "basic"
    strat = normalize_fuzz_strategies(strategies, fuzz_mode=mode)
    languages = (
        resolve_explore_languages(
            extras,
            language_selection_explicit=language_selection_explicit,
        )
        if mode == "multilingual"
        else None
    )
    strat_note = "/".join(strat)
    if mode == "multilingual":
        lang_groups = 1 + len(languages or [])
        lang_note = f", languages=English+{languages}"
        seed_note = (
            f"{num_seeds} seed(s) per language ({lang_groups} language group(s)) "
            f"rotating {strat_note}"
        )
    else:
        lang_note = ""
        seed_note = f"{num_seeds} seed(s) rotating {strat_note}"
    _report(
        f"Rewording prompt into {seed_note} via LLMFuzzer "
        f"(runtime utility LLM, black-box, fuzz_mode={mode}{lang_note}) ..."
    )
    # Sequential paraphraser: finish all rewording before any retrieval
    # wait_for deadline is calculated or started.
    query_seeds = reword_prompt_seeds(
        prompt,
        n=num_seeds,
        fuzz_mode=mode,
        languages=languages,
        strategies=strat,
    )
    _report(
        "Seeds: "
        + "; ".join(
            f"[{_language_group_name(s.language)}/{s.strategy or '?'}] {s.text[:60]}"
            for s in query_seeds
        )
    )

    # --- Phase 1: one bounded concurrent retrieval batch ---
    # Per-call timers start inside _retrieve_jobs_async only after a worker
    # slot is acquired — never during rewording.
    jobs = [
        (seed_index, qs, llm_index, llm)
        for seed_index, qs in enumerate(query_seeds)
        for llm_index, llm in enumerate(retrieval_llms)
    ]
    total = len(jobs)
    global_workers = retrieval_global_concurrency(workers)
    global_workers = min(global_workers, total) if total else 1
    per_provider = min(retrieval_per_provider_concurrency(), global_workers)
    _report(
        f"Retrieving raw answers with global={global_workers} "
        f"per-provider={per_provider} across {total} queries "
        f"({len(query_seeds)} seeds × {len(retrieval_llms)} LLMs) ..."
    )
    raw_results = _run_coro(
        _retrieve_jobs_async(
            jobs,
            max_tokens=max_tokens,
            max_workers=global_workers,
            per_provider_concurrency=per_provider,
            progress=_report,
        )
    )
    record_retrieval_health(retrieval_llms, raw_results)

    # --- Phase 2: compile / organise / label / translate (before analysis) ---
    _report("Compiling, organising, and labeling raw LLM responses ...")
    local_fuzzer = None
    try:
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

        local_fuzzer = LLMFuzzer.for_runtime()
    except Exception as exc:
        logger.warning("Utility fuzzer unavailable for compile translation (%s)", exc)

    corpus = compile_raw_responses(
        prompt,
        query_seeds,
        raw_results,
        retrieval_llms,
        fuzz_mode=mode,
        languages=languages,
        fuzzer=local_fuzzer,
        progress=_report,
        # Translations hit one utility LLM; keep them at the per-provider cap
        # unless the caller forced a global worker count of 1.
        workers=per_provider if workers is None else global_workers,
    )

    compiled_results = corpus.results

    # --- Phase 3: analysis only after the corpus is fully compiled ---
    summary: Optional[str] = None
    claims_summary: Optional[str] = None
    if summarize:
        summary_llm = get_summary_llm()
        _report(
            f"Synthesising combined summary via {summary_llm.label} "
            f"(num_ctx={summary_llm.spec.num_ctx}) ..."
        )
        summary = _synthesize_summary(prompt, compiled_results, summary_llm)
        if summary:
            summary = localize_text_for_report(summary)
        _report(
            f"Synthesising corroborated claims brief via {summary_llm.label} ..."
        )
        claims_summary = _synthesize_claims_summary(
            prompt, compiled_results, summary_llm, impact_definition=resolved_impact
        )
        if claims_summary:
            claims_summary = localize_text_for_report(claims_summary)

    # --- Phase 4: render (raw compiled body first, analysis last) ---
    markdown = render_markdown(corpus, summary, retrieval_llms)
    return ExploreResult(
        prompt=prompt,
        seeds=corpus.seeds,
        results=compiled_results,
        markdown=markdown,
        summary=summary,
        claims_summary=claims_summary,
        llm_labels=list(corpus.llm_labels),
        llm_statuses=list(llm_statuses),
    )


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug or "exploration")[:max_len]


_KIND_FROM_LABEL = {
    "Closed API": "closed",
    "Open API": "open",
    "Local": "local",
}
_EXPLORATION_TITLE_RE = re.compile(
    r"^#\s+Topic exploration:\s*(?P<prompt>.+?)\s*$", re.M
)
_FUZZ_MODE_RE = re.compile(r"^_Fuzz mode:\s*`(?P<mode>[^`]+)`_\s*$", re.M)
_LANGUAGES_RE = re.compile(
    r"^_Languages:\s*English\s*\+\s*(?P<langs>.+?)\s*\(", re.M
)
_SEED_ITEM_RE = re.compile(
    r"^(?P<n>\d+)\.\s*(?:`(?P<strategy>[^`]+)`\s+)?(?P<text>.+?)\s*$"
)
_QUERY_HEADER_RE = re.compile(
    r"^####\s+Query\s+(?P<n>\d+)(?:\s*\[(?P<strategy>[^\]]+)\])?:\s*(?P<text>.+?)\s*$"
)
_SOURCE_HEADER_RE = re.compile(
    r"^#####\s+(?P<label>.+?)\s+_?\((?P<kind>Closed API|Open API|Local)\)_?\s*$"
)


def _strip_details_blocks(text: str) -> str:
    return re.sub(
        r"\n*<details>\s*\n<summary>.*?</summary>\s*\n.*?\n</details>\s*",
        "\n",
        text or "",
        flags=re.S,
    ).strip()


def _split_source_label(
    source_label: str, language_groups: List[str]
) -> tuple[str, Optional[str]]:
    """Split ``ChatGPT (Spanish)`` into llm label + language when annotated."""
    label = (source_label or "").strip()
    for lang in sorted(
        (g for g in language_groups if _is_foreign_language(g)),
        key=len,
        reverse=True,
    ):
        suffix = f" ({lang})"
        if label.endswith(suffix):
            return label[: -len(suffix)].strip(), lang
    return label, None


def parse_exploration_markdown(markdown: str) -> CompiledCorpus:
    """Rebuild a :class:`CompiledCorpus` from a rendered ``exploration.md``.

    Parses the title, fuzz-mode metadata, seed catalogue, and detailed findings.
    LLM answer bodies may contain ``##`` / ``###`` headings, so structural
    parsing keys off ``#### Query`` / ``##### source`` markers and known
    language-group names from the seed catalogue.
    """
    text = (markdown or "").lstrip("\ufeff")
    title = _EXPLORATION_TITLE_RE.search(text)
    if not title:
        raise ValueError(
            "not an exploration report: missing `# Topic exploration: ...` title"
        )
    prompt = title.group("prompt").strip()

    mode_match = _FUZZ_MODE_RE.search(text)
    fuzz_mode = (mode_match.group("mode") if mode_match else "basic").strip().lower()

    languages: Optional[List[str]] = None
    langs_match = _LANGUAGES_RE.search(text)
    if langs_match:
        languages = [
            part.strip()
            for part in langs_match.group("langs").split(",")
            if part.strip()
        ]

    seeds_match = re.search(
        r"^##\s+Reworded query seeds\s*$", text, flags=re.M
    )
    detailed_match = re.search(
        r"^##\s+Detailed findings by language, query, and source\s*$",
        text,
        flags=re.M,
    )
    if not detailed_match:
        raise ValueError(
            "not an exploration report: missing "
            "`## Detailed findings by language, query, and source`"
        )

    language_groups: List[str] = ["English"]
    seed_by_lang: Dict[str, List[tuple[Optional[str], str]]] = {}
    if seeds_match:
        seeds_body = text[seeds_match.end() : detailed_match.start()]
        current_lang: Optional[str] = None
        for line in seeds_body.splitlines():
            if line.startswith("### "):
                current_lang = line[4:].strip()
                if current_lang and current_lang not in language_groups:
                    language_groups.append(current_lang)
                seed_by_lang.setdefault(current_lang or "English", [])
                continue
            item = _SEED_ITEM_RE.match(line.strip())
            if item and current_lang is not None:
                seed_by_lang.setdefault(current_lang, []).append(
                    (item.group("strategy"), item.group("text").strip())
                )

    summary_match = re.search(r"^##\s+Summary of findings\s*$", text, flags=re.M)
    detailed_end = summary_match.start() if summary_match else len(text)
    detailed_body = text[detailed_match.end() : detailed_end]

    queries: List[CompiledQuery] = []
    llm_labels: List[str] = []
    seen_llm: set[str] = set()
    current_lang = "English"
    current_query: Optional[CompiledQuery] = None
    seed_index = 0
    pending_header: Optional[re.Match] = None
    pending_lines: List[str] = []

    def _flush_pending() -> None:
        nonlocal pending_header, pending_lines, current_query
        if pending_header is None or current_query is None:
            pending_header = None
            pending_lines = []
            return
        source_label = pending_header.group("label").strip()
        kind = _KIND_FROM_LABEL.get(pending_header.group("kind"), "open")
        llm_label, lang_from_label = _split_source_label(source_label, language_groups)
        language = lang_from_label or (
            current_lang if _is_foreign_language(current_lang) else None
        )
        body = _strip_details_blocks("\n".join(pending_lines))
        error: Optional[str] = None
        text_body = body
        if body.startswith("> Retrieval failed:"):
            error = body.split(":", 1)[1].strip()
            text_body = ""
        elif body.startswith("> (no content returned)"):
            text_body = ""
        elif body.startswith("> (foreign-language response pruned"):
            text_body = ""
            error = "foreign-language response pruned"
        result = RetrievalResult(
            seed=current_query.text,
            llm_label=llm_label,
            provider="",
            model="",
            kind=kind,
            text=text_body,
            error=error,
            language=language,
            strategy=current_query.strategy,
            seed_index=current_query.seed_index,
            llm_index=len(current_query.results),
        )
        current_query.results.append(result)
        if llm_label not in seen_llm:
            seen_llm.add(llm_label)
            llm_labels.append(llm_label)
        pending_header = None
        pending_lines = []

    for raw_line in detailed_body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("### ") and line[4:].strip() in language_groups:
            _flush_pending()
            current_lang = line[4:].strip()
            current_query = None
            continue
        query_match = _QUERY_HEADER_RE.match(line)
        if query_match:
            _flush_pending()
            current_query = CompiledQuery(
                seed_index=seed_index,
                text=query_match.group("text").strip(),
                language=(
                    current_lang if _is_foreign_language(current_lang) else None
                ),
                strategy=query_match.group("strategy"),
                language_group=current_lang,
                results=[],
            )
            queries.append(current_query)
            seed_index += 1
            continue
        source_match = _SOURCE_HEADER_RE.match(line)
        if source_match:
            _flush_pending()
            if current_query is None:
                current_query = CompiledQuery(
                    seed_index=seed_index,
                    text="",
                    language=(
                        current_lang if _is_foreign_language(current_lang) else None
                    ),
                    strategy=None,
                    language_group=current_lang,
                    results=[],
                )
                queries.append(current_query)
                seed_index += 1
            pending_header = source_match
            pending_lines = []
            continue
        if pending_header is not None:
            pending_lines.append(raw_line)
    _flush_pending()

    if not queries and seed_by_lang:
        # Degenerate report with seeds but no findings — still return metadata.
        for lang_name, items in seed_by_lang.items():
            for strategy, seed_text in items:
                queries.append(
                    CompiledQuery(
                        seed_index=len(queries),
                        text=seed_text,
                        language=(
                            lang_name if _is_foreign_language(lang_name) else None
                        ),
                        strategy=strategy,
                        language_group=lang_name,
                        results=[],
                    )
                )

    return CompiledCorpus(
        prompt=prompt,
        fuzz_mode=fuzz_mode,
        languages=languages,
        queries=queries,
        llm_labels=llm_labels,
    )


@dataclass
class _RenderLlm:
    """Minimal LLM stand-in for :func:`render_markdown` source grouping."""

    label: str
    kind: str = "closed"


@dataclass
class ModelRerunResult:
    """Outcome of overwriting selected models in an existing exploration."""

    explore: ExploreResult
    overwritten_labels: List[str]
    failures: List[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def _corpus_query_seeds(corpus: CompiledCorpus) -> List[Any]:
    from moyo.publicside.barrierprobe.llm_fuzzer import QuerySeed

    return [
        QuerySeed(text=query.text, language=query.language, strategy=query.strategy)
        for query in corpus.queries
    ]


def _render_llms_for_corpus(
    corpus: CompiledCorpus, live: Sequence[LLMClient]
) -> List[Any]:
    by_label = {llm.label: llm for llm in live}
    kinds = {row.llm_label: row.kind for row in corpus.results if row.llm_label}
    ordered: List[Any] = []
    seen: set[str] = set()
    for label in list(corpus.llm_labels) + [llm.label for llm in live]:
        if not label or label in seen:
            continue
        seen.add(label)
        if label in by_label:
            ordered.append(by_label[label])
        else:
            ordered.append(_RenderLlm(label=label, kind=kinds.get(label) or "closed"))
    return ordered


def merge_model_results(
    corpus: CompiledCorpus,
    new_results: Sequence[RetrievalResult],
    labels: Sequence[str],
) -> CompiledCorpus:
    """Replace per-query answers for ``labels`` with ``new_results``."""
    wanted = {str(label).strip() for label in labels if str(label).strip()}
    by_seed: Dict[int, List[RetrievalResult]] = {}
    for row in new_results:
        by_seed.setdefault(int(row.seed_index), []).append(row)
    for query in corpus.queries:
        kept = [row for row in query.results if row.llm_label not in wanted]
        incoming = list(by_seed.get(int(query.seed_index), []))
        combined = kept + incoming
        combined.sort(key=lambda row: (row.llm_index, row.llm_label or ""))
        query.results = combined
    for label in labels:
        text = str(label).strip()
        if text and text not in corpus.llm_labels:
            corpus.llm_labels.append(text)
    return corpus


def selected_model_failures(
    results: Sequence[RetrievalResult], labels: Sequence[str]
) -> List[str]:
    wanted = {str(label).strip() for label in labels if str(label).strip()}
    failures: List[str] = []
    seen: set[str] = set()
    for row in results:
        if row.llm_label not in wanted:
            continue
        seen.add(row.llm_label)
        if row.ok:
            continue
        reason = (row.error or "no content returned").strip() or "no content returned"
        failures.append(
            f"{row.llm_label} [seed {row.seed_index}]: {reason[:160]}"
        )
    for label in labels:
        text = str(label).strip()
        if text and text not in seen:
            failures.append(f"{text}: no retrieval rows returned")
    return failures


def rerun_exploration_models(
    markdown: str,
    retrieval_llms: Sequence[LLMClient],
    *,
    progress: Optional[ProgressFn] = None,
    workers: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> ModelRerunResult:
    """Re-query selected LLMs against existing seeds and overwrite their answers.

    Other models' bodies are kept. Summary synthesis is skipped; the report
    rebuild from parse regenerates claims from the updated markdown.
    """

    def _report(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    llms = [llm for llm in retrieval_llms if llm is not None]
    if not llms:
        raise ValueError("rerun_exploration_models requires at least one retrieval LLM")
    corpus = parse_exploration_markdown(markdown)
    labels = [llm.label for llm in llms]
    label_index = {label: i for i, label in enumerate(corpus.llm_labels)}
    for llm in llms:
        if llm.label not in label_index:
            label_index[llm.label] = len(label_index)

    query_seeds = _corpus_query_seeds(corpus)
    if not query_seeds:
        raise ValueError("exploration.md has no query seeds to rerun")

    jobs = [
        (query.seed_index, seed, label_index[llm.label], llm)
        for query, seed in zip(corpus.queries, query_seeds)
        for llm in llms
    ]
    total = len(jobs)
    global_workers = retrieval_global_concurrency(workers)
    global_workers = min(global_workers, total) if total else 1
    per_provider = min(retrieval_per_provider_concurrency(), global_workers)
    _report(
        f"Rerunning {len(llms)} model(s) across {len(query_seeds)} seed(s) "
        f"({total} retrievals): {', '.join(labels)}"
    )
    raw_results = _run_coro(
        _retrieve_jobs_async(
            jobs,
            max_tokens=max_tokens,
            max_workers=global_workers,
            per_provider_concurrency=per_provider,
            progress=_report,
        )
    )
    record_retrieval_health(llms, raw_results)

    foreign = [row for row in raw_results if row.ok and _is_foreign_language(row.language)]
    if foreign:
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

        fuzzer = LLMFuzzer.for_runtime()
        _report(f"Translating {len(foreign)} foreign-language rerun answer(s)")
        _parallel_localize_results(foreign, fuzzer, workers=workers, progress=_report)

    merge_model_results(corpus, raw_results, labels)
    markdown_out = render_markdown(
        corpus, None, _render_llms_for_corpus(corpus, llms)
    )
    explore = ExploreResult(
        prompt=corpus.prompt,
        seeds=corpus.seeds,
        results=list(corpus.results),
        markdown=markdown_out,
        llm_labels=list(corpus.llm_labels),
        llm_statuses=retrieval_statuses_from_result(
            ExploreResult(
                prompt=corpus.prompt,
                seeds=corpus.seeds,
                results=list(raw_results),
                markdown=markdown_out,
                llm_labels=labels,
            )
        ),
    )
    return ModelRerunResult(
        explore=explore,
        overwritten_labels=labels,
        failures=selected_model_failures(raw_results, labels),
    )


def summarize_exploration(
    exploration_path: str | Path,
    output_path: Optional[str] = None,
    default_llm: Optional[LLMClient] = None,
    impact_definition: Optional[str] = None,
    impact_definition_files: Optional[List[str]] = None,
    progress: Optional[ProgressFn] = None,
) -> ExploreResult:
    """Synthesise ``summary.md`` from an existing ``exploration.md``.

    Does not reword, retrieve, or translate — only parses the report and runs
    the claims-summary synthesis used by :func:`explore_topic`.
    """
    def _report(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    path = Path(exploration_path)
    if not path.is_file():
        raise FileNotFoundError(f"exploration report not found: {path}")

    _report(f"Parsing exploration report {path} ...")
    corpus = parse_exploration_markdown(path.read_text(encoding="utf-8"))
    usable = [r for r in corpus.results if r.ok]
    _report(
        f"Parsed prompt={corpus.prompt!r}; "
        f"{len(corpus.queries)} quer(ies), {len(usable)} usable answer(s) "
        f"from {len(corpus.llm_labels)} LLM label(s)"
    )
    if not usable:
        raise ValueError(
            f"no usable retrieval answers found in {path}; cannot synthesise summary"
        )

    summary_llm = get_summary_llm(default_llm)
    resolved_impact = build_impact_definition(
        extra=impact_definition, extra_files=impact_definition_files
    )
    _report(
        f"Synthesising corroborated claims brief via {summary_llm.label} "
        f"(num_ctx={summary_llm.spec.num_ctx}) ..."
    )
    claims_summary = _synthesize_claims_summary(
        corpus.prompt,
        corpus.results,
        summary_llm,
        impact_definition=resolved_impact,
    )
    if not claims_summary:
        raise RuntimeError("claims summary synthesis returned no content")
    claims_summary = localize_text_for_report(claims_summary)

    summary_target = (
        Path(output_path) if output_path else path.parent / "summary.md"
    )
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.write_text(claims_summary.rstrip() + "\n", encoding="utf-8")
    _report(f"Wrote claims summary to {summary_target}")

    return ExploreResult(
        prompt=corpus.prompt,
        seeds=corpus.seeds,
        results=corpus.results,
        markdown=path.read_text(encoding="utf-8"),
        claims_summary=claims_summary,
        output_path=str(path),
        summary_path=str(summary_target),
        llm_labels=list(corpus.llm_labels),
    )


def _resolve_explore_output_directory(output_directory: Optional[str]) -> Path:
    if output_directory:
        return Path(output_directory)
    from moyo.project import load_saved_project, resolve_public_sources_dir

    saved = load_saved_project()
    if saved is not None:
        saved.ensure()
        return saved.public_sources_dir
    return resolve_public_sources_dir(create=True)


def explore_and_save(
    prompt: str,
    output_directory: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> ExploreResult:
    """Run :func:`explore_topic` and persist ``exploration.md``.

    If ``output_path`` is given, the markdown is written there. Otherwise it is
    written to ``<output_directory>/<slug>/exploration.md``. When
    ``output_directory`` is omitted, the current project's ``public_sources/``
    is used. Explore does not write ``summary.md`` (use ``moyo-gather summarize``
    only if you need that).
    """
    # Explore output is exploration.md only — skip summary synthesis by default.
    kwargs.setdefault("summarize", False)
    result = explore_topic(prompt, **kwargs)

    if output_path:
        target = Path(output_path)
    else:
        slug = _slugify(prompt)
        target = _resolve_explore_output_directory(output_directory) / slug / "exploration.md"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.markdown, encoding="utf-8")
    result.output_path = str(target)
    logger.info("Wrote exploration report to %s", target)
    write_llm_retrieval_check(result, target.parent)
    result.summary_path = None

    return result


def normalize_prompts(prompts: List[str] | str) -> List[str]:
    """Deduplicate non-empty prompts while preserving order."""
    if isinstance(prompts, str):
        items = [prompts]
    else:
        items = list(prompts)
    seen: set[str] = set()
    out: List[str] = []
    for raw in items:
        prompt = " ".join(str(raw or "").split()).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        out.append(prompt)
    return out


def explore_and_save_many(
    prompts: List[str] | str,
    output_directory: Optional[str] = None,
    **kwargs,
) -> List[ExploreResult]:
    """Run :func:`explore_and_save` once per prompt (sequential).

    Each prompt writes to its own ``<output_directory>/<slug>/`` folder.
    ``output_path`` is not supported for batch runs (pass prompts one at a time
    if you need a fixed path).
    """
    if "output_path" in kwargs and kwargs["output_path"] is not None:
        raise ValueError(
            "output_path is not supported for multi-prompt exploration; "
            "omit it so each prompt writes under output_directory/<slug>/"
        )
    all_prompts = normalize_prompts(prompts)
    if not all_prompts:
        raise ValueError("No prompts provided")

    progress = kwargs.get("progress")
    results: List[ExploreResult] = []
    total = len(all_prompts)
    for i, prompt in enumerate(all_prompts, start=1):
        if progress:
            progress(f"[{i}/{total}] Exploring prompt: {prompt}")
        results.append(
            explore_and_save(
                prompt,
                output_directory=output_directory,
                **kwargs,
            )
        )
    return results
