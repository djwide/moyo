"""Naive-prompt topic exploration via multiple LLMs.

Takes a plain-language request from a non-technical user (e.g. "give me all the
info you can on the recipe for Coca-Cola"), rewords it into several effective
retrieval queries via the local Ollama fuzzer (``llama3.1:8b`` through
:mod:`moyo.publicside.barrierprobe.llm_fuzzer` — black-box, no target concept),
then explores each reworded query against every configured retrieval LLM
(closed API, open API and local) in parallel. The result is a single markdown
document summarising everything learned, marked by source of retrieval.

Public entry points:

- :func:`reword_prompt` -- naive prompt -> N reworded query "seeds" (local fuzzer).
- :func:`explore_topic` -- run the full fan-out and build the markdown report.
- :func:`explore_and_save` -- as above, then persist the markdown to disk.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

from moyo.llm.client import LLMClient
from moyo.llm.registry import get_default_llm, get_retrieval_llms

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

RETRIEVAL_SYSTEM = (
    "You are a knowledgeable research assistant. Answer with factual, specific "
    "information about the query. Use short paragraphs or bullet points. If you "
    "are uncertain or lack reliable information, say so briefly rather than "
    "inventing details."
)
SUMMARY_SYSTEM = (
    "You are a research analyst. You synthesise multiple sources into one clear, "
    "well-organised markdown summary, attributing claims to their source when "
    "sources disagree or when a claim is notable."
)
CLAIMS_SUMMARY_SYSTEM = (
    "You are a research analyst who extracts and ranks factual claims from "
    "multi-source retrieval. You are concise, sceptical of single-source "
    "colour, and explicit about corroboration."
)

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

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


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


# --- Prompt rewording -------------------------------------------------------
def _augment_seeds(prompt: str, n: int) -> List[str]:
    """Deterministic seed list when the local fuzzer is unavailable."""
    from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

    return LLMFuzzer._augment_reword_seeds(prompt, [], n)[:n]


def reword_prompt(
    prompt: str,
    llm: Optional[LLMClient] = None,
    n: int = 5,
    fuzzer: Optional[Any] = None,
    fuzz_mode: str = "basic",
) -> List[str]:
    """Reword a naive ``prompt`` into ``n`` distinct retrieval queries.

    Black-box: uses :class:`~moyo.publicside.barrierprobe.llm_fuzzer.LLMFuzzer`
    with the locally running Ollama model ``llama3.1:8b``. No target concept is
    supplied — explore only diversifies the user's request for retrieval.

    ``fuzz_mode`` ``basic`` paraphrases only; ``full`` rotates translate,
    abstract, summarize, and typo. ``llm`` is ignored (kept for call-site
    compatibility). Pass ``fuzzer`` to inject a preconfigured :class:`LLMFuzzer`
    (e.g. in tests).
    """
    del llm  # explore rewording is local-fuzzer-only
    mode = (fuzz_mode or "basic").strip().lower()
    if fuzzer is None:
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer, LLMFuzzerConfig

        fuzzer = LLMFuzzer(LLMFuzzerConfig(fuzz_mode=mode))
    try:
        return fuzzer.reword_for_retrieval(prompt, n=n, fuzz_mode=mode)
    except Exception as exc:
        logger.warning("Local fuzzer rewording failed (%s); using deterministic seeds", exc)
        return _augment_seeds(prompt, n)


# --- Report localization ----------------------------------------------------
def localize_text_for_report(text: str, fuzzer: Optional[Any] = None) -> str:
    """Translate non-English text to English and annotate the source language."""
    body = (text or "").strip()
    if not body:
        return ""
    if fuzzer is None:
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

        fuzzer = LLMFuzzer.local_ollama()
    try:
        return fuzzer.text_for_report(body)
    except Exception as exc:
        logger.warning("Report localization failed (%s); keeping original text", exc)
        return body


def localize_results_for_report(
    results: List[RetrievalResult],
    fuzzer: Optional[Any] = None,
    progress: Optional[ProgressFn] = None,
) -> None:
    """In-place: rewrite retrieval bodies to English with language annotations."""
    from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

    if fuzzer is None:
        fuzzer = LLMFuzzer.local_ollama()

    cache: dict[str, str] = {}
    pending = [r for r in results if r.ok]
    total = len(pending)
    for i, result in enumerate(pending, 1):
        raw = result.text
        if raw not in cache:
            if progress:
                progress(f"Localizing result {i}/{total} for report ...")
            cache[raw] = localize_text_for_report(raw, fuzzer=fuzzer)
        result.text = cache[raw]


# --- Retrieval fan-out ------------------------------------------------------
def retrieve(seed: str, llm: LLMClient, max_tokens: Optional[int] = None) -> RetrievalResult:
    """Query a single LLM with a single reworded seed."""
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
    )
    try:
        result.text = llm.complete(prompt, system=RETRIEVAL_SYSTEM, max_tokens=max_tokens) or ""
    except Exception as exc:
        result.error = str(exc)
        logger.warning("Retrieval failed for %s via %s: %s", seed[:60], llm.label, exc)
    return result


def _synthesize_summary(
    prompt: str, results: List[RetrievalResult], llm: LLMClient
) -> Optional[str]:
    """Ask the default LLM to consolidate all findings into one summary."""
    usable = [r for r in results if r.ok]
    if not usable:
        return None

    corpus = _corpus_blocks(usable)
    ask = (
        f'The user originally asked: "{prompt}".\n\n'
        "Below are answers gathered from several different LLMs for several "
        "reworded versions of the request. Synthesise them into one cohesive "
        "markdown summary of everything learned about the subject. Organise by "
        "theme, note where sources agree or disagree, and flag anything that "
        "seems uncertain or speculative.\n\n"
        f"{corpus}"
    )
    try:
        return llm.complete(ask, system=SUMMARY_SYSTEM, max_tokens=1500)
    except Exception as exc:
        logger.warning("Summary synthesis failed (%s)", exc)
        return None


def _corpus_blocks(results: List[RetrievalResult], per_source_chars: int = 1500) -> str:
    blocks = []
    for r in results:
        if not r.ok:
            continue
        snippet = r.text.strip()
        if len(snippet) > per_source_chars:
            snippet = snippet[:per_source_chars] + " ..."
        blocks.append(f"[Source: {r.llm_label} | query: {r.seed}]\n{snippet}")
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
    source_labels = sorted({r.llm_label for r in usable})
    corpus = _corpus_blocks(usable)

    ask = (
        f'The user originally asked: "{prompt}".\n\n'
        "Below are answers from several retrieval LLMs (each distinct "
        f"`Source:` label is one source). Successful sources: "
        f"{', '.join(source_labels)}.\n\n"
        "Write a *concise* markdown claims brief for a busy reader. Rules:\n"
        "1. Extract atomic factual claims. Merge paraphrases of the same fact.\n"
        "2. Corroboration count = number of *distinct Source labels* that "
        "assert the claim (same LLM across different queries counts once).\n"
        "3. Only treat a claim as corroborated if at least 2 distinct sources "
        "support it.\n"
        "4. Rank primarily by corroboration (higher first), then by impact.\n"
        f"5. High-impact definition: {impact_definition}\n"
        "6. Keep the whole brief short: aim for roughly 15–25 bullets max. "
        "Omit low-impact filler even if corroborated.\n"
        "7. Use this structure exactly:\n"
        f"# Claims summary: {prompt}\n\n"
        "_Ranked by corroboration, then impact. A source is a distinct "
        "retrieval LLM._\n\n"
        "## High-impact corroborated claims\n"
        "Numbered list. Each item: **claim** — *N sources* (`LabelA`, "
        "`LabelB`, ...). One short clause on why it is high-impact.\n\n"
        "## Other corroborated claims\n"
        "Bullets for corroborated but lower-impact facts. Same attribution "
        "format. Skip this section if empty.\n\n"
        "## Contested or single-source (notable only)\n"
        "Only include if high-impact *and* either contested across sources "
        "or supported by a single source. Mark as contested or single-source. "
        "Skip if nothing qualifies.\n\n"
        f"{corpus}"
    )
    try:
        return llm.complete(ask, system=CLAIMS_SUMMARY_SYSTEM, max_tokens=1400)
    except Exception as exc:
        logger.warning("Claims summary synthesis failed (%s)", exc)
        return None


# --- CLI / GUI retrieval summary --------------------------------------------
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


def format_retrieval_table(result: ExploreResult) -> str:
    """Render a per-LLM success/failure summary for explore CLI/GUI output."""
    from collections import OrderedDict

    by_label: OrderedDict[str, List[RetrievalResult]] = OrderedDict()
    for label in result.llm_labels:
        by_label[label] = []
    for r in result.results:
        by_label.setdefault(r.llm_label, []).append(r)

    rows = []
    for label, items in by_label.items():
        n_ok = sum(1 for r in items if r.ok)
        n = len(items)
        if n_ok == n and n > 0:
            status, reason = "ok", ""
        elif n_ok == 0:
            status = "fail"
            sample = next((r for r in items if r.error), None)
            empty = sample is None and any(not r.text.strip() for r in items)
            reason = _short_error(sample.error if sample else None, empty=empty)
        else:
            status = "partial"
            sample = next((r for r in items if not r.ok), None)
            empty = bool(sample and not sample.error and not sample.text.strip())
            reason = (
                f"{n_ok}/{n} ok; "
                + _short_error(sample.error if sample else None, empty=empty)
            )
        rows.append((label, f"{n_ok}/{n}", status, reason))

    headers = ("LLM", "ok", "status", "reason")
    widths = [len(h) for h in headers]
    for label, ok_s, status, reason in rows:
        widths[0] = max(widths[0], len(label))
        widths[1] = max(widths[1], len(ok_s))
        widths[2] = max(widths[2], len(status))
        widths[3] = max(widths[3], len(reason))

    def fmt(cols: tuple) -> str:
        return (
            f"{cols[0]:<{widths[0]}}  "
            f"{cols[1]:<{widths[1]}}  "
            f"{cols[2]:<{widths[2]}}  "
            f"{cols[3]}"
        ).rstrip()

    lines = [fmt(headers), fmt(tuple("-" * w for w in widths))]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


# --- Markdown rendering -----------------------------------------------------
def _kind_label(kind: str) -> str:
    return {"closed": "Closed API", "open": "Open API", "local": "Local"}.get(kind, kind)


def render_markdown(
    prompt: str,
    seeds: List[str],
    results: List[RetrievalResult],
    summary: Optional[str],
    llms: List[LLMClient],
    fuzz_mode: str = "basic",
) -> str:
    lines: List[str] = []
    lines.append(f"# Topic exploration: {prompt}")
    lines.append("")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    mode = (fuzz_mode or "basic").strip().lower()
    lines.append(f"_Fuzz mode: `{mode}`_")
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

    # Reworded seeds.
    lines.append("## Reworded query seeds")
    lines.append("")
    for i, seed in enumerate(seeds, 1):
        lines.append(f"{i}. {seed}")
    lines.append("")

    # Synthesised summary.
    if summary:
        lines.append("## Summary of findings")
        lines.append("")
        lines.append(summary.strip())
        lines.append("")

    # Per-seed detail, marked by source.
    lines.append("## Detailed findings by query and source")
    lines.append("")
    for i, seed in enumerate(seeds, 1):
        lines.append(f"### Seed {i}: {seed}")
        lines.append("")
        seed_results = [r for r in results if r.seed == seed]
        for r in seed_results:
            header = f"#### {r.llm_label}  _({_kind_label(r.kind)})_"
            lines.append(header)
            lines.append("")
            if r.error:
                lines.append(f"> Retrieval failed: {r.error}")
            elif not r.text.strip():
                lines.append("> (no content returned)")
            else:
                lines.append(r.text.strip())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --- Orchestration ----------------------------------------------------------
def explore_topic(
    prompt: str,
    default_llm: Optional[LLMClient] = None,
    retrieval_llms: Optional[List[LLMClient]] = None,
    num_seeds: int = 5,
    summarize: bool = True,
    max_tokens: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
    workers: Optional[int] = None,
    impact_definition: Optional[str] = None,
    impact_definition_files: Optional[List[str]] = None,
    fuzz_mode: str = "basic",
) -> ExploreResult:
    """Run the full naive-prompt exploration and return an :class:`ExploreResult`.

    Retrieval calls (seed × LLM) are independent and run concurrently. ``workers``
    caps how many run at once (default: one per configured retrieval LLM). Pass
    ``workers=1`` to force the old sequential behaviour. Rewording and summary
    synthesis stay serial on the default LLM.

    ``fuzz_mode`` ``basic`` (default) paraphrases the prompt into seeds;
    ``full`` rotates translate / abstract / summarize / typo.

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

    resolved_impact = build_impact_definition(
        extra=impact_definition, extra_files=impact_definition_files
    )

    mode = (fuzz_mode or "basic").strip().lower()
    _report(
        f"Rewording prompt into {num_seeds} query seeds via local LLMFuzzer "
        f"(Ollama llama3.1:8b, black-box, fuzz_mode={mode}) ..."
    )
    seeds = reword_prompt(prompt, n=num_seeds, fuzz_mode=mode)
    _report(f"Seeds: {seeds}")

    jobs = [(seed, llm) for seed in seeds for llm in retrieval_llms]
    total = len(jobs)
    max_workers = len(retrieval_llms) if workers is None else max(1, int(workers))
    max_workers = min(max_workers, total) if total else 1
    _report(f"Retrieving with {max_workers} worker(s) across {total} queries ...")

    # Preserve seed×LLM order in the report regardless of completion order.
    ordered: List[Optional[RetrievalResult]] = [None] * total
    done = 0
    lock = threading.Lock()

    def _run(index: int, seed: str, llm: LLMClient) -> None:
        nonlocal done
        result = retrieve(seed, llm, max_tokens=max_tokens)
        with lock:
            ordered[index] = result
            done += 1
            _report(f"[{done}/{total}] {llm.label}: {seed[:70]}")

    if max_workers == 1:
        for i, (seed, llm) in enumerate(jobs):
            _run(i, seed, llm)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_run, i, seed, llm) for i, (seed, llm) in enumerate(jobs)
            ]
            for fut in as_completed(futures):
                fut.result()  # surface unexpected worker exceptions

    results: List[RetrievalResult] = [r for r in ordered if r is not None]

    _report("Localizing foreign-language retrieval results to English for report ...")
    try:
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

        localize_results_for_report(
            results, fuzzer=LLMFuzzer.local_ollama(), progress=_report
        )
    except Exception as exc:
        logger.warning("Batch localization skipped (%s)", exc)

    summary: Optional[str] = None
    claims_summary: Optional[str] = None
    if summarize:
        _report("Synthesising combined summary ...")
        summary = _synthesize_summary(prompt, results, default_llm)
        if summary:
            summary = localize_text_for_report(summary)
        _report("Synthesising corroborated claims brief ...")
        claims_summary = _synthesize_claims_summary(
            prompt, results, default_llm, impact_definition=resolved_impact
        )
        if claims_summary:
            claims_summary = localize_text_for_report(claims_summary)

    markdown = render_markdown(
        prompt, seeds, results, summary, retrieval_llms, fuzz_mode=mode
    )
    return ExploreResult(
        prompt=prompt,
        seeds=seeds,
        results=results,
        markdown=markdown,
        summary=summary,
        claims_summary=claims_summary,
        llm_labels=[llm.label for llm in retrieval_llms],
    )


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug or "exploration")[:max_len]


def explore_and_save(
    prompt: str,
    output_directory: str = "data/public_sources",
    output_path: Optional[str] = None,
    **kwargs,
) -> ExploreResult:
    """Run :func:`explore_topic` and persist the markdown report.

    If ``output_path`` is given, the markdown is written there. Otherwise it is
    written to ``<output_directory>/<slug>/exploration.md``. A concise
    corroborated-claims brief is written beside it as ``summary.md`` when
    synthesis succeeds.
    """
    result = explore_topic(prompt, **kwargs)

    if output_path:
        target = Path(output_path)
    else:
        slug = _slugify(prompt)
        target = Path(output_directory) / slug / "exploration.md"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.markdown, encoding="utf-8")
    result.output_path = str(target)
    logger.info("Wrote exploration report to %s", target)

    if result.claims_summary:
        summary_target = target.parent / "summary.md"
        summary_target.write_text(result.claims_summary.rstrip() + "\n", encoding="utf-8")
        result.summary_path = str(summary_target)
        logger.info("Wrote claims summary to %s", summary_target)

    return result
