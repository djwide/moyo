"""Prompt-language helpers for English-only report presentation.

Findings may originate from foreign-language prompting. Reports always present
the English form of each finding and annotate the prompt language separately.
"""

from __future__ import annotations

import re
from typing import Any, Callable


_ENGLISH_HINT_WORDS = frozenset(
    {
        "the",
        "and",
        "of",
        "to",
        "in",
        "a",
        "is",
        "that",
        "for",
        "on",
        "with",
        "as",
        "was",
        "were",
        "by",
        "from",
        "at",
        "an",
        "be",
        "or",
        "which",
        "this",
        "are",
        "have",
        "has",
        "had",
        "not",
        "but",
        "their",
        "they",
        "his",
        "her",
        "its",
        "who",
        "whom",
        "into",
        "about",
        "after",
        "before",
        "during",
        "while",
        "when",
        "where",
        "what",
        "than",
        "then",
        "also",
        "been",
        "being",
        "would",
        "could",
        "should",
        "may",
        "might",
        "will",
        "can",
        "said",
        "according",
        "report",
        "reports",
        "commission",
        "president",
        "assassination",
    }
)

_LANGUAGES_LINE_RE = re.compile(
    r"_Languages:\s*(.+?)_",
    re.I,
)


def is_foreign_language(language: str | None) -> bool:
    lang = (language or "").strip().lower()
    return bool(lang) and lang not in {"english", "en", "eng"}


def prompt_language(finding: dict[str, Any]) -> str | None:
    """Language the model was prompted in (seed language), if known."""
    lang = (finding.get("language") or finding.get("prompt_language") or "").strip()
    if not lang:
        # Fall back to trailing "(Language)" on source_model labels.
        sm = (finding.get("source_model") or "").strip()
        if sm.endswith(")") and " (" in sm:
            base, last = sm.rsplit(" (", 1)
            tag = last[:-1].strip()
            if tag and not any(ch.isdigit() for ch in tag) and len(tag) < 40:
                if is_foreign_language(tag):
                    return tag
        return None
    if is_foreign_language(lang):
        return lang
    return None


def looks_like_english(text: str) -> bool:
    """Heuristic: mostly Latin letters plus common English function words."""
    sample = (text or "").strip()
    if not sample:
        return True
    # Any CJK / Hangul / Kana / Arabic / Cyrillic → not English for report display.
    if any(
        ("\u4e00" <= ch <= "\u9fff")
        or ("\u3400" <= ch <= "\u4dbf")
        or ("\u3040" <= ch <= "\u30ff")
        or ("\uac00" <= ch <= "\ud7af")
        or ("\u0600" <= ch <= "\u06ff")
        or ("\u0400" <= ch <= "\u04ff")
        for ch in sample
    ):
        return False
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
        ascii_letters = sum(
            1 for ch in sample if ("A" <= ch <= "Z") or ("a" <= ch <= "z")
        )
        return (not letters) or (ascii_letters / max(letters, 1) > 0.92)
    hits = sum(1 for t in tokens if t in _ENGLISH_HINT_WORDS)
    return hits / len(tokens) >= 0.12


def parse_languages_line(text: str) -> list[str]:
    """Parse ``_Languages: English + Spanish, French, Mandarin Chinese_``."""
    m = _LANGUAGES_LINE_RE.search(text or "")
    if not m:
        return []
    raw = m.group(1).strip()
    # "English + Spanish, French, Mandarin Chinese (…)" → drop parenthetical note
    raw = re.sub(r"\([^)]*\)\s*$", "", raw).strip()
    parts: list[str] = []
    for bit in re.split(r"\s*\+\s*|,|;", raw):
        name = bit.strip().strip("`").strip()
        if name and name.lower() not in {"full strategy set per language"}:
            parts.append(name)
    # Stable unique, English first
    out: list[str] = []
    seen: set[str] = set()
    for name in parts:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    eng = [x for x in out if not is_foreign_language(x)]
    foreign = [x for x in out if is_foreign_language(x)]
    return eng + foreign


def languages_from_findings(findings: list[dict[str, Any]]) -> list[str]:
    """Distinct prompt languages observed on findings (English first)."""
    seen: set[str] = set()
    foreign: list[str] = []
    has_english = False
    for f in findings or []:
        lang = (f.get("language") or f.get("prompt_language") or "").strip()
        if not lang:
            pl = prompt_language(f)
            lang = pl or ""
        if not lang:
            continue
        key = lang.lower()
        if key in seen:
            continue
        seen.add(key)
        if is_foreign_language(lang):
            foreign.append(lang)
        else:
            has_english = True
    out: list[str] = []
    if has_english or foreign:
        out.append("English")
    out.extend(foreign)
    return out


def language_annotation(language: str | None) -> str:
    lang = (language or "").strip()
    if not is_foreign_language(lang):
        return ""
    return f"via {lang} prompting"


TranslateFn = Callable[[str], str]


def _english_peer_claim(
    finding: dict[str, Any],
    by_cluster: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    cid = finding.get("cluster_id")
    if not cid:
        return None
    peers = by_cluster.get(cid) or []
    # Prefer peers tagged English whose claim text looks English.
    for p in peers:
        if p.get("claim_id") == finding.get("claim_id"):
            continue
        if is_foreign_language(p.get("language")):
            continue
        if looks_like_english(str(p.get("claim") or "")):
            return p
    # Any English-looking peer claim.
    for p in peers:
        if p.get("claim_id") == finding.get("claim_id"):
            continue
        if looks_like_english(str(p.get("claim") or "")):
            return p
    return None


def englishize_findings(
    findings: list[dict[str, Any]],
    *,
    translate: TranslateFn | None = None,
) -> list[dict[str, Any]]:
    """Return findings with English claim/excerpt text + prompt-language notes.

    Non-English claim text is replaced by an English cluster peer when
    available, otherwise by ``translate`` when provided. Foreign-language
    evidence excerpts are not shown; the English claim stands as the finding.
    """
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for f in findings or []:
        cid = f.get("cluster_id")
        if cid:
            by_cluster.setdefault(str(cid), []).append(f)

    out: list[dict[str, Any]] = []
    for f in findings or []:
        row = dict(f)
        pl = prompt_language(row)
        if pl:
            row["prompt_language"] = pl
            row["language_annotation"] = language_annotation(pl)
        else:
            row["prompt_language"] = None
            row["language_annotation"] = ""

        claim = str(row.get("claim") or "").strip()
        excerpt = str(row.get("raw_excerpt") or "").strip()
        claim_en = looks_like_english(claim)
        excerpt_en = looks_like_english(excerpt)
        row["english_pending"] = False

        if not claim_en:
            peer = _english_peer_claim(row, by_cluster)
            if peer and looks_like_english(str(peer.get("claim") or "")):
                row["claim_original"] = claim
                row["claim"] = str(peer.get("claim")).strip()
                claim = row["claim"]
                claim_en = True
                if looks_like_english(str(peer.get("raw_excerpt") or "")):
                    row["raw_excerpt"] = str(peer.get("raw_excerpt")).strip()
                    excerpt = row["raw_excerpt"]
                    excerpt_en = True
            elif translate is not None:
                translated = (translate(claim) or "").strip()
                if translated and looks_like_english(translated):
                    row["claim_original"] = claim
                    row["claim"] = translated
                    claim = translated
                    claim_en = True

        if not claim_en:
            # Never present foreign-language claim text in report products.
            lang = pl or (row.get("language") or "non-English")
            start = row.get("raw_start_line")
            end = row.get("raw_end_line") or start
            loc = f" (exploration lines {start}–{end})" if start else ""
            row["claim_original"] = claim
            row["claim"] = (
                f"Finding recovered via {lang} prompting{loc}; "
                f"English claim text was not available at report build time."
            )
            row["english_pending"] = True
            claim_en = True
            if not row.get("language_annotation"):
                row["language_annotation"] = language_annotation(pl or lang)
            if pl or is_foreign_language(lang):
                row["prompt_language"] = pl or lang
        if not excerpt_en:
            # Never present foreign-language evidence in the report body.
            row["raw_excerpt_original"] = excerpt
            row["raw_excerpt"] = ""
            if pl and not row.get("language_annotation"):
                row["language_annotation"] = language_annotation(pl)

        out.append(row)
    return out


def default_translate_fn() -> TranslateFn | None:
    """Best-effort local translator; returns None if unavailable."""
    try:
        from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer

        fuzzer = LLMFuzzer.local_ollama()
    except Exception:
        return None

    def _translate(text: str) -> str:
        try:
            return (fuzzer.localize_to_english(text).english or "").strip()
        except Exception:
            return text

    return _translate
