"""GCS folder names for report orders.

Bucket objects live under ``reports/<stamp>_<topic>_<order-suffix>/`` so the
prefix sorts chronologically in Cloud Storage, then by the prompt's primary
subject (Theranos, SenteGuard, Coca-Cola). Firestore still keys documents by
``orderId``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_SORT_STAMP_RE = re.compile(r"^(\d{8}T\d{6}Z)(?:_|$|\s)", re.I)

# Prompt scaffolding and generic descriptors — not the subject of the report.
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "about",
        "be",
        "been",
        "being",
        "background",
        "can",
        "companies",
        "company",
        "controversies",
        "controversy",
        "could",
        "describe",
        "detail",
        "details",
        "did",
        "do",
        "does",
        "explain",
        "fact",
        "facts",
        "for",
        "founder",
        "founders",
        "from",
        "give",
        "had",
        "happen",
        "happened",
        "happens",
        "has",
        "have",
        "her",
        "his",
        "history",
        "how",
        "if",
        "in",
        "info",
        "information",
        "into",
        "involved",
        "involving",
        "is",
        "it",
        "its",
        "known",
        "lesser",
        "little",
        "me",
        "my",
        "of",
        "oinvolved",
        "on",
        "or",
        "our",
        "over",
        "please",
        "product",
        "products",
        "recipe",
        "recipes",
        "related",
        "s",
        "secret",
        "secrets",
        "should",
        "show",
        "some",
        "stories",
        "story",
        "summarize",
        "tell",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "to",
        "under",
        "viability",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
        "would",
        "your",
    }
)


def _tokens(text: str) -> list[str]:
    """Keep Title-Case hyphenates (Coca-Cola); split lesser-known compounds."""
    out: list[str] = []
    for raw in _WORD_RE.findall(text or ""):
        if "-" in raw:
            parts = raw.split("-")
            if all(p[:1].isupper() for p in parts if p):
                out.append(raw)
            else:
                out.extend(p for p in parts if p)
        else:
            out.append(raw)
    return out


def _is_year_or_number(tok: str) -> bool:
    compact = tok.replace("-", "")
    return compact.isdigit() or bool(_YEAR_RE.match(tok))


def _is_content(tok: str) -> bool:
    if not tok or _is_year_or_number(tok):
        return False
    return tok.lower() not in _STOP


def _brand_like(tok: str) -> bool:
    letters = [c for c in tok if c.isalpha()]
    if len(letters) < 2:
        return False
    if "-" in tok and any(p[:1].isupper() for p in tok.split("-") if p):
        return True
    if tok.isupper() and 2 <= len(letters) <= 6:
        return True
    return any(c.isupper() for c in tok[1:])


def _slug_parts(tokens: list[str]) -> list[str]:
    parts: list[str] = []
    for tok in tokens:
        parts.extend(re.findall(r"[a-z0-9]+", tok.lower()))
    return parts


def slugify_topic(text: str, *, max_len: int = 48) -> str:
    """Primary subject of a prompt as a filesystem-safe slug.

    ``Tell me about SenteGuard founder`` → ``senteguard``
    ``What are KFC's secret 11 Herbs and Spices`` → ``kfc``
    ``what is the recipe for Coca-Cola`` → ``coca_cola``
    """
    tokens = _tokens(text)
    remaining = [t for t in tokens if _is_content(t)]
    if not remaining:
        fallback = _slug_parts(tokens)[:3]
        slug = "_".join(fallback) if fallback else "report"
        slug = slug[:max_len].strip("_")
        return slug or "report"

    start = next(i for i, tok in enumerate(tokens) if _is_content(tok))
    first = tokens[start]
    span = [first]
    if not _brand_like(first):
        for tok in tokens[start + 1 :]:
            if not _is_content(tok):
                break
            if _brand_like(tok):
                break
            span.append(tok)
            if len(span) >= 3:
                break

    slug = "_".join(_slug_parts(span))[:max_len].strip("_")
    return slug or "report"


def slugify_first_words(
    text: str,
    *,
    words: int = 3,
    max_len: int = 48,
) -> str:
    """Backward-compatible alias for :func:`slugify_topic`."""
    del words
    return slugify_topic(text, max_len=max_len)


def order_id_suffix(order_id: str, *, length: int = 8) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", (order_id or "").lower())
    if not compact:
        return "order"
    return compact[-max(1, int(length)) :]


def utc_sort_stamp(*, when: datetime | None = None) -> str:
    """Compact UTC stamp that sorts lexicographically (``YYYYMMDDTHHMMSSZ``)."""
    dt = when or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def sort_stamp_from_text(value: str | None) -> str | None:
    """Return a leading or embedded sort stamp from a folder name, title, or order id."""
    text = (value or "").strip()
    if not text:
        return None
    leading = _SORT_STAMP_RE.match(text)
    if leading:
        return leading.group(1).upper()
    match = re.search(r"(\d{8}T\d{6}Z)", text, re.I)
    if match:
        return match.group(1).upper()
    return None


def with_leading_sort_stamp(text: str, stamp: str | None = None) -> str:
    """Prefix ``text`` with a sort stamp unless one is already present."""
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    existing = sort_stamp_from_text(cleaned)
    if existing and cleaned.upper().startswith(existing):
        return cleaned
    prefix = (stamp or utc_sort_stamp()).strip()
    return f"{prefix} {cleaned}"


def order_storage_folder(
    order_id: str,
    prompts: list[str] | None = None,
    *,
    words: int = 3,
    when: datetime | None = None,
) -> str:
    """GCS folder: ``<stamp>_<topic>_<order-suffix>`` for chronological sorting."""
    del words
    prompt = ""
    for raw in prompts or []:
        prompt = str(raw).strip()
        if prompt:
            break
    label = slugify_topic(prompt)
    suffix = order_id_suffix(order_id)
    stamp = sort_stamp_from_text(order_id) or utc_sort_stamp(when=when)
    return f"{stamp}_{label}_{suffix}"
