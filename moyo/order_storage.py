"""GCS folder names for report orders.

Bucket objects live under ``reports/<topic>_<order-suffix>/`` so the prefix
is the prompt's primary subject (Theranos, SenteGuard, Coca-Cola) instead of
a generic ``ord_xxx`` id or the first few prompt words. Firestore still keys
documents by ``orderId``.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

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


def order_storage_folder(
    order_id: str,
    prompts: list[str] | None = None,
    *,
    words: int = 3,
) -> str:
    """GCS folder name: primary prompt topic plus a short unique order suffix."""
    del words
    prompt = ""
    for raw in prompts or []:
        prompt = str(raw).strip()
        if prompt:
            break
    label = slugify_topic(prompt)
    suffix = order_id_suffix(order_id)
    return f"{label}_{suffix}"
