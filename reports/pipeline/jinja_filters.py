"""Jinja filters that format values for print — never dump raw LLM strings."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def format_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "0"


def format_number(value: Any, digits: int = 1) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "—"
    if n == int(n):
        return f"{int(n):,}"
    return f"{n:,.{int(digits)}f}"


def format_score(value: Any, max_score: int = 5) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "—"
    return f"{n}/{int(max_score)}"


def display_status(value: Any) -> str:
    """Sentence-style label. Phrases that already contain spaces stay phrases."""
    raw = str(value or "").strip().replace("_", "-")
    if not raw:
        return ""
    if " " in raw and any(ch.islower() for ch in raw):
        return raw
    parts = [p for p in raw.replace(" ", "-").split("-") if p]
    if not parts:
        return ""
    head = parts[0].capitalize()
    tail = [p.lower() for p in parts[1:]]
    return "-".join([head, *tail])


def clip(value: Any, length: int = 150, end: str = "…") -> str:
    text = " ".join(str(value or "").split())
    limit = max(1, int(length))
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - len(end))].rsplit(" ", 1)
    head = (cut[0] if cut else text[:limit]).rstrip(".,;: ")
    return (head or text[:limit]).rstrip() + end


def sentence_case(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isupper() and " " in text:
        return text.capitalize()
    return text


def format_timestamp(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    return d.strftime("%-d %b %Y · %H:%M UTC")


_CITE_REF_RE = re.compile(r"\b(S\d+)\b")


def link_cites(value: Any) -> Any:
    """Turn compact source refs such as ``S21`` into in-document links."""
    from markupsafe import Markup, escape

    text = str(value or "")
    if not text:
        return ""
    linked = _CITE_REF_RE.sub(
        r'<a class="cite-ref" href="#cite-\1">\1</a>',
        str(escape(text)),
    )
    return Markup(linked)


def register_filters(env: Any) -> None:
    env.filters["format_int"] = format_int
    env.filters["format_number"] = format_number
    env.filters["format_score"] = format_score
    env.filters["display_status"] = display_status
    env.filters["clip"] = clip
    env.filters["sentence_case"] = sentence_case
    env.filters["format_timestamp"] = format_timestamp
    env.filters["link_cites"] = link_cites
