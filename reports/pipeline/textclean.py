"""Markdown handling for report output.

Model answers arrive as markdown. The reports are PDFs, so markdown syntax must
never reach the page: text is either flattened to plain prose
(:func:`strip_markdown` / :func:`plain_text`) or converted into real HTML
formatting (:func:`markdown_to_html`, exposed to templates as the ``md`` filter).

No hyperlinks are emitted — the one-pager PDF is required to be link-free, and
resolved URLs are printed in the report's Sources section instead.
"""

from __future__ import annotations

import html
import re
from typing import Any

_FENCE_LINE_RE = re.compile(r"^\s*```+[^\n]*$")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?")
_RULE_RE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
_BULLET_RE = re.compile(r"^(\s{0,8})[-*+•]\s+")
_ORDERED_RE = re.compile(r"^(\s{0,8})(\d{1,3})[.)]\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.S)
_ITALIC_RE = re.compile(r"(?<![\w*_])([*_])(?=[^\s*_])(.+?)(?<=\S)\1(?![\w*_])", re.S)
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.S)
_CODE_RE = re.compile(r"`+([^`]+)`+")
_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!~|>])")
_WS_RE = re.compile(r"[ \t]+")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_BLANKS_RE = re.compile(r"\n{3,}")


def _strip_inline(text: str) -> str:
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(lambda m: m.group(1).strip() or m.group(2), text)
    text = _CODE_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\2", text)
    text = _STRIKE_RE.sub(r"\1", text)
    # Unpaired emphasis markers left behind by truncated model output.
    text = re.sub(r"\*{1,3}", "", text)
    text = _ESCAPE_RE.sub(r"\1", text)
    return text


def _table_row_to_text(line: str) -> str:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return "  ".join(c for c in cells if c)


def strip_markdown(text: Any) -> str:
    """Flatten markdown to plain prose, preserving line and paragraph breaks."""
    raw = str(text or "")
    if not raw.strip():
        return ""
    out: list[str] = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _FENCE_LINE_RE.match(line) or _RULE_RE.match(line):
            continue
        if _TABLE_SEP_RE.match(line) and "|" in line:
            continue
        line = _QUOTE_RE.sub("", line)
        line = _HEADING_RE.sub("", line)
        line = _BULLET_RE.sub(r"\1", line)
        line = _ORDERED_RE.sub(r"\1\2. ", line)
        if line.count("|") >= 2:
            line = _table_row_to_text(line)
        out.append(_strip_inline(line).rstrip())
    joined = "\n".join(out)
    joined = _TRAILING_WS_RE.sub("\n", joined)
    joined = _BLANKS_RE.sub("\n\n", joined)
    return joined.strip()


def plain_text(text: Any) -> str:
    """Single-line plain prose — for table cells, headings, and metadata."""
    flat = strip_markdown(text).replace("\n", " ")
    return _WS_RE.sub(" ", flat).strip()


def _inline_html(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = _IMAGE_RE.sub(r"\1", escaped)
    escaped = _LINK_RE.sub(lambda m: m.group(1).strip() or m.group(2), escaped)
    escaped = _CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD_RE.sub(r"<strong>\2</strong>", escaped)
    escaped = _ITALIC_RE.sub(r"<em>\2</em>", escaped)
    escaped = _STRIKE_RE.sub(r"<s>\1</s>", escaped)
    escaped = re.sub(r"\*{1,3}", "", escaped)
    escaped = _ESCAPE_RE.sub(r"\1", escaped)
    return escaped.strip()


def markdown_to_html(text: Any) -> str:
    """Render markdown as report HTML: emphasis, lists, and paragraphs.

    Everything is escaped first, so model output can never inject markup, and
    no ``<a>`` elements are produced.
    """
    raw = str(text or "")
    if not raw.strip():
        return ""

    blocks: list[str] = []
    list_items: list[str] = []
    list_tag = ""
    para: list[str] = []

    def flush_para() -> None:
        if para:
            body = _inline_html(" ".join(para))
            if body:
                blocks.append(f"<p>{body}</p>")
            para.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if list_items:
            items = "".join(f"<li>{item}</li>" for item in list_items if item)
            if items:
                blocks.append(f"<{list_tag}>{items}</{list_tag}>")
            list_items.clear()
        list_tag = ""

    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _FENCE_LINE_RE.match(line) or _RULE_RE.match(line):
            continue
        if _TABLE_SEP_RE.match(line) and "|" in line:
            continue
        if not line.strip():
            flush_para()
            flush_list()
            continue
        line = _QUOTE_RE.sub("", line)
        heading = _HEADING_RE.match(line)
        if heading:
            flush_para()
            flush_list()
            body = _inline_html(line[heading.end() :])
            if body:
                blocks.append(f'<p class="md-heading">{body}</p>')
            continue
        bullet = _BULLET_RE.match(line)
        ordered = None if bullet else _ORDERED_RE.match(line)
        if bullet or ordered:
            flush_para()
            tag = "ul" if bullet else "ol"
            if list_tag and list_tag != tag:
                flush_list()
            list_tag = tag
            match = bullet or ordered
            list_items.append(_inline_html(line[match.end() :]))
            continue
        flush_list()
        if line.count("|") >= 2:
            line = _table_row_to_text(line)
        para.append(line.strip())

    flush_para()
    flush_list()
    return "".join(blocks)


_MARKDOWN_RESIDUE_RE = re.compile(
    r"\*\*|(?<![\w:/])__|^\s{0,3}#{1,6}\s|\]\(https?://", re.M
)


def find_markdown_residue(html_text: str, *, limit: int = 5) -> list[str]:
    """Return snippets where markdown syntax survived into rendered HTML."""
    text = re.sub(r"(?s)<(script|style)\b.*?</\1>", " ", html_text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    hits: list[str] = []
    for match in _MARKDOWN_RESIDUE_RE.finditer(text):
        start = max(0, match.start() - 40)
        hits.append(" ".join(text[start : match.end() + 40].split()))
        if len(hits) >= limit:
            break
    return hits
