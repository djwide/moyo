"""Deterministic citation extraction from exploration chunk text."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://[^\s\]\)<>\"']+", re.IGNORECASE)
# Match Sources/References headers even when they trail a prose sentence
# (e.g. "...members. Primary sources for verification include:").
_CITATION_SECTION_RE = re.compile(
    r"(?im)(?:#{1,6}\s*)?(?:\*{0,2})?(?:(?:key|primary|further)\s+)?"
    r"(?:sources?|references?|citations?|bibliography|works cited|documents)"
    r"(?:\s+for\s+verification)?(?:\s+include)?(?:\s+see)?"
    # Allow both `**Sources:**` (colon inside bold) and `**Sources**:`.
    r"(?:\*{0,2})?\s*:?\s*(?:\*{0,2})?\s*$"
)
_INLINE_SOURCES_RE = re.compile(
    r"(?i)(?:\*{0,2}(?:(?:key|primary|further)\s+)?"
    r"(?:sources?|references?|citations?)"
    r"(?:\s+for\s+verification)?(?:\s+include)?"
    r"\*{0,2}\s*:)\s*(.+)$"
)
_CITATION_BULLET_RE = re.compile(
    r"(?m)^\s*(?:[-*•]|\d+[.)]|\[\d+\])\s+(?P<item>.+?)\s*$"
)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
# Numbered reference entries: "[3] https://…", "(3) Texas Tribune…", "3. FEC…"
_REF_ENTRY_RE = re.compile(
    r"^\s*(?:\[(?P<b>\d{1,3})\]|\((?P<p>\d{1,3})\)|(?P<n>\d{1,3})[.)])\s+(?P<body>\S.*)$"
)
# Inline markers on a fact line ("…missed 186 votes.[9]"); not markdown links.
_INLINE_MARKER_RE = re.compile(r"\[(\d{1,3})\](?!\()")


def _balance_wrappers(text: str, *, close_open_pairs: bool = False) -> str:
    """Repair brackets and quotes left dangling by sentence-level splitting.

    ``close_open_pairs`` appends the missing closer instead of leaving a clipped
    parenthetical; only safe once any URL has been split off the string.
    """
    for _ in range(4):
        before = text
        for opener, closer in (("(", ")"), ("[", "]")):
            opens, closes = text.count(opener), text.count(closer)
            if text.endswith(closer) and opens < closes:
                text = text[: -len(closer)].rstrip(" .,;")
            elif text.startswith(opener) and closes < opens:
                text = text[len(opener) :].lstrip()
            elif close_open_pairs and opens > closes:
                text = text.rstrip(" .,;") + closer * (opens - closes)
        if text.count('"') % 2 == 1 and text.endswith('"'):
            text = text[:-1].rstrip(" .,;")
        if text == before:
            break
    return text.strip()


def clean_citation(raw: str) -> str:
    text = (raw or "").strip().rstrip(".,;")
    text = re.sub(r"\*+", "", text).strip()
    return _balance_wrappers(text)


def citation_section_items(text: str) -> list[str]:
    """Ordered entries listed under a Sources / References heading."""
    lines = (text or "").splitlines()
    items: list[str] = []
    i = 0
    while i < len(lines):
        if not _CITATION_SECTION_RE.search(lines[i].strip()):
            i += 1
            continue
        i += 1
        blank_streak = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                blank_streak += 1
                i += 1
                if blank_streak >= 2:
                    break
                continue
            blank_streak = 0
            if line.startswith("#") or _CITATION_SECTION_RE.search(line):
                break
            bullet = _CITATION_BULLET_RE.match(lines[i])
            if not bullet:
                # Only list items under Sources/References count as citations.
                break
            item = clean_citation(bullet.group("item"))
            # Skip category labels like "Official Reports:" and keep leaf citations.
            if item and not item.endswith(":"):
                items.append(item)
            i += 1
    return items


def extract_reference_map(text: str) -> dict[str, str]:
    """Map numbered reference markers (``[9]``) to their citation text.

    Explicitly numbered entries win. When a Sources list is unnumbered but the
    answer carries inline markers — the common search-augmented format — entries
    are mapped by list position, and only if the list is long enough to cover
    every marker used.
    """
    body = text or ""
    lines = body.splitlines()
    in_section = False
    out: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _CITATION_SECTION_RE.search(line):
            in_section = True
            continue
        entry = _REF_ENTRY_RE.match(raw_line)
        if not entry:
            if line.startswith("#"):
                in_section = False
            continue
        item = clean_citation(entry.group("body"))
        if not item or len(item) < 3:
            continue
        if not in_section and not _urls_in(item) and not _MD_LINK_RE.search(item):
            continue
        number = entry.group("b") or entry.group("p") or entry.group("n")
        out.setdefault(str(int(number)), item)

    markers = {int(m.group(1)) for m in _INLINE_MARKER_RE.finditer(body)}
    markers = {n for n in markers if n >= 1}
    if markers and not markers.issubset({int(k) for k in out}):
        items = citation_section_items(body)
        if items and max(markers) <= len(items):
            for n in sorted(markers):
                out.setdefault(str(n), items[n - 1])
    return out


def citations_from_markers(
    text: str,
    reference_map: dict[str, str] | None,
    *,
    limit: int = 8,
) -> list[str]:
    """Resolve inline ``[n]`` markers in ``text`` against a reference map."""
    if not reference_map or not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _INLINE_MARKER_RE.finditer(text):
        key = str(int(match.group(1)))
        cited = reference_map.get(key)
        if not cited or cited.lower() in seen:
            continue
        seen.add(cited.lower())
        out.append(cited)
        if len(out) >= limit:
            break
    return out


def split_citation(raw: str) -> tuple[str, str]:
    """Split a citation string into ``(label, url)``; either may be empty."""
    text = clean_citation(str(raw or ""))
    if not text:
        return "", ""
    link = _MD_LINK_RE.search(text)
    if link:
        label = link.group(1).strip()
        url = link.group(2).strip()
        prefix = text[: link.start()].strip(" :-—")
        return _label_text(label or prefix or url), url
    urls = _urls_in(text)
    if not urls:
        return _label_text(text), ""
    url = urls[0]
    label = text.replace(url, "").strip(" :-—;,")
    return _label_text(label), url


def _label_text(label: str) -> str:
    text = _balance_wrappers(label.strip(" :-—;,"), close_open_pairs=True)
    for opener, closer in (("(", ")"), ("[", "]")):
        while (
            text.startswith(opener)
            and text.endswith(closer)
            and closer not in text[1:-1]
        ):
            text = text[1:-1].strip(" :-—;,")
    return text


def citation_entry(raw: str) -> dict[str, str]:
    """Normalize a citation string into ``label`` / ``url`` / display ``text``."""
    label, url = split_citation(raw)
    if not label and url:
        # Bare URL: name it by host — the full URL is printed alongside.
        host = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
        label = host or url
    return {"label": label, "url": url, "text": clean_citation(str(raw or ""))}


def resolve_claim_citations(
    *,
    claim: str = "",
    excerpt: str = "",
    llm_citations: list[str] | None = None,
    chunk_citations: list[str] | None = None,
    reference_map: dict[str, str] | None = None,
    limit: int = 6,
) -> list[str]:
    """Pick the citations that actually back one claim.

    Markers and Sources tails inside the claim's own excerpt win; only when the
    excerpt carries no citation of its own do the parent chunk's sources apply.
    """
    blob = "\n".join(part for part in (excerpt, claim) if part)
    specific = merge_citations(
        citations_from_markers(blob, reference_map),
        extract_citations(excerpt or ""),
        limit=limit,
    )
    if specific:
        return merge_citations(specific, llm_citations, limit=limit)
    return merge_citations(llm_citations, chunk_citations, limit=limit)


def _urls_in(text: str) -> list[str]:
    return [m.group(0).rstrip(".,);]") for m in _URL_RE.finditer(text or "")]


def extract_citations(text: str, limit: int = 24) -> list[str]:
    """Pull URLs and Sources/References entries from a model response chunk."""
    body = (text or "").strip()
    if not body:
        return []

    found: list[str] = []
    seen: set[str] = set()
    seen_urls: set[str] = set()

    def _remember_urls(item: str) -> None:
        for url in _urls_in(item):
            seen_urls.add(url.lower())

    def _add_simple(cleaned: str) -> None:
        cleaned = clean_citation(cleaned)
        if not cleaned or len(cleaned) < 3:
            return
        key = cleaned.lower()
        if key in seen:
            return
        if _URL_RE.fullmatch(cleaned) and cleaned.lower() in seen_urls:
            return
        # Prefer labeled form over a prior bare URL for the same target.
        urls = _urls_in(cleaned)
        if urls and " — " in cleaned:
            for url in urls:
                bare = url.lower()
                if bare in seen:
                    found[:] = [f for f in found if f.lower() != bare]
                    seen.discard(bare)
        seen.add(key)
        _remember_urls(cleaned)
        found.append(cleaned)

    def _add(item: str) -> None:
        cleaned = clean_citation(item)
        if not cleaned or len(cleaned) < 3:
            return

        md_links = list(_MD_LINK_RE.finditer(cleaned))
        if md_links:
            if len(md_links) == 1:
                match = md_links[0]
                label, url = match.group(1).strip(), match.group(2).strip()
                prefix = cleaned[: match.start()].strip(" :-—")
                suffix = cleaned[match.end() :].strip(" :-—")
                if prefix:
                    display = prefix.rstrip(":").strip() or label
                else:
                    display = label
                if suffix:
                    display = f"{display}; {suffix}".strip(" ;") if display else suffix
                _add_simple(f"{display} — {url}" if display else url)
            else:
                for match in md_links:
                    label, url = match.group(1).strip(), match.group(2).strip()
                    _add_simple(f"{label} — {url}" if label else url)
            return

        _add_simple(cleaned)

    # Prefer structured Sources/References lists first (richer labels).
    lines = body.splitlines()
    for item in citation_section_items(body):
        _add(item)

    # Inline "Sources: ..." tails on fact lines (common in Gemini answers).
    for line in lines:
        raw_line = line.strip()
        m = _INLINE_SOURCES_RE.search(raw_line)
        if not m:
            continue
        # Skip section headers whose citation items are list bullets below.
        if _CITATION_SECTION_RE.search(raw_line) and not m.group(1).strip():
            continue
        # If the "Sources:" marker is at end-of-line (list follows), skip here.
        marker_at_end = _CITATION_SECTION_RE.search(raw_line)
        if marker_at_end and marker_at_end.end() >= len(raw_line):
            continue
        tail = clean_citation(m.group(1))
        if not tail:
            continue
        # Split lightly on semicolons for multiple named sources.
        parts = [p.strip() for p in re.split(r"\s*;\s*", tail) if p.strip()]
        if len(parts) <= 1:
            _add(tail)
        else:
            for part in parts:
                _add(part)

    # Then add any body URLs not already represented.
    for match in _URL_RE.finditer(body):
        _add(match.group(0))

    return found[:limit]


def merge_citations(*groups: list[str] | None, limit: int = 24) -> list[str]:
    """Deduplicate citation strings, preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    seen_urls: set[str] = set()
    for group in groups:
        for item in group or []:
            cleaned = clean_citation(str(item))
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            if _URL_RE.fullmatch(cleaned) and cleaned.lower() in seen_urls:
                continue
            seen.add(key)
            for url in _urls_in(cleaned):
                seen_urls.add(url.lower())
            out.append(cleaned)
            if len(out) >= limit:
                return out
    return out


def _chunk_field(chunk, name: str):
    value = getattr(chunk, name, None)
    if value is None and isinstance(chunk, dict):
        value = chunk.get(name)
    return value


def attach_chunk_citations(
    claims: list[dict],
    chunks: list,
    *,
    overwrite: bool = False,
    limit: int = 6,
) -> list[dict]:
    """Fill claim ``citations`` from the parent chunk's sources.

    Claims whose excerpt carries its own ``[n]`` markers get just those
    references; the rest inherit the chunk's Sources list.
    """
    by_id: dict[str, list[str]] = {}
    refs_by_id: dict[str, dict[str, str]] = {}
    for ch in chunks:
        cid = _chunk_field(ch, "chunk_id")
        if not cid:
            continue
        text = _chunk_field(ch, "text") or ""
        cites = _chunk_field(ch, "citations")
        if cites is None:
            cites = extract_citations(text)
        refs = _chunk_field(ch, "citation_refs")
        if not refs:
            refs = extract_reference_map(text)
        by_id[str(cid)] = list(cites or [])
        refs_by_id[str(cid)] = dict(refs or {})

    for claim in claims:
        existing = list(claim.get("citations") or [])
        if existing and not overwrite:
            continue
        chunk_id = str(claim.get("chunk_id") or "")
        resolved = resolve_claim_citations(
            claim=str(claim.get("claim") or ""),
            excerpt=str(claim.get("raw_excerpt") or ""),
            llm_citations=existing,
            chunk_citations=by_id.get(chunk_id, []),
            reference_map=refs_by_id.get(chunk_id),
            limit=limit,
        )
        if resolved or overwrite:
            claim["citations"] = resolved
    return claims
