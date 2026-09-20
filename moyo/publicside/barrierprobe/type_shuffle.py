"""Optional type-similar token shuffle for barrier-probe fuzzing.

Discovers homogeneous slots in a phrase (coordinated lists, generic shapes,
optional embedding clusters) and enumerates substitutions / permutations.
No domain dictionaries — spices, names, vault paths, and IDs are all the
same operator.

This is an a-la-carte strategy (``shuffle``), not part of the default
paraphrase / abstract / summarize rotation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "with",
        "that",
        "this",
        "these",
        "those",
        "from",
        "as",
        "by",
        "it",
        "its",
        "not",
        "at",
        "have",
        "has",
        "had",
        "you",
        "your",
        "we",
        "they",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "into",
        "onto",
        "over",
        "under",
        "than",
        "then",
        "so",
        "if",
        "but",
        "nor",
        "both",
        "each",
        "other",
        "such",
        "per",
        "via",
        "et",
        "ou",
        "und",
        "oder",
    }
)

_CONJ_WORDS = "and|or|et|ou|und|oder"
_ITEM = (
    r"[A-Za-z0-9$][\w./$%:'-]*(?:\s+[A-Za-z0-9$][\w./$%:'-]*){0,4}"
)
_COMMA_LIST = re.compile(
    rf"(?P<region>(?:{_ITEM},\s*){{1,}}(?:(?:{_CONJ_WORDS})\s+)?{_ITEM})",
    re.I,
)
_SPLIT_ITEMS = re.compile(
    rf"\s*,\s*(?:(?:{_CONJ_WORDS})\s+)?|\s+(?:{_CONJ_WORDS})\s+",
    re.I,
)
_CONJ = re.compile(rf"\b({_CONJ_WORDS})\b", re.I)
_TOKEN_PAIR = re.compile(
    rf"\b(?P<a>[\w./$%:'-]+)\s+(?P<conj>{_CONJ_WORDS})\s+(?P<b>[\w./$%:'-]+)\b",
    re.I,
)
_AMOUNT = re.compile(
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|trillion))?"
    r"|\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|trillion)\b",
    re.I,
)
_PERCENT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:percent|per\s*cent)\b|\b\d+(?:\.\d+)?%",
    re.I,
)
_PATH = re.compile(
    r"(?:s3|gs|https?|ftp)://[^\s,;]+"
    r"|(?:(?:[A-Za-z0-9_.-]+/){2,}[A-Za-z0-9_.-]+)",
    re.I,
)
_ID = re.compile(
    r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9_-]{4,}\b"
    r"|\b\d{2,4}[-/]\d{2,4}[-/]\d{2,4}\b"
)
_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'./_-]*")

SHAPE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("amount", _AMOUNT),
    ("percent", _PERCENT),
    ("path", _PATH),
    ("id", _ID),
    ("name", _NAME),
)

EmbedFn = Callable[[List[str]], List[List[float]]]
NeighborFn = Callable[[str], List[str]]

DEFAULT_BUDGET = 16
_CLUSTER_COSINE = 0.72


@dataclass(frozen=True)
class Slot:
    start: int
    end: int
    text: str
    kind: str
    group_id: str


@dataclass(frozen=True)
class ListRegion:
    start: int
    end: int
    items: Tuple[Slot, ...]
    conjunction: str


def similar_texts_from_hits(hits: Optional[Sequence[object]]) -> List[str]:
    """Pull plain text out of fuzzer similar-phrase hits."""
    texts: List[str] = []
    for hit in hits or []:
        if isinstance(hit, str):
            if hit.strip():
                texts.append(hit.strip())
            continue
        if not isinstance(hit, dict):
            continue
        text = hit.get("text") or ""
        if not text:
            meta = hit.get("metadata") or {}
            if isinstance(meta, dict):
                text = meta.get("text") or meta.get("text_preview") or ""
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def discover_slots(
    phrase: str,
    *,
    embed_fn: Optional[EmbedFn] = None,
) -> Tuple[List[ListRegion], List[Slot]]:
    """Find list regions and interchangeable slots in ``phrase``."""
    if not (phrase or "").strip():
        return [], []
    covered: List[Tuple[int, int]] = []
    regions: List[ListRegion] = []
    slots: List[Slot] = []

    for region in _comma_list_regions(phrase):
        regions.append(region)
        slots.extend(region.items)
        covered.append((region.start, region.end))

    for region in _pair_list_regions(phrase, covered):
        regions.append(region)
        slots.extend(region.items)
        covered.append((region.start, region.end))

    next_group = 0
    for kind, pattern in SHAPE_PATTERNS:
        group_id = f"shape:{kind}:{next_group}"
        found = 0
        for match in pattern.finditer(phrase):
            start, end = match.start(), match.end()
            if _overlaps(start, end, covered):
                continue
            text = match.group(0)
            if _is_stopword(text):
                continue
            slots.append(
                Slot(start=start, end=end, text=text, kind=kind, group_id=group_id)
            )
            covered.append((start, end))
            found += 1
        if found:
            next_group += 1

    leftover = _content_spans(phrase, covered)
    clustered = _cluster_spans(leftover, embed_fn=embed_fn)
    slots.extend(clustered)
    return regions, slots


def collect_shuffle_variants(
    phrase: str,
    *,
    similar_texts: Optional[Sequence[str]] = None,
    neighbor_fn: Optional[NeighborFn] = None,
    embed_fn: Optional[EmbedFn] = None,
    budget: int = DEFAULT_BUDGET,
) -> List[str]:
    """Enumerate distinct type-similar rewrites of ``phrase`` (not including it)."""
    regions, slots = discover_slots(phrase, embed_fn=embed_fn)
    if not slots and not regions:
        return []

    harvest_texts = list(similar_texts or [])
    if neighbor_fn:
        probes: List[str] = []
        for region in regions:
            if region.items:
                probes.append(region.items[0].text)
        seen_groups = set()
        for slot in slots:
            if slot.group_id in seen_groups:
                continue
            seen_groups.add(slot.group_id)
            probes.append(slot.text)
        for probe in probes[:4]:
            try:
                harvest_texts.extend(neighbor_fn(probe) or [])
            except Exception:
                logger.debug("shuffle neighbor lookup failed for %r", probe, exc_info=True)

    candidates = _candidates_by_kind(harvest_texts, embed_fn=embed_fn)
    variants: List[str] = []
    seen = {phrase.strip().lower()}

    def _add(text: str) -> None:
        key = (text or "").strip()
        if not key:
            return
        low = key.lower()
        if low in seen:
            return
        seen.add(low)
        variants.append(key)

    for region in regions:
        items = [slot.text for slot in region.items]
        for perm in _limited_permutations(items):
            _add(_replace_span(phrase, region.start, region.end, _format_list(perm, region.conjunction)))
            if len(variants) >= budget:
                return variants
        pool = _pool_for_kind("list", candidates, items, embed_fn)
        for i, item in enumerate(items):
            for cand in pool:
                if cand.lower() == item.lower() or cand.lower() in {x.lower() for x in items}:
                    continue
                new_items = list(items)
                new_items[i] = cand
                _add(
                    _replace_span(
                        phrase,
                        region.start,
                        region.end,
                        _format_list(new_items, region.conjunction),
                    )
                )
                if len(variants) >= budget:
                    return variants

    by_group: Dict[str, List[Slot]] = {}
    for slot in slots:
        by_group.setdefault(slot.group_id, []).append(slot)

    for group_id, group in by_group.items():
        if group_id.startswith("list:"):
            continue
        group = sorted(group, key=lambda s: s.start)
        values = [s.text for s in group]
        if len(group) >= 2:
            for perm in _limited_permutations(values):
                _add(_apply_span_values(phrase, group, perm))
                if len(variants) >= budget:
                    return variants
        kind = group[0].kind
        current = {v.lower() for v in values}
        pool = _pool_for_kind(kind, candidates, values, embed_fn)
        for i, slot in enumerate(group):
            for cand in pool:
                if cand.lower() == slot.text.lower():
                    continue
                if len(group) >= 2 and cand.lower() in current:
                    continue
                new_values = list(values)
                new_values[i] = cand
                _add(_apply_span_values(phrase, group, new_values))
                if len(variants) >= budget:
                    return variants

    return variants[:budget]


def apply_shuffle(
    phrase: str,
    *,
    repeat_index: int = 0,
    similar_texts: Optional[Sequence[str]] = None,
    neighbor_fn: Optional[NeighborFn] = None,
    embed_fn: Optional[EmbedFn] = None,
    budget: int = DEFAULT_BUDGET,
) -> Optional[str]:
    """Return the ``repeat_index``-th shuffle variant, cycling if needed."""
    variants = collect_shuffle_variants(
        phrase,
        similar_texts=similar_texts,
        neighbor_fn=neighbor_fn,
        embed_fn=embed_fn,
        budget=budget,
    )
    if not variants:
        return None
    idx = int(repeat_index) % len(variants)
    return variants[idx]


def _comma_list_regions(phrase: str) -> List[ListRegion]:
    regions: List[ListRegion] = []
    covered: List[Tuple[int, int]] = []
    list_n = 0
    for match in _COMMA_LIST.finditer(phrase):
        start, end = match.start(), match.end()
        if _overlaps(start, end, covered):
            continue
        region_text = match.group("region")
        items = [part.strip() for part in _SPLIT_ITEMS.split(region_text) if part.strip()]
        items = [item for item in items if not _is_stopword(item)]
        if len(items) < 2:
            continue
        conj = _conjunction(region_text)
        group_id = f"list:{list_n}"
        spans = _sequential_item_slots(phrase, start, items, group_id)
        if len(spans) < 2:
            continue
        regions.append(
            ListRegion(start=start, end=end, items=tuple(spans), conjunction=conj)
        )
        covered.append((start, end))
        list_n += 1
    return regions


def _pair_list_regions(
    phrase: str, covered: Sequence[Tuple[int, int]]
) -> List[ListRegion]:
    regions: List[ListRegion] = []
    list_n = 1000
    for match in _TOKEN_PAIR.finditer(phrase):
        start, end = match.start(), match.end()
        if _overlaps(start, end, covered):
            continue
        left, right = match.group("a"), match.group("b")
        if _is_stopword(left) or _is_stopword(right):
            continue
        if left.lower() == right.lower():
            continue
        group_id = f"list:{list_n}"
        a_start = match.start("a")
        b_start = match.start("b")
        items = (
            Slot(start=a_start, end=a_start + len(left), text=left, kind="list", group_id=group_id),
            Slot(start=b_start, end=b_start + len(right), text=right, kind="list", group_id=group_id),
        )
        regions.append(
            ListRegion(
                start=start,
                end=end,
                items=items,
                conjunction=match.group("conj"),
            )
        )
        list_n += 1
    return regions


def _sequential_item_slots(
    phrase: str, region_start: int, items: Sequence[str], group_id: str
) -> List[Slot]:
    slots: List[Slot] = []
    cursor = region_start
    lower = phrase.lower()
    for item in items:
        pos = lower.find(item.lower(), cursor)
        if pos < 0:
            pos = lower.find(item.lower(), region_start)
        if pos < 0:
            continue
        end = pos + len(item)
        slots.append(
            Slot(start=pos, end=end, text=phrase[pos:end], kind="list", group_id=group_id)
        )
        cursor = end
    return slots


def _content_spans(
    phrase: str, covered: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for match in _WORD.finditer(phrase):
        text = match.group(0)
        if _is_stopword(text) or len(text) < 2:
            continue
        start, end = match.start(), match.end()
        if _overlaps(start, end, covered):
            continue
        spans.append((start, end, text))
    return spans


def _cluster_spans(
    spans: Sequence[Tuple[int, int, str]],
    *,
    embed_fn: Optional[EmbedFn],
) -> List[Slot]:
    if embed_fn is None or len(spans) < 2:
        return []
    texts = [s[2] for s in spans]
    try:
        vecs = embed_fn(texts)
    except Exception:
        logger.debug("shuffle clustering embed failed", exc_info=True)
        return []
    if not vecs or len(vecs) != len(spans):
        return []
    parent = list(range(len(spans)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if _cosine(vecs[i], vecs[j]) >= _CLUSTER_COSINE:
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(spans)):
        groups.setdefault(find(i), []).append(i)
    slots: List[Slot] = []
    cluster_n = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        group_id = f"cluster:{cluster_n}"
        cluster_n += 1
        for i in members:
            start, end, text = spans[i]
            slots.append(
                Slot(start=start, end=end, text=text, kind="cluster", group_id=group_id)
            )
    return slots


def _candidates_by_kind(
    texts: Sequence[str],
    *,
    embed_fn: Optional[EmbedFn],
) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    seen: Dict[str, set] = {}
    for text in texts:
        if not text:
            continue
        regions, slots = discover_slots(text, embed_fn=None)
        for region in regions:
            for item in region.items:
                _add_candidate(pools, seen, "list", item.text)
        for slot in slots:
            _add_candidate(pools, seen, slot.kind, slot.text)
    return pools


def _add_candidate(
    pools: Dict[str, List[str]],
    seen: Dict[str, set],
    kind: str,
    value: str,
) -> None:
    key = (value or "").strip()
    if not key or _is_stopword(key):
        return
    bucket = seen.setdefault(kind, set())
    low = key.lower()
    if low in bucket:
        return
    bucket.add(low)
    pools.setdefault(kind, []).append(key)


def _pool_for_kind(
    kind: str,
    candidates: Dict[str, List[str]],
    current: Sequence[str],
    embed_fn: Optional[EmbedFn],
) -> List[str]:
    pool = list(candidates.get(kind, []))
    current_low = {c.lower() for c in current}
    uniq: List[str] = []
    seen = set()
    for item in pool:
        low = item.lower()
        if low in seen or low in current_low:
            continue
        seen.add(low)
        uniq.append(item)
    if embed_fn and uniq and current:
        uniq = _rank_candidates(current[0], uniq, embed_fn)
    return uniq[:12]


def _rank_candidates(
    probe: str, candidates: Sequence[str], embed_fn: EmbedFn
) -> List[str]:
    try:
        vecs = embed_fn([probe, *candidates])
    except Exception:
        logger.debug("shuffle candidate ranking embed failed", exc_info=True)
        return list(candidates)
    if not vecs or len(vecs) != len(candidates) + 1:
        return list(candidates)
    probe_vec = vecs[0]
    scored = [
        (_cosine(probe_vec, vecs[i + 1]), cand) for i, cand in enumerate(candidates)
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1].lower()))
    return [cand for _, cand in scored]


def _limited_permutations(items: Sequence[str]) -> List[List[str]]:
    values = list(items)
    n = len(values)
    if n < 2:
        return []
    seen = {tuple(v.lower() for v in values)}
    out: List[List[str]] = []

    def push(perm: List[str]) -> None:
        key = tuple(v.lower() for v in perm)
        if key in seen:
            return
        seen.add(key)
        out.append(perm)

    for k in range(1, n):
        push(values[k:] + values[:k])
    for i in range(n - 1):
        swapped = list(values)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        push(swapped)
    if n > 2:
        push(list(reversed(values)))
    return out


def _format_list(items: Sequence[str], conjunction: str) -> str:
    clean = [item.strip() for item in items if item and item.strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    conj = conjunction or "and"
    if len(clean) == 2:
        return f"{clean[0]} {conj} {clean[1]}"
    return ", ".join(clean[:-1]) + f", {conj} {clean[-1]}"


def _replace_span(phrase: str, start: int, end: int, new: str) -> str:
    return phrase[:start] + new + phrase[end:]


def _apply_span_values(phrase: str, slots: Sequence[Slot], values: Sequence[str]) -> str:
    repl = [(slot.start, slot.end, value) for slot, value in zip(slots, values)]
    out = phrase
    for start, end, value in sorted(repl, key=lambda row: -row[0]):
        out = out[:start] + value + out[end:]
    return out


def _conjunction(region: str) -> str:
    match = _CONJ.search(region)
    return match.group(1) if match else "and"


def _overlaps(start: int, end: int, covered: Iterable[Tuple[int, int]]) -> bool:
    for cs, ce in covered:
        if start < ce and end > cs:
            return True
    return False


def _is_stopword(text: str) -> bool:
    return text.strip().lower() in STOPWORDS


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(a[i] * a[i] for i in range(n)) ** 0.5
    nb = sum(b[i] * b[i] for i in range(n)) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)
