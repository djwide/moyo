"""Feed approved phrases + a public pack to Kimi; parse how they differ."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

from moyo.privateside.phrases.schema import LABELS, PhraseRecord

CompleteFn = Callable[..., str]
ProgressFn = Callable[[str], None]

CHAR_BUDGET = 80_000
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_ID_RE = re.compile(r"ph_[a-f0-9]{12}", re.I)

SYSTEM = """You compare an organization's approved private sensitive phrases against a public-sources pack.

This is a qualitative judgment, not embedding similarity. Decide whether each private phrase is already attested in the public pack (paraphrase counts) or still private-only.

Return JSON only, no markdown:
{
  "headline": "at most three sentences on how the two corpora differ",
  "only_private": [{"id": "ph_...", "text": "phrase", "reason": "why this is not in public"}],
  "overlap": [{"id": "ph_...", "text": "phrase", "quote": "short public excerpt", "reason": "how it appears"}],
  "only_public": [{"text": "public claim with no private counterpart", "quote": "excerpt"}],
  "caveats": ["truncation, language mix, or uncertainty"]
}

Use the given phrase ids. Every private phrase must appear in only_private or overlap, never both.
only_public is public material that does not correspond to any listed private phrase.
Do not invent private phrases that were not provided.
If the public pack was truncated, say so in caveats."""


@dataclass
class PackedSide:
    """One side of the prompt (private phrases or public pack)."""

    kind: str
    path: str
    text: str
    chars: int
    truncated: bool
    item_count: int = 0
    omitted_chars: int = 0
    omitted_items: int = 0


@dataclass
class CompareItem:
    """One phrase or public-only claim in the compare result."""

    id: str = ""
    text: str = ""
    label: str = ""
    verdict: str = ""
    quote: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompareItem":
        return cls(
            id=str(data.get("id") or ""),
            text=str(data.get("text") or "").strip(),
            label=str(data.get("label") or ""),
            verdict=str(data.get("verdict") or ""),
            quote=str(data.get("quote") or "").strip(),
            reason=str(data.get("reason") or "").strip(),
        )


@dataclass
class CompareResult:
    """Structured Kimi delta between private phrases and a public pack."""

    headline: str = ""
    only_private: list[CompareItem] = field(default_factory=list)
    overlap: list[CompareItem] = field(default_factory=list)
    only_public: list[CompareItem] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    phrase_rows: list[CompareItem] = field(default_factory=list)
    packing: dict[str, Any] = field(default_factory=dict)
    model: str = "kimi-k2.6"
    created_at: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "only_private": [i.to_dict() for i in self.only_private],
            "overlap": [i.to_dict() for i in self.overlap],
            "only_public": [i.to_dict() for i in self.only_public],
            "caveats": list(self.caveats),
            "phrase_rows": [i.to_dict() for i in self.phrase_rows],
            "packing": self.packing,
            "model": self.model,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompareResult":
        def _items(key: str) -> list[CompareItem]:
            rows = data.get(key) or []
            return [CompareItem.from_dict(r) for r in rows if isinstance(r, dict)]

        return cls(
            headline=str(data.get("headline") or "").strip(),
            only_private=_items("only_private"),
            overlap=_items("overlap"),
            only_public=_items("only_public"),
            caveats=[str(c).strip() for c in (data.get("caveats") or []) if str(c).strip()],
            phrase_rows=_items("phrase_rows"),
            packing=dict(data.get("packing") or {}),
            model=str(data.get("model") or "kimi-k2.6"),
            created_at=str(data.get("created_at") or ""),
        )


def kimi_compare_spec():
    from moyo.llm.client import LLMSpec

    return LLMSpec.from_dict(
        {
            "provider": "custom",
            "model": "kimi-k2.6",
            "base_url": "https://api.moonshot.ai/v1",
            "api_key": "$MOONSHOT_API_KEY",
            "temperature": 0.2,
            "max_tokens": 6000,
            "timeout": 180,
            "label": "Kimi (naive compare)",
        }
    )


def kimi_compare_complete(prompt: str, system: str | None = None) -> str:
    from moyo.llm.client import LLMClient, llm_spec_has_auth

    spec = kimi_compare_spec()
    if not llm_spec_has_auth(spec):
        raise RuntimeError(
            "MOONSHOT_API_KEY is not set. Kimi is required to compare corpora."
        )
    client = LLMClient(spec)
    if not client.is_available():
        raise RuntimeError(client.init_error or "Kimi client is unavailable.")
    return client.complete(prompt, system=system or SYSTEM, max_tokens=6000)


def run_naive_compare(
    *,
    project=None,
    phrases_dir: Union[str, Path, None] = None,
    exploration_path: Union[str, Path, None] = None,
    sources_dir: Union[str, Path, None] = None,
    complete: CompleteFn | None = None,
    progress: ProgressFn | None = None,
    persist: bool = True,
    char_budget: int = CHAR_BUDGET,
) -> CompareResult:
    """Pack approved phrases + public sources, ask Kimi, return structured buckets."""
    from moyo.privateside.phrases.store import PhraseStore

    store_root = Path(phrases_dir) if phrases_dir else None
    if store_root is None:
        if project is None:
            raise ValueError("Select a project (or pass phrases_dir).")
        store_root = Path(project.phrases_dir)
    store = PhraseStore(store_root)
    phrases = store.load_approved()
    if not phrases:
        raise ValueError(
            "No approved private phrases. Extract and approve them in Private Data Input."
        )

    public = resolve_public_pack(
        project,
        exploration_path=exploration_path,
        sources_dir=sources_dir,
    )
    packed_private, packed_public, user_prompt = pack_compare_prompt(
        phrases, public, char_budget=char_budget
    )
    if progress:
        progress(
            f"Packed {packed_private.item_count} phrases "
            f"({packed_private.chars} chars) + {packed_public.kind} "
            f"({packed_public.chars} chars"
            f"{', truncated' if packed_public.truncated else ''})."
        )
        progress("Calling Kimi…")

    fn = complete or _default_complete()
    raw = fn(user_prompt, SYSTEM)
    parsed = parse_compare_payload(raw)
    if parsed is None:
        snippet = (raw or "").strip()[:400]
        raise RuntimeError(f"Kimi did not return compare JSON. Preview:\n{snippet}")

    result = assemble_result(
        phrases,
        parsed,
        packed_private=packed_private,
        packed_public=packed_public,
        raw=raw,
        char_budget=char_budget,
    )
    if persist and project is not None:
        path = save_result(project, result)
        if progress:
            progress(f"Saved {path}")
    return result


def pack_compare_prompt(
    phrases: Iterable[PhraseRecord],
    public: PackedSide,
    *,
    char_budget: int = CHAR_BUDGET,
) -> tuple[PackedSide, PackedSide, str]:
    """Build the user prompt. Phrases go in first; public gets the remainder."""
    records = [p for p in phrases if p.text]
    header = (
        "Compare these approved private phrases to the public pack. JSON only.\n\n"
        "PRIVATE PHRASES (id, label, text):\n"
    )
    footer_prefix = "\n\nPUBLIC PACK"
    reserved = len(header) + len(footer_prefix) + 80
    private_budget = max(2_000, min(len(_format_phrases(records)) + 1, char_budget // 4))
    public_budget = max(1_000, char_budget - reserved - private_budget)

    private_text, private_kept, omitted = _fit_phrases(records, private_budget)
    packed_private = PackedSide(
        kind="phrases",
        path="",
        text=private_text,
        chars=len(private_text),
        truncated=omitted > 0,
        item_count=private_kept,
        omitted_items=omitted,
    )
    leftover = char_budget - reserved - packed_private.chars
    public_budget = max(1_000, leftover)
    packed_public = _truncate_side(public, public_budget)

    loc = packed_public.path or packed_public.kind
    user = (
        f"{header}{packed_private.text}\n\n"
        f"PUBLIC PACK ({packed_public.kind}: {loc}"
        f"{'; truncated' if packed_public.truncated else ''}):\n"
        f"{packed_public.text}"
    )
    return packed_private, packed_public, user


def resolve_public_pack(
    project=None,
    *,
    exploration_path: Union[str, Path, None] = None,
    sources_dir: Union[str, Path, None] = None,
) -> PackedSide:
    """Prefer ``extracted.json``; else exploration.md; else sources.json."""
    extracted = _find_extracted_file(project, sources_dir, exploration_path)
    if extracted is not None:
        packed = _pack_extracted_json(extracted)
        if packed.item_count or packed.text.strip():
            return packed

    explicit = Path(exploration_path) if exploration_path else None
    if explicit and explicit.is_file():
        if explicit.name == "extracted.json":
            packed = _pack_extracted_json(explicit)
            if packed.item_count or packed.text.strip():
                return packed
        return _read_text_side(explicit, kind="exploration")

    if project is not None and not sources_dir:
        found = list(project.find_explorations() or [])
        if found:
            return _read_text_side(found[0], kind="exploration")

    directory = Path(sources_dir) if sources_dir else None
    if directory is None and project is not None:
        dirs = list(project.find_sources_dirs() or [])
        directory = dirs[0] if dirs else Path(project.public_sources_dir)
    if directory is None:
        raise ValueError(
            "No public pack. Extract relevant text on Build Public Corpus, "
            "or gather public sources."
        )

    if directory.is_file() and directory.name == "extracted.json":
        packed = _pack_extracted_json(directory)
        if packed.item_count or packed.text.strip():
            return packed

    if directory.is_file() and directory.name == "exploration.md":
        return _read_text_side(directory, kind="exploration")

    expl = directory / "exploration.md" if directory.is_dir() else None
    if expl is not None and expl.is_file():
        return _read_text_side(expl, kind="exploration")
    if directory.is_dir():
        nested = sorted(directory.rglob("exploration.md"))
        if nested:
            return _read_text_side(nested[0], kind="exploration")

    packed = _pack_sources_json(directory)
    if packed.item_count == 0 and not packed.text.strip():
        raise ValueError(
            f"No extracted.json, exploration.md, or sources.json under {directory}. "
            "Run Extract relevant text on Build Public Corpus first."
        )
    return packed


def _find_extracted_file(
    project,
    sources_dir: Union[str, Path, None],
    exploration_path: Union[str, Path, None],
) -> Optional[Path]:
    from moyo.publicside.gatherpublicsources.extract import EXTRACTED_FILE_NAME, extracted_path

    candidates: list[Path] = []
    explicit = Path(exploration_path) if exploration_path else None
    if explicit:
        if explicit.is_file() and explicit.name == EXTRACTED_FILE_NAME:
            candidates.append(explicit)
        elif explicit.is_dir():
            candidates.append(extracted_path(explicit))
    if sources_dir:
        src = Path(sources_dir)
        candidates.append(src if src.is_file() else extracted_path(src))
        if src.is_dir():
            candidates.extend(sorted(src.rglob(EXTRACTED_FILE_NAME)))
    if project is not None:
        found = getattr(project, "find_extracted", lambda: None)()
        if found:
            candidates.append(Path(found))
        public = getattr(project, "public_sources_dir", None)
        if public:
            candidates.append(extracted_path(public))
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        return path
    return None


def parse_compare_payload(raw: str) -> dict[str, Any] | None:
    """Parse Kimi JSON into headline / bucket lists."""
    payload = _load_json(raw)
    if not isinstance(payload, dict):
        return None
    headline = str(payload.get("headline") or payload.get("summary") or "").strip()
    only_private = _bucket_rows(
        payload.get("only_private") or payload.get("private_only") or []
    )
    overlap = _bucket_rows(payload.get("overlap") or payload.get("shared") or [])
    only_public = _bucket_rows(
        payload.get("only_public") or payload.get("public_only") or []
    )
    caveats = payload.get("caveats") or payload.get("notes") or []
    if isinstance(caveats, str):
        caveats = [caveats]
    caveat_list = [str(c).strip() for c in caveats if str(c).strip()]
    if not headline and not only_private and not overlap and not only_public:
        return None
    return {
        "headline": headline,
        "only_private": only_private,
        "overlap": overlap,
        "only_public": only_public,
        "caveats": caveat_list,
    }


def assemble_result(
    phrases: list[PhraseRecord],
    parsed: dict[str, Any],
    *,
    packed_private: PackedSide,
    packed_public: PackedSide,
    raw: str,
    char_budget: int = CHAR_BUDGET,
) -> CompareResult:
    """Attach PhraseStore labels and fill unscored private rows."""
    by_id = {p.id: p for p in phrases}
    by_text = {_key(p.text): p for p in phrases}
    overlap_ids: set[str] = set()
    private_ids: set[str] = set()

    overlap_items = []
    for row in parsed.get("overlap") or []:
        rec = _match_phrase(row, by_id, by_text)
        item = _item_from_row(row, rec, verdict="overlap")
        overlap_items.append(item)
        if rec:
            overlap_ids.add(rec.id)

    only_private_items = []
    for row in parsed.get("only_private") or []:
        rec = _match_phrase(row, by_id, by_text)
        if rec and rec.id in overlap_ids:
            continue
        item = _item_from_row(row, rec, verdict="private-only")
        only_private_items.append(item)
        if rec:
            private_ids.add(rec.id)

    only_public_items = [
        CompareItem(
            text=str(row.get("text") or "").strip(),
            verdict="public-only",
            quote=str(row.get("quote") or "").strip(),
            reason=str(row.get("reason") or "").strip(),
        )
        for row in parsed.get("only_public") or []
        if str(row.get("text") or "").strip()
    ]

    phrase_rows: list[CompareItem] = []
    for rec in phrases:
        if rec.id in overlap_ids:
            matched = next((i for i in overlap_items if i.id == rec.id), None)
            phrase_rows.append(
                matched
                or CompareItem(
                    id=rec.id, text=rec.text, label=rec.label, verdict="overlap"
                )
            )
        elif rec.id in private_ids:
            matched = next((i for i in only_private_items if i.id == rec.id), None)
            phrase_rows.append(
                matched
                or CompareItem(
                    id=rec.id, text=rec.text, label=rec.label, verdict="private-only"
                )
            )
        else:
            phrase_rows.append(
                CompareItem(
                    id=rec.id,
                    text=rec.text,
                    label=rec.label,
                    verdict="unscored",
                    reason="Kimi did not classify this phrase",
                )
            )

    packing = {
        "budget": char_budget,
        "used": packed_private.chars + packed_public.chars,
        "private": {
            "kind": packed_private.kind,
            "chars": packed_private.chars,
            "items": packed_private.item_count,
            "truncated": packed_private.truncated,
            "omitted_items": packed_private.omitted_items,
        },
        "public": {
            "kind": packed_public.kind,
            "path": packed_public.path,
            "chars": packed_public.chars,
            "items": packed_public.item_count,
            "truncated": packed_public.truncated,
            "omitted_chars": packed_public.omitted_chars,
        },
    }
    return CompareResult(
        headline=str(parsed.get("headline") or "").strip(),
        only_private=only_private_items,
        overlap=overlap_items,
        only_public=only_public_items,
        caveats=list(parsed.get("caveats") or []),
        phrase_rows=phrase_rows,
        packing=packing,
        model="kimi-k2.6",
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        raw=raw,
    )


def private_only_by_label(result: CompareResult) -> dict[str, int]:
    """Counts of private-only phrase rows keyed by PhraseStore label."""
    counts = {name: 0 for name in LABELS}
    for row in result.phrase_rows:
        if row.verdict != "private-only":
            continue
        label = row.label if row.label in counts else "other"
        counts[label] += 1
    return counts


def save_result(project, result: CompareResult) -> Path:
    directory = Path(project.compare_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "last.json"
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_result(project) -> Optional[CompareResult]:
    path = Path(project.compare_dir) / "last.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return CompareResult.from_dict(data)


def _default_complete() -> CompleteFn:
    try:
        from moyo.llm.testing import is_test_mode

        if is_test_mode():
            return _offline_complete
    except Exception:
        pass
    return kimi_compare_complete


def _offline_complete(prompt: str, system: str | None = None) -> str:
    ids = _ID_RE.findall(prompt or "")
    only_private = [
        {"id": i, "text": i, "reason": "test-mode stub; no live Kimi call"}
        for i in ids
    ]
    return json.dumps(
        {
            "headline": "Test-mode compare: no live Kimi call.",
            "only_private": only_private,
            "overlap": [],
            "only_public": [],
            "caveats": ["MOYO_TEST_MODE stub"],
        }
    )


def _format_phrases(records: list[PhraseRecord]) -> str:
    return "\n".join(f"{p.id}\t{p.label}\t{p.text}" for p in records)


def _fit_phrases(
    records: list[PhraseRecord], budget: int
) -> tuple[str, int, int]:
    lines: list[str] = []
    size = 0
    for rec in records:
        line = f"{rec.id}\t{rec.label}\t{rec.text}"
        extra = len(line) + (1 if lines else 0)
        if lines and size + extra > budget:
            omitted = len(records) - len(lines)
            return "\n".join(lines), len(lines), omitted
        lines.append(line)
        size += extra
    return "\n".join(lines), len(lines), 0


def _truncate_side(side: PackedSide, budget: int) -> PackedSide:
    text = side.text or ""
    if len(text) <= budget:
        return PackedSide(
            kind=side.kind,
            path=side.path,
            text=text,
            chars=len(text),
            truncated=False,
            item_count=side.item_count,
        )
    marker = "\n\n[... truncated {n} chars ...]\n\n"
    keep = max(200, budget - len(marker) - 12)
    head = keep * 2 // 3
    tail = keep - head
    omitted = len(text) - head - tail
    body = text[:head] + marker.format(n=omitted) + text[-tail:]
    return PackedSide(
        kind=side.kind,
        path=side.path,
        text=body,
        chars=len(body),
        truncated=True,
        item_count=side.item_count,
        omitted_chars=omitted,
    )


def _read_text_side(path: Path, *, kind: str) -> PackedSide:
    text = path.read_text(encoding="utf-8", errors="replace")
    return PackedSide(
        kind=kind,
        path=str(path),
        text=text,
        chars=len(text),
        truncated=False,
        item_count=1,
    )


def _pack_extracted_json(path: Path) -> PackedSide:
    """Pack post-extraction ``extracted.json`` into compare prompt text."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PackedSide(
            kind="extracted",
            path=str(path),
            text="",
            chars=0,
            truncated=False,
            item_count=0,
        )
    if isinstance(data, dict):
        rows = data.get("sources") or data.get("phrases") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    parts: list[str] = []
    count = 0
    for raw in rows:
        if isinstance(raw, str):
            text = raw.strip()
            title = ""
        elif isinstance(raw, dict):
            title = str(raw.get("title") or "").strip()
            text = str(raw.get("content") or raw.get("text") or "").strip()
        else:
            continue
        if not text:
            continue
        block = "\n".join(x for x in (f"## {title}" if title else "", text) if x)
        parts.append(block)
        count += 1
    body = "\n\n".join(parts)
    return PackedSide(
        kind="extracted",
        path=str(path),
        text=body,
        chars=len(body),
        truncated=False,
        item_count=count,
    )


def _pack_sources_json(directory: Path) -> PackedSide:
    files: list[Path] = []
    if directory.is_file() and directory.suffix.lower() == ".json":
        files = [directory]
    elif directory.is_dir():
        files = sorted(p for p in directory.rglob("sources.json") if p.is_file())
    parts: list[str] = []
    count = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for raw in data:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            content = str(raw.get("content") or "").strip()
            url = str(raw.get("url") or raw.get("source_url") or "").strip()
            if not (title or content):
                continue
            block = "\n".join(x for x in (f"## {title}" if title else "", url, content) if x)
            parts.append(block)
            count += 1
    text = "\n\n".join(parts)
    return PackedSide(
        kind="sources",
        path=str(directory),
        text=text,
        chars=len(text),
        truncated=False,
        item_count=count,
    )


def _bucket_rows(rows: Any) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    for row in rows:
        if isinstance(row, str):
            text = row.strip()
            if text:
                out.append({"id": "", "text": text, "quote": "", "reason": ""})
            continue
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("phrase") or "").strip()
        if not text and not row.get("id"):
            continue
        out.append(
            {
                "id": str(row.get("id") or "").strip(),
                "text": text,
                "quote": str(row.get("quote") or row.get("excerpt") or "").strip(),
                "reason": str(row.get("reason") or "").strip(),
            }
        )
    return out


def _match_phrase(
    row: dict[str, str],
    by_id: dict[str, PhraseRecord],
    by_text: dict[str, PhraseRecord],
) -> PhraseRecord | None:
    rec = by_id.get(row.get("id") or "")
    if rec:
        return rec
    return by_text.get(_key(row.get("text") or ""))


def _item_from_row(
    row: dict[str, str], rec: PhraseRecord | None, *, verdict: str
) -> CompareItem:
    if rec:
        return CompareItem(
            id=rec.id,
            text=rec.text,
            label=rec.label,
            verdict=verdict,
            quote=row.get("quote") or "",
            reason=row.get("reason") or "",
        )
    return CompareItem(
        id=row.get("id") or "",
        text=row.get("text") or "",
        verdict=verdict,
        quote=row.get("quote") or "",
        reason=row.get("reason") or "",
    )


def _key(text: str) -> str:
    return " ".join((text or "").lower().split())


def _load_json(raw: str) -> Any | None:
    text = (raw or "").strip()
    if not text:
        return None
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
