"""Kimi extracts relevant public passages, with optional user direction.

Mirrors private-side ``phrases.filter``: raw gather documents plus an optional
``direction`` block, then JSON excerpts that become the public corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Union

from shared_utils.ids import generate_id

from .schema import PublicSource

CompleteFn = Callable[..., str]
# current, total, message — callers may also accept a single string.
ProgressFn = Callable[..., None]
PathLike = Union[str, Path]

LABELS = (
    "fact",
    "identifier",
    "quote",
    "claim",
    "personnel",
    "other",
)
_ALLOWED = set(LABELS)
EXTRACTED_FILE_NAME = "extracted.json"


def extracted_path(root: PathLike) -> Path:
    """Canonical post-extraction corpus path under a gather output directory."""
    dest = Path(root)
    if dest.is_file():
        return dest
    return dest / EXTRACTED_FILE_NAME


def load_extracted_sources(path: PathLike) -> list[PublicSource]:
    """Load the post-extraction corpus written by :func:`save_extracted`."""
    src = Path(path)
    if src.is_dir():
        src = extracted_path(src)
    if not src.is_file():
        raise FileNotFoundError(
            f"Extracted corpus not found: {src}. "
            "Run Extract relevant text on Build Public Corpus first."
        )
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read extracted corpus {src}: {exc}") from exc
    if isinstance(data, dict):
        rows = data.get("sources") or data.get("phrases") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    sources: list[PublicSource] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        if "content" not in raw and "text" in raw:
            raw = {**raw, "content": raw["text"]}
        if "id" not in raw:
            raw = {**raw, "id": generate_id("extracted")}
        if "title" not in raw:
            raw = {**raw, "title": "extracted"}
        if "source_type" not in raw:
            from .schema import SourceType

            raw = {**raw, "source_type": SourceType.WEB_SEARCH}
        try:
            sources.append(PublicSource(**raw))
        except Exception:
            continue
    return sources


def emit_progress(
    progress: ProgressFn | None,
    current: int,
    total: int,
    message: str = "",
) -> None:
    """Notify GUI/CLI of extract progress. Accepts (current, total, msg) or (msg,)."""
    if progress is None:
        return
    try:
        progress(current, total, message)
    except TypeError:
        progress(message or f"{current}/{total}")


def format_extract_progress(
    current: int,
    total: int,
    message: str = "",
    *,
    width: int = 24,
) -> str:
    """Single-line status: ``[####----] 3/12 (25.0%) Source 1/4``."""
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    frac = current / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    extra = " ".join((message or "").split())
    if len(extra) > 56:
        extra = extra[:53] + "..."
    suffix = f" {extra}" if extra else ""
    return f"[{bar}] {current}/{total} ({100.0 * frac:5.1f}%){suffix}"


def cli_extract_progress(current: int, total: int, message: str = "") -> None:
    """Overwrite one stderr line with an ASCII completion bar."""
    import sys

    line = f"\r  extract {format_extract_progress(current, total, message)}"
    print(line, end="", file=sys.stderr, flush=True)
    if int(current) >= max(1, int(total)):
        print(file=sys.stderr)


def count_extract_windows(sources: Sequence[PublicSource]) -> int:
    """How many Kimi calls extract will make for these sources."""
    from moyo.privateside.phrases.filter import _windows

    return sum(len(_windows(source.content)) for source in sources)


SYSTEM = """You extract relevant passages from a public document for an information-barrier corpus.

Keep specific facts, names, dates, identifiers, quotes, claims, locations, and
operational details that could overlap with private information.

Drop boilerplate: navigation, cookie banners, "click here", marketing slogans,
retrieval-failed notes, LLM self-talk, headings without facts, and generic filler.

The user may give extra direction after the source, labelled direction. Follow it.

Return JSON only, no markdown:
{"phrases": [{"text": "verbatim or tight excerpt", "label": "fact|identifier|quote|claim|personnel|other", "reason": "short why"}]}

If nothing is relevant, return {"phrases": []}."""


def extract_relevant_passages(
    text: str,
    *,
    direction: str | None = None,
    complete: CompleteFn | None = None,
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Ask Kimi for relevant passages in ``text``. ``complete`` is for tests."""
    from moyo.privateside.phrases.filter import kimi_complete, _windows

    windows = _windows(text)
    if not windows:
        return []
    fn = complete or kimi_complete
    found: list[dict[str, Any]] = []
    extra = (direction or "").strip() or None
    total = len(windows)
    emit_progress(progress, 0, total, "Starting extract…")
    for i, window in enumerate(windows, start=1):
        emit_progress(progress, i, total, f"Window {i}/{total}")
        raw = fn(_user_prompt(window, extra), SYSTEM)
        found.extend(parse_passage_payload(raw))
    return _dedupe(found)


def extract_from_sources(
    sources: Sequence[PublicSource],
    *,
    direction: str | None = None,
    complete: CompleteFn | None = None,
    progress: ProgressFn | None = None,
) -> list[PublicSource]:
    """Replace raw gather documents with extracted relevant passages."""
    from moyo.privateside.phrases.filter import kimi_complete, _windows

    extra = (direction or "").strip() or None
    fn = complete or kimi_complete
    plan: list[tuple[PublicSource, list[str]]] = [
        (source, _windows(source.content)) for source in sources
    ]
    total_windows = sum(len(windows) for _, windows in plan)
    n_sources = len(plan)
    if total_windows == 0:
        emit_progress(progress, 0, 1, "No text to extract")
        return []

    emit_progress(
        progress,
        0,
        total_windows,
        f"Extracting {total_windows} windows from {n_sources} sources…",
    )
    out: list[PublicSource] = []
    done = 0
    for source_i, (source, windows) in enumerate(plan, start=1):
        found: list[dict[str, Any]] = []
        title = source.title or source.id
        if not windows:
            continue
        for window_i, window in enumerate(windows, start=1):
            done += 1
            emit_progress(
                progress,
                done,
                total_windows,
                f"Source {source_i}/{n_sources} window {window_i}/{len(windows)}: {title}",
            )
            raw = fn(_user_prompt(window, extra), SYSTEM)
            found.extend(parse_passage_payload(raw))
        out.extend(_passages_to_sources(source, _dedupe(found)))
    emit_progress(
        progress,
        total_windows,
        total_windows,
        f"Kept {len(out)} passages from {n_sources} sources",
    )
    return out


def run_public_extract(
    sources_dir: PathLike,
    *,
    direction: str | None = None,
    output: PathLike | None = None,
    complete: CompleteFn | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Load gather output, extract, and write ``extracted.json``."""
    from .load import load_public_sources

    root = Path(sources_dir)
    emit_progress(progress, 0, 1, f"Loading sources from {root}…")
    sources = load_public_sources(root)
    if not sources:
        raise RuntimeError(
            "No sources found. Run Gather Public Sources first "
            "(sources.json from crawl, or exploration.md from explore)."
        )
    extracted = extract_from_sources(
        sources,
        direction=direction,
        complete=complete,
        progress=progress,
    )
    if not extracted:
        raise RuntimeError(
            "Kimi kept no relevant passages. Relax the direction and try again."
        )
    dest = save_extracted(
        output or extracted_path(root),
        extracted,
        direction=direction,
    )
    return {"path": str(dest), "count": len(extracted)}


def save_extracted(
    path: PathLike,
    sources: Sequence[PublicSource],
    *,
    direction: str | None = None,
) -> Path:
    """Write extracted passages next to gather output for inspection."""
    dest = Path(path)
    if dest.is_dir() or dest.suffix == "":
        dest = dest / EXTRACTED_FILE_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "direction": (direction or "").strip(),
        "count": len(sources),
        "sources": [_source_dict(s) for s in sources],
    }
    dest.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def parse_passage_payload(raw: str) -> list[dict[str, Any]]:
    """Parse Kimi JSON into ``{text, label, reason}`` rows."""
    from moyo.privateside.phrases.filter import _load_json, normalize_phrase

    payload = _load_json(raw)
    if payload is None:
        return []
    if isinstance(payload, dict):
        rows = payload.get("phrases") or payload.get("items") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, str):
            text = normalize_phrase(row)
            label, reason = "other", "kimi"
        elif isinstance(row, dict):
            text = normalize_phrase(str(row.get("text") or row.get("phrase") or ""))
            label = str(row.get("label") or "other").strip().lower()
            reason = str(row.get("reason") or "kimi")
        else:
            continue
        if not text:
            continue
        if label not in _ALLOWED:
            label = "other"
        out.append({"text": text, "label": label, "reason": reason, "score": 1.0})
    return out


def _user_prompt(window: str, direction: str | None = None) -> str:
    prompt = (
        "Extract relevant passages from this public text. "
        "JSON only.\n\n"
        f"{window}"
    )
    extra = (direction or "").strip()
    if extra:
        prompt += f"\n\ndirection:\n{extra}"
    return prompt


def _passages_to_sources(
    source: PublicSource, rows: Sequence[dict[str, Any]]
) -> list[PublicSource]:
    out: list[PublicSource] = []
    for row in rows:
        tags = list(source.tags or [])
        if "extracted" not in tags:
            tags.append("extracted")
        if row["label"] not in tags:
            tags.append(row["label"])
        meta = dict(source.metadata or {})
        meta.update(
            {
                "extracted": True,
                "source_id": source.id,
                "label": row["label"],
                "reason": row.get("reason") or "",
            }
        )
        out.append(
            PublicSource(
                id=generate_id("extracted"),
                title=source.title,
                content=row["text"],
                source_type=source.source_type,
                url=source.url,
                source_url=source.source_url,
                published_date=source.published_date,
                author=source.author,
                organization=source.organization,
                language=source.language,
                metadata=meta,
                tags=tags,
                relevance_score=source.relevance_score,
                confidence_score=source.confidence_score,
            )
        )
    return out


def _source_dict(source: PublicSource) -> Dict[str, Any]:
    if hasattr(source, "model_dump"):
        return source.model_dump(mode="json")
    return source.dict()


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = row["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
