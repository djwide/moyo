"""[1] Deterministic parser / chunker for exploration.md."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .citations import extract_citations, extract_reference_map


QUERY_HEADING_RE = re.compile(
    r"^####\s+Query\s+(\d+)\s*(?:\[[^\]]*\])?:\s*(.+?)\s*$"
)
MODEL_HEADING_RE = re.compile(r"^#####\s+(.+?)\s*$")
LANG_HEADING_RE = re.compile(r"^###\s+(.+?)\s*$")
FAILED_RE = re.compile(r"retrieval failed|request failed|error calling", re.I)


@dataclass
class Chunk:
    chunk_id: str
    query_id: str
    query_text: str
    source_model: str
    language: str | None
    text: str
    start_line: int
    end_line: int
    approx_tokens: int
    failed: bool = False
    citations: list[str] = field(default_factory=list)
    # Numbered reference markers ("[9]") mapped to their source string, so a
    # claim can inherit only the citations its own excerpt points at.
    citation_refs: dict[str, str] = field(default_factory=dict)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _clean_model_label(raw: str) -> str:
    # "ChatGPT (OpenAI gpt-4o)  _(Closed API)_" → "ChatGPT (OpenAI gpt-4o)"
    label = re.sub(r"\s*_\([^)]*\)_\s*$", "", raw).strip()
    return label


def parse_exploration(
    path: Path,
    *,
    max_tokens: int = 15_000,
    include_failed: bool = False,
) -> list[Chunk]:
    """Split exploration.md by Query / model headings; preserve all text."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    language: str | None = None
    # Collect (line_idx, kind, payload)
    markers: list[tuple[int, str, object]] = []
    for i, line in enumerate(lines):
        lm = LANG_HEADING_RE.match(line)
        if lm and i > 0:
            # Language sections under "Detailed findings..." use ### English etc.
            # Skip early ### under Reworded query seeds by requiring we have seen
            # the detailed-findings section or a Query heading later.
            language_candidate = lm.group(1).strip()
            if language_candidate.lower() in {
                "english",
                "spanish",
                "french",
                "mandarin chinese",
                "chinese",
                "german",
                "japanese",
                "korean",
                "portuguese",
                "italian",
                "arabic",
                "hindi",
                "russian",
            } or language_candidate.endswith("Chinese"):
                markers.append((i, "lang", language_candidate))
                continue
        qm = QUERY_HEADING_RE.match(line)
        if qm:
            markers.append((i, "query", (qm.group(1), qm.group(2).strip())))
            continue
        mm = MODEL_HEADING_RE.match(line)
        if mm:
            markers.append((i, "model", _clean_model_label(mm.group(1))))

    # Build query ranges with language context
    query_events: list[tuple[int, str, str, str | None]] = []
    current_lang: str | None = None
    for idx, kind, payload in markers:
        if kind == "lang":
            current_lang = str(payload)
        elif kind == "query":
            qnum, qtext = payload  # type: ignore[misc]
            query_events.append((idx, f"Q{int(qnum)}", qtext, current_lang))

    chunks: list[Chunk] = []
    chunk_n = 0
    global_q = 0

    for qi, (q_start, q_id_local, q_text, lang) in enumerate(query_events):
        q_end = query_events[qi + 1][0] if qi + 1 < len(query_events) else len(lines)
        global_q += 1
        # Prefer sequential global ids for schema uniqueness across languages
        query_id = f"Q{global_q}"

        model_starts = [
            (idx, label)
            for idx, kind, label in markers
            if kind == "model" and q_start < idx < q_end
        ]
        if not model_starts:
            text = "\n".join(lines[q_start:q_end])
            failed = bool(FAILED_RE.search(text))
            if failed and not include_failed:
                continue
            chunk_n += 1
            chunks.append(
                Chunk(
                    chunk_id=f"CHK{chunk_n:04d}",
                    query_id=query_id,
                    query_text=q_text,
                    source_model="unknown",
                    language=lang,
                    text=text,
                    start_line=q_start + 1,
                    end_line=q_end,
                    approx_tokens=_approx_tokens(text),
                    failed=failed,
                    citations=extract_citations(text),
                    citation_refs=extract_reference_map(text),
                )
            )
            continue

        for mi, (m_start, model) in enumerate(model_starts):
            m_end = model_starts[mi + 1][0] if mi + 1 < len(model_starts) else q_end
            text = "\n".join(lines[m_start:m_end])
            failed = bool(FAILED_RE.search(text[:500]))
            if failed and not include_failed:
                continue
            parts = _split_if_large(text, max_tokens=max_tokens)
            # Reference list belongs to the whole answer: a split part can hold
            # the markers while the Sources block lands in a later part.
            reference_map = extract_reference_map(text)
            for part_i, part in enumerate(parts):
                chunk_n += 1
                suffix = f".{part_i + 1}" if len(parts) > 1 else ""
                chunks.append(
                    Chunk(
                        chunk_id=f"CHK{chunk_n:04d}{suffix}",
                        query_id=query_id,
                        query_text=q_text,
                        source_model=model,
                        language=lang,
                        text=part,
                        start_line=m_start + 1,
                        end_line=m_end,
                        approx_tokens=_approx_tokens(part),
                        failed=failed,
                        citations=extract_citations(part) or extract_citations(text),
                        citation_refs=reference_map,
                    )
                )
    return chunks


def _split_if_large(text: str, max_tokens: int) -> list[str]:
    if _approx_tokens(text) <= max_tokens:
        return [text]
    paras = re.split(r"\n{2,}", text)
    parts: list[str] = []
    buf: list[str] = []
    for p in paras:
        trial = "\n\n".join(buf + [p])
        if buf and _approx_tokens(trial) > max_tokens:
            parts.append("\n\n".join(buf))
            buf = [p]
        else:
            buf.append(p)
    if buf:
        parts.append("\n\n".join(buf))
    return parts or [text]


def write_chunks_manifest(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks_manifest(path: Path) -> list[Chunk]:
    """Load chunks.jsonl written by ``write_chunks_manifest``."""
    chunks: list[Chunk] = []
    if not path.exists():
        return chunks
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or ""
            citations = row.get("citations")
            if not isinstance(citations, list):
                citations = extract_citations(text)
            refs = row.get("citation_refs")
            if not isinstance(refs, dict):
                refs = extract_reference_map(text)
            chunks.append(
                Chunk(
                    chunk_id=str(row.get("chunk_id") or ""),
                    query_id=str(row.get("query_id") or ""),
                    query_text=str(row.get("query_text") or ""),
                    source_model=str(row.get("source_model") or "unknown"),
                    language=row.get("language"),
                    text=text,
                    start_line=int(row.get("start_line") or 1),
                    end_line=int(row.get("end_line") or 1),
                    approx_tokens=int(row.get("approx_tokens") or _approx_tokens(text)),
                    failed=bool(row.get("failed")),
                    citations=[str(c) for c in citations if str(c).strip()],
                    citation_refs={str(k): str(v) for k, v in refs.items() if v},
                )
            )
    return chunks


def topic_from_exploration(path: Path) -> str:
    first = path.read_text(encoding="utf-8").splitlines()[:5]
    for line in first:
        if line.startswith("# Topic exploration:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("# "):
            return line[2:].strip()
    return path.parent.name.replace("_", " ")


_STRATEGY_RE = re.compile(
    r"`(paraphrase|translate|summarize|typo|abstract)`",
    re.I,
)
_FUZZ_MODE_RE = re.compile(r"_Fuzz mode:\s*`([^`]+)`_", re.I)
_TECHNIQUES_RE = re.compile(
    r"_Techniques(?:\s*\([^)]*\))?:\s*(.+?)_",
    re.I,
)
_RETRIEVAL_MODEL_RE = re.compile(r"`([^`]+)`")


def exploration_run_meta(path: Path) -> dict:
    """Pull fuzz mode, strategies, languages, and retrieval models from exploration.md."""
    from .language import parse_languages_line

    text = path.read_text(encoding="utf-8")
    fuzz_mode = "basic"
    m = _FUZZ_MODE_RE.search(text)
    if m:
        fuzz_mode = m.group(1).strip().lower()

    strategies: list[str] = []
    tech = _TECHNIQUES_RE.search(text)
    if tech:
        strategies = list(
            dict.fromkeys(s.lower() for s in _STRATEGY_RE.findall(tech.group(1)))
        )
    if not strategies:
        # Fall back to strategy tags on reworded seeds
        seed_section = text
        if "## Reworded query seeds" in text:
            seed_section = text.split("## Reworded query seeds", 1)[1]
            seed_section = seed_section.split("## Detailed findings", 1)[0]
        strategies = list(
            dict.fromkeys(s.lower() for s in _STRATEGY_RE.findall(seed_section))
        )
    if not strategies:
        if fuzz_mode == "multilingual":
            strategies = ["paraphrase", "abstract", "summarize"]
        else:
            strategies = ["paraphrase", "translate", "summarize"]

    languages = parse_languages_line(text)

    models: list[str] = []
    # Prefer the Retrieval sources block (before Detailed findings)
    sources_block = text
    if "## Retrieval sources" in text:
        tail = text.split("## Retrieval sources", 1)[1]
        sources_block = tail.split("## ", 1)[0]
        for line in sources_block.splitlines():
            if "`" not in line:
                continue
            for label in _RETRIEVAL_MODEL_RE.findall(line):
                cleaned = _clean_model_label(label)
                if cleaned and cleaned not in models and cleaned.lower() != fuzz_mode:
                    models.append(cleaned)

    return {
        "fuzz_mode": fuzz_mode,
        "strategies": strategies,
        "languages": languages,
        "models_tested": models,
    }
