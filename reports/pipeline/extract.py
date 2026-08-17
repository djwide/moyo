"""[2] Per-chunk LLM claim extraction → claims.jsonl."""

from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .citations import attach_chunk_citations, resolve_claim_citations
from .parse import Chunk
from .textclean import plain_text


def _done_path_for(claims_path: Path) -> Path:
    return claims_path.with_name("extract_done.jsonl")


STATUSES = {
    "CORROBORATED",
    "CONTESTED",
    "OUTLIER",
    "UNVERIFIED",
    "MODEL-SPECIFIC",
}

REFUSAL_RE = re.compile(
    r"(?i)\b("
    r"i('m| am) sorry[,.]?\s*(but\s+)?i (cannot|can't|am unable to)|"
    r"i cannot (provide|assist|help)|"
    r"i can't (provide|assist|help)|"
    r"i am unable to provide|"
    r"cannot provide (assistance|information)|"
    r"i will not (speculate|fabricate)|"
    r"i do not have reliable|"
    r"this request involves (collecting )?sensitive|"
    r"i cannot provide information on individuals"
    r")\b"
)

_SOURCES_BLOCK_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?(?:sources?|references?|further reading)\s*:?\s*$"
)


def _progress_bar(done: int, total: int, *, width: int = 28, extra: str = "") -> None:
    """Overwrite a single stderr line with an ASCII progress bar."""
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    frac = done / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {extra}" if extra else ""
    print(
        f"\r  extract [{bar}] {done}/{total} ({100.0 * frac:5.1f}%){suffix}",
        end="",
        file=sys.stderr,
        flush=True,
    )
    if done >= total:
        print(file=sys.stderr)


def _clamp(n: Any, lo: int = 1, hi: int = 5) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = lo
    return max(lo, min(hi, v))


def _strip_for_extract_prompt(text: str) -> str:
    """Drop model header + trailing Sources/URL laundry lists from prompt text.

    Line offsets in the claim still refer to the original exploration.md chunk.
    """
    lines = text.splitlines()
    # Drop leading ##### model heading (and blank lines after it)
    while lines and (lines[0].startswith("#####") or not lines[0].strip()):
        lines.pop(0)
    body = "\n".join(lines).rstrip()
    match = _SOURCES_BLOCK_RE.search(body)
    if match:
        tail = body[match.end() :]
        # Only strip if the remainder looks like a citation list, not prose
        urlish = len(re.findall(r"https?://|\.gov|\.org|\.com", tail, re.I))
        bullets = len(re.findall(r"(?m)^\s*[-*•]", tail))
        if urlish + bullets >= 2 and len(tail.strip()) < 2500:
            body = body[: match.start()].rstrip()
    return body


def _load_prompt(prompt_path: Path, chunk: Chunk) -> str:
    tmpl = prompt_path.read_text(encoding="utf-8")
    chunk_text = _strip_for_extract_prompt(chunk.text)
    return (
        tmpl.replace("{{ query_id }}", chunk.query_id)
        .replace("{{ query_text }}", chunk.query_text or "")
        .replace("{{ source_model }}", chunk.source_model)
        .replace("{{ line_offset }}", str(chunk.start_line))
        .replace("{{ language }}", chunk.language or "")
        .replace("{{ chunk_text }}", chunk_text)
    )


def _looks_like_refusal(chunk: Chunk) -> bool:
    # Check body without the model header
    body = _strip_for_extract_prompt(chunk.text)
    if len(body) < 40:
        return True
    head = body[:800]
    if REFUSAL_RE.search(head):
        # Real findings sometimes open with a soft hedge then list facts
        bullets = len(re.findall(r"(?m)^\s*[-*•]", body))
        if bullets >= 3 and len(body) > 400:
            return False
        return True
    return False


def _language_allowed(chunk: Chunk, allowed: list[str] | None) -> bool:
    if not allowed:
        return True
    lang = (chunk.language or "English").strip().lower()
    allow = {a.strip().lower() for a in allowed if a and str(a).strip()}
    if not allow:
        return True
    # Treat missing language as English for filtering purposes
    if not chunk.language:
        return "english" in allow or "en" in allow
    return lang in allow or any(lang.startswith(a) for a in allow)


def select_chunks_for_extract(
    chunks: list[Chunk],
    *,
    chunk_config: dict | None = None,
) -> tuple[list[Chunk], dict[str, int]]:
    """Apply cheap gates before paid extraction. Returns (kept, skip_counts)."""
    cfg = chunk_config or {}
    min_tokens = int(cfg.get("min_tokens") or 0)
    skip_refusals = bool(cfg.get("skip_refusals", True))
    # Prefer skip_refusals; accept plan alias skip_refusal_patterns
    if "skip_refusal_patterns" in cfg:
        skip_refusals = bool(cfg.get("skip_refusal_patterns"))
    languages = cfg.get("languages")
    if isinstance(languages, str):
        languages = [languages]
    allowed = list(languages) if languages else None

    kept: list[Chunk] = []
    skips = {"tiny": 0, "refusal": 0, "language": 0}
    for ch in chunks:
        if allowed is not None and not _language_allowed(ch, allowed):
            skips["language"] += 1
            continue
        if min_tokens > 0 and int(ch.approx_tokens) < min_tokens:
            skips["tiny"] += 1
            continue
        if skip_refusals and _looks_like_refusal(ch):
            skips["refusal"] += 1
            continue
        kept.append(ch)
    return kept, skips


def _salvage_json_objects(text: str) -> list[dict]:
    """Pull complete ``{...}`` objects from truncated / messy JSON array text."""
    out: list[dict] = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            # Remainder is truncated mid-object; keep what we already have
            break
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def _parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array of claim objects; tolerate fences and truncated LLM output."""
    if not text or not str(text).strip():
        return []
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start < 0:
        # Model may have returned a single object
        if "{" in text:
            return _salvage_json_objects(text)
        return []
    end = text.rfind("]")
    blob = text[start : end + 1] if end > start else text[start:]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        salvaged = _salvage_json_objects(text[start:])
        if salvaged:
            print(
                f"  warn: truncated/invalid JSON; salvaged {len(salvaged)} object(s)",
                file=sys.stderr,
            )
        return salvaged
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _normalize_claim(
    raw: dict,
    *,
    claim_id: str,
    chunk: Chunk,
    require_raw_excerpt: bool,
) -> dict | None:
    claim = plain_text(raw.get("claim") or "")
    if not claim:
        return None
    excerpt = (raw.get("raw_excerpt") or "").strip()
    if require_raw_excerpt and not excerpt:
        # fall back to a short slice of the chunk
        excerpt = chunk.text[:400].strip()
        if not excerpt:
            return None
    status = str(raw.get("status") or "UNVERIFIED").upper().replace(" ", "-")
    if status not in STATUSES:
        status = "UNVERIFIED"
    try:
        start = int(raw.get("raw_start_line") or chunk.start_line)
    except (TypeError, ValueError):
        start = chunk.start_line
    try:
        end = int(raw.get("raw_end_line") or chunk.end_line)
    except (TypeError, ValueError):
        end = chunk.end_line
    return {
        "claim_id": claim_id,
        "claim": claim,
        "source_model": raw.get("source_model") or chunk.source_model,
        "query_id": raw.get("query_id") or chunk.query_id,
        "category": raw.get("category") or "unclassified",
        "sensitivity": _clamp(raw.get("sensitivity", 3)),
        "specificity": _clamp(raw.get("specificity", 3)),
        "novelty": _clamp(raw.get("novelty", 3)),
        "confidence": _clamp(raw.get("confidence", 3)),
        "corroboration": max(0, int(raw.get("corroboration") or 1)),
        "interestingness": _clamp(raw.get("interestingness", 3)),
        "status": status,
        "raw_excerpt": excerpt,
        "raw_start_line": start,
        "raw_end_line": max(start, end),
        "language": chunk.language,
        "chunk_id": chunk.chunk_id,
        # Real-world citations for this claim: reference markers in its own
        # excerpt first, then LLM picks, then the chunk's Sources list.
        "citations": resolve_claim_citations(
            claim=claim,
            excerpt=excerpt,
            llm_citations=raw.get("citations") or [],
            chunk_citations=chunk.citations,
            reference_map=chunk.citation_refs,
        ),
    }


def heuristic_extract(chunk: Chunk, start_id: int) -> list[dict]:
    """Deterministic fallback extractor for smoke tests / offline runs."""
    claims: list[dict] = []
    # Bullet / bold-ish lines with substance
    candidates: list[str] = []
    for line in chunk.text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("#####"):
            continue
        if s.startswith(("-", "*", "•")) or "**" in s:
            cleaned = re.sub(r"^[-*•]\s*", "", s)
            cleaned = re.sub(r"\*\*", "", cleaned).strip()
            if 40 <= len(cleaned) <= 400:
                candidates.append(cleaned)
    body = _strip_for_extract_prompt(chunk.text)
    # Dense paragraphs with numbers / recipe cues, then any substantial prose
    # so a blank extractor LLM still yields an inventory from exploration.md.
    for para in re.split(r"\n{2,}", body):
        p = " ".join(para.split())
        if len(p) < 80 or p.startswith(">") or p.lower().startswith("sources"):
            continue
        if re.search(
            r"\b(7X|Merchandise|mg|oil|formula|cocaine|Stepan| Pemberton|\d+\s*g)\b",
            p,
            re.I,
        ):
            candidates.append(p[:350])
    if not candidates:
        for para in re.split(r"\n{2,}", body):
            p = " ".join(para.split())
            if 80 <= len(p) <= 800 and not p.startswith(">"):
                candidates.append(p[:400])
    if not candidates:
        compact = " ".join(body.split())
        if len(compact) >= 40:
            candidates.append(compact[:400])

    seen: set[str] = set()
    n = start_id
    for c in candidates:
        key = c[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        sens = 4 if re.search(r"7X|Merchandise|formula|cocaine|oil", c, re.I) else 2
        spec = 5 if re.search(r"\d", c) else 3
        status = "OUTLIER" if sens >= 4 and spec >= 4 else "UNVERIFIED"
        if "alleged" in c.lower() or "believed" in c.lower():
            status = "UNVERIFIED"
        claims.append(
            _normalize_claim(
                {
                    "claim": c,
                    "sensitivity": sens,
                    "specificity": spec,
                    "novelty": min(5, sens),
                    "confidence": 2 if "alleged" in c.lower() else 3,
                    "interestingness": sens,
                    "status": status,
                    "category": "proprietary_adjacent" if sens >= 4 else "public_fact",
                    "raw_excerpt": c[:500],
                },
                claim_id=f"C{n:04d}",
                chunk=chunk,
                require_raw_excerpt=True,
            )
        )
        n += 1
        if len(claims) >= 8:
            break
    return [c for c in claims if c]


def extract_chunk_llm(
    chunk: Chunk,
    *,
    client: Any,
    prompt_path: Path,
    require_raw_excerpt: bool,
    start_id: int,
) -> list[dict]:
    prompt = _load_prompt(prompt_path, chunk)
    try:
        text = client.complete(prompt)
    except Exception as exc:
        print(
            f"  warn: extract failed for {chunk.chunk_id}: {exc}",
            file=sys.stderr,
        )
        return []
    try:
        raw_claims = _parse_json_array(text or "")
    except Exception as exc:
        print(
            f"  warn: JSON parse failed for {chunk.chunk_id}: {exc}",
            file=sys.stderr,
        )
        return []
    out: list[dict] = []
    n = start_id
    for raw in raw_claims:
        c = _normalize_claim(
            raw,
            claim_id=f"C{n:04d}",
            chunk=chunk,
            require_raw_excerpt=require_raw_excerpt,
        )
        if c:
            out.append(c)
            n += 1
    return out


def _max_claim_counter(claims: list[dict]) -> int:
    """Highest numeric suffix in claim_id values (C0007 → 7)."""
    best = 0
    for c in claims:
        cid = str(c.get("claim_id") or "")
        m = re.search(r"(\d+)$", cid)
        if m:
            best = max(best, int(m.group(1)))
    return best


def load_extract_done(path: Path) -> set[str]:
    """Load completed chunk_ids from extract_done.jsonl."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("chunk_id")
            if cid:
                done.add(str(cid))
    return done


def extract_all(
    chunks: list[Chunk],
    *,
    out_path: Path,
    prompt_path: Path,
    config: dict,
    dry_run: bool = False,
    chunk_config: dict | None = None,
) -> list[dict]:
    """Extract claims from all chunks; write claims.jsonl; return claims.

    Resumes automatically when ``claims.jsonl`` / ``extract_done.jsonl`` already
    exist: finished chunk_ids are skipped and new claims are appended. Delete
    both files to force a full re-extract.
    """
    require_raw = bool(config.get("require_raw_excerpt", True))
    workers = int(config.get("workers", 4))

    selected, skips = select_chunks_for_extract(chunks, chunk_config=chunk_config)
    skipped_n = sum(skips.values())
    if skipped_n:
        print(
            f"  skipped {skips['refusal']} refusal / {skips['tiny']} tiny / "
            f"{skips['language']} language "
            f"({len(selected)}/{len(chunks)} chunks to extract)",
            file=sys.stderr,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = _done_path_for(out_path)

    existing = load_claims(out_path) if out_path.exists() else []
    already_done = load_extract_done(done_path)
    already_done |= {
        str(c["chunk_id"]) for c in existing if c.get("chunk_id")
    }

    pending = [ch for ch in selected if ch.chunk_id not in already_done]
    claims = list(existing)
    claim_counter = _max_claim_counter(claims) + 1

    if already_done:
        print(
            f"  resume: {len(already_done)} chunk(s) done, "
            f"{len(pending)} remaining "
            f"({len(claims)} claims on disk)",
            file=sys.stderr,
        )

    def _write_issues(rows: list[dict]) -> None:
        issues_path = out_path.with_name("extract_issues.json")
        if rows:
            issues_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif issues_path.exists():
            issues_path.unlink()

    def _finalize(result: list[dict], extra_issues: list[dict] | None = None) -> list[dict]:
        """Attach chunk Sources/URLs onto claims and rewrite claims.jsonl."""
        attach_chunk_citations(result, chunks)
        if result:
            with out_path.open("w", encoding="utf-8") as f:
                for c in result:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        issues = list(extra_issues or [])
        _write_issues(issues)
        return result

    issues: list[dict] = []
    total = len(selected)
    if total == 0:
        salvage = [
            ch
            for ch in chunks
            if not ch.failed and len(_strip_for_extract_prompt(ch.text).strip()) >= 40
        ]
        if salvage and not claims:
            print(
                f"  no chunks left after gates; heuristic salvage on "
                f"{len(salvage)}/{len(chunks)} unfailed chunk(s)",
                file=sys.stderr,
            )
            issues.append(
                {
                    "stage": "extract",
                    "reason": (
                        f"gates dropped all {len(chunks)} chunks; "
                        f"heuristic salvage n={len(salvage)}"
                    ),
                }
            )
            n = _max_claim_counter(claims) + 1
            for ch in salvage:
                for raw in heuristic_extract(ch, n):
                    raw = dict(raw)
                    raw.pop("claim_id", None)
                    raw["claim_id"] = f"C{n:04d}"
                    n += 1
                    claims.append(raw)
            return _finalize(claims, issues)
        if not claims:
            print(
                "  no chunks left after gates; writing empty claims.jsonl",
                file=sys.stderr,
            )
            out_path.write_text("", encoding="utf-8")
            done_path.write_text("", encoding="utf-8")
        return _finalize(claims, issues)

    if not pending:
        print("  resume: all selected chunks already extracted", file=sys.stderr)
        return _finalize(claims, issues)

    write_lock = threading.Lock()
    # Ensure files exist for append; keep prior content when resuming
    if not out_path.exists():
        out_path.write_text("", encoding="utf-8")
    if not done_path.exists():
        done_path.write_text("", encoding="utf-8")

    def _commit_chunk(cid: str, batch: list[dict]) -> list[dict]:
        """Assign ids, append claims + done marker; return numbered batch."""
        nonlocal claim_counter
        numbered: list[dict] = []
        with write_lock:
            for raw in batch:
                raw = dict(raw)
                raw["claim_id"] = f"C{claim_counter:04d}"
                claim_counter += 1
                numbered.append(raw)
            with out_path.open("a", encoding="utf-8") as cf:
                for c in numbered:
                    cf.write(json.dumps(c, ensure_ascii=False) + "\n")
                cf.flush()
            with done_path.open("a", encoding="utf-8") as df:
                df.write(
                    json.dumps({"chunk_id": cid, "n_claims": len(numbered)})
                    + "\n"
                )
                df.flush()
            claims.extend(numbered)
        return numbered

    already_n = len(selected) - len(pending)
    _progress_bar(already_n, total, extra="resuming" if already_n else "starting")

    try:
        from moyo.llm.testing import is_test_mode
        if is_test_mode():
            dry_run = True
    except Exception:
        pass

    def _heuristic_pending() -> None:
        finished = already_n
        for chunk in pending:
            batch = heuristic_extract(chunk, 1)
            for b in batch:
                b.pop("claim_id", None)
            numbered = _commit_chunk(chunk.chunk_id, batch)
            finished += 1
            _progress_bar(
                finished, total, extra=f"{chunk.chunk_id} +{len(numbered)}"
            )

    if dry_run:
        _heuristic_pending()
        return _finalize(claims, issues)

    from moyo.llm.client import LLMClient, LLMSpec, llm_spec_has_auth
    from moyo.llm.vertex import is_vertex_openai_url

    base_url = config.get("base_url") or "https://api.moonshot.ai/v1"
    api_key = config.get("api_key")
    if "api_key" not in config and not is_vertex_openai_url(base_url):
        api_key = "$MOONSHOT_API_KEY"

    spec = LLMSpec.from_dict(
        {
            "provider": config.get("provider", "custom"),
            "model": config.get("model", "kimi-k2.6"),
            "base_url": base_url,
            "api_key": api_key,
            "temperature": float(config.get("temperature", 0.2)),
            "max_tokens": int(config.get("max_tokens", 2500)),
            "num_ctx": config.get("num_ctx"),
            "timeout": int(config.get("timeout", 120)),
        }
    )
    client: Any = None
    if not llm_spec_has_auth(spec):
        print(
            f"  warn: extractor LLM has no API key for {spec.provider}/{spec.model}; "
            "using heuristic extract",
            file=sys.stderr,
        )
        issues.append(
            {
                "stage": "extract",
                "source": f"{spec.provider}/{spec.model}",
                "reason": "extractor API key missing; heuristic fallback",
            }
        )
        _heuristic_pending()
        return _finalize(claims, issues)
    try:
        client = LLMClient(spec)
        if not client.is_available():
            raise RuntimeError(
                f"Extractor LLM unavailable ({spec.provider}/{spec.model})"
            )
    except Exception as exc:
        print(
            f"  warn: extractor LLM unavailable ({exc}); using heuristic extract",
            file=sys.stderr,
        )
        issues.append(
            {
                "stage": "extract",
                "source": f"{spec.provider}/{spec.model}",
                "reason": f"extractor unavailable ({exc}); heuristic fallback"[:240],
            }
        )
        _heuristic_pending()
        return _finalize(claims, issues)

    def _work(ch: Chunk) -> tuple[str, list[dict], str | None]:
        reason: str | None = None
        try:
            batch = extract_chunk_llm(
                ch,
                client=client,
                prompt_path=prompt_path,
                require_raw_excerpt=require_raw,
                start_id=1,
            )
        except Exception as exc:
            batch = []
            reason = f"extractor error: {exc}"[:240]
        for b in batch:
            b.pop("claim_id", None)
        if batch:
            return ch.chunk_id, batch, reason
        salvaged = heuristic_extract(ch, 1)
        for b in salvaged:
            b.pop("claim_id", None)
        if salvaged:
            note = reason or "empty or unparseable extractor response"
            return (
                ch.chunk_id,
                salvaged,
                f"{note}; heuristic fallback ({len(salvaged)} claim(s))",
            )
        return (
            ch.chunk_id,
            [],
            reason or "empty or unparseable extractor response",
        )

    finished = already_n
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_work, ch): ch.chunk_id for ch in pending}
        for fut in as_completed(futs):
            cid = futs[fut]
            try:
                cid, batch, reason = fut.result()
            except Exception as exc:
                batch = []
                reason = f"extractor worker crashed: {exc}"[:240]
            if reason:
                issues.append(
                    {
                        "stage": "extract",
                        "chunk_id": cid,
                        "source": next(
                            (c.source_model for c in pending if c.chunk_id == cid),
                            "",
                        ),
                        "reason": reason,
                    }
                )
                print(f"  warn: extract {cid}: {reason}", file=sys.stderr)
            numbered = _commit_chunk(cid, batch)
            finished += 1
            _progress_bar(
                finished, total, extra=f"{cid} +{len(numbered)}"
            )

    return _finalize(claims, issues)


def load_claims(path: Path) -> list[dict]:
    claims = []
    if not path.exists():
        return claims
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    claims.append(json.loads(line))
                except json.JSONDecodeError:
                    # Truncated last line from a hard interrupt — skip it
                    continue
    return claims
