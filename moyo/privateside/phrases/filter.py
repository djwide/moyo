"""Kimi extracts sensitive phrases and drops framing language."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from moyo.privateside.phrases.schema import LABELS

CompleteFn = Callable[..., str]

_WS_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_ALLOWED = set(LABELS)

SYSTEM = """You extract sensitive, company-valuable phrases from an internal document.

Keep facts, names, codenames, credentials, identifiers, money, dates, locations,
personnel details, and operational specifics that would harm the company if public.

Drop framing and boilerplate: headings, page numbers, table of contents,
confidentiality legends, "this document is…", "for internal use only",
"the purpose of this memorandum", generic legal filler, and empty pleasantries.

Return JSON only, no markdown:
{"phrases": [{"text": "verbatim or tight excerpt", "label": "credential|identifier|financial|project|personnel|operational|other", "reason": "short why"}]}

If nothing is sensitive, return {"phrases": []}."""


def normalize_phrase(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def kimi_phrase_spec():
    from moyo.llm.client import LLMSpec

    return LLMSpec.from_dict(
        {
            "provider": "custom",
            "model": "kimi-k2.6",
            "base_url": "https://api.moonshot.ai/v1",
            "api_key": "$MOONSHOT_API_KEY",
            "temperature": 0.2,
            "max_tokens": 2500,
            "timeout": 120,
            "label": "Kimi (phrase extract)",
        }
    )


def kimi_complete(prompt: str, system: str | None = None) -> str:
    from moyo.llm.client import LLMClient, llm_spec_has_auth

    spec = kimi_phrase_spec()
    if not llm_spec_has_auth(spec):
        raise RuntimeError(
            "MOONSHOT_API_KEY is not set. Kimi is required to extract sensitive phrases."
        )
    client = LLMClient(spec)
    if not client.is_available():
        raise RuntimeError(client.init_error or "Kimi client is unavailable.")
    return client.complete(prompt, system=system, max_tokens=2500)


def extract_sensitive_phrases(
    text: str,
    *,
    direction: str | None = None,
    complete: CompleteFn | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Ask Kimi for sensitive phrases in ``text``. ``complete`` is for tests."""
    windows = _windows(text)
    if not windows:
        return []
    fn = complete or kimi_complete
    found: list[dict[str, Any]] = []
    extra = (direction or "").strip() or None
    for i, window in enumerate(windows, start=1):
        if progress:
            progress(f"Kimi extract {i}/{len(windows)}…")
        raw = fn(_user_prompt(window, extra), SYSTEM)
        found.extend(parse_phrase_payload(raw))
    return _dedupe(found)


def parse_phrase_payload(raw: str) -> list[dict[str, Any]]:
    """Parse Kimi JSON into ``{text, label, reason}`` rows."""
    payload = _load_json(raw)
    if payload is None:
        return []
    rows: list[Any]
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
        "Extract sensitive phrases from this internal text. "
        "JSON only.\n\n"
        f"{window}"
    )
    extra = (direction or "").strip()
    if extra:
        prompt += f"\n\ndirection:\n{extra}"
    return prompt


def _windows(text: str, size: int = 3500, overlap: int = 200) -> list[str]:
    body = (text or "").strip()
    if not body:
        return []
    try:
        from shared_utils.chunking import chunk_text

        parts = [
            normalize_phrase(p)
            for p in chunk_text(body, chunk_size=size, overlap=overlap)
            if normalize_phrase(p)
        ]
        if parts:
            return parts
    except Exception:
        pass
    return [normalize_phrase(body)]


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
