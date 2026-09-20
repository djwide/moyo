"""Keep reasoning / scratchpad out of retrieval evidence.

Visible assistant ``content`` is what scans store, extract, and chart.
Provider reasoning (``reasoning_content``, ``<think>`` blocks, scratchpad
preambles) is retained only on the redacted provider trace.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_THINK_BLOCK_RE = re.compile(
    r"<\s*(?:think|thinking|thought|reasoning|scratchpad)\s*>.*?"
    r"<\s*/\s*(?:think|thinking|thought|reasoning|scratchpad)\s*>",
    re.I | re.S,
)
_THINK_OPEN_RE = re.compile(
    r"<\s*(?:think|thinking|thought|reasoning|scratchpad)\s*>.*$",
    re.I | re.S,
)
_SCRATCHPAD_HEAD_RE = re.compile(
    r"(?is)^\s*(?:"
    r"(?:okay,?\s+)?let me (?:think|reason|work)\b.{0,80}\n|"
    r"reasoning\s*:?\s*\n|"
    r"internal (?:reasoning|monologue|scratchpad)\s*:?\s*\n|"
    r"chain[- ]of[- ]thought\s*:?\s*\n|"
    r"hidden (?:reasoning|thinking)\s*:?\s*\n"
    r")"
)
_SECRET_KEY_RE = re.compile(
    r"(authorization|api[_-]?key|x-api-key|x-goog-api-key|cookie|set-cookie|"
    r"proxy-authorization|x-auth|secret|passwd|password|token)$",
    re.I,
)

_SECRET_KEYS = {
    "authorization",
    "api-key",
    "api_key",
    "x-api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "proxy-authorization",
    "x-auth-token",
}


def openai_reasoning_text(message: Any) -> str:
    """Hidden reasoning on an OpenAI-compatible chat message, if any."""
    if message is None:
        return ""
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def strip_reasoning_spill(text: Optional[str]) -> str:
    """Remove think-tags and scratchpad preambles from visible answer text."""
    body = str(text or "")
    if not body.strip():
        return ""
    body = _THINK_BLOCK_RE.sub("", body)
    body = _THINK_OPEN_RE.sub("", body)
    body = _SCRATCHPAD_HEAD_RE.sub("", body)
    return body.strip()


def visible_answer_body(text: Optional[str]) -> str:
    """Answer text with model headings and hidden reasoning removed."""
    lines = str(text or "").splitlines()
    while lines and (lines[0].startswith("#####") or not lines[0].strip()):
        lines.pop(0)
    return strip_reasoning_spill("\n".join(lines))


def is_substantive_answer(text: Optional[str], *, error: Optional[str] = None) -> bool:
    """True when the visible body is usable evidence (not empty / failed)."""
    if (error or "").strip():
        return False
    body = visible_answer_body(text)
    if body.startswith(">") and (
        "retrieval failed" in body.lower() or "no content returned" in body.lower()
    ):
        return False
    return len(body) >= 40


def redact_secrets(value: Any, *, _depth: int = 0) -> Any:
    """Drop auth headers and secret-like keys from a nested payload."""
    if _depth > 8:
        return "[truncated]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SECRET_KEYS or _SECRET_KEY_RE.search(name):
                out[name] = "[redacted]"
            else:
                out[name] = redact_secrets(item, _depth=_depth + 1)
        return out
    if isinstance(value, list):
        return [redact_secrets(item, _depth=_depth + 1) for item in value[:200]]
    if isinstance(value, tuple):
        return [redact_secrets(item, _depth=_depth + 1) for item in value[:200]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    dump = _safe_model_dump(value)
    if dump is not None:
        return redact_secrets(dump, _depth=_depth + 1)
    text = str(value)
    return text if len(text) < 4000 else text[:4000] + "…"


def _safe_model_dump(value: Any) -> Optional[dict[str, Any]]:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                data = fn() if attr != "model_dump" else fn(exclude_none=False)
            except TypeError:
                try:
                    data = fn()
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(data, dict):
                return data
    return None


def provider_payload(response: Any) -> Any:
    """JSON-safe, header-redacted provider body."""
    dump = _safe_model_dump(response)
    if dump is not None:
        dump.pop("request", None)
        headers = dump.get("headers")
        if isinstance(headers, dict):
            dump["headers"] = redact_secrets(headers)
        return redact_secrets(dump)
    return redact_secrets(response)
