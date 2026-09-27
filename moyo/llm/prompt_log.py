"""Append-only log of every full prompt sent through :class:`LLMClient`.

Activate with :func:`prompt_log_path` (context manager) or ``MOYO_PROMPT_LOG``.
Each call writes one JSON line: model, system, full user prompt, timestamps.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_PATH: ContextVar[Optional[Path]] = ContextVar("moyo_prompt_log_path", default=None)
_LOCK = threading.Lock()
_SEQ = 0


def active_prompt_log_path() -> Optional[Path]:
    """Resolved log path for this thread/context, or None when logging is off."""
    path = _PATH.get()
    if path is not None:
        return path
    raw = (os.environ.get("MOYO_PROMPT_LOG") or "").strip()
    return Path(raw) if raw else None


@contextmanager
def prompt_log_path(path: Path | str | None) -> Iterator[Optional[Path]]:
    """Record LLM prompts to ``path`` for the duration of the block."""
    if path is None:
        token = _PATH.set(None)
        try:
            yield None
        finally:
            _PATH.reset(token)
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_text("", encoding="utf-8")
    token = _PATH.set(dest)
    try:
        yield dest
    finally:
        _PATH.reset(token)


def record_llm_prompt(
    *,
    prompt: str,
    system: Optional[str],
    provider: str,
    model: str,
    label: str,
    temperature: Any = None,
    max_tokens: Any = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Append one full prompt record when a log path is active."""
    path = active_prompt_log_path()
    if path is None:
        return
    global _SEQ
    with _LOCK:
        _SEQ += 1
        seq = _SEQ
        row: dict[str, Any] = {
            "seq": seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "monotonic": time.monotonic(),
            "provider": provider,
            "model": model,
            "label": label,
            "system": system or "",
            "prompt": prompt if isinstance(prompt, str) else str(prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if meta:
            row["meta"] = meta
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
