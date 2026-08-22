"""Shared HTTP helpers for public-source adapters."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "moyo-gather/0.1 (public-source crawler; +https://github.com)"
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def env(name: str) -> Optional[str]:
    value = (os.environ.get(name) or "").strip()
    return value or None


def optional_http_url(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return None


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is None else value.astimezone().replace(tzinfo=None)
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    # GDELT seendate: 20240101T120000Z already rewritten above if it ended with Z
    for fmt in (
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y%m%dT%H%M%S%z",
        "%Y%m%dT%H%M%S",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None


def reconstruct_inverted_abstract(index: Any) -> str:
    """Rebuild OpenAlex ``abstract_inverted_index`` into readable text."""
    if not isinstance(index, dict) or not index:
        return ""
    positions: dict[int, str] = {}
    for word, locs in index.items():
        if not isinstance(locs, list):
            continue
        for loc in locs:
            try:
                positions[int(loc)] = str(word)
            except (TypeError, ValueError):
                continue
    if not positions:
        return ""
    return " ".join(positions[i] for i in range(max(positions) + 1) if i in positions)


async def get_json(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    retries: int = 3,
) -> Any:
    merged = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        merged.update(headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
                response = await client.get(url, params=params, headers=merged)
            if response.status_code in (429, 502, 503) and attempt + 1 < retries:
                await _backoff(attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= retries:
                break
            await _backoff(attempt)
    logger.warning("GET JSON failed for %s: %s", url, last_exc)
    return None


async def get_text(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    retries: int = 3,
) -> str:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
                response = await client.get(url, params=params, headers=merged)
            if response.status_code in (429, 502, 503) and attempt + 1 < retries:
                await _backoff(attempt)
                continue
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= retries:
                break
            await _backoff(attempt)
    logger.warning("GET text failed for %s: %s", url, last_exc)
    return ""


async def _backoff(attempt: int) -> None:
    import asyncio

    await asyncio.sleep(min(2 ** attempt, 8))
