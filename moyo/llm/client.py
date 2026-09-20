"""Provider-agnostic LLM client used across moyo.

All generative LLM usage should funnel through :class:`LLMClient` so that the
default model can be swapped in one place. The client mirrors the working
provider call patterns from
:mod:`moyo.publicside.barrierprobe.llm_fuzzer` but exposes a single,
uniform :meth:`LLMClient.complete` entry point.
"""

from __future__ import annotations

import logging
import os
import random
import re
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Snapshot retrieval: one hard per-call deadline, no client-side retries.
# SDK retries are also disabled in LLMClient so this is the only retry layer.
# Prompt rewording uses PARAPHRASER_TIMEOUT on the utility LLM and finishes
# before any of these retrieval deadlines are calculated or started.
SNAPSHOT_TIMEOUT = 120
SNAPSHOT_MAX_ATTEMPTS = 1
RETRIEVAL_TIMEOUT_DEFAULT = 120
RETRIEVAL_TIMEOUT_WEB_SEARCH = 300
PARAPHRASER_TIMEOUT = 120


# --- Rate-limit / transient retry -------------------------------------------
# Hard billing/auth failures sometimes arrive as HTTP 429; do not retry those.
_NON_RETRYABLE_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "no credits remaining",
    "never purchased credits",
    "incorrect api key",
    "invalid_api_key",
    "invalid_authentication",
    "invalid authentication",
    "authentication_error",
    "permission-denied",
    "doesn't have any credits",
    "does not have any credits",
    "unauthorized",
)

_RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "resource_exhausted",
    "overloaded",
    "temporarily unavailable",
    "server is busy",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection error",
    "503",
    "529",
    "502",
)


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True for transient rate-limit / overload / network errors."""
    text = str(exc).lower()
    if any(marker in text for marker in _NON_RETRYABLE_MARKERS):
        return False
    # Free-tier hard stop (Gemini often reports limit: 0 with a 429).
    if re.search(r"limit:\s*0\b", text):
        return False

    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    if status in (429, 502, 503, 529):
        # 429 with a non-retryable marker already returned False above.
        return True

    name = type(exc).__name__.lower()
    if any(tok in name for tok in ("ratelimit", "overloaded", "timeout", "apiconnection")):
        return True

    return any(marker in text for marker in _RETRYABLE_MARKERS)


def format_llm_error(exc: BaseException) -> str:
    """Human-readable LLM failure, including the TCP/TLS cause when present.

    OpenAI's ``APIConnectionError`` string is just ``Connection error.``; the
    useful detail lives on ``__cause__`` (ConnectTimeout, NameResolutionError).
    """
    text = (str(exc) or "").strip() or type(exc).__name__
    cause = exc.__cause__ or getattr(exc, "__context__", None)
    if cause is None:
        return text
    cause_text = (str(cause) or "").strip()
    cause_name = type(cause).__name__
    if not cause_text or cause_text in text:
        if cause_name.lower() not in text.lower():
            return f"{text} ({cause_name})"
        return text
    return f"{text} ({cause_name}: {cause_text})"


def retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    """Seconds to wait before the next attempt (honours provider hints when present)."""
    text = str(exc)
    match = re.search(r"retry in\s+([\d.]+)\s*s", text, re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 0.25, 120.0)

    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None:
        retry_after = None
        try:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            retry_after = None
        if retry_after is not None:
            try:
                return min(float(retry_after), 120.0)
            except (TypeError, ValueError):
                pass

    # Exponential backoff with light jitter: 1, 2, 4, ... capped.
    return min((2 ** attempt) + random.uniform(0, 0.5), 60.0)


# --- .env loading -----------------------------------------------------------
# API keys are read from ``os.environ`` (directly here and via ``$VAR``
# references in config). Nothing else in moyo exports a ``.env`` file into the
# process environment, so we do it once here (without overriding real env vars)
# so keys placed in ``.env`` "just work" for every LLM provider.
_ENV_FILE_LOADED = False

# Secret Manager / Cloud Run env vars often include a trailing newline.
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")


def sanitize_secret_environ() -> None:
    """Strip whitespace and newlines from API keys already in ``os.environ``."""
    for key, val in list(os.environ.items()):
        if not isinstance(val, str) or not val:
            continue
        upper = key.upper()
        if not any(upper.endswith(suf) for suf in _SECRET_ENV_SUFFIXES):
            continue
        cleaned = val.strip()
        if cleaned != val:
            os.environ[key] = cleaned


def _load_env_file(path: Optional[str] = None) -> None:
    env_path = Path(path or os.environ.get("MOYO_ENV_FILE", ".env"))
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if val and val[0] in "\"'":
                quote = val[0]
                end = val.find(quote, 1)
                val = val[1:end] if end != -1 else val[1:]
            else:
                # Strip trailing inline comments (whitespace + '#').
                for sep in (" #", "\t#"):
                    idx = val.find(sep)
                    if idx != -1:
                        val = val[:idx]
                val = val.strip()
            # Real environment variables take precedence over .env.
            os.environ.setdefault(key, val)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load env file %s: %s", env_path, exc)


def ensure_env_loaded() -> None:
    """Load ``.env`` into ``os.environ`` once (idempotent).

    Always strips trailing whitespace from ``*_API_KEY`` / ``*_TOKEN`` /
    ``*_SECRET`` env vars (Secret Manager values often include a newline).
    """
    global _ENV_FILE_LOADED
    if not _ENV_FILE_LOADED:
        _ENV_FILE_LOADED = True
        _load_env_file()
    sanitize_secret_environ()


# --- Provider classification ------------------------------------------------
# Used to group results in reports by "source of retrieval".
CLOSED_API_PROVIDERS = {"openai", "anthropic"}
OPEN_API_PROVIDERS = {"custom"}
LOCAL_PROVIDERS = {"ollama", "echo"}

_ENV_KEY_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
    "ollama": "llama3.1:8b",
    "custom": "gpt-4o",
    "echo": "echo",
}


def _is_kimi_k25_or_k26(model: str) -> bool:
    """True for Moonshot Kimi K2.5 / K2.6 (supports toggling thinking)."""
    name = (model or "").lower()
    return (
        name.startswith("kimi-k2.5")
        or name.startswith("kimi-k2.6")
        or "/kimi-k2.5" in name
        or "/kimi-k2.6" in name
    )


def _is_kimi_k3(model: str) -> bool:
    name = (model or "").lower()
    return name.startswith("kimi-k3") or "/kimi-k3" in name


def _is_openai_max_completion_tokens_model(model: str) -> bool:
    """Models that reject ``max_tokens`` and require ``max_completion_tokens``."""
    name = (model or "").lower()
    if name.startswith(("o1", "o3", "o4")):
        return True
    if name.startswith("gpt-5") or "/gpt-5" in name:
        return True
    return False


def _is_openai_fixed_sampling_model(model: str) -> bool:
    """Models that reject non-default temperature / sampling knobs."""
    return _is_openai_max_completion_tokens_model(model)


def _is_anthropic_no_temperature_model(model: str) -> bool:
    """Claude 4.7+ / Opus 5 / Sonnet 5 reject temperature/top_p/top_k."""
    name = (model or "").lower().replace("_", "-")
    markers = (
        "opus-5",
        "opus-4-7",
        "opus-4.7",
        "opus-4-8",
        "opus-4.8",
        "sonnet-5",
        "fable-5",
        "mythos-5",
        "mythos-preview",
    )
    return any(marker in name for marker in markers)


def _is_deepseek_v4(model: str) -> bool:
    name = (model or "").lower()
    return "deepseek-v4" in name or "deepseek/deepseek-v4" in name


def _is_gemini_model(model: str, base_url: Optional[str] = None) -> bool:
    name = (model or "").lower()
    url = (base_url or "").lower()
    return "gemini" in name or "generativelanguage.googleapis.com" in url


def _is_dashscope_url(base_url: Optional[str]) -> bool:
    return "dashscope" in (base_url or "").lower()


def _is_openrouter_url(base_url: Optional[str]) -> bool:
    return "openrouter.ai" in (base_url or "").lower()


def _is_xai_url(base_url: Optional[str]) -> bool:
    return "api.x.ai" in (base_url or "").lower()


def _is_moonshot_url(base_url: Optional[str]) -> bool:
    url = (base_url or "").lower()
    return "moonshot" in url or "kimi.ai" in url


def _uses_responses_web_search(provider: str, base_url: Optional[str] = None) -> bool:
    """OpenAI + xAI expose hosted web_search on the Responses API."""
    if provider == "openai":
        return True
    return provider == "custom" and _is_xai_url(base_url)


def _is_reasoning_budget_model(model: str, base_url: Optional[str] = None) -> bool:
    """Models whose reasoning tokens share the completion budget with content."""
    if _is_gemini_model(model, base_url):
        return True
    if _is_openai_max_completion_tokens_model(model):
        return True
    if _is_kimi_k3(model):
        return True
    if _is_deepseek_v4(model):
        return True
    if _is_anthropic_no_temperature_model(model):
        return True
    name = (model or "").lower()
    return "sonar-reasoning" in name


def _omit_temperature_for_model(model: str) -> bool:
    """True when the request must not include a temperature field."""
    name = (model or "").lower()
    if _is_openai_fixed_sampling_model(name):
        return True
    if _is_anthropic_no_temperature_model(name):
        return True
    if _is_kimi_k3(name):
        return True
    return False


def _fixed_temperature_for_model(model: str) -> Optional[float]:
    """Return a forced temperature, or None when temperature should be omitted/left alone."""
    name = (model or "").lower()
    if _omit_temperature_for_model(name):
        return None
    # K2.5/K2.6 non-thinking mode (our default) requires temperature=0.6.
    if _is_kimi_k25_or_k26(name):
        return 0.6
    # Other Moonshot Kimi K2.x only accept temperature=1.
    if name.startswith("kimi-k2") or "/kimi-k2" in name:
        return 1.0
    return None


# Some OpenAI-compatible providers reject tiny completion caps (e.g. Perplexity
# requires max_tokens >= 16).
MIN_COMPLETION_TOKENS = 16
# Reasoning models share the completion budget with hidden thinking tokens.
MIN_REASONING_COMPLETION_TOKENS = 1024


def _openai_extra_body_for_model(model: str) -> Dict[str, Any]:
    """Provider-specific OpenAI-compatible request fields.

    Kimi K2.5/K2.6 default to thinking mode. Reasoning tokens count against
    ``max_tokens``, so short caps often return empty ``content``. Disable
    thinking so retrieval replies land in ``content``.

    Kimi K3 always thinks. Do not send OpenAI ``reasoning_effort`` or K2
    ``thinking`` toggles — Moonshot rejects them with tokenization failed.
    """
    if _is_kimi_k25_or_k26(model):
        return {"thinking": {"type": "disabled"}}
    if _is_kimi_k3(model):
        return {}
    if _is_deepseek_v4(model):
        return {"reasoning": {"effort": "low"}}
    return {}


def _openai_create_extras(
    model: str,
    base_url: Optional[str] = None,
    *,
    web_search: bool = False,
) -> Dict[str, Any]:
    """Extra kwargs for ``chat.completions.create`` beyond messages/tokens.

    When ``web_search`` is true, turns on provider web-search options where the
    Chat Completions endpoint exposes a simple flag (Qwen ``enable_search``,
    Gemini ``web_search_options``, OpenRouter ``web`` plugin). OpenAI/xAI/
    Anthropic use dedicated tool paths elsewhere in this module.
    """
    extras: Dict[str, Any] = {}
    extra_body = dict(_openai_extra_body_for_model(model))
    if web_search:
        if _is_dashscope_url(base_url):
            extra_body["enable_search"] = True
        if _is_openrouter_url(base_url):
            # Works for any OpenRouter model, including ones without tool calling.
            extra_body["plugins"] = [{"id": "web"}]
    if extra_body:
        for key in ("reasoning_effort",):
            if key in extra_body:
                extras[key] = extra_body.pop(key)
        if extra_body:
            extras["extra_body"] = extra_body
    # gemini-3.1-pro-preview (and other Gemini thinking models) spend max_tokens on
    # internal reasoning first; low reasoning effort keeps short replies usable.
    # ``none`` is rejected by Pro-class aliases that require thinking mode.
    if _is_gemini_model(model, base_url):
        extras["reasoning_effort"] = "low"
        if web_search:
            extras["web_search_options"] = {}
    if _is_openai_max_completion_tokens_model(model):
        extras.setdefault("reasoning_effort", "low")
    return extras


def _anthropic_web_search_tools() -> List[Dict[str, Any]]:
    """Hosted Anthropic web_search server tool (executed by Anthropic)."""
    return [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
        }
    ]


def _responses_output_text(response: Any) -> str:
    """Visible text from an OpenAI/xAI Responses API result."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if isinstance(item, dict):
            item_type = item.get("type")
            content = item.get("content") or []
        else:
            content = getattr(item, "content", None) or []
        if item_type != "message":
            continue
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in (None, "output_text", "text"):
                    chunk = str(part.get("text") or "").strip()
                    if chunk:
                        parts.append(chunk)
            else:
                part_type = getattr(part, "type", None)
                if part_type in (None, "output_text", "text"):
                    chunk = str(getattr(part, "text", "") or "").strip()
                    if chunk:
                        parts.append(chunk)
    return "\n".join(parts).strip()


def _openai_message_text(message: Any) -> str:
    """Visible assistant text only — never fall back to reasoning/scratchpad."""
    if message is None:
        return ""
    from moyo.llm.content_filter import strip_reasoning_spill

    content = getattr(message, "content", None)
    visible = ""
    if isinstance(content, str) and content.strip():
        visible = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif isinstance(part, dict):
                part_type = str(part.get("type") or "").lower()
                if part_type in {"reasoning", "thinking", "thought"}:
                    continue
                text = str(part.get("text") or "").strip()
                if text:
                    parts.append(text)
            else:
                part_type = str(getattr(part, "type", "") or "").lower()
                if part_type in {"reasoning", "thinking", "thought"}:
                    continue
                text = str(getattr(part, "text", "") or "").strip()
                if text:
                    parts.append(text)
        if parts:
            visible = "\n".join(parts)
    elif isinstance(content, str):
        visible = content.strip()
    return strip_reasoning_spill(visible)


def _anthropic_message_text(response: Any) -> str:
    """Concatenate text blocks; Opus 5 may lead with thinking blocks."""
    blocks = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type and block_type != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _citations_from_response(response: Any) -> List[str]:
    """Pull citation URLs from OpenAI-compatible responses (e.g. Perplexity)."""
    raw = getattr(response, "citations", None)
    if raw is None:
        extra = getattr(response, "model_extra", None) or {}
        if isinstance(extra, dict):
            raw = extra.get("citations")
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            url = item.strip()
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("href") or "").strip()
        else:
            url = ""
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _with_provider_citations(content: str, response: Any) -> str:
    """Append a Sources list when the provider returns structured citations."""
    citations = _citations_from_response(response)
    if not citations:
        return content
    # Avoid duplicating URLs already present in the answer body.
    missing = [c for c in citations if c not in content]
    if not missing:
        return content
    lines = "\n".join(f"- {c}" for c in missing)
    body = (content or "").rstrip()
    if body:
        return f"{body}\n\nSources:\n{lines}"
    return f"Sources:\n{lines}"


def _is_fixed_temperature_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "temperature" in text and (
        "deprecated" in text or "not supported" in text or "unsupported" in text
    ):
        return True
    if "invalid temperature" not in text:
        return False
    return "only 1" in text or "only 0.6" in text or "only 0.60" in text


def _is_max_tokens_unsupported_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "max_tokens" in text and (
        "max_completion_tokens" in text or "unsupported parameter" in text
    )

def _default_label(provider: str, model: str) -> str:
    """Human-readable label preferring the model name over the transport provider."""
    if model:
        # OpenRouter-style "org/model" -> short model name for reports.
        short = model.rsplit("/", 1)[-1]
        return short
    return provider or "llm"


def classify_provider(provider: str) -> str:
    """Return ``"closed"``, ``"open"`` or ``"local"`` for a provider name."""
    p = (provider or "").lower()
    if p in CLOSED_API_PROVIDERS:
        return "closed"
    if p in OPEN_API_PROVIDERS:
        return "open"
    return "local"


@dataclass
class CompletionResult:
    """Normalized visible text plus the sanitized provider payload.

    ``text`` is what scans extract and chart. ``provider_record`` is the
    redacted HTTP/SDK body, kept even when parsing yields empty text so a
    200 with an unexpected shape is not lost at the adapter boundary.
    """

    text: str
    provider_record: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None


@dataclass
class LLMSpec:
    """Description of a single LLM endpoint.

    ``provider`` is one of ``openai``, ``anthropic``, ``ollama``, ``custom`` or
    ``echo``. Missing ``model`` falls back to a per-provider default and missing
    ``api_key`` is resolved from the conventional environment variable.
    """

    provider: str
    model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    label: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 120
    max_retries: int = 3
    web_search: bool = False
    # Ollama context-window allocation (tokens). None = server/model default
    # (commonly 2048–4096). Raise for long summarise prompts.
    num_ctx: Optional[int] = None

    def __post_init__(self) -> None:
        ensure_env_loaded()
        self.provider = (self.provider or "echo").lower()
        if not self.model:
            self.model = _DEFAULT_MODEL_BY_PROVIDER.get(self.provider, "")
        if not self.api_key and self.provider in _ENV_KEY_BY_PROVIDER:
            self.api_key = os.environ.get(_ENV_KEY_BY_PROVIDER[self.provider])
        if isinstance(self.api_key, str):
            self.api_key = self.api_key.strip() or None
        if not self.label:
            self.label = _default_label(self.provider, self.model)
        if self.num_ctx is not None:
            self.num_ctx = int(self.num_ctx)

    @property
    def kind(self) -> str:
        """Coarse classification: ``closed`` / ``open`` / ``local``."""
        return classify_provider(self.provider)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMSpec":
        """Build a spec from a plain dict (e.g. loaded from JSON config).

        An ``api_key`` value of the form ``"$ENV_VAR"`` is resolved from the
        environment, which keeps secrets out of committed config files.
        """
        ensure_env_loaded()
        api_key = data.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("$"):
            api_key = os.environ.get(api_key[1:])
        if isinstance(api_key, str):
            api_key = api_key.strip() or None
        num_ctx = data.get("num_ctx")
        return cls(
            provider=data.get("provider", "echo"),
            model=data.get("model", "") or "",
            api_key=api_key,
            base_url=data.get("base_url"),
            label=data.get("label"),
            temperature=float(data.get("temperature", 0.7)),
            max_tokens=int(data.get("max_tokens", 1000)),
            timeout=int(data.get("timeout", 120)),
            max_retries=int(data.get("max_retries", 3)),
            web_search=bool(data.get("web_search", False)),
            num_ctx=int(num_ctx) if num_ctx is not None else None,
        )


def is_slow_search_retrieval(spec: LLMSpec) -> bool:
    """True for hosted search models that often miss the snapshot cap."""
    if _is_dashscope_url(spec.base_url) or _is_xai_url(spec.base_url):
        return True
    url = (spec.base_url or "").lower()
    model = (spec.model or "").lower()
    if "perplexity" in url:
        return "reason" in model or "sonar-reasoning" in model
    return False


def apply_retrieval_timeout(
    spec: LLMSpec,
    override: Optional[int] = None,
    *,
    web_search: bool = False,
) -> int:
    """Resolve per-call timeout for retrieval fan-out."""
    if web_search:
        return RETRIEVAL_TIMEOUT_WEB_SEARCH
    if override is not None:
        return max(1, int(override))
    return RETRIEVAL_TIMEOUT_DEFAULT


def llm_spec_has_auth(spec: LLMSpec) -> bool:
    """True when the spec can authenticate (API key or Vertex ADC)."""
    if (spec.api_key or "").strip():
        return True
    try:
        from moyo.llm.vertex import is_vertex_openai_url

        return is_vertex_openai_url(spec.base_url)
    except Exception:
        return False


class LLMClient:
    """A uniform wrapper over one LLM endpoint described by an :class:`LLMSpec`."""

    def __init__(self, spec: LLMSpec):
        self.spec = spec
        self._init_error: Optional[str] = None
        self._client = self._init_client()
        self._tls = threading.local()

    def last_provider_record(self) -> Optional[Dict[str, Any]]:
        """Thread-local redacted provider payload from the most recent complete()."""
        record = getattr(self._tls, "last_record", None)
        return record if isinstance(record, dict) else None

    def _store_provider_record(self, **fields: Any) -> None:
        from moyo.llm.content_filter import provider_payload, redact_secrets

        record = {
            "provider": self.spec.provider,
            "model": self.spec.model,
            "label": self.label,
            **fields,
        }
        raw = record.get("raw")
        if raw is not None:
            record["raw"] = provider_payload(raw)
        record = redact_secrets(record)
        self._tls.last_record = record

    def _text_from_stored_response(
        self,
        response: Any,
        parse: Any,
        **meta: Any,
    ) -> str:
        """Persist ``response`` first, then parse. Empty parse does not drop raw."""
        from moyo.llm.content_filter import strip_reasoning_spill

        self._store_provider_record(raw=response, content="", **meta)
        try:
            content = strip_reasoning_spill(parse() or "")
        except Exception as exc:
            err = format_llm_error(exc)
            logger.warning(
                "%s accepted the request but the response body could not be parsed: %s",
                self.label,
                err,
            )
            self._store_provider_record(
                raw=response,
                content="",
                parse_error=err,
                **meta,
            )
            return ""
        self._store_provider_record(raw=response, content=content, **meta)
        return content

    # -- introspection ------------------------------------------------------
    @property
    def label(self) -> str:
        return self.spec.label or self.spec.provider

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def is_available(self) -> bool:
        """Whether this client can currently be used to generate text."""
        if self.spec.provider == "echo":
            return True
        if self._client is None:
            return False
        if self.spec.provider == "ollama":
            try:
                return bool(self._client.is_available())
            except Exception:
                return False
        return True

    # -- construction -------------------------------------------------------
    def _init_client(self) -> Any:
        provider = self.spec.provider
        try:
            if provider == "echo":
                return "echo"
            if provider == "ollama":
                # Reuse the stdlib-only Ollama client already in the codebase.
                from moyo.publicside.barrierprobe.llm_fuzzer import OllamaClient

                return OllamaClient(
                    self.spec.model,
                    base_url=self.spec.base_url,
                    timeout=self.spec.timeout,
                )
            if provider in ("openai", "custom"):
                from openai import OpenAI

                kwargs: Dict[str, Any] = {
                    "timeout": self._http_timeout(),
                    # LLMClient.complete implements backoff; do not stack SDK retries.
                    "max_retries": 0,
                }
                vertex = False
                try:
                    from moyo.llm.vertex import (
                        is_vertex_openai_url,
                        openai_compatible_http_client,
                        vertex_api_key,
                        vertex_openai_headers,
                    )

                    vertex = is_vertex_openai_url(self.spec.base_url)
                    kwargs["http_client"] = openai_compatible_http_client(
                        self._http_timeout(), vertex=vertex
                    )
                except Exception as exc:
                    logger.warning("Could not build httpx client for LLM (%s)", exc)

                if vertex:
                    kwargs["api_key"] = vertex_api_key
                    kwargs["base_url"] = self.spec.base_url
                    kwargs["default_headers"] = vertex_openai_headers()
                elif provider == "custom":
                    if not self.spec.base_url:
                        self._init_error = (
                            "provider 'custom' requires a base_url pointing at an "
                            "OpenAI-compatible endpoint (e.g. http://localhost:8000/v1)"
                        )
                        return None
                    kwargs["base_url"] = self.spec.base_url
                    kwargs["api_key"] = self.spec.api_key or "not-needed"
                else:
                    if self.spec.api_key:
                        kwargs["api_key"] = self.spec.api_key
                    if self.spec.base_url:
                        kwargs["base_url"] = self.spec.base_url
                return OpenAI(**kwargs)
            if provider == "anthropic":
                from anthropic import Anthropic

                # Let Anthropic construct the timeout using its own HTTP
                # transport. Newer SDK builds use ``httpx2`` and reject an
                # ``httpx.Timeout`` instance created for the OpenAI client.
                kwargs = {
                    "timeout": float(self.spec.timeout or 120),
                    "max_retries": 0,
                }
                if self.spec.api_key:
                    kwargs["api_key"] = self.spec.api_key
                return Anthropic(**kwargs)
            self._init_error = f"unsupported LLM provider: {provider}"
            return None
        except ImportError as exc:
            self._init_error = f"missing dependency for provider '{provider}': {exc}"
            logger.warning(self._init_error)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            self._init_error = f"failed to initialize provider '{provider}': {exc}"
            logger.warning(self._init_error)
            return None

    def _http_timeout(self) -> Any:
        """SDK timeout. OpenAI's default connect=5s is too tight through Cloud NAT."""
        read = max(float(self.spec.timeout or 120), 30.0)
        connect = max(30.0, min(read, 60.0))
        try:
            from httpx import Timeout

            return Timeout(connect=connect, read=read, write=read, pool=read)
        except Exception:
            return read

    # -- generation ---------------------------------------------------------
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: Optional[int] = None,
    ) -> str:
        """Generate a completion for ``prompt``.

        Prefer :meth:`complete_result` at retrieval boundaries so the sanitized
        provider body is retained even when parsed text is empty.
        """
        return self.complete_result(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        ).text

    def complete_result(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: Optional[int] = None,
    ) -> CompletionResult:
        """Return normalized text and the sanitized provider payload together."""
        temperature = self.spec.temperature if temperature is None else temperature
        omit_temperature = _omit_temperature_for_model(self.spec.model)
        fixed = _fixed_temperature_for_model(self.spec.model)
        if omit_temperature:
            temperature = None  # type: ignore[assignment]
        elif fixed is not None:
            temperature = fixed
        max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens
        max_tokens = max(MIN_COMPLETION_TOKENS, int(max_tokens))
        if _is_reasoning_budget_model(self.spec.model, self.spec.base_url):
            max_tokens = max(max_tokens, MIN_REASONING_COMPLETION_TOKENS)
        provider = self.spec.provider

        try:
            from moyo.llm.testing import is_test_mode
            if provider == "echo" or is_test_mode():
                text = self._echo(prompt, system)
                self._store_provider_record(content=text, raw={"echo": True})
                return CompletionResult(text=text, provider_record=self.last_provider_record())
        except Exception:
            if provider == "echo":
                text = self._echo(prompt, system)
                self._store_provider_record(content=text, raw={"echo": True})
                return CompletionResult(text=text, provider_record=self.last_provider_record())

        if self._client is None:
            raise RuntimeError(self._init_error or f"LLM provider '{provider}' unavailable")

        max_retries = self.spec.max_retries if retries is None else max(0, int(retries))
        attempts = max(1, int(max_retries) + 1)
        last_exc: Optional[BaseException] = None
        use_max_completion_tokens = _is_openai_max_completion_tokens_model(self.spec.model)
        self._tls.last_record = None
        for attempt in range(attempts):
            try:
                text = self._complete_once(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_max_completion_tokens=use_max_completion_tokens,
                )
                record = self.last_provider_record()
                parse_error = None
                if isinstance(record, dict):
                    parse_error = record.get("parse_error")
                    if parse_error is not None:
                        parse_error = str(parse_error)
                return CompletionResult(
                    text=text or "",
                    provider_record=record,
                    parse_error=parse_error,
                )
            except Exception as exc:
                last_exc = exc
                if (
                    not use_max_completion_tokens
                    and provider in ("openai", "custom")
                    and _is_max_tokens_unsupported_error(exc)
                ):
                    logger.warning(
                        "%s rejected max_tokens; retrying with max_completion_tokens",
                        self.label,
                    )
                    use_max_completion_tokens = True
                    continue
                if temperature is not None and _is_fixed_temperature_error(exc):
                    logger.warning(
                        "%s rejected temperature=%s; retrying without temperature",
                        self.label,
                        temperature,
                    )
                    temperature = None  # type: ignore[assignment]
                    omit_temperature = True
                    continue
                forced = _fixed_temperature_for_model(self.spec.model)
                if (
                    forced is not None
                    and temperature is not None
                    and _is_fixed_temperature_error(exc)
                    and temperature != forced
                ):
                    logger.warning(
                        "%s rejected temperature=%s; retrying with temperature=%s",
                        self.label,
                        temperature,
                        forced,
                    )
                    temperature = forced
                    continue
                if attempt + 1 >= attempts or not is_retryable_llm_error(exc):
                    record = self.last_provider_record()
                    if record:
                        return CompletionResult(
                            text="",
                            provider_record=record,
                            parse_error=format_llm_error(exc),
                        )
                    raise
                delay = retry_delay_seconds(exc, attempt)
                logger.warning(
                    "Transient LLM error from %s (attempt %d/%d); retrying in %.1fs: %s",
                    self.label,
                    attempt + 1,
                    attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)
        assert last_exc is not None
        record = self.last_provider_record()
        if record:
            return CompletionResult(
                text="",
                provider_record=record,
                parse_error=format_llm_error(last_exc),
            )
        raise last_exc

    def _complete_once(
        self,
        prompt: str,
        system: Optional[str],
        temperature: Optional[float],
        max_tokens: int,
        *,
        use_max_completion_tokens: bool = False,
    ) -> str:
        provider = self.spec.provider

        if provider == "ollama":
            raw_out = self._client.generate(
                prompt,
                system=system,
                temperature=0.7 if temperature is None else temperature,
                max_tokens=max_tokens,
                num_ctx=self.spec.num_ctx,
            )
            raw_payload: Any = (
                {"text": raw_out} if isinstance(raw_out, str) else raw_out
            )
            return self._text_from_stored_response(
                raw_payload,
                lambda: raw_out if isinstance(raw_out, str) else str(raw_out or ""),
            )

        web_search = bool(self.spec.web_search)

        if web_search and _uses_responses_web_search(provider, self.spec.base_url):
            return self._complete_via_responses(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if provider in ("openai", "custom"):
            if (
                web_search
                and _is_moonshot_url(self.spec.base_url)
                and not _is_kimi_k3(self.spec.model)
            ):
                return self._complete_moonshot_with_web_search(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_max_completion_tokens=use_max_completion_tokens,
                )
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            create_kwargs: Dict[str, Any] = {
                "model": self.spec.model,
                "messages": messages,
            }
            if use_max_completion_tokens or _is_openai_max_completion_tokens_model(
                self.spec.model
            ):
                create_kwargs["max_completion_tokens"] = max_tokens
            else:
                create_kwargs["max_tokens"] = max_tokens
            if temperature is not None and not _omit_temperature_for_model(self.spec.model):
                create_kwargs["temperature"] = temperature
            create_kwargs.update(
                _openai_create_extras(
                    self.spec.model, self.spec.base_url, web_search=web_search
                )
            )
            response = self._client.chat.completions.create(**create_kwargs)

            def _parse_openai() -> str:
                from moyo.llm.content_filter import strip_reasoning_spill

                message = response.choices[0].message if response.choices else None
                content = strip_reasoning_spill(_openai_message_text(message))
                return _with_provider_citations(content, response)

            return self._text_from_stored_response(response, _parse_openai)

        if provider == "anthropic":
            kwargs: Dict[str, Any] = {}
            if system:
                kwargs["system"] = system
            create_kwargs: Dict[str, Any] = {
                "model": self.spec.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            }
            if web_search:
                create_kwargs["tools"] = _anthropic_web_search_tools()
            if temperature is not None and not _omit_temperature_for_model(self.spec.model):
                create_kwargs["temperature"] = temperature
            response = self._client.messages.create(**create_kwargs)
            return self._text_from_stored_response(
                response,
                lambda: _anthropic_message_text(response),
            )

        raise RuntimeError(f"unsupported LLM provider: {provider}")

    def _complete_via_responses(
        self,
        prompt: str,
        *,
        system: Optional[str],
        temperature: Optional[float],
        max_tokens: int,
    ) -> str:
        """OpenAI / xAI Responses API with hosted ``web_search`` enabled."""
        create_kwargs: Dict[str, Any] = {
            "model": self.spec.model,
            "input": prompt,
            "tools": [{"type": "web_search"}],
            "max_output_tokens": max_tokens,
        }
        if system:
            create_kwargs["instructions"] = system
        if temperature is not None and not _omit_temperature_for_model(self.spec.model):
            create_kwargs["temperature"] = temperature
        if _is_openai_max_completion_tokens_model(self.spec.model):
            create_kwargs["reasoning"] = {"effort": "low"}
        responses = getattr(self._client, "responses", None)
        if responses is None or not hasattr(responses, "create"):
            raise RuntimeError(
                f"{self.label} requires the Responses API for web search, but the "
                "SDK client has no responses.create"
            )
        response = responses.create(**create_kwargs)
        return self._text_from_stored_response(
            response,
            lambda: _responses_output_text(response),
        )

    def _complete_moonshot_with_web_search(
        self,
        prompt: str,
        *,
        system: Optional[str],
        temperature: Optional[float],
        max_tokens: int,
        use_max_completion_tokens: bool,
    ) -> str:
        """Moonshot Kimi K2: declare ``$web_search`` and echo tool results server-side.

        Kimi K3 does not use this path: its tokenizer rejects the builtin tool
        schema (HTTP 400 ``tokenization failed``).
        """
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        tools = [
            {
                "type": "builtin_function",
                "function": {"name": "$web_search"},
            }
        ]
        last_content = ""
        response = None
        message = None
        for _ in range(4):
            create_kwargs: Dict[str, Any] = {
                "model": self.spec.model,
                "messages": messages,
                "tools": tools,
            }
            if use_max_completion_tokens or _is_openai_max_completion_tokens_model(
                self.spec.model
            ):
                create_kwargs["max_completion_tokens"] = max_tokens
            else:
                create_kwargs["max_tokens"] = max_tokens
            if temperature is not None and not _omit_temperature_for_model(self.spec.model):
                create_kwargs["temperature"] = temperature
            create_kwargs.update(
                _openai_create_extras(
                    self.spec.model, self.spec.base_url, web_search=True
                )
            )
            response = self._client.chat.completions.create(**create_kwargs)
            self._store_provider_record(raw=response, content="")
            message = response.choices[0].message if response.choices else None
            if message is None:
                break
            last_content = _openai_message_text(message)
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                return self._finish_moonshot_response(
                    _with_provider_citations(last_content, response),
                    response,
                    message,
                )
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": getattr(message, "content", None) or "",
            }
            serialized_calls = []
            for call in tool_calls:
                fn = getattr(call, "function", None)
                serialized_calls.append(
                    {
                        "id": getattr(call, "id", ""),
                        "type": getattr(call, "type", None) or "builtin_function",
                        "function": {
                            "name": getattr(fn, "name", "") if fn else "",
                            "arguments": getattr(fn, "arguments", "") if fn else "",
                        },
                    }
                )
            assistant_msg["tool_calls"] = serialized_calls
            messages.append(assistant_msg)
            for call in serialized_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["function"]["name"],
                        "content": call["function"]["arguments"],
                    }
                )
        return self._finish_moonshot_response(last_content, response, message)

    def _finish_moonshot_response(
        self,
        text: str,
        response: Any,
        message: Any,
    ) -> str:
        from moyo.llm.content_filter import openai_reasoning_text, strip_reasoning_spill

        content = strip_reasoning_spill(text)
        self._store_provider_record(
            content=content,
            reasoning=openai_reasoning_text(message) or None,
            raw=response,
        )
        return content

    # -- offline stub -------------------------------------------------------
    def _echo(self, prompt: str, system: Optional[str]) -> str:
        """Deterministic offline response.

        Useful for tests and air-gapped smoke runs. When the prompt looks like a
        request for ``N`` reworded variants, it emits a numbered list so the
        exploration pipeline can be exercised without any network access.
        """
        lowered = prompt.lower()
        import re

        match = re.search(r"give me (\d+) different", lowered)
        if match:
            n = int(match.group(1))
            topic = self._extract_quoted(prompt) or "the topic"
            variants = [
                f"{topic} overview and key facts",
                f"detailed explanation of {topic}",
                f"history and background of {topic}",
                f"technical specifics of {topic}",
                f"common questions and answers about {topic}",
                f"authoritative sources on {topic}",
            ]
            return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(variants[:n]))

        return textwrap.dedent(
            f"""\
            [echo:{self.spec.model}] Offline stub response.
            This provider does not contact any network; it returns placeholder
            text so the pipeline can be tested end to end. Query was:
            {prompt.strip()[:500]}"""
        )

    @staticmethod
    def _extract_quoted(text: str) -> Optional[str]:
        import re

        match = re.search(r'"([^"]+)"', text)
        return match.group(1) if match else None
