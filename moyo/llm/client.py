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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    """Load ``.env`` into ``os.environ`` once (idempotent)."""
    global _ENV_FILE_LOADED
    if _ENV_FILE_LOADED:
        return
    _ENV_FILE_LOADED = True
    _load_env_file()


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


def _fixed_temperature_for_model(model: str) -> Optional[float]:
    """Return a forced temperature for models that reject other values."""
    name = (model or "").lower()
    # K2.5/K2.6 non-thinking mode (our default) requires temperature=0.6.
    if _is_kimi_k25_or_k26(name):
        return 0.6
    # Other Moonshot Kimi K2.x / K3 only accept temperature=1.
    if (
        name.startswith("kimi-k2")
        or name.startswith("kimi-k3")
        or "/kimi-k2" in name
        or "/kimi-k3" in name
    ):
        return 1.0
    return None


# Some OpenAI-compatible providers reject tiny completion caps (e.g. Perplexity
# requires max_tokens >= 16).
MIN_COMPLETION_TOKENS = 16


def _is_gemini_model(model: str, base_url: Optional[str] = None) -> bool:
    name = (model or "").lower()
    url = (base_url or "").lower()
    return "gemini" in name or "generativelanguage.googleapis.com" in url


def _openai_extra_body_for_model(model: str) -> Dict[str, Any]:
    """Provider-specific OpenAI-compatible request fields.

    Kimi K2.5/K2.6 default to thinking mode. Reasoning tokens count against
    ``max_tokens``, so short caps often return empty ``content``. Disable
    thinking so retrieval replies land in ``content``.
    """
    if _is_kimi_k25_or_k26(model):
        return {"thinking": {"type": "disabled"}}
    return {}


def _openai_create_extras(
    model: str, base_url: Optional[str] = None
) -> Dict[str, Any]:
    """Extra kwargs for ``chat.completions.create`` beyond messages/tokens."""
    extras: Dict[str, Any] = {}
    extra_body = _openai_extra_body_for_model(model)
    if extra_body:
        extras["extra_body"] = extra_body
    # gemini-3.1-pro-preview (and other Gemini thinking models) spend max_tokens on
    # internal reasoning first; low reasoning effort keeps short replies usable.
    # ``none`` is rejected by Pro-class aliases that require thinking mode.
    if _is_gemini_model(model, base_url):
        extras["reasoning_effort"] = "low"
    return extras


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
    if "invalid temperature" not in text:
        return False
    return "only 1" in text or "only 0.6" in text or "only 0.60" in text


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
            num_ctx=int(num_ctx) if num_ctx is not None else None,
        )


class LLMClient:
    """A uniform wrapper over one LLM endpoint described by an :class:`LLMSpec`."""

    def __init__(self, spec: LLMSpec):
        self.spec = spec
        self._init_error: Optional[str] = None
        self._client = self._init_client()

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

                kwargs: Dict[str, Any] = {}
                if provider == "custom":
                    if not self.spec.base_url:
                        self._init_error = (
                            "provider 'custom' requires a base_url pointing at an "
                            "OpenAI-compatible endpoint (e.g. http://localhost:8000/v1)"
                        )
                        return None
                    kwargs["base_url"] = self.spec.base_url
                    # Many self-hosted servers ignore the key but the SDK
                    # requires a non-empty value.
                    kwargs["api_key"] = self.spec.api_key or "not-needed"
                else:
                    if self.spec.api_key:
                        kwargs["api_key"] = self.spec.api_key
                    if self.spec.base_url:
                        kwargs["base_url"] = self.spec.base_url
                return OpenAI(**kwargs)
            if provider == "anthropic":
                from anthropic import Anthropic

                kwargs = {}
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

        Transient rate-limit / overload / network errors are retried with
        backoff (honouring provider "retry in Ns" hints when present). Hard
        failures such as exhausted credits or invalid API keys are not retried.

        ``retries`` overrides ``spec.max_retries`` for this call (use ``0`` for
        a single-shot preflight probe).

        Raises ``RuntimeError`` if the client could not be initialized (e.g. a
        missing API key or unreachable endpoint) so callers can record the
        failure per source instead of crashing the whole run.
        """
        temperature = self.spec.temperature if temperature is None else temperature
        fixed = _fixed_temperature_for_model(self.spec.model)
        if fixed is not None:
            temperature = fixed
        max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens
        max_tokens = max(MIN_COMPLETION_TOKENS, int(max_tokens))
        # Gemini thinking models count reasoning toward max_tokens; tiny caps
        # often return empty content with finish_reason=length.
        if _is_gemini_model(self.spec.model, self.spec.base_url) and max_tokens < 256:
            max_tokens = 256
        provider = self.spec.provider

        if provider == "echo":
            return self._echo(prompt, system)

        if self._client is None:
            raise RuntimeError(self._init_error or f"LLM provider '{provider}' unavailable")

        max_retries = self.spec.max_retries if retries is None else max(0, int(retries))
        attempts = max(1, int(max_retries) + 1)
        last_exc: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                return self._complete_once(
                    prompt, system=system, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:
                last_exc = exc
                # Some providers reject non-fixed temperatures without it being in the model id.
                forced = _fixed_temperature_for_model(self.spec.model)
                if forced is None and _is_fixed_temperature_error(exc):
                    forced = 1.0
                if _is_fixed_temperature_error(exc) and forced is not None and temperature != forced:
                    logger.warning(
                        "%s rejected temperature=%s; retrying with temperature=%s",
                        self.label,
                        temperature,
                        forced,
                    )
                    temperature = forced
                    continue
                if attempt + 1 >= attempts or not is_retryable_llm_error(exc):
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
        raise last_exc

    def _complete_once(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        provider = self.spec.provider

        if provider == "ollama":
            return self._client.generate(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                num_ctx=self.spec.num_ctx,
            )

        if provider in ("openai", "custom"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            create_kwargs: Dict[str, Any] = {
                "model": self.spec.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            create_kwargs.update(
                _openai_create_extras(self.spec.model, self.spec.base_url)
            )
            response = self._client.chat.completions.create(**create_kwargs)
            content = (response.choices[0].message.content or "").strip()
            return _with_provider_citations(content, response)

        if provider == "anthropic":
            kwargs: Dict[str, Any] = {}
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(
                model=self.spec.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return (response.content[0].text if response.content else "").strip()

        raise RuntimeError(f"unsupported LLM provider: {provider}")

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
