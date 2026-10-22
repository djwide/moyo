"""Vertex AI Gemini for Cloud Run processing (ADC, no AI Studio API key).

Retrieval ("what does AI know") uses Gemini on AI
Studio. This module is for extract, cluster, synthesize, reword, and
other pipeline jobs that should not impersonate a public Gemini model.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from moyo.llm.client import LLMSpec, _is_gemini_model, ensure_env_loaded
from moyo.llm.utility import running_in_cloud

logger = logging.getLogger(__name__)

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Gemini 3.x publisher models use the ``global`` location (see ``vertex_openai_base_url``).
# Override with ``MOYO_VERTEX_GEMINI_MODEL`` / ``MOYO_VERTEX_UTILITY_MODEL``.
VERTEX_DEFAULT_GEMINI_MODEL = "google/gemini-3.1-pro-preview"
# Hosted Ollama-replacement jobs (reword / extract / cluster / …).
# Retrieval Gemini is not rewritten here; it stays on AI Studio.
# Vertex model for Flash retrieval slots that must run on Vertex.
VERTEX_FLASH_GEMINI_MODEL = "google/gemini-3.8-flash"
# Processing model. 3.8 Flash stalled for minutes at a time on 2026-10-22; Flash-Lite
# answered in about a second, so it is the default and 3.8 Flash is the fallback.
VERTEX_UTILITY_GEMINI_MODEL = "google/gemini-3.1-flash-lite"

# Sized for a full retrieval fan-out (e.g. 7 models × 3 seeds = 21 POSTs)
# plus paraphraser / utility calls on the same process. Too-low caps make
# extra requests wait on local sockets and burn the per-call timeout.
HTTPX_MAX_CONNECTIONS = 100
HTTPX_MAX_KEEPALIVE_CONNECTIONS = 50


def vertex_enabled() -> bool:
    flag = os.environ.get("MOYO_VERTEX_GEMINI", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def vertex_project() -> str:
    return (
        os.environ.get("MOYO_VERTEX_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or "senteguard-website"
    ).strip()


def vertex_location() -> str:
    return (
        os.environ.get("MOYO_VERTEX_LOCATION")
        or os.environ.get("MOYO_CLOUD_REGION")
        or "global"
    ).strip()


def vertex_openai_base_url(project: Optional[str] = None, location: Optional[str] = None) -> str:
    project = project or vertex_project()
    location = location or vertex_location()
    if location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/"
        f"{project}/locations/{location}/endpoints/openapi"
    )


def is_vertex_openai_url(url: Optional[str]) -> bool:
    text = (url or "").lower()
    return "aiplatform.googleapis.com" in text and "/endpoints/openapi" in text


def vertex_access_token() -> Optional[str]:
    """OAuth token from the Cloud Run job service account (ADC)."""
    try:
        import google.auth
        from google.auth.transport.requests import Request

        creds, _ = google.auth.default(scopes=[VERTEX_SCOPE])
        if not creds.valid:
            creds.refresh(Request())
        if not creds.token:
            return None
        return creds.token
    except Exception as exc:
        logger.warning("Vertex ADC token unavailable: %s", exc)
        return None


def vertex_api_key() -> str:
    """Callable API key for the OpenAI SDK (refreshes ADC each request)."""
    token = vertex_access_token()
    if not token:
        raise RuntimeError(
            "Vertex ADC token unavailable. Grant the Cloud Run job SA "
            "roles/aiplatform.user (moyo-worker@senteguard-website.iam.gserviceaccount.com)."
        )
    return token


def _prefixed_gemini_id(model: str) -> str:
    text = (model or "").strip()
    if not text:
        return VERTEX_DEFAULT_GEMINI_MODEL
    if text.startswith("google/") or "/" in text:
        return text
    return f"google/{text}"


def vertex_gemini_model(current: str = "") -> str:
    """Stable Vertex model id for a Gemini *retrieval* slot.

    AI Studio retrieval ids stay on AI Studio; when pipeline code must run
    Gemini on Vertex we map catalog names to Vertex publisher models.
    Flash catalog slots stay on Flash; everything else uses Pro unless
    ``MOYO_VERTEX_GEMINI_MODEL`` overrides.
    """
    override = (os.environ.get("MOYO_VERTEX_GEMINI_MODEL") or "").strip()
    if override:
        return _prefixed_gemini_id(override)
    name = (current or "").lower()
    if "flash" in name:
        return VERTEX_FLASH_GEMINI_MODEL
    return VERTEX_DEFAULT_GEMINI_MODEL


def vertex_utility_model() -> str:
    """Vertex model for Cloud Run reword / summary / extract / cluster / synthesize."""
    override = (os.environ.get("MOYO_VERTEX_UTILITY_MODEL") or "").strip()
    if override:
        return _prefixed_gemini_id(override)
    return VERTEX_UTILITY_GEMINI_MODEL


# Pipeline processing calls (reword / extract / cluster / summaries / organize)
# on the Vertex utility model stream so a stalled request fails on the idle
# timeout instead of waiting out the whole deadline. On 2026-10-22 global
# gemini-3.8-flash intermittently sent no first byte for minutes, and four
# 180s attempts turned one extract call into a 10-minute stall. The model is
# global-only, so the retry goes to a different model rather than a region.
VERTEX_PROCESSING_FALLBACK_MODEL = VERTEX_FLASH_GEMINI_MODEL
PROCESSING_IDLE_TIMEOUT = 45.0
PROCESSING_DEADLINE = 120.0
PROCESSING_MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class ProcessingPolicy:
    """Fail-fast settings for one Vertex processing call."""

    idle_timeout: float
    deadline: float
    max_attempts: int
    fallback_model: Optional[str]


def _env_float(name: str, default: float) -> float:
    try:
        value = float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def processing_policy(base_url: Optional[str], model: str, *, timeout: float = 0) -> Optional[ProcessingPolicy]:
    """Policy for Vertex utility-model calls; ``None`` leaves the call unchanged.

    Retrieval Gemini (Pro, AI Studio, web search) never matches.
    """
    if not is_vertex_openai_url(base_url) or model != vertex_utility_model():
        return None
    deadline = _env_float("MOYO_PROCESSING_DEADLINE", PROCESSING_DEADLINE)
    if timeout and timeout > 0:
        deadline = min(deadline, float(timeout))
    try:
        attempts = int((os.environ.get("MOYO_PROCESSING_MAX_ATTEMPTS") or "").strip() or PROCESSING_MAX_ATTEMPTS)
    except ValueError:
        attempts = PROCESSING_MAX_ATTEMPTS
    fallback = (os.environ.get("MOYO_VERTEX_FALLBACK_MODEL") or VERTEX_PROCESSING_FALLBACK_MODEL).strip()
    if fallback.lower() in {"", "0", "none", "off"}:
        fallback_model = None
    else:
        fallback_model = _prefixed_gemini_id(fallback)
        if fallback_model == model:
            fallback_model = None
    return ProcessingPolicy(
        idle_timeout=min(_env_float("MOYO_PROCESSING_IDLE_TIMEOUT", PROCESSING_IDLE_TIMEOUT), deadline),
        deadline=deadline,
        max_attempts=max(1, attempts),
        fallback_model=fallback_model,
    )


def rewrite_gemini_spec_for_vertex(spec: LLMSpec) -> LLMSpec:
    """Point a Gemini spec at Vertex's OpenAI-compatible endpoint.

    Used for pipeline processing that already lives on Vertex. Retrieval
    AI Studio Gemini calls stay on AI Studio and must not go through
    this rewrite.
    """
    ensure_env_loaded()
    if not running_in_cloud() or not vertex_enabled():
        return spec
    if is_vertex_openai_url(spec.base_url):
        return spec
    if not _is_gemini_model(spec.model, spec.base_url):
        return spec
    model = vertex_gemini_model(spec.model)
    logger.info("Cloud Gemini via Vertex: %s @ %s", model, vertex_openai_base_url())
    return LLMSpec(
        provider="custom",
        model=model,
        api_key=None,
        base_url=vertex_openai_base_url(),
        label=spec.label or f"Gemini (Vertex {model})",
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=spec.timeout,
        max_retries=spec.max_retries,
    )


def vertex_openai_headers() -> dict[str, str]:
    return {"x-goog-user-project": vertex_project()}


def httpx_connection_limits() -> Any:
    """Pool large enough that concurrent retrieval POSTs are not socket-blocked."""
    import httpx

    return httpx.Limits(
        max_connections=HTTPX_MAX_CONNECTIONS,
        max_keepalive_connections=HTTPX_MAX_KEEPALIVE_CONNECTIONS,
    )


def openai_compatible_http_client(timeout: Any, *, vertex: bool = False) -> Any:
    """httpx client matching working curl (HTTP/1.1) from the NAT VM.

    Retrieval uses the sync OpenAI SDK via a thread pool, so this is
    ``httpx.Client`` (not ``AsyncClient``) with the same high connection
    limits you would set on an async client sending ~21 concurrent POSTs.
    """
    import httpx

    kwargs: dict[str, Any] = {
        "timeout": timeout,
        "http2": False,
        "follow_redirects": True,
        "limits": httpx_connection_limits(),
    }
    if vertex:
        kwargs["headers"] = vertex_openai_headers()
    return httpx.Client(**kwargs)


def openai_compatible_async_http_client(timeout: Any, *, vertex: bool = False) -> Any:
    """Async counterpart with the same pool limits as the sync retrieval client."""
    import httpx

    kwargs: dict[str, Any] = {
        "timeout": timeout,
        "http2": False,
        "follow_redirects": True,
        "limits": httpx_connection_limits(),
    }
    if vertex:
        kwargs["headers"] = vertex_openai_headers()
    return httpx.AsyncClient(**kwargs)
