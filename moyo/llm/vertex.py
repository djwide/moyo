"""Vertex AI Gemini for Cloud Run (ADC, no AI Studio API key)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from moyo.llm.client import LLMSpec, _is_gemini_model, ensure_env_loaded
from moyo.llm.utility import running_in_cloud

logger = logging.getLogger(__name__)

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# AI Studio preview ids 404 on Vertex us-central1. Override with
# ``MOYO_VERTEX_GEMINI_MODEL`` (e.g. ``google/gemini-1.5-pro``).
VERTEX_DEFAULT_GEMINI_MODEL = "google/gemini-2.5-pro"


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
        or "us-central1"
    ).strip()


def vertex_openai_base_url(project: Optional[str] = None, location: Optional[str] = None) -> str:
    project = project or vertex_project()
    location = location or vertex_location()
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
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


def vertex_gemini_model(_current: str = "") -> str:
    """Stable Vertex model id. AI Studio preview names are not used as-is."""
    override = (os.environ.get("MOYO_VERTEX_GEMINI_MODEL") or "").strip()
    if override:
        if override.startswith("google/") or "/" in override:
            return override
        return f"google/{override}"
    return VERTEX_DEFAULT_GEMINI_MODEL


def rewrite_gemini_spec_for_vertex(spec: LLMSpec) -> LLMSpec:
    """Point Gemini retrieval at Vertex's OpenAI-compatible endpoint.

    Auth is applied later by :class:`~moyo.llm.client.LLMClient` via ADC
    (no AI Studio API key, token refreshed per request).
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


def openai_compatible_http_client(timeout: Any, *, vertex: bool = False) -> Any:
    """httpx client matching working curl (HTTP/1.1) from the NAT VM."""
    import httpx

    kwargs: dict[str, Any] = {
        "timeout": timeout,
        "http2": False,
        "follow_redirects": True,
        "limits": httpx.Limits(max_keepalive_connections=8, max_connections=20),
    }
    if vertex:
        kwargs["headers"] = vertex_openai_headers()
    return httpx.Client(**kwargs)
