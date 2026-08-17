"""Cheap utility LLM for rewording, translation, clustering, and summaries.

Local desktop runs keep Ollama (``llama3.1:8b``). Cloud Run has no Ollama, so
those jobs use Vertex Gemini Flash (``google/gemini-2.5-flash`` via ADC).
Override with ``MOYO_UTILITY_*`` / ``MOYO_VERTEX_UTILITY_MODEL``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from moyo.llm.client import LLMClient, LLMSpec, ensure_env_loaded

logger = logging.getLogger(__name__)

LOCAL_UTILITY_MODEL = "llama3.1:8b"
LOCAL_UTILITY_BASE_URL = "http://localhost:11434"

CLOUD_KIMI_PROVIDER = "custom"
CLOUD_KIMI_MODEL = "kimi-k2.6"
CLOUD_KIMI_BASE_URL = "https://api.moonshot.ai/v1"


def running_in_cloud() -> bool:
    """True on Cloud Run jobs/services or when ``MOYO_CLOUD_RUNTIME`` is set."""
    flag = os.environ.get("MOYO_CLOUD_RUNTIME", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(
        os.environ.get("CLOUD_RUN_JOB")
        or os.environ.get("CLOUD_RUN_EXECUTION")
        or os.environ.get("K_SERVICE")
    )


def _stale_cloud_utility_pin(model: str, base_url: str) -> bool:
    """True for leftover Kimi / OpenRouter-Llama utility env from older images."""
    name = (model or "").lower()
    url = (base_url or "").lower()
    return (
        "moonshot.ai" in url
        or name.startswith("kimi")
        or "kimi-k2" in name
        or "openrouter.ai" in url
        or "llama-3.1-8b" in name
    )


def utility_llm_spec() -> LLMSpec:
    """Spec for reword / translate / cluster / summary (not retrieval fan-out)."""
    ensure_env_loaded()
    try:
        from moyo.llm.testing import is_test_mode, test_llm_spec

        if is_test_mode():
            return test_llm_spec()
    except Exception:
        pass

    if not running_in_cloud():
        return LLMSpec(
            provider="ollama",
            model=os.environ.get("MOYO_UTILITY_MODEL") or LOCAL_UTILITY_MODEL,
            base_url=(
                os.environ.get("MOYO_UTILITY_BASE_URL")
                or os.environ.get("MOYO_OLLAMA_BASE_URL")
                or LOCAL_UTILITY_BASE_URL
            ),
            label=f"Llama 3.1 8B (local utility)",
            temperature=0.3,
            max_tokens=800,
            timeout=120,
        )

    from moyo.llm.client import _is_gemini_model
    from moyo.llm.vertex import (
        is_vertex_openai_url,
        vertex_openai_base_url,
        vertex_utility_model,
    )

    explicit_provider = (os.environ.get("MOYO_UTILITY_PROVIDER") or "").strip()
    explicit_model = (os.environ.get("MOYO_UTILITY_MODEL") or "").strip()
    explicit_base = (os.environ.get("MOYO_UTILITY_BASE_URL") or "").strip()
    explicit_key = (os.environ.get("MOYO_UTILITY_API_KEY") or "").strip()

    use_vertex_flash = (not explicit_model) or _stale_cloud_utility_pin(
        explicit_model, explicit_base
    )
    if use_vertex_flash:
        provider = "custom"
        model = vertex_utility_model()
        base_url = vertex_openai_base_url()
        api_key = None
    else:
        provider = explicit_provider or "custom"
        model = explicit_model
        base_url = explicit_base
        if _is_gemini_model(model, base_url) and not is_vertex_openai_url(base_url):
            base_url = vertex_openai_base_url()
            api_key = None
        else:
            api_key = (
                explicit_key
                or os.environ.get("MOONSHOT_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
                or None
            )
            if isinstance(api_key, str):
                api_key = api_key.strip() or None

    spec = LLMSpec(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        label=f"Cloud utility ({model})",
        temperature=0.3,
        max_tokens=800,
        timeout=120,
    )
    logger.info(
        "Cloud utility LLM: %s/%s @ %s (api_key=%s)",
        spec.provider,
        spec.model,
        spec.base_url,
        "yes" if spec.api_key else "vertex-adc" if is_vertex_openai_url(spec.base_url) else "NO",
    )
    return spec


def get_utility_llm() -> LLMClient:
    return LLMClient(utility_llm_spec())


def _utility_api_key_env_ref(spec: LLMSpec) -> Optional[str]:
    """``$ENV_VAR`` form so cluster YAML never stores the raw secret."""
    key = (spec.api_key or "").strip()
    if not key:
        return None
    pairs = (
        ("MOYO_UTILITY_API_KEY", "$MOYO_UTILITY_API_KEY"),
        ("MOONSHOT_API_KEY", "$MOONSHOT_API_KEY"),
        ("OPENROUTER_API_KEY", "$OPENROUTER_API_KEY"),
    )
    for env_name, ref in pairs:
        if key == (os.environ.get(env_name) or "").strip():
            return ref
    return None


def utility_cluster_config(base: Optional[dict] = None) -> dict:
    """Overlay hosted-utility settings onto reports ``cluster:`` config."""
    cfg = dict(base or {})
    spec = utility_llm_spec()
    cfg["provider"] = spec.provider
    cfg["model"] = spec.model
    if spec.base_url:
        cfg["base_url"] = spec.base_url
    key_ref = _utility_api_key_env_ref(spec)
    if key_ref:
        cfg["api_key"] = key_ref
    elif spec.api_key:
        cfg["api_key"] = spec.api_key
    else:
        cfg.pop("api_key", None)
    cfg.setdefault("temperature", 0.1)
    cfg.setdefault("max_tokens", 4000)
    cfg.setdefault("timeout", 180)
    cfg.pop("num_ctx", None)
    return cfg


def vertex_flash_hosted_config(base: Optional[dict] = None) -> dict:
    """Vertex Gemini Flash overlay for Cloud Run report stages (ADC, no API key)."""
    from moyo.llm.vertex import vertex_openai_base_url, vertex_utility_model

    cfg = dict(base or {})
    cfg["provider"] = "custom"
    cfg["model"] = vertex_utility_model()
    cfg["base_url"] = vertex_openai_base_url()
    cfg.pop("api_key", None)
    cfg.pop("num_ctx", None)
    cfg.setdefault("timeout", 180)
    return cfg


def kimi_hosted_config(base: Optional[dict] = None) -> dict:
    """Moonshot Kimi overlay (unused by the worker; Vertex Flash is default)."""
    cfg = dict(base or {})
    cfg["provider"] = CLOUD_KIMI_PROVIDER
    cfg["model"] = (
        os.environ.get("MOYO_CLOUD_KIMI_MODEL") or CLOUD_KIMI_MODEL
    ).strip()
    cfg["base_url"] = CLOUD_KIMI_BASE_URL
    cfg["api_key"] = "$MOONSHOT_API_KEY"
    cfg.pop("num_ctx", None)
    cfg.setdefault("timeout", 180)
    return cfg


def cloud_paid_llm_config(base: Optional[dict] = None) -> dict:
    """Optional OpenAI/Anthropic overlay (unused by the worker)."""
    cfg = dict(base or {})
    choice = (os.environ.get("MOYO_CLOUD_REPORT_PROVIDER") or "openai").strip().lower()
    cfg.pop("base_url", None)
    cfg.pop("num_ctx", None)
    if choice in {"anthropic", "claude"}:
        cfg["provider"] = "anthropic"
        cfg["model"] = (
            os.environ.get("MOYO_CLOUD_REPORT_MODEL") or "claude-sonnet-4-6"
        ).strip()
        cfg["api_key"] = "$ANTHROPIC_API_KEY"
    else:
        cfg["provider"] = "openai"
        cfg["model"] = (os.environ.get("MOYO_CLOUD_REPORT_MODEL") or "gpt-4o").strip()
        cfg["api_key"] = "$OPENAI_API_KEY"
    cfg.setdefault("timeout", 120)
    return cfg
