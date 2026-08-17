"""Cheap utility LLM for rewording, translation, clustering, and summaries.

Local desktop runs keep Ollama (``llama3.1:8b``). Cloud Run has no Ollama, so
the same jobs use a hosted OpenAI-compatible model (default: OpenRouter
Llama 3.1 8B Instruct).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from moyo.llm.client import LLMClient, LLMSpec, ensure_env_loaded

logger = logging.getLogger(__name__)

LOCAL_UTILITY_MODEL = "llama3.1:8b"
LOCAL_UTILITY_BASE_URL = "http://localhost:11434"

# Closest cheap hosted analogue of the local 8B Ollama model.
CLOUD_UTILITY_PROVIDER = "custom"
CLOUD_UTILITY_MODEL = "meta-llama/llama-3.1-8b-instruct"
CLOUD_UTILITY_BASE_URL = "https://openrouter.ai/api/v1"

MOONSHOT_FALLBACK_MODEL = "kimi-k2.6"
MOONSHOT_FALLBACK_BASE_URL = "https://api.moonshot.ai/v1"


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

    provider = (os.environ.get("MOYO_UTILITY_PROVIDER") or "").strip()
    model = (os.environ.get("MOYO_UTILITY_MODEL") or "").strip()
    base_url = (os.environ.get("MOYO_UTILITY_BASE_URL") or "").strip()
    explicit_key = (os.environ.get("MOYO_UTILITY_API_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    moonshot_key = (os.environ.get("MOONSHOT_API_KEY") or "").strip()
    api_key = explicit_key or openrouter_key

    # Dockerfile pins OpenRouter. If that key is missing, use Moonshot
    # (already required for extract) rather than calling OpenRouter unauthenticated.
    openrouter_like = (not base_url) or ("openrouter.ai" in base_url)
    if not api_key and moonshot_key and openrouter_like and not explicit_key:
        provider = "custom"
        model = MOONSHOT_FALLBACK_MODEL
        base_url = MOONSHOT_FALLBACK_BASE_URL
        api_key = moonshot_key
    else:
        provider = provider or CLOUD_UTILITY_PROVIDER
        model = model or CLOUD_UTILITY_MODEL
        base_url = base_url or CLOUD_UTILITY_BASE_URL

    spec = LLMSpec(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key or None,
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
        "yes" if spec.api_key else "NO",
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
        ("OPENROUTER_API_KEY", "$OPENROUTER_API_KEY"),
        ("MOONSHOT_API_KEY", "$MOONSHOT_API_KEY"),
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
    cfg.setdefault("temperature", 0.1)
    cfg.setdefault("max_tokens", 4000)
    cfg.setdefault("timeout", 180)
    return cfg
