"""Default-LLM resolution and the retrieval-LLM registry.

This module is the single place moyo decides *which* LLM is the "default" and
*which* LLMs to fan a query out to during public exploration.

Hot-swapping the default model across moyo:

- Persistent: set ``MOYO_LLM_PROVIDER`` / ``MOYO_LLM_MODEL`` /
  ``MOYO_LLM_API_KEY`` / ``MOYO_LLM_BASE_URL`` (env or ``.env``).
- Runtime / in-process: call :func:`set_default_llm`.

Configuring the retrieval LLMs (the closed/open/local models a query is sent
to), in order of precedence:

1. ``MOYO_RETRIEVAL_LLMS`` env var holding a JSON list of specs.
2. A JSON file at ``MOYO_RETRIEVAL_LLMS_FILE`` (default
   ``config/retrieval_llms.json``) with either a top-level list or a
   ``{"retrieval_llms": [...]}`` object.
3. Fallback: just the default LLM.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Set, Union

from moyo.llm.client import LLMClient, LLMSpec, apply_retrieval_timeout

logger = logging.getLogger(__name__)

_DEFAULT_RETRIEVAL_CONFIG = "config/retrieval_llms.json"

# In-process override for the default LLM (highest precedence).
_default_override: Optional[LLMSpec] = None


def default_spec() -> LLMSpec:
    """Resolve the current default LLM spec.

    Precedence: runtime override (:func:`set_default_llm`) > ``--test`` /
    ``MOYO_TEST_MODE`` (echo) > ``MOYO_LLM_*`` settings / ``.env`` > field
    defaults.
    """
    if _default_override is not None:
        return _default_override

    try:
        from moyo.llm.testing import is_test_mode, test_llm_spec
        if is_test_mode():
            return test_llm_spec()
    except Exception:
        pass

    try:
        from moyo.config.settings import get_settings

        llm = get_settings().llm
        return LLMSpec(
            provider=llm.provider,
            model=llm.model,
            api_key=llm.api_key,
            base_url=getattr(llm, "base_url", None),
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            timeout=llm.timeout,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Falling back to echo default LLM (%s)", exc)
        return LLMSpec(provider="echo")


def get_default_llm() -> LLMClient:
    """Return an :class:`LLMClient` for the current default LLM."""
    return LLMClient(default_spec())


def set_default_llm(spec: Union[LLMSpec, dict, None]) -> None:
    """Hot-swap the process-wide default LLM.

    Pass ``None`` to clear the override and fall back to settings/env.
    """
    global _default_override
    if spec is None:
        _default_override = None
    elif isinstance(spec, LLMSpec):
        _default_override = spec
    else:
        _default_override = LLMSpec.from_dict(spec)


def _load_retrieval_entries() -> List[dict]:
    raw = os.environ.get("MOYO_RETRIEVAL_LLMS")
    if raw:
        try:
            parsed = json.loads(raw)
            entries = parsed.get("retrieval_llms", []) if isinstance(parsed, dict) else parsed
            if isinstance(entries, list) and entries:
                return [d for d in entries if isinstance(d, dict)]
        except Exception as exc:
            logger.warning("Ignoring invalid MOYO_RETRIEVAL_LLMS: %s", exc)

    path = Path(os.environ.get("MOYO_RETRIEVAL_LLMS_FILE", _DEFAULT_RETRIEVAL_CONFIG))
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data.get("retrieval_llms", []) if isinstance(data, dict) else data
            if isinstance(entries, list) and entries:
                return [d for d in entries if isinstance(d, dict)]
        except Exception as exc:
            logger.warning("Ignoring invalid retrieval-LLM config %s: %s", path, exc)

    return []


def _load_retrieval_specs(*, include_optional: bool = False) -> List[LLMSpec]:
    entries = _load_retrieval_entries()
    specs: List[LLMSpec] = []
    for data in entries:
        if data.get("optional") and not include_optional:
            continue
        try:
            specs.append(LLMSpec.from_dict(data))
        except Exception as exc:
            logger.warning("Ignoring invalid retrieval LLM spec %s: %s", data.get("model"), exc)
    return specs or [default_spec()]


def get_retrieval_specs(*, include_optional: bool = False) -> List[LLMSpec]:
    """Return the configured retrieval-LLM specs (see module docstring).

    Local Ollama is intentionally excluded: it is used for prompt rewording
    and report clustering, not as a ``moyo-gather explore`` retrieval target.
    Optional storefront extras are omitted unless ``include_optional`` is true
    (used when an order names specific extra model ids).
    """
    try:
        from moyo.llm.testing import is_test_mode, test_llm_spec
        if is_test_mode():
            return [test_llm_spec()]
    except Exception:
        pass
    specs = _load_retrieval_specs(include_optional=include_optional)
    kept: List[LLMSpec] = []
    for spec in specs:
        provider = (spec.provider or "").lower()
        if provider == "ollama":
            logger.info(
                "Skipping local Ollama retrieval LLM %s (not used for explore fan-out)",
                spec.label or spec.model,
            )
            continue
        try:
            from moyo.llm.vertex import rewrite_gemini_spec_for_vertex

            spec = rewrite_gemini_spec_for_vertex(spec)
        except Exception as exc:
            logger.warning("Vertex Gemini rewrite skipped: %s", exc)
        kept.append(spec)
    return kept if kept else specs


def retrieval_model_id(spec: LLMSpec) -> str:
    """Stable id for storefront / order filtering (``provider:model``)."""
    provider = (spec.provider or "echo").strip().lower()
    model = (spec.model or "").strip()
    return f"{provider}:{model}"


def get_retrieval_llms(
    model_ids: Optional[List[str]] = None,
    *,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    web_search_model_ids: Optional[Set[str]] = None,
    require_match: bool = False,
) -> List[LLMClient]:
    """Return an :class:`LLMClient` for each configured retrieval LLM.

    When ``model_ids`` is set, only specs whose :func:`retrieval_model_id`
    or label appears in that list are included (order preserved), including
    optional extras. Unknown ids are ignored. If nothing matches, falls back
    to the default (non-optional) configured set unless ``require_match``.

    ``web_search_model_ids`` lists retrieval ids that should use hosted web
    search (300s timeout, or higher when ``timeout`` is set). Reasoning-budget
    models use 240s. Others use 120s. Pass ``timeout`` (e.g. admin model
    reruns at 480s) to raise the per-call floor. The same per-model caps apply
    to Exposure Data, Snapshot, and Basis unless overridden. ``max_retries``
    overrides each spec.
    """
    wanted = {str(x).strip() for x in (model_ids or []) if str(x).strip()}
    web_search_wanted = {str(x).strip() for x in (web_search_model_ids or set()) if str(x).strip()}
    specs = get_retrieval_specs(include_optional=bool(wanted))
    if wanted:
        filtered = [
            spec
            for spec in specs
            if retrieval_model_id(spec) in wanted or (spec.label or "").strip() in wanted
        ]
        if not filtered:
            if require_match:
                raise ValueError(
                    "No retrieval LLMs matched " + ", ".join(sorted(wanted))
                )
            logger.warning(
                "No retrieval LLMs matched model_ids=%s; using default configured set",
                sorted(wanted),
            )
            filtered = get_retrieval_specs(include_optional=False)
        specs = filtered
    clients: List[LLMClient] = []
    for spec in specs:
        model_id = retrieval_model_id(spec)
        use_web_search = model_id in web_search_wanted
        spec = replace(
            spec,
            web_search=use_web_search,
            timeout=apply_retrieval_timeout(
                spec, timeout, web_search=use_web_search
            ),
        )
        if max_retries is not None:
            spec = replace(spec, max_retries=max(0, int(max_retries)))
        clients.append(LLMClient(spec))
    return clients
