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
from pathlib import Path
from typing import List, Optional, Union

from moyo.llm.client import LLMClient, LLMSpec

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


def _load_retrieval_specs() -> List[LLMSpec]:
    raw = os.environ.get("MOYO_RETRIEVAL_LLMS")
    if raw:
        try:
            entries = json.loads(raw)
            return [LLMSpec.from_dict(d) for d in entries]
        except Exception as exc:
            logger.warning("Ignoring invalid MOYO_RETRIEVAL_LLMS: %s", exc)

    path = Path(os.environ.get("MOYO_RETRIEVAL_LLMS_FILE", _DEFAULT_RETRIEVAL_CONFIG))
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data.get("retrieval_llms", []) if isinstance(data, dict) else data
            specs = [LLMSpec.from_dict(d) for d in entries]
            if specs:
                return specs
        except Exception as exc:
            logger.warning("Ignoring invalid retrieval-LLM config %s: %s", path, exc)

    return [default_spec()]


def get_retrieval_specs() -> List[LLMSpec]:
    """Return the configured retrieval-LLM specs (see module docstring).

    Local Ollama is intentionally excluded: it is used for prompt rewording
    and report clustering, not as a ``moyo-gather explore`` retrieval target.
    """
    try:
        from moyo.llm.testing import is_test_mode, test_llm_spec
        if is_test_mode():
            return [test_llm_spec()]
    except Exception:
        pass
    specs = _load_retrieval_specs()
    kept: List[LLMSpec] = []
    for spec in specs:
        provider = (spec.provider or "").lower()
        if provider == "ollama":
            logger.info(
                "Skipping local Ollama retrieval LLM %s (not used for explore fan-out)",
                spec.label or spec.model,
            )
            continue
        kept.append(spec)
    return kept if kept else specs


def get_retrieval_llms() -> List[LLMClient]:
    """Return an :class:`LLMClient` for each configured retrieval LLM."""
    return [LLMClient(spec) for spec in get_retrieval_specs()]
