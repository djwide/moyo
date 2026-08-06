"""Shared, hot-swappable LLM layer for moyo.

This package provides a single, small abstraction over every generative LLM
provider moyo talks to:

- closed hosted APIs (OpenAI, Anthropic),
- "open" / self-hosted OpenAI-compatible endpoints (vLLM, LM Studio, Together,
  Groq, OpenRouter, DeepSeek, llama.cpp server, ...),
- local models via Ollama,
- an offline ``echo`` provider for tests / air-gapped runs.

The goal is that any part of moyo that needs to *generate* text can go through
:class:`~moyo.llm.client.LLMClient`, and the project-wide default model can be
hot-swapped from one place -- either via ``MOYO_LLM_*`` settings/``.env`` or at
runtime with :func:`~moyo.llm.registry.set_default_llm`.
"""

from moyo.llm.client import (
    LLMClient,
    LLMSpec,
    classify_provider,
)
from moyo.llm.registry import (
    default_spec,
    get_default_llm,
    get_retrieval_llms,
    set_default_llm,
)

__all__ = [
    "LLMClient",
    "LLMSpec",
    "classify_provider",
    "default_spec",
    "get_default_llm",
    "get_retrieval_llms",
    "set_default_llm",
]
