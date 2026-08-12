"""Fake deterministic LLM clients for ``--test`` / offline runs.

Enable with :func:`enable_test_mode` (or env ``MOYO_TEST_MODE=1``). Every
generative path that checks :func:`is_test_mode` (or constructs clients via
this module) returns stable, hash-seeded text and never opens a network
socket.

Duck-typed interfaces covered:

- :class:`moyo.llm.client.LLMClient` (``complete``) via forced ``echo`` specs
- :class:`~moyo.publicside.barrierprobe.llm_fuzzer.OllamaClient` (``generate``)
- :class:`~moyo.publicside.barrierprobe.llm_fuzzer.LocalLLMClient`
  (``transform_text``)
- OpenAI / Anthropic-style helper clients used by red-team modules
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENV_FLAG = "MOYO_TEST_MODE"
_test_mode: bool = False


def is_test_mode() -> bool:
    """Return True when CLI ``--test`` or ``MOYO_TEST_MODE`` is active."""
    if _test_mode:
        return True
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def enable_test_mode(enabled: bool = True) -> None:
    """Turn on (or off) process-wide fake LLM mode.

    When enabled:
    - sets ``MOYO_TEST_MODE=1``
    - forces the default LLM + retrieval registry to the offline ``echo`` provider
    """
    global _test_mode
    _test_mode = bool(enabled)
    if enabled:
        os.environ[_ENV_FLAG] = "1"
        # Force retrieval fan-out to a single offline stub so explore never
        # hits config/retrieval_llms.json live endpoints.
        os.environ["MOYO_RETRIEVAL_LLMS"] = json.dumps(
            [
                {
                    "provider": "echo",
                    "model": "echo-test",
                    "label": "test-echo",
                }
            ]
        )
        try:
            from moyo.llm.client import LLMSpec
            from moyo.llm.registry import set_default_llm

            set_default_llm(
                LLMSpec(provider="echo", model="echo-test", label="test-echo")
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not force default LLM to echo under --test: %s", exc)
        logger.info("LLM test mode enabled (fake deterministic clients; no API calls)")
    else:
        os.environ.pop(_ENV_FLAG, None)
        os.environ.pop("MOYO_RETRIEVAL_LLMS", None)
        try:
            from moyo.llm.registry import set_default_llm

            set_default_llm(None)
        except Exception:
            pass


def click_test_option(fn=None):
    """Decorator / option factory: ``@click.option('--test', ...)``.

    Use on a Click group or command::

        @click.group()
        @click_test_option
        def cli(test: bool):
            if test:
                enable_test_mode()
    """
    import click

    option = click.option(
        "--test",
        "test_mode",
        is_flag=True,
        default=False,
        help=(
            "Use fake deterministic LLM clients (no network / API keys). "
            "Also settable via MOYO_TEST_MODE=1."
        ),
    )
    if fn is None:
        return option
    return option(fn)


# ---------------------------------------------------------------------------
# Deterministic text generation
# ---------------------------------------------------------------------------


def _stable_int(*parts: str, mod: int = 10_000) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def fake_complete(
    prompt: str,
    system: Optional[str] = None,
    *,
    model: str = "echo-test",
) -> str:
    """Return a deterministic offline completion for ``prompt``."""
    lowered = (prompt or "").lower()
    topic = _extract_quoted(prompt) or _first_sentence(prompt) or "the topic"
    seed = _stable_int(model, system or "", prompt)

    # Reword / N-variants prompts (explore seed generation).
    match = re.search(r"give me (\d+) different", lowered)
    if match:
        n = int(match.group(1))
        variants = [
            f"{topic} overview and key facts",
            f"detailed explanation of {topic}",
            f"history and background of {topic}",
            f"technical specifics of {topic}",
            f"common questions about {topic}",
            f"authoritative sources on {topic}",
            f"risks and controversies around {topic}",
            f"recent developments regarding {topic}",
        ]
        # Rotate start index by seed so different prompts differ stably.
        start = seed % len(variants)
        ordered = variants[start:] + variants[:start]
        return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(ordered[:n]))

    # Translation-ish prompts.
    if "translate" in lowered or "into english" in lowered:
        body = prompt.strip()[-400:]
        return f"[test-translate:{seed}] {body}"

    # Red-team / probe rephrase prompts (one variant per line).
    if "rephrased" in lowered or "variants" in lowered or "follow-up" in lowered:
        n = 3
        m = re.search(r"generate\s+(\d+)", lowered)
        if m:
            n = max(1, min(10, int(m.group(1))))
        return "\n".join(
            f"Test probe {seed + i}: what is known about {topic}?"
            for i in range(n)
        )

    # Hypothesis brainstorming.
    if "exploratory question" in lowered or "proprietary" in lowered:
        n = 5
        m = re.search(r"generate\s+(\d+)", lowered)
        if m:
            n = max(1, min(20, int(m.group(1))))
        return "\n".join(
            f"What internal details exist for {topic} (aspect {seed + i})?"
            for i in range(n)
        )

    # JSON claim extraction (reports) — unused when --test maps to dry-run,
    # but kept so callers that still hit complete() get parseable JSON.
    if "claim" in lowered and ("json" in lowered or "extract" in lowered):
        return json.dumps(
            [
                {
                    "claim_id": f"T{seed % 10000:04d}",
                    "claim": f"Test claim about {topic}",
                    "source_model": "echo-test",
                    "category": "test",
                    "sensitivity": 1 + (seed % 5),
                    "specificity": 1 + ((seed // 5) % 5),
                    "novelty": 1 + ((seed // 25) % 5),
                    "confidence": 1 + ((seed // 125) % 5),
                    "corroboration": 1,
                    "interestingness": 1 + ((seed // 625) % 5),
                    "status": "UNVERIFIED",
                    "raw_excerpt": topic[:200],
                }
            ]
        )

    return textwrap.dedent(
        f"""\
        [test:{model}:{seed}] Offline stub response.
        No network call was made. Query preview:
        {(prompt or '').strip()[:500]}"""
    )


def _extract_quoted(text: str) -> Optional[str]:
    match = re.search(r'"([^"]+)"', text or "")
    return match.group(1) if match else None


def _first_sentence(text: str) -> Optional[str]:
    if not text:
        return None
    line = text.strip().splitlines()[0].strip()
    return line[:120] if line else None


# ---------------------------------------------------------------------------
# Duck-typed fake clients
# ---------------------------------------------------------------------------


@dataclass
class FakeDeterministicLLM:
    """Stand-in for OllamaClient / LocalLLMClient / OpenAI-style helpers."""

    model_name: str = "echo-test"
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    # -- OllamaClient-compatible -------------------------------------------
    def is_available(self) -> bool:
        return True

    def list_models(self) -> List[str]:
        return [self.model_name]

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **_kwargs: Any,
    ) -> str:
        text = fake_complete(prompt, system, model=self.model_name)
        self.call_log.append({"prompt": prompt, "system": system, "response": text})
        return text

    # -- LocalLLMClient-compatible -----------------------------------------
    def transform_text(
        self,
        original_text: str,
        target_concept: str,
        similar_phrases: Optional[List[str]] = None,
    ) -> str:
        seed = _stable_int(original_text, target_concept)
        phrases = similar_phrases or []
        hint = phrases[0] if phrases else target_concept
        return f"{original_text} ⟶ {hint} [test:{seed}]"

    # -- LLMClient-compatible ----------------------------------------------
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: Optional[int] = None,
    ) -> str:
        return self.generate(prompt, system=system)

    # -- Minimal OpenAI SDK duck type (chat.completions.create) ------------
    @property
    def chat(self) -> "_FakeChatNamespace":
        return _FakeChatNamespace(self)

    # -- Minimal Anthropic SDK duck type (messages.create) -----------------
    @property
    def messages(self) -> "_FakeMessagesNamespace":
        return _FakeMessagesNamespace(self)


@dataclass
class _FakeChoiceMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeChoiceMessage


@dataclass
class _FakeUsage:
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeChatResponse:
    choices: List[_FakeChoice]
    usage: _FakeUsage


class _FakeChatNamespace:
    def __init__(self, parent: FakeDeterministicLLM):
        self._parent = parent
        self.completions = self

    def create(self, *, model: str, messages: List[Dict[str, str]], **_kwargs: Any) -> _FakeChatResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        text = self._parent.generate(user, system=system)
        return _FakeChatResponse(
            choices=[_FakeChoice(message=_FakeChoiceMessage(content=text))],
            usage=_FakeUsage(total_tokens=len(text.split())),
        )


@dataclass
class _FakeAnthropicBlock:
    text: str


@dataclass
class _FakeAnthropicResponse:
    content: List[_FakeAnthropicBlock]
    usage: _FakeUsage


class _FakeMessagesNamespace:
    def __init__(self, parent: FakeDeterministicLLM):
        self._parent = parent

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        **_kwargs: Any,
    ) -> _FakeAnthropicResponse:
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        text = self._parent.generate(user, system=system)
        return _FakeAnthropicResponse(
            content=[_FakeAnthropicBlock(text=text)],
            usage=_FakeUsage(
                input_tokens=len(user.split()),
                output_tokens=len(text.split()),
            ),
        )


def fake_llm_client(model_name: str = "echo-test") -> FakeDeterministicLLM:
    """Convenience constructor used by CLIs and library code under ``--test``."""
    return FakeDeterministicLLM(model_name=model_name)


def test_llm_spec():
    """Return an :class:`LLMSpec` for the offline echo provider."""
    from moyo.llm.client import LLMSpec

    return LLMSpec(provider="echo", model="echo-test", label="test-echo")
