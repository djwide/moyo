"""Target LLM client - wraps the LLM being tested (not the helper LLM).

This module is deliberately separate from the helper LLMs in llm_fuzzer.py.
The target is the system under audit; helper LLMs are used to craft probes.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import TargetLLMConfig

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of sending a single probe to the target LLM."""

    probe: str
    response: str
    latency_ms: float
    tokens_used: int
    timestamp: str
    strategy: str = ""
    secret_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe": self.probe,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "secret_id": self.secret_id,
            "metadata": self.metadata,
        }


class TargetLLMClient:
    """Client for querying the target LLM under test.

    Supports three backends:
    - openai: OpenAI chat completions API
    - anthropic: Anthropic messages API
    - rest: Generic POST endpoint (custom/internal deployments)

    All interactions are recorded for audit trail purposes.
    """

    def __init__(self, config: TargetLLMConfig):
        self.config = config
        self._client = self._init_client()
        self.interaction_log: List[ProbeResult] = []

    def _init_client(self) -> Any:
        provider = self.config.provider
        if provider == "openai":
            try:
                from openai import OpenAI
                kwargs: Dict[str, Any] = {"timeout": self.config.timeout}
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                return OpenAI(**kwargs)
            except ImportError:
                logger.error("openai package not installed: pip install openai")
                return None
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                kwargs = {"timeout": self.config.timeout}
                if self.config.api_key:
                    kwargs["api_key"] = self.config.api_key
                return Anthropic(**kwargs)
            except ImportError:
                logger.error("anthropic package not installed: pip install anthropic")
                return None
        elif provider == "rest":
            try:
                import httpx
                return httpx.Client(timeout=self.config.timeout)
            except ImportError:
                logger.error("httpx package not installed: pip install httpx")
                return None
        else:
            logger.error(f"Unknown target provider: {provider!r}. Use 'openai', 'anthropic', or 'rest'.")
            return None

    def send_probe(
        self,
        prompt: str,
        system: Optional[str] = None,
        strategy: str = "",
        secret_id: Optional[str] = None,
    ) -> ProbeResult:
        """Send a single probe to the target LLM and return the result."""
        effective_system = system or self.config.system_prompt
        t0 = time.monotonic()

        response_text = ""
        tokens_used = 0
        error_meta: Dict[str, Any] = {}

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response_text, tokens_used = self._call_provider(prompt, effective_system)
                break
            except Exception as exc:
                logger.warning(f"Target LLM call failed (attempt {attempt}/{self.config.max_retries}): {exc}")
                if attempt < self.config.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    response_text = f"[ERROR] {exc}"
                    error_meta = {"error": str(exc)}

        latency_ms = (time.monotonic() - t0) * 1000
        result = ProbeResult(
            probe=prompt,
            response=response_text,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
            secret_id=secret_id,
            metadata=error_meta,
        )
        self.interaction_log.append(result)
        logger.debug(f"Probe sent ({latency_ms:.0f}ms). Response length: {len(response_text)}")
        return result

    def _call_provider(self, prompt: str, system: Optional[str]) -> tuple[str, int]:
        """Dispatch to the correct provider backend. Returns (response_text, tokens_used)."""
        if self.config.provider == "openai":
            return self._call_openai(prompt, system)
        elif self.config.provider == "anthropic":
            return self._call_anthropic(prompt, system)
        elif self.config.provider == "rest":
            return self._call_rest(prompt, system)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _call_openai(self, prompt: str, system: Optional[str]) -> tuple[str, int]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return text, tokens

    def _call_anthropic(self, prompt: str, system: Optional[str]) -> tuple[str, int]:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""
        tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
        return text, tokens

    def _call_rest(self, prompt: str, system: Optional[str]) -> tuple[str, int]:
        body: Dict[str, Any] = {self.config.rest_prompt_field: prompt}
        if system:
            body["system"] = system
        resp = self._client.post(str(self.config.base_url), json=body)
        resp.raise_for_status()
        data = resp.json()
        # Navigate dot-path to extract response text
        text = data
        for key in self.config.rest_response_path.split("."):
            if isinstance(text, list):
                text = text[int(key)]
            else:
                text = text.get(key, "")
        return str(text), 0

    def send_probe_batch(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        strategy: str = "",
        secret_id: Optional[str] = None,
        delay_seconds: float = 0.5,
    ) -> List[ProbeResult]:
        """Send multiple probes sequentially with optional rate-limiting delay."""
        results = []
        for prompt in prompts:
            results.append(self.send_probe(prompt, system=system, strategy=strategy, secret_id=secret_id))
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        return results

    def save_interaction_log(self, path: str) -> None:
        """Persist the full interaction log to a JSON-lines file."""
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for result in self.interaction_log:
                fh.write(json.dumps(result.to_dict()) + "\n")
        logger.info(f"Saved {len(self.interaction_log)} interactions to {path}")
