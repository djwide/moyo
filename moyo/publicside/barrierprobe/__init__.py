"""Probe information barriers between public and private corpora.

Heavy modules (numpy / FAISS / embeddings) are imported lazily so Cloud Run
explore and model-rerun can load ``llm_fuzzer`` without those extras.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .schema import BarrierProbeConfig, BarrierProbeResult

_LAZY_ATTRS = {
    "BarrierAnalyzer": (".barrier_analyzer", "BarrierAnalyzer"),
    "DistributionLayer": (".distribution", "DistributionLayer"),
    "build_distribution_layer": (".distribution", "build_distribution_layer"),
    "LLMFuzzer": (".llm_fuzzer", "LLMFuzzer"),
    "LLMFuzzerConfig": (".llm_fuzzer", "LLMFuzzerConfig"),
    "OllamaClient": (".llm_fuzzer", "OllamaClient"),
    "fuzz_public_toward_private_phrases": (
        ".llm_fuzzer",
        "fuzz_public_toward_private_phrases",
    ),
    "FuzzOrchestrator": (".fuzz_orchestrator", "FuzzOrchestrator"),
    "OrchestratorConfig": (".fuzz_orchestrator", "OrchestratorConfig"),
}

__all__ = [
    "BarrierAnalyzer",
    "BarrierProbeConfig",
    "BarrierProbeResult",
    "DistributionLayer",
    "build_distribution_layer",
    "LLMFuzzer",
    "LLMFuzzerConfig",
    "FuzzOrchestrator",
    "OrchestratorConfig",
    "fuzz_public_toward_private_phrases",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(__all__))
