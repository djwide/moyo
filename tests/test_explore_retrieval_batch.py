"""Provider health cache, cloud preflight skip, and bounded retrieval batch."""

from __future__ import annotations

import threading
import time

from moyo.llm.client import LLMClient, LLMSpec
from moyo.publicside.barrierprobe.llm_fuzzer import QuerySeed
from moyo.publicside.gatherpublicsources.explorer import (
    ProviderCircuitBreaker,
    RetrievalResult,
    _retrieve_jobs_async,
    _run_coro,
    clear_provider_health_cache,
    probe_llm,
    record_retrieval_health,
    scheduler_provider_key,
    skip_llm_preflight,
)


def test_skip_llm_preflight_in_cloud(monkeypatch):
    monkeypatch.delenv("MOYO_SKIP_LLM_PREFLIGHT", raising=False)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    assert skip_llm_preflight() is True
    monkeypatch.setenv("MOYO_SKIP_LLM_PREFLIGHT", "0")
    assert skip_llm_preflight() is False
    monkeypatch.setenv("MOYO_SKIP_LLM_PREFLIGHT", "1")
    monkeypatch.delenv("MOYO_CLOUD_RUNTIME", raising=False)
    assert skip_llm_preflight() is True


def test_probe_llm_cached_globally(monkeypatch):
    clear_provider_health_cache()
    llm = LLMClient(LLMSpec(provider="echo", model="echo", label="Echo"))
    calls = {"n": 0}

    def fake_complete(*_a, **_k):
        calls["n"] += 1
        return "ok"

    monkeypatch.setattr(llm, "complete", fake_complete)
    first = probe_llm(llm)
    second = probe_llm(llm)
    assert first.status == "ok"
    assert second.status == "ok"
    assert calls["n"] == 1


def test_record_retrieval_health_updates_cache():
    clear_provider_health_cache()
    llm = LLMClient(LLMSpec(provider="echo", model="echo", label="Echo"))
    record_retrieval_health(
        [llm],
        [
            RetrievalResult(
                seed="q",
                llm_label=llm.label,
                provider="echo",
                model="echo",
                kind="local",
                error="boom",
            )
        ],
    )
    cached = probe_llm(llm)
    assert cached.status == "fail"
    assert "boom" in cached.reason


def test_retrieve_batch_times_out(monkeypatch):
    llm = LLMClient(LLMSpec(provider="echo", model="echo", timeout=1, label="Echo"))
    released = threading.Event()

    def hang(*_a, **_k):
        released.wait(timeout=5)
        return RetrievalResult(
            seed="hello",
            llm_label=llm.label,
            provider="echo",
            model="echo",
            kind="local",
            text="late",
        )

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.explorer.retrieve", hang
    )
    qs = QuerySeed(text="hello", language=None, strategy="original")
    started = time.monotonic()
    try:
        results = _run_coro(
            _retrieve_jobs_async(
                [(0, qs, 0, llm)],
                max_tokens=None,
                max_workers=1,
                circuit=ProviderCircuitBreaker(),
                progress=None,
            )
        )
    finally:
        released.set()
    elapsed = time.monotonic() - started
    assert elapsed < 4
    assert results
    assert results[0].error
    assert "timed out" in results[0].error


def _echo_llm(label: str, provider: str = "echo", base_url: str | None = None) -> LLMClient:
    return LLMClient(
        LLMSpec(
            provider=provider,
            model=label,
            label=label,
            timeout=30,
            base_url=base_url,
            api_key="sk-test",
        )
    )


def _ok_result(llm: LLMClient, seed: str = "hello") -> RetrievalResult:
    return RetrievalResult(
        seed=seed,
        llm_label=llm.label,
        provider=llm.spec.provider,
        model=llm.spec.model,
        kind=llm.kind,
        text="ok",
    )


def _fail_result(llm: LLMClient, seed: str = "hello") -> RetrievalResult:
    return RetrievalResult(
        seed=seed,
        llm_label=llm.label,
        provider=llm.spec.provider,
        model=llm.spec.model,
        kind=llm.kind,
        error="timed out after 1s",
    )


def test_scheduler_provider_key_uses_host_for_custom():
    openai = _echo_llm("GPT", provider="openai")
    grok = _echo_llm("Grok", provider="custom", base_url="https://api.x.ai/v1")
    llama = _echo_llm(
        "Llama", provider="custom", base_url="https://openrouter.ai/api/v1"
    )
    deepseek = _echo_llm(
        "DeepSeek", provider="custom", base_url="https://openrouter.ai/api/v1"
    )
    assert scheduler_provider_key(openai) == "openai"
    assert scheduler_provider_key(grok) == "api.x.ai"
    assert scheduler_provider_key(llama) == scheduler_provider_key(deepseek) == "openrouter.ai"


def test_per_provider_concurrency_cap(monkeypatch):
    openai = _echo_llm("GPT", provider="openai")
    grok = _echo_llm("Grok", provider="custom", base_url="https://api.x.ai/v1")
    current = {"n": 0, "peak": 0, "by_key": {}}
    lock = threading.Lock()

    def fake_retrieve(seed, llm, *_args, **_kwargs):
        key = scheduler_provider_key(llm)
        with lock:
            current["n"] += 1
            current["peak"] = max(current["peak"], current["n"])
            by_key = current["by_key"].setdefault(key, {"n": 0, "peak": 0})
            by_key["n"] += 1
            by_key["peak"] = max(by_key["peak"], by_key["n"])
        time.sleep(0.05)
        with lock:
            current["n"] -= 1
            current["by_key"][key]["n"] -= 1
        return _ok_result(llm, seed)

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.explorer.retrieve", fake_retrieve
    )
    qs = QuerySeed(text="hello", language=None, strategy="original")
    jobs = [
        (0, qs, 0, openai),
        (1, qs, 1, grok),
        (2, qs, 0, openai),
        (3, qs, 1, grok),
        (4, qs, 0, openai),
        (5, qs, 1, grok),
    ]
    results = _run_coro(
        _retrieve_jobs_async(
            jobs,
            max_tokens=None,
            max_workers=12,
            per_provider_concurrency=1,
            circuit=ProviderCircuitBreaker(),
            progress=None,
        )
    )
    assert len(results) == 6
    assert all(row.ok for row in results)
    assert current["peak"] <= 2
    assert current["by_key"]["openai"]["peak"] == 1
    assert current["by_key"]["api.x.ai"]["peak"] == 1


def test_provider_circuit_breaker_skips_remaining(monkeypatch):
    llm = _echo_llm("GPT", provider="openai")
    calls = {"n": 0}

    def fake_retrieve(seed, llm, *_args, **_kwargs):
        calls["n"] += 1
        return _fail_result(llm, seed)

    monkeypatch.setattr(
        "moyo.publicside.gatherpublicsources.explorer.retrieve", fake_retrieve
    )
    qs = QuerySeed(text="hello", language=None, strategy="original")
    breaker = ProviderCircuitBreaker(threshold=2)
    results = _run_coro(
        _retrieve_jobs_async(
            [(i, qs, 0, llm) for i in range(4)],
            max_tokens=None,
            max_workers=12,
            per_provider_concurrency=1,
            circuit=breaker,
            progress=None,
        )
    )
    assert len(results) == 4
    assert calls["n"] == 2
    skipped = [row for row in results if "circuit open" in (row.error or "")]
    assert len(skipped) == 2
    assert breaker.skip_reason("openai")
