"""Provider health cache, cloud preflight skip, and bounded retrieval batch."""

from __future__ import annotations

import time

from moyo.llm.client import LLMClient, LLMSpec
from moyo.publicside.barrierprobe.llm_fuzzer import QuerySeed
from moyo.publicside.gatherpublicsources.explorer import (
    RetrievalResult,
    _retrieve_jobs_async,
    _run_coro,
    clear_provider_health_cache,
    probe_llm,
    record_retrieval_health,
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
    import threading

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
