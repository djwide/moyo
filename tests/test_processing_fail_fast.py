"""Fail-fast Vertex processing calls, parallel scan workers, and cloud extract tuning."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import moyo.llm.client as client_mod
from moyo.llm.client import LLMClient, LLMSpec
from moyo.llm.vertex import (
    VERTEX_PROCESSING_FALLBACK_MODEL,
    processing_policy,
    vertex_openai_base_url,
    vertex_utility_model,
)


def _chunk(text: str | None = None, finish: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(
        model="m",
        usage=None,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
    )


class _FakeCompletions:
    """Stalls (ReadTimeout) for models in ``stall``; streams ``reply`` otherwise."""

    def __init__(self, stall: set[str], reply: str = '["ok"]'):
        self.stall = stall
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] in self.stall:
            raise httpx.ReadTimeout("stalled")
        assert kwargs.get("stream") is True
        return iter([_chunk(self.reply[:2]), _chunk(self.reply[2:], "stop")])


def _vertex_client(monkeypatch, completions: _FakeCompletions) -> LLMClient:
    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(client_mod, "_processing_primary_down_until", {})
    spec = LLMSpec(
        provider="custom",
        model=vertex_utility_model(),
        base_url=vertex_openai_base_url(),
        timeout=180,
        max_retries=3,
    )
    client = LLMClient(spec)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "MOYO_PROCESSING_DEADLINE",
        "MOYO_PROCESSING_IDLE_TIMEOUT",
        "MOYO_PROCESSING_MAX_ATTEMPTS",
        "MOYO_VERTEX_FALLBACK_MODEL",
        "MOYO_VERTEX_UTILITY_MODEL",
        "MOYO_TEST_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_policy_only_for_vertex_utility_model():
    policy = processing_policy(vertex_openai_base_url(), vertex_utility_model(), timeout=180)
    assert policy is not None
    assert policy.deadline == 120
    assert policy.idle_timeout == 45
    assert policy.max_attempts == 2
    assert policy.fallback_model == VERTEX_PROCESSING_FALLBACK_MODEL
    assert processing_policy(vertex_openai_base_url(), "google/gemini-3.1-pro-preview") is None
    assert processing_policy("https://api.moonshot.ai/v1", vertex_utility_model()) is None


def test_policy_deadline_never_exceeds_spec_timeout(monkeypatch):
    policy = processing_policy(vertex_openai_base_url(), vertex_utility_model(), timeout=60)
    assert policy.deadline == 60
    monkeypatch.setenv("MOYO_VERTEX_FALLBACK_MODEL", "off")
    assert processing_policy(vertex_openai_base_url(), vertex_utility_model()).fallback_model is None


def test_streamed_reply_is_joined(monkeypatch):
    completions = _FakeCompletions(stall=set())
    client = _vertex_client(monkeypatch, completions)
    result = client.complete_result("prompt")
    assert result.text == '["ok"]'
    assert len(completions.calls) == 1
    assert completions.calls[0]["timeout"].read == 45


def test_stalled_primary_retries_once_on_fallback(monkeypatch):
    completions = _FakeCompletions(stall={vertex_utility_model()})
    client = _vertex_client(monkeypatch, completions)
    result = client.complete_result("prompt")
    assert result.text == '["ok"]'
    assert [call["model"] for call in completions.calls] == [
        vertex_utility_model(),
        VERTEX_PROCESSING_FALLBACK_MODEL,
    ]


def test_breaker_skips_stalled_primary_on_next_call(monkeypatch):
    completions = _FakeCompletions(stall={vertex_utility_model()})
    client = _vertex_client(monkeypatch, completions)
    client.complete_result("first")
    completions.calls.clear()
    client.complete_result("second")
    assert [call["model"] for call in completions.calls] == [VERTEX_PROCESSING_FALLBACK_MODEL]


def test_attempts_capped_when_everything_stalls(monkeypatch):
    completions = _FakeCompletions(
        stall={vertex_utility_model(), VERTEX_PROCESSING_FALLBACK_MODEL}
    )
    client = _vertex_client(monkeypatch, completions)
    with pytest.raises(httpx.ReadTimeout):
        client.complete_result("prompt")
    assert len(completions.calls) == 2


def test_cloud_extract_tuning_shrinks_chunks(monkeypatch):
    from cloud_worker import apply_cloud_extract_tuning

    monkeypatch.delenv("MOYO_CLOUD_CHUNK_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MOYO_CLOUD_EXTRACT_WORKERS", raising=False)
    cfg = {"chunk": {"target_tokens": 8000, "max_tokens": 15000}, "extract": {"workers": 4, "max_tokens": 6000}}
    apply_cloud_extract_tuning(cfg)
    assert cfg["chunk"] == {"target_tokens": 2500, "max_tokens": 2500}
    assert cfg["extract"]["workers"] == 8
    assert cfg["extract"]["max_tokens"] == 8000


def test_single_cluster_batch_skips_merge_pass(tmp_path):
    from reports.pipeline.cluster import DEFAULT_CLUSTER_PROMPT, group_claim_ids_with_ollama

    calls: list[str] = []

    class _Client:
        def complete(self, prompt: str) -> str:
            calls.append(prompt)
            return '{"groups": [["C1"], ["C2"], ["C3"]]}'

    claims = [{"claim_id": f"C{i}", "claim": f"fact {i}"} for i in (1, 2, 3)]
    groups = group_claim_ids_with_ollama(claims, client=_Client(), prompt_path=DEFAULT_CLUSTER_PROMPT)
    assert sorted(map(tuple, groups)) == [("C1",), ("C2",), ("C3",)]
    assert len(calls) == 1


def test_early_summaries_copy_by_claim_text(monkeypatch):
    import sys

    sys.path.insert(0, str(client_mod.Path(__file__).resolve().parents[1] / "reports"))
    import build_report

    def _fake_attach(claims, **_kwargs):
        for claim in claims:
            claim["summary"] = f"sum:{claim['claim']}"
        return len(claims)

    monkeypatch.setattr("pipeline.summarize.attach_claim_summaries", _fake_attach)
    early = build_report.start_early_summaries(
        [{"claim_id": "a", "claim": "Fact  one"}, {"claim_id": "b", "claim": "Fact two"}],
        dry_run=False,
    )
    clustered = [{"claim_id": "x", "claim": "fact one"}, {"claim_id": "y", "claim": "merged new text"}]
    assert build_report.apply_early_summaries(early, clustered) == 1
    assert clustered[0]["summary"] == "sum:Fact  one"
    assert "summary" not in clustered[1]


def test_pool_runs_orders_in_isolated_children(monkeypatch):
    import pool_worker

    monkeypatch.delenv("MOYO_POOL_MAX_MESSAGES", raising=False)
    assert pool_worker.pool_max_messages() == 4
    monkeypatch.setenv("MOYO_POOL_MAX_MESSAGES", "6")
    assert pool_worker.pool_max_messages() == 6
    env = pool_worker.child_order_env("ord_1", {"KEEP": "1"})
    assert env == {"KEEP": "1", "ORDER_ID": "ord_1", "MOYO_ORDER_ALREADY_CLAIMED": "1"}

    seen: dict = {}

    def _run(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs["env"]
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pool_worker.subprocess, "run", _run)
    assert pool_worker._run_claimed_order("ord_2") == 1
    assert seen["argv"][-1].endswith("cloud_worker.py")
    assert seen["env"]["ORDER_ID"] == "ord_2"

    monkeypatch.setattr(pool_worker.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=-9))
    with pytest.raises(RuntimeError):
        pool_worker._run_claimed_order("ord_3")
