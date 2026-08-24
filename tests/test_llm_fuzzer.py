"""White-box LLM fuzzer scores every strategy per round and prunes the weakest."""

from __future__ import annotations

import pytest

from moyo.publicside.barrierprobe.llm_fuzzer import (
    LLMFuzzer,
    LLMFuzzerConfig,
    _split_tournament_log,
)


class _DummyIndex:
    embedding_model = "all-MiniLM-L6-v2"


def _vec(text: str) -> list[float]:
    t = (text or "").lower()
    if t == "target concept":
        return [1.0, 0.0]
    if "para" in t:
        return [0.98, 0.10]
    if "summ" in t:
        return [0.80, 0.40]
    if "trans" in t:
        return [0.10, 0.99]
    if "worse" in t:
        return [0.05, 0.99]
    return [0.55, 0.55]


def _install_fuzzer_fakes(monkeypatch, fuzzer: LLMFuzzer) -> list[dict]:
    calls: list[dict] = []

    def fake_embed(texts, model_name=None, **kwargs):
        return [_vec(t) for t in texts]

    def fake_query(prompt, system=None):
        strategy = "unknown"
        for line in (prompt or "").splitlines():
            if "fuzz strategy for this step:" in line.lower():
                strategy = line.split(":", 1)[1].strip().lower()
                break
        parent = ""
        for line in (prompt or "").splitlines():
            if line.lower().startswith("original phrase:"):
                parent = line.split(":", 1)[1].strip()
                break
        calls.append({"strategy": strategy, "parent": parent, "prompt": prompt})
        return f"{strategy}_out"

    monkeypatch.setattr("moyo.publicside.barrierprobe.llm_fuzzer.embed", fake_embed)
    monkeypatch.setattr("moyo.publicside.barrierprobe.llm_fuzzer.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(fuzzer, "query_llm", fake_query)
    monkeypatch.setattr(fuzzer, "find_similar_phrases", lambda *a, **k: [])
    return calls


def test_prune_weakest_drops_lowest_and_later_on_ties():
    active = ["paraphrase", "translate", "summarize"]
    trials = [
        {"strategy": "paraphrase", "similarity": 0.9},
        {"strategy": "translate", "similarity": 0.4},
        {"strategy": "summarize", "similarity": 0.4},
    ]
    dropped = LLMFuzzer._prune_weakest_strategy(active, trials)
    assert dropped == "summarize"
    assert active == ["paraphrase", "translate"]


def test_fuzz_phrase_runs_strategies_independently_then_prunes(monkeypatch):
    fuzzer = LLMFuzzer(
        LLMFuzzerConfig(
            llm_provider="test",
            max_iterations=2,
            target_similarity=0.999,
        )
    )
    calls = _install_fuzzer_fakes(monkeypatch, fuzzer)
    original = "secret vault path"

    phrase, similarity, history, interactions = fuzzer.fuzz_phrase(
        original, "target concept", _DummyIndex()
    )
    trials, tournament = _split_tournament_log(interactions)

    round1 = [c for c in calls if c["parent"] == original]
    assert {c["strategy"] for c in round1} == {"paraphrase", "translate", "summarize"}
    assert len(round1) == 3

    assert tournament["pruned_strategies"][0] == "translate"
    assert "translate" not in tournament["surviving_strategies"]
    assert tournament["surviving_strategies"] == ["paraphrase"]

    round2_parents = {c["parent"] for c in calls[3:]}
    assert original not in round2_parents
    assert all("para" in p for p in round2_parents)

    assert "para" in phrase
    assert similarity > 0.9
    assert history[0] == original
    assert any("para" in step for step in history[1:])
    assert {t["strategy"] for t in trials if t["round"] == 1} == {
        "paraphrase",
        "translate",
        "summarize",
    }


def test_fuzz_phrase_does_not_move_when_all_strategies_are_worse(monkeypatch):
    fuzzer = LLMFuzzer(
        LLMFuzzerConfig(
            llm_provider="test",
            max_iterations=1,
            target_similarity=0.99,
            fuzz_strategies=["paraphrase", "translate", "summarize"],
        )
    )
    _install_fuzzer_fakes(monkeypatch, fuzzer)

    def worse_query(prompt, system=None):
        return "worse_out"

    monkeypatch.setattr(fuzzer, "query_llm", worse_query)
    original = "already close secret"

    phrase, similarity, history, interactions = fuzzer.fuzz_phrase(
        original, "target concept", _DummyIndex()
    )
    _, tournament = _split_tournament_log(interactions)
    assert phrase == original
    assert history == [original]
    assert tournament["pruned_strategies"]
    assert similarity == pytest.approx(_cosine(_vec(original), _vec("target concept")))


def _cosine(a, b):
    denom = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    return sum(x * y for x, y in zip(a, b)) / denom
