"""White-box orchestrator: call plan, live nodes, pruning."""

from __future__ import annotations

import pytest

from moyo.publicside.barrierprobe.fuzz_orchestrator import (
    FuzzOrchestrator,
    OrchestratorConfig,
    SearchNode,
    build_call_plan,
    prune_nodes,
)
from moyo.publicside.barrierprobe.llm_fuzzer import (
    DEFAULT_TRANSLATE_LANGUAGES,
    LLMFuzzer,
    LLMFuzzerConfig,
    WHITEBOX_FUZZ_STRATEGIES,
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
    if "abs" in t:
        return [0.70, 0.50]
    if "trans" in t or "spanish" in t:
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
        language = None
        for line in (prompt or "").splitlines():
            lower = line.lower()
            if "fuzz strategy for this step:" in lower:
                strategy = line.split(":", 1)[1].strip().lower()
            if lower.startswith("strategy instructions:") and "into " in lower:
                # "Translate the phrase into Spanish."
                after = line.split("into", 1)[1]
                language = after.split(".", 1)[0].strip()
        parent = ""
        for line in (prompt or "").splitlines():
            if line.lower().startswith("original phrase:"):
                parent = line.split(":", 1)[1].strip()
                break
        calls.append(
            {
                "strategy": strategy,
                "language": language,
                "parent": parent,
                "prompt": prompt,
            }
        )
        if strategy == "translate" and language:
            return f"{language}_trans_out"
        return f"{strategy}_out"

    monkeypatch.setattr("moyo.publicside.barrierprobe.llm_fuzzer.embed", fake_embed)
    monkeypatch.setattr(
        "moyo.publicside.barrierprobe.llm_fuzzer.time.sleep", lambda *_a, **_k: None
    )
    monkeypatch.setattr(fuzzer, "query_llm", fake_query)
    monkeypatch.setattr(fuzzer, "find_similar_phrases", lambda *a, **k: [])
    return calls


def test_default_call_plan_is_each_once_plus_four_languages():
    plan = build_call_plan(WHITEBOX_FUZZ_STRATEGIES)
    assert [c.label for c in plan] == [
        "paraphrase",
        "translate:Spanish",
        "translate:Chinese",
        "translate:French",
        "translate:Japanese",
    ]
    assert tuple(DEFAULT_TRANSLATE_LANGUAGES) == (
        "Spanish",
        "Chinese",
        "French",
        "Japanese",
    )


def test_call_plan_breadth_repeats_every_operator_and_language():
    plan = build_call_plan(
        ["paraphrase", "translate"],
        translate_languages=["Spanish", "French"],
        calls_per_strategy=3,
    )
    labels = [c.label for c in plan]
    assert labels.count("paraphrase") == 3
    assert labels.count("translate:Spanish") == 3
    assert labels.count("translate:French") == 3
    assert len(plan) == 9


def test_prune_nodes_keeps_top_k_unique():
    nodes = [
        SearchNode("alpha", 0.4, "a"),
        SearchNode("beta", 0.9, "b"),
        SearchNode("BETA", 0.8, "b2"),  # duplicate of beta
        SearchNode("gamma", 0.7, "c"),
        SearchNode("delta", 0.2, "d"),
    ]
    kept, pruned = prune_nodes(nodes, keep_k=2)
    assert [n.phrase for n in kept] == ["beta", "gamma"]
    assert all(n.alive for n in kept)
    assert not any(n.alive for n in pruned)
    assert any(n.phrase == "alpha" for n in pruned)


def test_modify_text_is_a_single_rewrite(monkeypatch):
    fuzzer = LLMFuzzer(LLMFuzzerConfig(llm_provider="test"))
    _install_fuzzer_fakes(monkeypatch, fuzzer)
    raw, prompt, response = fuzzer.modify_text(
        "secret vault path",
        "paraphrase",
        target_concept="target concept",
    )
    assert response == "paraphrase_out"
    assert "paraphrase" in prompt.lower()
    assert "Do not copy the target concept verbatim" in prompt
    assert raw


def test_modify_text_translate_names_the_language(monkeypatch):
    fuzzer = LLMFuzzer(LLMFuzzerConfig(llm_provider="test"))
    _install_fuzzer_fakes(monkeypatch, fuzzer)
    _raw, prompt, response = fuzzer.modify_text(
        "secret vault path",
        "translate",
        language="Japanese",
        target_concept="target concept",
    )
    assert "japanese" in prompt.lower()
    assert "Japanese_trans_out" in response


def test_orchestrator_expands_live_nodes_then_prunes(monkeypatch):
    fuzzer = LLMFuzzer(
        LLMFuzzerConfig(
            llm_provider="test",
            max_iterations=2,
            target_similarity=0.999,
            whitebox_strategies=["paraphrase", "summarize", "translate"],
            translate_languages=["Spanish"],
            calls_per_strategy=1,
            keep_k=2,
        )
    )
    calls = _install_fuzzer_fakes(monkeypatch, fuzzer)
    orch = FuzzOrchestrator(
        fuzzer, OrchestratorConfig.from_fuzzer_config(fuzzer.config)
    )
    result = orch.run("secret vault path", "target concept", _DummyIndex())

    round1 = [c for c in calls if c["parent"] == "secret vault path"]
    assert {c["strategy"] for c in round1} == {"paraphrase", "summarize", "translate"}
    assert any(c["strategy"] == "translate" and c["language"] == "Spanish" for c in round1)
    assert len(round1) == 3

    assert len(result.live_nodes) == 2
    assert all(n.alive for n in result.live_nodes)
    # paraphrase is closest to the target vector
    assert "para" in result.fuzzed_phrase
    assert result.final_similarity > 0.9
    assert result.history[0] == "secret vault path"

    # Round 2 parents are the surviving live nodes, not only the original seed.
    round2 = [c for c in calls if c not in round1]
    assert round2
    assert all(c["parent"] != "secret vault path" or "para" in c["parent"] for c in round2)


def test_fuzz_phrase_delegates_to_orchestrator(monkeypatch):
    fuzzer = LLMFuzzer(
        LLMFuzzerConfig(
            llm_provider="test",
            max_iterations=1,
            target_similarity=0.999,
            whitebox_strategies=["paraphrase", "summarize"],
            translate_languages=["Spanish"],
            keep_k=1,
            calls_per_strategy=1,
        )
    )
    _install_fuzzer_fakes(monkeypatch, fuzzer)
    phrase, similarity, history, interactions = fuzzer.fuzz_phrase(
        "secret vault path", "target concept", _DummyIndex()
    )
    _, meta = _split_tournament_log(interactions)
    assert meta.get("orchestrator") is True
    assert meta.get("keep_k") == 1
    assert meta.get("winning_prompt_chain")
    assert meta["winning_prompt_chain"][0]["role"] == "seed"
    assert "para" in phrase
    assert similarity > 0.9
    assert history[0] == "secret vault path"


def test_all_worse_answers_can_keep_the_seed(monkeypatch):
    fuzzer = LLMFuzzer(
        LLMFuzzerConfig(
            llm_provider="test",
            max_iterations=1,
            target_similarity=0.99,
            whitebox_strategies=["paraphrase"],
            keep_k=1,
        )
    )
    _install_fuzzer_fakes(monkeypatch, fuzzer)
    monkeypatch.setattr(fuzzer, "query_llm", lambda prompt, system=None: "worse_out")
    original = "already close secret"
    orch = FuzzOrchestrator(
        fuzzer, OrchestratorConfig.from_fuzzer_config(fuzzer.config)
    )
    result = orch.run(original, "target concept", _DummyIndex())
    assert result.fuzzed_phrase == original
    assert result.live_nodes[0].phrase == original
    assert result.final_similarity == pytest.approx(
        _cosine(_vec(original), _vec("target concept"))
    )


def test_orchestrator_records_prompt_chain_with_public_neighbors(monkeypatch):
    fuzzer = LLMFuzzer(
        LLMFuzzerConfig(
            llm_provider="test",
            max_iterations=1,
            target_similarity=0.999,
            whitebox_strategies=["paraphrase"],
            keep_k=1,
            calls_per_strategy=1,
        )
    )
    _install_fuzzer_fakes(monkeypatch, fuzzer)
    public_hit = {
        "text": "GitHub gist showing Vault path prod/secrets/db-root",
        "similarity": 0.87,
        "metadata": {
            "title": "misconfigured Vault path",
            "role": "planted_leak",
            "source": "mock_data.json",
        },
    }
    monkeypatch.setattr(
        fuzzer, "find_similar_phrases", lambda *a, **k: [public_hit]
    )
    orch = FuzzOrchestrator(
        fuzzer, OrchestratorConfig.from_fuzzer_config(fuzzer.config)
    )
    result = orch.run("secret vault path", "target concept", _DummyIndex())

    chain = result.winning_prompt_chain
    assert chain[0]["role"] == "seed"
    assert chain[0]["phrase"] == "secret vault path"
    assert chain[0]["prompt"] is None
    rewrite = next(hop for hop in chain if hop["role"] == "rewrite")
    assert rewrite["strategy"] == "paraphrase"
    assert rewrite["parent_phrase"] == "secret vault path"
    assert rewrite["prompt"]
    assert "secret vault path" in rewrite["prompt"]
    assert rewrite["public_neighbors"]
    assert rewrite["public_neighbors"][0]["role"] == "planted_leak"
    assert "Vault path" in rewrite["public_neighbors"][0]["text"]
    assert result.live_prompt_chains
    assert result.live_prompt_chains[0]["chain"][0]["role"] == "seed"
    _, meta = _split_tournament_log(result.interactions)
    assert meta["winning_prompt_chain"][0]["role"] == "seed"


def _cosine(a, b):
    denom = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
    return sum(x * y for x, y in zip(a, b)) / denom


def test_fuzz_public_toward_private_uses_each_private_as_target(monkeypatch):
    fuzzer = LLMFuzzer(
        LLMFuzzerConfig(
            llm_provider="test",
            max_iterations=1,
            target_similarity=0.999,
            whitebox_strategies=["paraphrase"],
            keep_k=1,
            public_seeds_per_phrase=1,
        )
    )
    _install_fuzzer_fakes(monkeypatch, fuzzer)
    public_hit = {
        "text": "kfc uses 11 herbs and spices",
        "similarity": 0.81,
        "above_threshold": True,
        "metadata": {},
    }
    seen_queries = []

    def fake_neighbors(query, index, k=None, enforce_threshold=True):
        seen_queries.append(query)
        return [public_hit]

    monkeypatch.setattr(fuzzer, "find_similar_phrases", fake_neighbors)
    private = "The blend is white pepper and celery salt."
    results = fuzzer.fuzz_public_toward_private(
        [private],
        _DummyIndex(),
        public_seeds_per_phrase=1,
    )
    assert seen_queries[0] == private
    assert len(results) == 1
    row = results[0]
    assert row["direction"] == "public_toward_private"
    assert row["private_phrase"] == private
    assert row["target_concept"] == private
    assert row["original_phrase"] == public_hit["text"]
    assert row["public_seed"] == public_hit["text"]
    assert row["public_seed_similarity"] == 0.81
    assert row["fuzzed_phrase"]

