"""Fuzzer modify_text stays a single rewrite; search lives in the orchestrator."""

from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer, LLMFuzzerConfig


def test_modify_text_does_not_search(monkeypatch):
    fuzzer = LLMFuzzer(LLMFuzzerConfig(llm_provider="test"))
    seen = []

    def fake_query(prompt, system=None):
        seen.append(prompt)
        return "rewritten phrase"

    monkeypatch.setattr(fuzzer, "query_llm", fake_query)
    monkeypatch.setattr(
        "moyo.publicside.barrierprobe.llm_fuzzer.time.sleep", lambda *_a, **_k: None
    )
    raw, prompt, response = fuzzer.modify_text(
        "original", "summarize", target_concept="target"
    )
    assert len(seen) == 1
    assert "summarize" in prompt.lower()
    assert response == "rewritten phrase"
    assert raw == "rewritten phrase"
