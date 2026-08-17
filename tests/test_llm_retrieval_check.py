from pathlib import Path

from moyo.publicside.gatherpublicsources.explorer import (
    ExploreResult,
    LLMStatus,
    RetrievalResult,
    render_llm_retrieval_check,
    write_llm_retrieval_check,
)


def test_write_llm_retrieval_check(tmp_path: Path):
    result = ExploreResult(
        prompt="What happened at Enron?",
        seeds=["What happened at Enron?"],
        results=[
            RetrievalResult(
                seed="What happened at Enron?",
                llm_label="ChatGPT (OpenAI gpt-4o)",
                provider="openai",
                model="gpt-4o",
                kind="closed",
                text="",
                error="401 invalid_api_key",
            ),
            RetrievalResult(
                seed="What happened at Enron?",
                llm_label="Kimi (Moonshot kimi-k2.6)",
                provider="custom",
                model="kimi-k2.6",
                kind="open",
                text="Enron used special purpose entities.",
            ),
        ],
        markdown="# Topic exploration: What happened at Enron?\n",
        llm_labels=["ChatGPT (OpenAI gpt-4o)", "Kimi (Moonshot kimi-k2.6)"],
        llm_statuses=[
            LLMStatus(name="ChatGPT (OpenAI gpt-4o)", status="fail", reason="401"),
            LLMStatus(name="Kimi (Moonshot kimi-k2.6)", status="ok"),
        ],
    )
    md_path = write_llm_retrieval_check(result, tmp_path)
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "# LLM Retrieval Check" in text
    assert "Preflight" in text
    assert "ChatGPT" in text
    payload = (tmp_path / "llm-retrieval-check.json").read_text(encoding="utf-8")
    assert "invalid_api_key" in payload or "401" in payload
    markdown, data = render_llm_retrieval_check(result)
    assert data["preflight_ok"] == 1
    assert data["preflight_total"] == 2
    assert data["retrieval_ok"] == 1
