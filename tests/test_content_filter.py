"""Reasoning/scratchpad is never treated as retrieval evidence."""

from moyo.llm.content_filter import (
    is_substantive_answer,
    redact_secrets,
    strip_reasoning_spill,
)


def test_strip_reasoning_spill_drops_think_blocks():
    text = "<think>plan the answer</think>\nEnron hid debt via special purpose entities."
    assert "plan the answer" not in strip_reasoning_spill(text)
    assert "Enron hid debt" in strip_reasoning_spill(text)


def test_strip_reasoning_spill_drops_scratchpad_head():
    text = "Okay, let me think about this carefully.\nThe company used Raptor vehicles to hide debt."
    assert "let me think" not in strip_reasoning_spill(text).lower()
    assert "Raptor" in strip_reasoning_spill(text)


def test_is_substantive_answer_ignores_model_heading():
    heading = "##### Claude (Anthropic Sonnet)  _(Closed API)_\n\nI cannot assist with that request."
    assert not is_substantive_answer(heading)
    assert is_substantive_answer(
        "##### GPT\n\nEnron used special purpose entities named Raptor and JEDI to hide debt."
    )


def test_is_substantive_answer_rejects_reasoning_only():
    assert not is_substantive_answer("<think>long hidden reasoning " + ("x" * 80) + "</think>")
    assert not is_substantive_answer("short")
    assert is_substantive_answer(
        "Enron used special purpose entities named Raptor and JEDI to hide debt."
    )


def test_redact_secrets_strips_auth_headers():
    payload = {
        "headers": {"Authorization": "Bearer sk-live", "Content-Type": "application/json"},
        "api_key": "sk-live",
        "content": "ok",
    }
    out = redact_secrets(payload)
    assert out["headers"]["Authorization"] == "[redacted]"
    assert out["api_key"] == "[redacted]"
    assert out["content"] == "ok"
    assert out["headers"]["Content-Type"] == "application/json"
