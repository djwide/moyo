"""Tests for LLM client rate-limit retry helpers."""

from moyo.llm.client import (
    LLMClient,
    LLMSpec,
    _fixed_temperature_for_model,
    _openai_extra_body_for_model,
    is_retryable_llm_error,
    retry_delay_seconds,
)


class _FakeStatusError(Exception):
    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code


def test_credit_exhaustion_is_not_retryable():
    exc = _FakeStatusError(
        "Error code: 429 - {'error': {'message': 'You have no credits remaining.', "
        "'code': 'credit_balance_exhausted'}}"
    )
    assert not is_retryable_llm_error(exc)


def test_invalid_api_key_is_not_retryable():
    exc = Exception("Error code: 401 - Incorrect API key provided.")
    assert not is_retryable_llm_error(exc)


def test_gemini_limit_zero_is_not_retryable():
    exc = _FakeStatusError(
        "You exceeded your current quota. Quota exceeded for metric: x, limit: 0, "
        "model: gemini-2.5-pro. Please retry in 15.3s."
    )
    assert not is_retryable_llm_error(exc)


def test_transient_rate_limit_is_retryable():
    exc = _FakeStatusError(
        "Error code: 429 - Rate limit reached for requests. Please retry in 2.5s."
    )
    assert is_retryable_llm_error(exc)
    assert 2.5 <= retry_delay_seconds(exc, 0) <= 3.0


def test_overloaded_and_503_are_retryable():
    assert is_retryable_llm_error(_FakeStatusError("overloaded_error", status_code=529))
    assert is_retryable_llm_error(_FakeStatusError("upstream", status_code=503))


def test_kimi_k26_disables_thinking_and_uses_non_thinking_temperature():
    assert _openai_extra_body_for_model("kimi-k2.6") == {"thinking": {"type": "disabled"}}
    assert _fixed_temperature_for_model("kimi-k2.6") == 0.6
    assert _fixed_temperature_for_model("moonshotai/kimi-k2.5") == 0.6
    assert _openai_extra_body_for_model("gpt-4o") == {}
    assert _fixed_temperature_for_model("kimi-k3") == 1.0


def test_complete_retries_then_succeeds(monkeypatch):
    client = LLMClient(LLMSpec(provider="echo", model="echo", max_retries=3))
    calls = {"n": 0}

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeStatusError("Rate limit reached. Please retry in 0.01s.")
        return "ok"

    monkeypatch.setattr(client, "_complete_once", flaky)
    monkeypatch.setattr(client, "_client", object())
    # Force non-echo path.
    client.spec.provider = "openai"

    assert client.complete("hi") == "ok"
    assert calls["n"] == 3


def test_complete_does_not_retry_hard_failures(monkeypatch):
    client = LLMClient(LLMSpec(provider="echo", model="echo", max_retries=3))
    calls = {"n": 0}

    def always_broke(*_args, **_kwargs):
        calls["n"] += 1
        raise _FakeStatusError(
            "Error code: 429 - You have no credits remaining. code credit_balance_exhausted"
        )

    monkeypatch.setattr(client, "_complete_once", always_broke)
    monkeypatch.setattr(client, "_client", object())
    client.spec.provider = "openai"

    try:
        client.complete("hi")
        assert False, "expected raise"
    except Exception as exc:
        assert "no credits remaining" in str(exc)
    assert calls["n"] == 1
