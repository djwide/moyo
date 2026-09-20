"""Tests for LLM client rate-limit retry helpers."""

import os

import pytest

from moyo.llm.client import (
    LLMClient,
    LLMSpec,
    _anthropic_message_text,
    _fixed_temperature_for_model,
    _is_anthropic_no_temperature_model,
    _is_openai_max_completion_tokens_model,
    _omit_temperature_for_model,
    _openai_create_extras,
    _openai_extra_body_for_model,
    _openai_message_text,
    format_llm_error,
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
    assert _omit_temperature_for_model("kimi-k3") is True
    assert _fixed_temperature_for_model("kimi-k3") is None
    assert _openai_extra_body_for_model("kimi-k3") == {}


def test_gpt5_uses_max_completion_tokens_and_omits_temperature():
    assert _is_openai_max_completion_tokens_model("gpt-5.6-sol") is True
    assert _omit_temperature_for_model("gpt-5.6-sol") is True
    extras = _openai_create_extras("gpt-5.6-sol")
    assert extras.get("reasoning_effort") == "low"


def test_claude_opus_5_omits_temperature():
    assert _is_anthropic_no_temperature_model("claude-opus-5") is True
    assert _omit_temperature_for_model("claude-opus-5") is True


def test_openai_message_text_ignores_reasoning_content():
    class Msg:
        content = ""
        reasoning_content = "secret scratchpad"

    assert _openai_message_text(Msg()) == ""


def test_complete_result_keeps_raw_when_visible_text_empty(monkeypatch):
    client = LLMClient(LLMSpec(provider="openai", model="gpt-4o", api_key="sk-test"))

    class Msg:
        content = ""
        reasoning_content = "hidden scratchpad"

    class Choice:
        message = Msg()
        finish_reason = "stop"

    class Resp:
        choices = [Choice()]
        usage = None

        def model_dump(self, **kwargs):
            return {
                "id": "chatcmpl-keep",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "",
                            "reasoning_content": "hidden scratchpad",
                        },
                    }
                ],
            }

    class FakeCompletions:
        def create(self, **kwargs):
            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(client, "_client", FakeClient())
    out = client.complete_result("hi", retries=0)
    assert out.text == ""
    assert out.provider_record["raw"]["id"] == "chatcmpl-keep"
    assert client.complete("hi", retries=0) == ""


def test_complete_result_keeps_raw_when_parse_raises(monkeypatch):
    client = LLMClient(LLMSpec(provider="openai", model="gpt-4o", api_key="sk-test"))

    class Resp:
        choices = object()

        def model_dump(self, **kwargs):
            return {"id": "chatcmpl-odd", "output": [{"type": "unknown"}]}

    class FakeCompletions:
        def create(self, **kwargs):
            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(client, "_client", FakeClient())
    out = client.complete_result("hi", retries=0)
    assert out.text == ""
    assert out.provider_record["raw"]["id"] == "chatcmpl-odd"
    assert out.parse_error


def test_anthropic_message_text_skips_thinking_blocks():
    class Block:
        def __init__(self, typ, text=None):
            self.type = typ
            self.text = text

    class Resp:
        content = [Block("thinking", "secret"), Block("text", "OK")]

    assert _anthropic_message_text(Resp()) == "OK"


def test_complete_omits_temperature_for_opus_5(monkeypatch):
    client = LLMClient(
        LLMSpec(
            provider="anthropic",
            model="claude-opus-5",
            api_key="sk-test",
            web_search=True,
        )
    )
    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Block:
                type = "text"
                text = "OK"

            class Resp:
                content = [Block()]

            return Resp()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(client, "_client", FakeClient())
    assert client.complete("hi", max_tokens=16, retries=0) == "OK"
    assert "temperature" not in captured
    assert captured["max_tokens"] >= 1024
    assert captured["tools"][0]["type"] == "web_search_20250305"


def test_complete_uses_responses_web_search_for_openai(monkeypatch):
    client = LLMClient(
        LLMSpec(
            provider="openai",
            model="gpt-5.6-sol",
            api_key="sk-test",
            web_search=True,
        )
    )
    captured: dict = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Resp:
                output_text = "OK"
                output = []

            return Resp()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(client, "_client", FakeClient())
    assert client.complete("hi", max_tokens=16, retries=0) == "OK"
    assert captured["tools"] == [{"type": "web_search"}]
    assert captured["max_output_tokens"] >= 1024
    assert "temperature" not in captured
    assert captured.get("reasoning") == {"effort": "low"}


def test_web_search_extras_for_qwen_gemini_openrouter():
    qwen = _openai_create_extras(
        "qwen-plus",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        web_search=True,
    )
    assert qwen["extra_body"]["enable_search"] is True

    gemini = _openai_create_extras(
        "gemini-3.1-pro-preview",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        web_search=True,
    )
    assert gemini["web_search_options"] == {}
    assert gemini["reasoning_effort"] == "low"

    openrouter = _openai_create_extras(
        "meta-llama/llama-3.3-70b-instruct",
        "https://openrouter.ai/api/v1",
        web_search=True,
    )
    assert openrouter["extra_body"]["plugins"] == [{"id": "web"}]

    assert "enable_search" not in _openai_create_extras(
        "qwen-plus", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    ).get("extra_body", {})


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


def test_sanitize_secret_environ_strips_newlines(monkeypatch):
    from moyo.llm.client import LLMSpec, sanitize_secret_environ

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dash\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-test\r\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai\n")
    sanitize_secret_environ()
    assert os.environ["DASHSCOPE_API_KEY"] == "sk-dash"
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test"
    spec = LLMSpec.from_dict(
        {"provider": "custom", "model": "qwen-plus", "api_key": "$DASHSCOPE_API_KEY"}
    )
    assert spec.api_key == "sk-dash"
    spec2 = LLMSpec(provider="openai", model="gpt-4o")
    assert spec2.api_key == "sk-openai"


def test_complete_honours_zero_retries(monkeypatch):
    client = LLMClient(LLMSpec(provider="echo", model="echo", max_retries=0))
    calls = {"n": 0}

    def always_transient(*_args, **_kwargs):
        calls["n"] += 1
        raise _FakeStatusError("Rate limit reached. Please retry in 0.01s.")

    monkeypatch.setattr(client, "_complete_once", always_transient)
    monkeypatch.setattr(client, "_client", object())
    client.spec.provider = "openai"
    try:
        client.complete("hi")
        assert False, "expected raise"
    except Exception as exc:
        assert "Rate limit" in str(exc)
    assert calls["n"] == 1


def test_openai_sdk_disables_retries(monkeypatch):
    openai = pytest.importorskip("openai")
    seen = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    client = LLMClient(LLMSpec(provider="openai", model="gpt-4o", api_key="sk-test"))
    assert client._client is not None
    assert seen.get("max_retries") == 0


def test_format_llm_error_includes_connection_cause():
    class ConnectTimeout(Exception):
        pass

    exc = Exception("Connection error.")
    exc.__cause__ = ConnectTimeout("timed out")
    text = format_llm_error(exc)
    assert "Connection error." in text
    assert "ConnectTimeout" in text
    assert "timed out" in text


def _fake_chat_client(captured: dict):
    class Msg:
        content = "OK"
        tool_calls = None

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]
        citations = None

    class FakeCompletions:
        def create(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    return FakeClient()


def test_complete_kimi_k3_skips_builtin_web_search(monkeypatch):
    client = LLMClient(
        LLMSpec(
            provider="custom",
            model="kimi-k3",
            api_key="sk-test",
            base_url="https://api.moonshot.ai/v1",
        )
    )
    captured: dict = {}
    monkeypatch.setattr(client, "_client", _fake_chat_client(captured))
    assert client.complete("hi", max_tokens=16, retries=0) == "OK"
    assert "tools" not in captured
    assert "reasoning_effort" not in captured
    assert "extra_body" not in captured
    assert "temperature" not in captured


def test_complete_kimi_k26_keeps_web_search_tools(monkeypatch):
    client = LLMClient(
        LLMSpec(
            provider="custom",
            model="kimi-k2.6",
            api_key="sk-test",
            base_url="https://api.moonshot.ai/v1",
            web_search=True,
        )
    )
    captured: dict = {}
    monkeypatch.setattr(client, "_client", _fake_chat_client(captured))
    assert client.complete("hi", max_tokens=16, retries=0) == "OK"
    assert captured["tools"][0]["function"]["name"] == "$web_search"
    assert captured.get("extra_body") == {"thinking": {"type": "disabled"}}


def test_apply_retrieval_timeout_web_search_and_defaults():
    from moyo.llm.client import (
        PARAPHRASER_TIMEOUT,
        RETRIEVAL_TIMEOUT_DEFAULT,
        RETRIEVAL_TIMEOUT_REASONING,
        RETRIEVAL_TIMEOUT_WEB_SEARCH,
        SNAPSHOT_TIMEOUT,
        apply_retrieval_timeout,
    )

    gpt = LLMSpec(provider="openai", model="gpt-4o", api_key="x")
    kimi = LLMSpec(provider="custom", model="kimi-k3", api_key="x")
    qwen = LLMSpec(provider="custom", model="qwen3.8-max", api_key="x")
    sol = LLMSpec(provider="openai", model="gpt-5.6-sol", api_key="x")
    assert RETRIEVAL_TIMEOUT_DEFAULT == 120
    assert SNAPSHOT_TIMEOUT == 120
    assert PARAPHRASER_TIMEOUT == 120
    assert RETRIEVAL_TIMEOUT_REASONING == 240
    assert RETRIEVAL_TIMEOUT_WEB_SEARCH == 300
    assert apply_retrieval_timeout(gpt, None) == RETRIEVAL_TIMEOUT_DEFAULT
    assert apply_retrieval_timeout(gpt, SNAPSHOT_TIMEOUT) == SNAPSHOT_TIMEOUT
    assert apply_retrieval_timeout(kimi, None) == RETRIEVAL_TIMEOUT_REASONING
    assert apply_retrieval_timeout(qwen, SNAPSHOT_TIMEOUT) == RETRIEVAL_TIMEOUT_REASONING
    assert apply_retrieval_timeout(sol, None) == RETRIEVAL_TIMEOUT_REASONING
    assert apply_retrieval_timeout(gpt, SNAPSHOT_TIMEOUT, web_search=True) == (
        RETRIEVAL_TIMEOUT_WEB_SEARCH
    )


def test_get_retrieval_llms_applies_per_spec_snapshot_timeouts(monkeypatch):
    from moyo.llm import registry as reg
    from moyo.llm.client import RETRIEVAL_TIMEOUT_WEB_SEARCH, SNAPSHOT_TIMEOUT
    from moyo.llm.registry import retrieval_model_id

    specs = [
        LLMSpec(provider="openai", model="gpt-4o", api_key="x", timeout=120),
        LLMSpec(
            provider="custom",
            model="qwen3.8-max",
            api_key="x",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            timeout=120,
        ),
    ]
    monkeypatch.setattr(reg, "get_retrieval_specs", lambda include_optional=False: list(specs))
    qwen_id = retrieval_model_id(specs[1])
    clients = reg.get_retrieval_llms(
        timeout=SNAPSHOT_TIMEOUT,
        max_retries=0,
        web_search_model_ids={qwen_id},
    )
    by_model = {c.spec.model: c for c in clients}
    assert by_model["gpt-4o"].spec.timeout == SNAPSHOT_TIMEOUT
    assert not by_model["gpt-4o"].spec.web_search
    assert by_model["qwen3.8-max"].spec.timeout == RETRIEVAL_TIMEOUT_WEB_SEARCH
    assert by_model["qwen3.8-max"].spec.web_search
    assert all(c.spec.max_retries == 0 for c in clients)


def test_complete_openai_skips_web_search_by_default(monkeypatch):
    client = LLMClient(LLMSpec(provider="openai", model="gpt-5.6-sol", api_key="sk-test"))
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Msg:
                content = "OK"
                tool_calls = None

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]
                citations = None

            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()
        responses = None

    monkeypatch.setattr(client, "_client", FakeClient())
    assert client.complete("hi", max_tokens=16, retries=0) == "OK"
    assert "tools" not in captured
