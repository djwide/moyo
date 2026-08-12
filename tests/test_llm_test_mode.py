"""Tests for --test / MOYO_TEST_MODE fake deterministic LLM clients."""

import os

import pytest

from moyo.llm.testing import (
    FakeDeterministicLLM,
    enable_test_mode,
    fake_complete,
    is_test_mode,
)


@pytest.fixture(autouse=True)
def _reset_test_mode():
    """Ensure each test starts with test mode off and cleans up after."""
    enable_test_mode(False)
    os.environ.pop("MOYO_TEST_MODE", None)
    os.environ.pop("MOYO_RETRIEVAL_LLMS", None)
    yield
    enable_test_mode(False)
    os.environ.pop("MOYO_TEST_MODE", None)
    os.environ.pop("MOYO_RETRIEVAL_LLMS", None)


def test_fake_complete_is_deterministic():
    a = fake_complete('Give me 3 different phrasings of "Coca-Cola recipe"')
    b = fake_complete('Give me 3 different phrasings of "Coca-Cola recipe"')
    assert a == b
    assert "1." in a and "2." in a and "3." in a


def test_enable_test_mode_forces_echo_default_and_retrieval():
    enable_test_mode(True)
    assert is_test_mode()

    from moyo.llm.registry import default_spec, get_retrieval_specs

    assert default_spec().provider == "echo"
    specs = get_retrieval_specs()
    assert len(specs) == 1
    assert specs[0].provider == "echo"


def test_llm_client_complete_stays_offline_under_test_mode():
    enable_test_mode(True)
    from moyo.llm.client import LLMClient, LLMSpec

    # Even a "live" provider spec must not dial out under --test.
    client = LLMClient(LLMSpec(provider="openai", model="gpt-4o", api_key="sk-fake"))
    text = client.complete("hello world")
    assert "[echo:" in text or "[test:" in text or "Offline stub" in text or "hello world" in text


def test_fuzzer_uses_fake_under_test_mode():
    enable_test_mode(True)
    from moyo.publicside.barrierprobe.llm_fuzzer import LLMFuzzer, LLMFuzzerConfig

    fuzzer = LLMFuzzer(LLMFuzzerConfig(llm_provider="ollama", model_name="llama3.1:8b"))
    assert isinstance(fuzzer.llm_client, FakeDeterministicLLM)
    out = fuzzer.query_llm("Please respond with ok")
    assert out and ("Offline stub" in out or "[test:" in out)


def test_target_llm_uses_fake_under_test_mode():
    enable_test_mode(True)
    from moyo.redteam.config import TargetLLMConfig
    from moyo.redteam.target_llm import TargetLLMClient

    target = TargetLLMClient(
        TargetLLMConfig(provider="openai", model="gpt-4o", api_key="sk-fake")
    )
    assert isinstance(target._client, FakeDeterministicLLM)
    result = target.send_probe("What is the secret?")
    assert result.response
    assert "[ERROR]" not in result.response


def test_env_flag_alone_activates_test_mode():
    os.environ["MOYO_TEST_MODE"] = "1"
    assert is_test_mode()
    from moyo.llm.registry import get_retrieval_specs

    assert get_retrieval_specs()[0].provider == "echo"
