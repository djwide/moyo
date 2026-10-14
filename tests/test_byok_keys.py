"""Customer BYO key decrypt and fallback (no Firestore)."""

from __future__ import annotations

import os

from moyo.byok_keys import (
    FALLBACK_MISSING_KEYS,
    FALLBACK_WRAP_NOT_CONFIGURED,
    apply_provider_env,
    byok_firestore_fields,
    decrypt_secret,
    encrypt_secret,
    resolve_customer_llm_keys,
    restore_provider_env,
)


WRAP = "test-wrap-key-not-for-prod"


def test_encrypt_decrypt_roundtrip():
    ciphertext, iv = encrypt_secret(WRAP, "sk-test-openai")
    assert decrypt_secret(WRAP, ciphertext, iv) == "sk-test-openai"


def test_credits_order_does_not_touch_keys():
    result = resolve_customer_llm_keys({"billingMode": "credits"}, wrap_secret=WRAP)
    assert result.attempted is False
    assert result.used is False
    assert result.env == {}
    assert byok_firestore_fields(result, now="t0") == {}


def test_byok_success_sets_provider_env():
    ciphertext, iv = encrypt_secret(WRAP, "sk-live")
    result = resolve_customer_llm_keys(
        {
            "billingMode": "byok",
            "byokUid": "uid_1",
            "byokProviders": ["openai"],
        },
        wrap_secret=WRAP,
        stored_providers={"openai": {"ciphertext": ciphertext, "iv": iv}},
    )
    assert result.used is True
    assert result.env["OPENAI_API_KEY"] == "sk-live"
    fields = byok_firestore_fields(result, now="t0")
    assert fields["byokAttempted"] is True
    assert fields["byokUsed"] is True
    assert fields["byokFallback"] is None


def test_missing_wrap_falls_back():
    result = resolve_customer_llm_keys(
        {
            "billingMode": "byok",
            "byokUid": "uid_1",
            "byokProviders": ["openai"],
        },
        wrap_secret="",
        stored_providers={},
    )
    assert result.used is False
    assert result.fallback == FALLBACK_WRAP_NOT_CONFIGURED
    fields = byok_firestore_fields(result, now="t0")
    assert fields["byokUsed"] is False
    assert fields["byokFallback"] == FALLBACK_WRAP_NOT_CONFIGURED
    assert fields["byokFallbackAt"] == "t0"


def test_missing_sealed_key_falls_back():
    ciphertext, iv = encrypt_secret(WRAP, "sk-live")
    result = resolve_customer_llm_keys(
        {
            "billingMode": "byok",
            "byokUid": "uid_1",
            "byokProviders": ["openai", "xai"],
        },
        wrap_secret=WRAP,
        stored_providers={"openai": {"ciphertext": ciphertext, "iv": iv}},
    )
    assert result.used is False
    assert result.fallback == FALLBACK_MISSING_KEYS
    assert "xai" in result.providers


def test_apply_and_restore_provider_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "platform")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    previous = apply_provider_env(
        {"OPENAI_API_KEY": "customer", "ANTHROPIC_API_KEY": "claude"}
    )
    assert os.environ["OPENAI_API_KEY"] == "customer"
    assert os.environ["ANTHROPIC_API_KEY"] == "claude"
    restore_provider_env(previous)
    assert os.environ["OPENAI_API_KEY"] == "platform"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_load_secrets_error_falls_back():
    def _boom(_uid: str):
        raise RuntimeError("no firestore")

    result = resolve_customer_llm_keys(
        {
            "billingMode": "byok",
            "byokUid": "uid_1",
            "byokProviders": ["openai"],
        },
        wrap_secret=WRAP,
        load_secrets=_boom,
    )
    assert result.used is False
    assert result.fallback == "secrets_unavailable"
