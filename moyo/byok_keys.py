"""Customer BYO LLM keys for the long-running worker pool.

Keys are stored AES-GCM sealed in Firestore ``moyomap_user_llm_secrets/{uid}``
(same wrap as the site ``MOYOMAP_KEY_WRAP`` / Node ``secretBox``). The Pub/Sub
message is only ``{orderId}``. This module decrypts for one ``run_order`` and
never logs plaintext.

If decrypt fails, the caller keeps the platform keys already on the process
and records a fallback on the order for the admin portal.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

logger = logging.getLogger("moyo.byok_keys")

SECRETS_COLLECTION = "moyomap_user_llm_secrets"
MIN_WRAP_CHARS = 16

PROVIDER_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

FALLBACK_MISSING_UID = "missing_uid"
FALLBACK_MISSING_PROVIDERS = "missing_providers"
FALLBACK_WRAP_NOT_CONFIGURED = "wrap_not_configured"
FALLBACK_MISSING_KEYS = "missing_keys"
FALLBACK_DECRYPT_FAILED = "decrypt_failed"
FALLBACK_SECRETS_UNAVAILABLE = "secrets_unavailable"


@dataclass(frozen=True)
class ByokResolution:
    attempted: bool
    used: bool
    fallback: str | None = None
    providers: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    detail: str | None = None


def wrap_key_bytes(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


def decrypt_secret(secret: str, ciphertext: str, iv: str) -> str:
    """Match Node ``secretBox.decryptSecret`` (AES-256-GCM, SHA-256 wrap)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    payload = base64.b64decode(ciphertext)
    iv_bytes = base64.b64decode(iv)
    if len(payload) < 17 or len(iv_bytes) != 12:
        raise ValueError("Invalid ciphertext.")
    plain = AESGCM(wrap_key_bytes(secret)).decrypt(iv_bytes, payload, None)
    return plain.decode("utf-8")


def encrypt_secret(secret: str, plain: str) -> tuple[str, str]:
    """Test helper with the same envelope as the site."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    iv = os.urandom(12)
    payload = AESGCM(wrap_key_bytes(secret)).encrypt(iv, plain.encode("utf-8"), None)
    return base64.b64encode(payload).decode("ascii"), base64.b64encode(iv).decode("ascii")


def wrap_secret_from_env() -> str:
    return (os.environ.get("MOYOMAP_KEY_WRAP") or "").strip()


def is_byok_order(data: Mapping[str, Any] | None) -> bool:
    return str((data or {}).get("billingMode") or "").strip().lower() == "byok"


def provider_ids_from_order(data: Mapping[str, Any] | None) -> list[str]:
    raw = (data or {}).get("byokProviders")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item or "").strip().lower()
        if name in PROVIDER_ENV and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def byok_firestore_fields(resolution: ByokResolution, *, now: str) -> dict[str, Any]:
    """Order fields the admin portal reads. Never includes key material."""
    if not resolution.attempted:
        return {}
    if resolution.used:
        return {
            "byokAttempted": True,
            "byokUsed": True,
            "byokFallback": None,
            "byokFallbackProviders": [],
            "byokFallbackAt": None,
            "byokFallbackDetail": None,
        }
    return {
        "byokAttempted": True,
        "byokUsed": False,
        "byokFallback": resolution.fallback,
        "byokFallbackProviders": list(resolution.providers),
        "byokFallbackAt": now,
        "byokFallbackDetail": (resolution.detail or "")[:500] or None,
    }


def apply_provider_env(env: Mapping[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for name, value in env.items():
        previous[name] = os.environ.get(name)
        os.environ[name] = value
    return previous


def restore_provider_env(previous: Mapping[str, str | None] | None) -> None:
    if not previous:
        return
    for name, old in previous.items():
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _fallback(
    code: str,
    providers: list[str],
    detail: str,
) -> ByokResolution:
    return ByokResolution(
        attempted=True,
        used=False,
        fallback=code,
        providers=providers,
        env={},
        detail=detail,
    )


def resolve_customer_llm_keys(
    data: Mapping[str, Any] | None,
    *,
    wrap_secret: str | None = None,
    stored_providers: Mapping[str, Any] | None = None,
    load_secrets: Callable[[str], Mapping[str, Any]] | None = None,
) -> ByokResolution:
    """Return customer provider env, or a fallback that keeps platform keys."""
    if not is_byok_order(data):
        return ByokResolution(attempted=False, used=False)
    uid = str((data or {}).get("byokUid") or "").strip()
    providers = provider_ids_from_order(data)
    if not uid:
        return _fallback(FALLBACK_MISSING_UID, providers, "Order is missing byokUid.")
    if not providers:
        return _fallback(
            FALLBACK_MISSING_PROVIDERS,
            [],
            "Order is missing byokProviders.",
        )
    secret = (wrap_secret if wrap_secret is not None else wrap_secret_from_env()).strip()
    if len(secret) < MIN_WRAP_CHARS:
        return _fallback(
            FALLBACK_WRAP_NOT_CONFIGURED,
            providers,
            "MOYOMAP_KEY_WRAP is not set on the worker.",
        )
    try:
        if stored_providers is not None:
            stored = stored_providers
        elif load_secrets is not None:
            stored = load_secrets(uid)
        else:
            return _fallback(
                FALLBACK_SECRETS_UNAVAILABLE,
                providers,
                "No secrets loader.",
            )
    except Exception as exc:
        logger.exception("could not load customer LLM secrets")
        return _fallback(
            FALLBACK_SECRETS_UNAVAILABLE,
            providers,
            f"Could not read sealed keys ({type(exc).__name__}).",
        )
    if not isinstance(stored, Mapping):
        stored = {}
    env: dict[str, str] = {}
    missing: list[str] = []
    for provider in providers:
        env_name = PROVIDER_ENV[provider]
        row = stored.get(provider) if isinstance(stored.get(provider), Mapping) else {}
        ciphertext = str(row.get("ciphertext") or "") if isinstance(row, Mapping) else ""
        iv = str(row.get("iv") or "") if isinstance(row, Mapping) else ""
        if not ciphertext or not iv:
            missing.append(provider)
            continue
        try:
            env[env_name] = decrypt_secret(secret, ciphertext, iv)
        except Exception:
            logger.exception("could not decrypt customer %s key", provider)
            return _fallback(
                FALLBACK_DECRYPT_FAILED,
                providers,
                f"Could not decrypt the {provider} key.",
            )
    if missing:
        return _fallback(
            FALLBACK_MISSING_KEYS,
            missing,
            "Missing sealed keys for: " + ", ".join(missing),
        )
    return ByokResolution(
        attempted=True,
        used=True,
        providers=providers,
        env=env,
    )
