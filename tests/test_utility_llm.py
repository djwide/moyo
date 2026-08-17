from moyo.llm.client import ensure_env_loaded
from moyo.llm.utility import running_in_cloud, utility_cluster_config, utility_llm_spec


def _clear_runtime(monkeypatch):
    ensure_env_loaded()
    try:
        from moyo.llm.testing import enable_test_mode

        enable_test_mode(False)
    except Exception:
        pass
    for key in (
        "MOYO_CLOUD_RUNTIME",
        "CLOUD_RUN_JOB",
        "CLOUD_RUN_EXECUTION",
        "K_SERVICE",
        "MOYO_TEST_MODE",
        "MOYO_UTILITY_PROVIDER",
        "MOYO_UTILITY_MODEL",
        "MOYO_UTILITY_BASE_URL",
        "MOYO_UTILITY_API_KEY",
        "OPENROUTER_API_KEY",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_running_in_cloud_flag(monkeypatch):
    _clear_runtime(monkeypatch)
    assert running_in_cloud() is False
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    assert running_in_cloud() is True


def test_local_utility_is_ollama(monkeypatch):
    _clear_runtime(monkeypatch)
    spec = utility_llm_spec()
    assert spec.provider == "ollama"
    assert "llama3.1" in spec.model


def test_cloud_utility_uses_openrouter(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    spec = utility_llm_spec()
    assert spec.provider == "custom"
    assert spec.model == "meta-llama/llama-3.1-8b-instruct"
    assert "openrouter" in (spec.base_url or "")
    assert spec.api_key == "sk-or-test"


def test_cloud_dockerfile_env_keeps_openrouter(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("MOYO_UTILITY_PROVIDER", "custom")
    monkeypatch.setenv("MOYO_UTILITY_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setenv("MOYO_UTILITY_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    spec = utility_llm_spec()
    assert spec.model == "meta-llama/llama-3.1-8b-instruct"
    assert spec.api_key == "sk-or-test"


def test_cloud_utility_falls_back_to_moonshot(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-test")
    spec = utility_llm_spec()
    assert spec.model == "kimi-k2.6"
    assert "moonshot" in (spec.base_url or "")
    assert spec.api_key == "sk-moon-test"


def test_cloud_falls_back_when_dockerfile_pins_openrouter_without_key(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("MOYO_UTILITY_PROVIDER", "custom")
    monkeypatch.setenv("MOYO_UTILITY_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setenv("MOYO_UTILITY_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-test")
    spec = utility_llm_spec()
    assert spec.model == "kimi-k2.6"
    assert "moonshot" in (spec.base_url or "")


def test_utility_cluster_config_uses_env_ref(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    cfg = utility_cluster_config({"batch_size": 10, "collapse": True})
    assert cfg["provider"] == "custom"
    assert cfg["api_key"] == "$OPENROUTER_API_KEY"
    assert cfg["batch_size"] == 10
    assert "sk-or-test" not in str(cfg)
