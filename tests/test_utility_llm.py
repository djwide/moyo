from moyo.llm.client import LLMSpec, ensure_env_loaded
from moyo.llm.utility import (
    cloud_paid_llm_config,
    running_in_cloud,
    utility_cluster_config,
    utility_llm_spec,
)


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
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MOYO_CLOUD_REPORT_PROVIDER",
        "MOYO_CLOUD_REPORT_MODEL",
        "MOYO_VERTEX_GEMINI",
        "MOYO_VERTEX_GEMINI_MODEL",
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


def test_cloud_paid_llm_defaults_to_openai(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    cfg = cloud_paid_llm_config({"workers": 4, "prompt": "prompts/extract_claims.md"})
    assert cfg["provider"] == "openai"
    assert cfg["model"] == "gpt-4o"
    assert cfg["api_key"] == "$OPENAI_API_KEY"
    assert cfg["workers"] == 4
    assert "base_url" not in cfg


def test_cloud_paid_llm_can_use_anthropic(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_REPORT_PROVIDER", "anthropic")
    cfg = cloud_paid_llm_config({})
    assert cfg["provider"] == "anthropic"
    assert cfg["api_key"] == "$ANTHROPIC_API_KEY"


def test_vertex_rewrites_gemini_in_cloud(monkeypatch):
    from moyo.llm.vertex import rewrite_gemini_spec_for_vertex

    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    spec = LLMSpec(
        provider="custom",
        model="gemini-3.1-pro-preview",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="AIza-old",
        label="Gemini (Google gemini-3.1-pro-preview)",
    )
    out = rewrite_gemini_spec_for_vertex(spec)
    assert "aiplatform.googleapis.com" in (out.base_url or "")
    assert "/endpoints/openapi" in (out.base_url or "")
    assert out.model == "google/gemini-3.1-pro-preview"
    assert out.api_key is None


def test_vertex_rewrite_skipped_on_desktop(monkeypatch):
    from moyo.llm.vertex import rewrite_gemini_spec_for_vertex

    _clear_runtime(monkeypatch)
    spec = LLMSpec(
        provider="custom",
        model="gemini-3.1-pro-preview",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="AIza-old",
    )
    out = rewrite_gemini_spec_for_vertex(spec)
    assert "generativelanguage.googleapis.com" in (out.base_url or "")


def test_is_vertex_openai_url():
    from moyo.llm.vertex import is_vertex_openai_url, vertex_openai_base_url

    assert is_vertex_openai_url(vertex_openai_base_url("senteguard-website", "us-central1"))
    assert not is_vertex_openai_url("https://generativelanguage.googleapis.com/v1beta/openai/")
