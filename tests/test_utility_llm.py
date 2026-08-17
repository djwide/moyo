from moyo.llm.client import LLMSpec, ensure_env_loaded, llm_spec_has_auth
from moyo.llm.utility import (
    cloud_paid_llm_config,
    kimi_hosted_config,
    running_in_cloud,
    utility_cluster_config,
    utility_llm_spec,
    vertex_flash_hosted_config,
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
        "MOYO_VERTEX_UTILITY_MODEL",
        "GOOGLE_CLOUD_PROJECT",
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


def test_cloud_utility_uses_vertex_flash(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    spec = utility_llm_spec()
    assert spec.provider == "custom"
    assert spec.model == "google/gemini-2.5-flash"
    assert "aiplatform.googleapis.com" in (spec.base_url or "")
    assert spec.api_key is None
    assert llm_spec_has_auth(spec)


def test_cloud_stale_kimi_env_still_uses_flash(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("MOYO_UTILITY_PROVIDER", "custom")
    monkeypatch.setenv("MOYO_UTILITY_MODEL", "kimi-k2.6")
    monkeypatch.setenv("MOYO_UTILITY_BASE_URL", "https://api.moonshot.ai/v1")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-test")
    spec = utility_llm_spec()
    assert spec.model == "google/gemini-2.5-flash"
    assert "aiplatform.googleapis.com" in (spec.base_url or "")
    assert spec.api_key is None


def test_cloud_stale_openrouter_env_uses_flash(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("MOYO_UTILITY_PROVIDER", "custom")
    monkeypatch.setenv("MOYO_UTILITY_MODEL", "meta-llama/llama-3.1-8b-instruct")
    monkeypatch.setenv("MOYO_UTILITY_BASE_URL", "https://openrouter.ai/api/v1")
    spec = utility_llm_spec()
    assert spec.model == "google/gemini-2.5-flash"
    assert "aiplatform.googleapis.com" in (spec.base_url or "")


def test_cloud_dockerfile_flash_env(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("MOYO_UTILITY_PROVIDER", "custom")
    monkeypatch.setenv("MOYO_UTILITY_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    spec = utility_llm_spec()
    assert spec.model == "google/gemini-2.5-flash"
    assert "aiplatform.googleapis.com" in (spec.base_url or "")
    assert spec.api_key is None


def test_utility_cluster_config_uses_vertex_flash(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    cfg = utility_cluster_config({"batch_size": 10, "collapse": True})
    assert cfg["provider"] == "custom"
    assert cfg["model"] == "google/gemini-2.5-flash"
    assert "aiplatform.googleapis.com" in (cfg.get("base_url") or "")
    assert "api_key" not in cfg
    assert cfg["batch_size"] == 10


def test_vertex_flash_hosted_config_replaces_ollama_and_kimi(monkeypatch):
    _clear_runtime(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    cfg = vertex_flash_hosted_config(
        {
            "provider": "custom",
            "model": "kimi-k2.6",
            "base_url": "https://api.moonshot.ai/v1",
            "api_key": "$MOONSHOT_API_KEY",
            "num_ctx": 16000,
            "collapse": True,
            "prompt": "prompts/extract_claims.md",
        }
    )
    assert cfg["provider"] == "custom"
    assert cfg["model"] == "google/gemini-2.5-flash"
    assert "aiplatform.googleapis.com" in (cfg.get("base_url") or "")
    assert "api_key" not in cfg
    assert cfg["collapse"] is True
    assert cfg["prompt"] == "prompts/extract_claims.md"
    assert "num_ctx" not in cfg


def test_kimi_hosted_config_replaces_ollama_cluster(monkeypatch):
    _clear_runtime(monkeypatch)
    cfg = kimi_hosted_config(
        {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434",
            "num_ctx": 16000,
            "collapse": True,
        }
    )
    assert cfg["provider"] == "custom"
    assert cfg["model"] == "kimi-k2.6"
    assert cfg["base_url"] == "https://api.moonshot.ai/v1"
    assert cfg["api_key"] == "$MOONSHOT_API_KEY"
    assert cfg["collapse"] is True
    assert "num_ctx" not in cfg


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
    assert out.model == "google/gemini-2.5-pro"
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


def test_vertex_gemini_model_honours_override(monkeypatch):
    from moyo.llm.vertex import vertex_gemini_model

    monkeypatch.setenv("MOYO_VERTEX_GEMINI_MODEL", "gemini-1.5-pro")
    assert vertex_gemini_model("gemini-3.1-pro-preview") == "google/gemini-1.5-pro"
    monkeypatch.setenv("MOYO_VERTEX_GEMINI_MODEL", "google/gemini-1.5-pro")
    assert vertex_gemini_model("ignored") == "google/gemini-1.5-pro"


def test_vertex_utility_model_honours_override(monkeypatch):
    from moyo.llm.vertex import vertex_utility_model

    monkeypatch.delenv("MOYO_VERTEX_UTILITY_MODEL", raising=False)
    assert vertex_utility_model() == "google/gemini-2.5-flash"
    monkeypatch.setenv("MOYO_VERTEX_UTILITY_MODEL", "gemini-2.0-flash")
    assert vertex_utility_model() == "google/gemini-2.0-flash"


def test_is_vertex_openai_url():
    from moyo.llm.vertex import is_vertex_openai_url, vertex_openai_base_url

    assert is_vertex_openai_url(vertex_openai_base_url("senteguard-website", "us-central1"))
    assert not is_vertex_openai_url("https://generativelanguage.googleapis.com/v1beta/openai/")


def test_llm_spec_has_auth_vertex_without_key():
    spec = LLMSpec(
        provider="custom",
        model="google/gemini-2.5-flash",
        base_url="https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi",
        api_key=None,
    )
    assert llm_spec_has_auth(spec)
    assert not llm_spec_has_auth(LLMSpec(provider="custom", model="kimi-k2.6"))
