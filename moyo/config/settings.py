"""Centralized configuration for moyo project.

Settings are loaded from (in increasing precedence): field defaults, a ``.env``
file in the working directory, and ``MOYO_*`` environment variables. See
``.env.example`` for the commonly-overridden values.
"""

from pathlib import Path
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s %(levelname)s %(name)s: %(message)s",
        description="Log format string"
    )
    structured: bool = Field(default=True, description="Use structured logging")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_size_mb: int = Field(default=100, description="Max log file size in MB")
    backup_count: int = Field(default=5, description="Number of backup files")
    
    model_config = ConfigDict(env_prefix="MOYO_LOG_")


class PrometheusConfig(BaseSettings):
    """Prometheus metrics configuration."""
    
    enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    port: int = Field(default=8000, description="Prometheus metrics port")
    path: str = Field(default="/metrics", description="Metrics endpoint path")
    namespace: str = Field(default="moyo", description="Metrics namespace")
    subsystem: str = Field(default="pipeline", description="Metrics subsystem")
    
    model_config = ConfigDict(env_prefix="MOYO_PROMETHEUS_")


class PipelineConfig(BaseSettings):
    """Pipeline configuration."""
    
    batch_size: int = Field(default=1000, description="Default batch size for processing")
    max_workers: int = Field(default=4, description="Maximum number of worker processes")
    chunk_size: int = Field(default=512, description="Default text chunk size in characters")
    overlap: int = Field(
        default=50,
        description="Chunk overlap in characters (~10% of chunk_size)",
    )
    min_chunk_length: int = Field(
        default=50,
        description="Drop section chunks shorter than this (boilerplate). "
        "Sentence/item chunks and atomic private secrets are kept.",
    )
    timeout_seconds: int = Field(default=300, description="Default timeout for operations")
    
    model_config = ConfigDict(env_prefix="MOYO_PIPELINE_")


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Default embedding model"
    )
    device: str = Field(
        default="auto",
        description="Device for embedding computation: auto | cuda | cpu",
    )
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    normalize: bool = Field(
        default=True,
        description="L2-normalize embeddings so FlatIP equals cosine similarity",
    )
    
    model_config = ConfigDict(env_prefix="MOYO_EMBEDDING_")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "cpu", "gpu"}
        normalized = (v or "auto").strip().lower()
        if normalized not in allowed:
            raise ValueError(f"device must be one of {sorted(allowed)}, got {v!r}")
        return "cuda" if normalized == "gpu" else normalized


class FAISSConfig(BaseSettings):
    """FAISS index configuration."""
    
    index_type: str = Field(default="FlatL2", description="Default FAISS index type")
    dimension: int = Field(default=768, description="Embedding dimension")
    nlist: int = Field(default=100, description="Number of clusters for IVF indices")
    nprobe: int = Field(default=10, description="Number of probes for IVF search")
    
    model_config = ConfigDict(env_prefix="MOYO_FAISS_")


class LLMConfig(BaseSettings):
    """LLM configuration for hypothesis generation."""
    
    provider: str = Field(default="openai", description="LLM provider: openai, anthropic, ollama, custom, echo")
    model: str = Field(default="gpt-4o", description="LLM model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="Base URL for ollama/custom (OpenAI-compatible) endpoints")
    max_tokens: int = Field(default=1000, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, description="Generation temperature")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    
    model_config = ConfigDict(env_prefix="MOYO_LLM_")


class RedTeamTargetConfig(BaseSettings):
    """Settings for the target LLM in red team sessions."""

    provider: str = Field(default="openai", description="Target LLM provider")
    model: str = Field(default="gpt-4o", description="Target LLM model")
    api_key: Optional[str] = Field(default=None, description="Target LLM API key")
    base_url: Optional[str] = Field(default=None, description="Base URL for REST targets")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    system_prompt: Optional[str] = Field(default=None, description="Known system prompt for target")

    model_config = ConfigDict(env_prefix="MOYO_TARGET_")


class RedTeamSettings(BaseSettings):
    """Settings for the red teaming module."""

    mode: str = Field(default="whitebox", description="Red team mode: whitebox or blackbox")
    target: RedTeamTargetConfig = Field(default_factory=RedTeamTargetConfig)
    helper_provider: str = Field(default="openai", description="Helper LLM provider")
    helper_model: str = Field(default="gpt-4o-mini", description="Helper LLM model")
    helper_api_key: Optional[str] = Field(default=None, description="Helper LLM API key")
    secrets_file: str = Field(default="data/secrets.json", description="Path to secrets file (whitebox mode)")
    attack_strategies: List[str] = Field(
        default=["direct", "indirect", "roleplay", "fewshot", "context", "authority"],
        description="Active attack strategies",
    )
    max_probes_per_secret: int = Field(default=5, description="Max probes per secret per strategy")
    similarity_threshold: float = Field(default=0.75, description="Cosine sim threshold for 'revealed'")
    blackbox_max_rounds: int = Field(default=10, description="Max rounds for black-box campaigns")
    hypothesis_source: str = Field(default="llm", description="Black-box hypothesis source")
    output_dir: str = Field(default="output/redteam", description="Red team output directory")

    model_config = ConfigDict(env_prefix="MOYO_REDTEAM_")


class Settings(BaseSettings):
    """Application settings with comprehensive configuration."""

    # Core settings
    data_dir: str = Field(default="data", description="Data directory")
    index_dir: str = Field(default="indexes", description="Legacy global index directory")
    cache_dir: str = Field(default="cache", description="Cache directory")
    output_dir: str = Field(default="output", description="Output directory")
    projects_dir: str = Field(
        default="projects",
        description="Directory of per-engagement project folders",
    )
    project: Optional[str] = Field(
        default=None,
        description="Current project slug (under projects_dir), or an absolute path",
    )

    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Component configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    red_team: RedTeamSettings = Field(default_factory=RedTeamSettings)

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v
    
    @field_validator('data_dir', 'index_dir', 'cache_dir', 'output_dir')
    @classmethod
    def ensure_directories_exist(cls, v):
        """Ensure directories exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    model_config = ConfigDict(
        env_prefix="MOYO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated env vars / .env keys
    )


def load_settings() -> Settings:
    """Load settings from defaults, ``.env``, and ``MOYO_*`` environment vars."""
    return Settings()


# Global settings instance
settings = load_settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from configuration."""
    global settings
    settings = load_settings()
    return settings
