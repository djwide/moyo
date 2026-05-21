"""Centralized configuration for moyo project."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
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
    chunk_size: int = Field(default=512, description="Default text chunk size")
    overlap: int = Field(default=50, description="Default chunk overlap")
    timeout_seconds: int = Field(default=300, description="Default timeout for operations")
    
    model_config = ConfigDict(env_prefix="MOYO_PIPELINE_")


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Default embedding model"
    )
    device: str = Field(default="cpu", description="Device for embedding computation")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    normalize: bool = Field(default=True, description="Normalize embeddings")
    
    model_config = ConfigDict(env_prefix="MOYO_EMBEDDING_")


class FAISSConfig(BaseSettings):
    """FAISS index configuration."""
    
    index_type: str = Field(default="FlatL2", description="Default FAISS index type")
    dimension: int = Field(default=384, description="Embedding dimension")
    nlist: int = Field(default=100, description="Number of clusters for IVF indices")
    nprobe: int = Field(default=10, description="Number of probes for IVF search")
    
    model_config = ConfigDict(env_prefix="MOYO_FAISS_")


class LLMConfig(BaseSettings):
    """LLM configuration for hypothesis generation."""
    
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    max_tokens: int = Field(default=1000, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, description="Generation temperature")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
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
    index_dir: str = Field(default="indexes", description="Index directory")
    cache_dir: str = Field(default="cache", description="Cache directory")
    output_dir: str = Field(default="output", description="Output directory")

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
    
    # Custom YAML configuration
    custom_config: Dict[str, Any] = Field(default_factory=dict)
    
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
        env_file_encoding="utf-8"
    )
    
    # @classmethod
    # def settings_customise_sources(
    #     cls,
    #     settings_cls,
    #     init_settings,
    #     env_settings,
    #     dotenv_settings,
    #     file_secret_settings,
    # ):
    #     """Customize settings sources to include YAML."""
    #     return (
    #         init_settings,
    #         yaml_settings_source,
    #         env_settings,
    #         dotenv_settings,
    #         file_secret_settings,
    #     )


def yaml_settings_source(settings_cls) -> Dict[str, Any]:
    """Load settings from YAML configuration files."""
    config_files = [
        "config.yaml",
        "moyo.yaml", 
        "config/settings.yaml",
        "config/config.yaml"
    ]
    
    config_data = {}
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data:
                        config_data.update(yaml_data)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
    
    return config_data


def load_settings(config_file: Optional[str] = None) -> Settings:
    """Load settings with optional custom config file."""
    if config_file and Path(config_file).exists():
        # Temporarily set environment variable for custom config
        os.environ["MOYO_CONFIG_FILE"] = config_file
    
    return Settings()


# Global settings instance
settings = load_settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings(config_file: Optional[str] = None) -> Settings:
    """Reload settings from configuration."""
    global settings
    settings = load_settings(config_file)
    return settings
