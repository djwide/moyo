"""Configuration models for the red teaming module."""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class TargetLLMConfig(BaseSettings):
    """Configuration for the target LLM being tested."""

    provider: str = Field(
        default="openai",
        description="LLM provider: 'openai', 'anthropic', or 'rest'",
    )
    model: str = Field(default="gpt-4o", description="Model name or ID")
    api_key: Optional[str] = Field(default=None, description="API key for the target LLM")
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for REST endpoints or proxied deployments",
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retry attempts on transient errors")
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to inject when querying the target (useful if known)",
    )
    # For generic REST: field path in the JSON body to place the prompt
    rest_prompt_field: str = Field(default="prompt", description="Body field for the user prompt")
    # For generic REST: dot-separated path to extract the response text
    rest_response_path: str = Field(
        default="choices.0.message.content",
        description="Dot-path to response text in the REST JSON response",
    )

    model_config = ConfigDict(env_prefix="MOYO_TARGET_")


class WhiteBoxConfig(BaseSettings):
    """Configuration for white-box red teaming (tester knows the secrets)."""

    secrets_file: str = Field(
        default="data/secrets.json",
        description="Path to the secrets file (JSON list, JSONL, YAML, or plain text)",
    )
    attack_strategies: List[str] = Field(
        default=["direct", "indirect", "roleplay", "fewshot", "context", "authority"],
        description="Attack strategies to use",
    )
    max_probes_per_secret: int = Field(
        default=5,
        description="Maximum number of probe variants to generate per secret",
    )
    similarity_threshold: float = Field(
        default=0.75,
        description="Cosine similarity threshold above which a response is considered a 'reveal'",
    )

    model_config = ConfigDict(env_prefix="MOYO_REDTEAM_WB_")


class BlackBoxConfig(BaseSettings):
    """Configuration for black-box red teaming (tester does not know the secrets)."""

    max_rounds: int = Field(
        default=10,
        description="Maximum number of iterative probing rounds",
    )
    hypothesis_source: str = Field(
        default="llm",
        description="Source for generating hypotheses: 'llm', 'public_corpus', or 'manual'",
    )
    manual_seeds: List[str] = Field(
        default=[],
        description="Manual seed queries when hypothesis_source='manual'",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Organizational domain hint (e.g. 'pharmaceutical research')",
    )
    specificity_threshold: float = Field(
        default=0.6,
        description="Specificity score above which a response is flagged as anomalous",
    )

    model_config = ConfigDict(env_prefix="MOYO_REDTEAM_BB_")


class RedTeamConfig(BaseSettings):
    """Top-level configuration for a red team session."""

    mode: str = Field(
        default="whitebox",
        description="Operation mode: 'whitebox' or 'blackbox'",
    )
    target: TargetLLMConfig = Field(default_factory=TargetLLMConfig)
    # Helper LLM used to generate/rephrase attack probes (not the target)
    helper_provider: str = Field(default="openai", description="Helper LLM provider")
    helper_model: str = Field(default="gpt-4o-mini", description="Helper LLM model")
    helper_api_key: Optional[str] = Field(default=None, description="Helper LLM API key")
    whitebox: WhiteBoxConfig = Field(default_factory=WhiteBoxConfig)
    blackbox: BlackBoxConfig = Field(default_factory=BlackBoxConfig)
    output_dir: str = Field(default="output/redteam", description="Directory for output files")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model used for response evaluation embeddings",
    )

    model_config = ConfigDict(env_prefix="MOYO_REDTEAM_")
