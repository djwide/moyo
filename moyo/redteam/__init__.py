"""Red teaming module for LLM proprietary information extraction testing.

This module provides two attack modes for testing whether a target LLM
will reveal an organization's classified or proprietary information:

- White box: The tester knows the secrets and crafts targeted extraction attacks.
- Black box: The tester does not know the secrets and explores blindly.
"""

from .config import RedTeamConfig, TargetLLMConfig, WhiteBoxConfig, BlackBoxConfig
from .target_llm import TargetLLMClient, ProbeResult

__all__ = [
    "RedTeamConfig",
    "TargetLLMConfig",
    "WhiteBoxConfig",
    "BlackBoxConfig",
    "TargetLLMClient",
    "ProbeResult",
]
