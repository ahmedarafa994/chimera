"""AutoDAN Engines - Integrated Attack Engines.

This module consolidates attack engines from various sources:
- overthink: Reasoning token exploitation (9 attack techniques)
- autoadv: Multi-turn conversation framework with pattern learning

These engines were migrated from app/engines/ to provide a unified
API surface under the autodan service.
"""

# Overthink engine - reasoning token exploitation
# AutoAdv engine - multi-turn conversation framework
# Note: AutoAdv is primarily script-based but exposes key classes
from .autoadv.attacker_llm import AttackerLLM
from .autoadv.config import DEFAULT_CONFIG as AUTOADV_DEFAULT_CONFIG
from .autoadv.conversation import multi_turn_conversation
from .autoadv.pattern_manager import PatternManager
from .autoadv.target_llm import TargetLLM
from .overthink import (
    OverthinkConfig,
    OverthinkEngine,
    OverthinkRequest,
    OverthinkResult,
)

__all__ = [
    # AutoAdv
    "AUTOADV_DEFAULT_CONFIG",
    "AttackerLLM",
    # Overthink
    "OverthinkConfig",
    "OverthinkEngine",
    "OverthinkRequest",
    "OverthinkResult",
    "PatternManager",
    "TargetLLM",
    "multi_turn_conversation",
]
