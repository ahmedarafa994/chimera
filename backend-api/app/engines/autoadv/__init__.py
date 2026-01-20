"""AutoAdv Engine Module - DEPRECATED.

.. deprecated::
    This module has been moved to `app.services.autodan.engines.autoadv`.
    Please update your imports to use the new path.
    This shim will be removed in a future version.

Example migration:
    # Old (deprecated):
    from app.engines.autoadv.attacker_llm import AttackerLLM

    # New:
    from app.services.autodan.engines.autoadv.attacker_llm import AttackerLLM
"""

import warnings

warnings.warn(
    "app.engines.autoadv is deprecated. "
    "Use app.services.autodan.engines.autoadv instead. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export key modules from the new location for backward compatibility
from app.services.autodan.engines.autoadv import logging_utils
from app.services.autodan.engines.autoadv.attacker_llm import AttackerLLM
from app.services.autodan.engines.autoadv.config import DEFAULT_CONFIG, DEFAULT_PATHS, TARGET_MODELS
from app.services.autodan.engines.autoadv.conversation import (
    multi_turn_conversation,
    save_conversation_log,
)
from app.services.autodan.engines.autoadv.pattern_manager import PatternManager
from app.services.autodan.engines.autoadv.target_llm import TargetLLM

__all__ = [
    "AttackerLLM",
    "DEFAULT_CONFIG",
    "DEFAULT_PATHS",
    "PatternManager",
    "TARGET_MODELS",
    "TargetLLM",
    "logging_utils",
    "multi_turn_conversation",
    "save_conversation_log",
]
