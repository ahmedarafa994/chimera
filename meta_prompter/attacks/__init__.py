"""
Chimera Attack Modules

Provides unified access to all adversarial attack implementations.

Available Attack Types:
- ExtendAttack: Token extension via poly-base ASCII obfuscation
- AutoDAN: Genetic algorithm-based adversarial prompt mutation
- HotFlip: Gradient-based token replacement (optional)

The attacks can be used individually or combined through the unified framework
for multi-vector adversarial testing.
"""

# AutoDAN attack module - genetic mutation with reasoning integration
from meta_prompter.attacks.autodan import (
    AutoDANAttack,
    AutoDANConfig,
    AutoDANReasoningWorkflow,
    MutationWithReasoningResult,
    ReasoningEnhancedMutation,
)
from meta_prompter.attacks.extend_attack import (
    AttackResult,
    ExtendAttack,
    ExtendAttackBuilder,
    ExtendAttackConfig,
    SelectionRules,
    SelectionStrategy,
    decode_attack,
    quick_attack,
)

# Import HotFlip if available
try:
    from meta_prompter.attacks.hotflip import HotFlipAttack
except ImportError:
    HotFlipAttack = None

__all__ = [
    "AttackResult",
    # AutoDAN exports
    "AutoDANAttack",
    "AutoDANConfig",
    "AutoDANReasoningWorkflow",
    # ExtendAttack exports
    "ExtendAttack",
    "ExtendAttackBuilder",
    "ExtendAttackConfig",
    # Optional attacks
    "HotFlipAttack",
    "MutationWithReasoningResult",
    "ReasoningEnhancedMutation",
    "SelectionRules",
    "SelectionStrategy",
    "decode_attack",
    "quick_attack",
]
