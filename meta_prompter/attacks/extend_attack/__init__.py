"""
ExtendAttack: Token Extension Attack via Poly-Base ASCII Obfuscation.

This module implements the ExtendAttack algorithm from the research paper
"ExtendAttack: Attacking Servers of LRMs via Extending Reasoning" (AAAI 2026).

ExtendAttack is a novel black-box attack method designed to maliciously occupy
server resources of Large Reasoning Models (LRMs) by extending their reasoning
processes through poly-base ASCII character obfuscation.

Key Results from Paper:
- 2.7x+ increase in response length for o3 model on HumanEval
- Maintains comparable accuracy (stealth preservation)
- Applicable to both direct prompting and indirect prompt injection
- Defeats current defenses: pattern matching, perplexity-based filtering, guardrails

Mathematical Formulations:
- Obfuscation ratio: ρ ∈ [0, 1]
- Character selection: k = ⌈|S_valid| · ρ⌉
- Base set: B = {2,...,9, 11,...,36} (excludes 10)
- Transformation: T(c_j) = <(n_j)val_n_j>
- Attack objectives: L(Y') >> L(Y), Latency(Y') >> Latency(Y), Acc(A') ≈ Acc(A)

Example Usage:
    >>> from meta_prompter.attacks.extend_attack import ExtendAttack
    >>> attacker = ExtendAttack(obfuscation_ratio=0.5)
    >>> result = attacker.attack("def strlen(s): return len(s)")
    >>> print(result.adversarial_query)

    # Using configuration presets
    >>> from meta_prompter.attacks.extend_attack import ExtendAttackConfig
    >>> config = ExtendAttackConfig.from_benchmark("humaneval", model="o3")
    >>> attacker = ExtendAttack.from_config(config)

    # Quick attack utility
    >>> from meta_prompter.attacks.extend_attack import quick_attack
    >>> adversarial = quick_attack("Hello world", ratio=0.3)

References:
    Zhu, Z., Liu, Y., Xu, Z., et al. (2025). ExtendAttack: Attacking Servers
    of LRMs via Extending Reasoning. arXiv:2506.13737v2
"""

# Core attack class
# Configuration
from .config import (
    AIME_2024_CONFIG,
    AIME_2025_CONFIG,
    AIME_WHITESPACE_CONFIG,
    BENCHMARK_CONFIGS,
    BIGCODEBENCH_CONFIG,
    DEFAULT_BASE_SET,
    GENERAL_CONFIG,
    HUMANEVAL_CONFIG,
    MAXIMUM_EXTENSION_CONFIG,
    STEALTH_CONFIG,
    BenchmarkConfig,
    ExtendAttackConfig,
    ModelFamily,
    get_benchmark_config,
    get_optimal_rho,
    list_available_benchmarks,
)
from .core import ExtendAttack, ExtendAttackBuilder, decode_attack, quick_attack

# Data models
from .models import (
    AttackMetrics,
    AttackResult,
    BatchAttackResult,
    IndirectInjectionResult,
    SelectionRules,
    SelectionStrategy,
    TransformInfo,
)

# N_note templates
from .n_notes import (
    AMBIGUOUS_N_NOTE,
    CODE_FOCUSED_N_NOTE,
    CONCISE_N_NOTE,
    DEFAULT_N_NOTE,
    DETAILED_N_NOTE,
    INSTRUCTIONAL_N_NOTE,
    MATHEMATICAL_N_NOTE,
    MINIMAL_N_NOTE,
    N_NOTE_TEMPLATES,
    NNoteBuilder,
    NNoteTemplate,
    NNoteVariant,
    get_n_note,
    get_n_note_for_benchmark,
    get_n_note_for_model,
    get_n_note_template,
)

# Transformers
from .transformers import (
    BaseConverter,
    CharacterTransformer,
    ObfuscationDecoder,
    estimate_token_increase,
)

__all__ = [
    # Core
    "ExtendAttack",
    "ExtendAttackBuilder",
    "quick_attack",
    "decode_attack",
    # Configuration
    "ExtendAttackConfig",
    "BenchmarkConfig",
    "ModelFamily",
    "DEFAULT_BASE_SET",
    "BENCHMARK_CONFIGS",
    "AIME_2024_CONFIG",
    "AIME_2025_CONFIG",
    "AIME_WHITESPACE_CONFIG",
    "HUMANEVAL_CONFIG",
    "BIGCODEBENCH_CONFIG",
    "GENERAL_CONFIG",
    "STEALTH_CONFIG",
    "MAXIMUM_EXTENSION_CONFIG",
    "get_benchmark_config",
    "get_optimal_rho",
    "list_available_benchmarks",
    # Models
    "SelectionStrategy",
    "SelectionRules",
    "TransformInfo",
    "AttackResult",
    "AttackMetrics",
    "BatchAttackResult",
    "IndirectInjectionResult",
    # N_notes
    "NNoteVariant",
    "NNoteTemplate",
    "NNoteBuilder",
    "N_NOTE_TEMPLATES",
    "DEFAULT_N_NOTE",
    "AMBIGUOUS_N_NOTE",
    "CONCISE_N_NOTE",
    "DETAILED_N_NOTE",
    "MATHEMATICAL_N_NOTE",
    "INSTRUCTIONAL_N_NOTE",
    "MINIMAL_N_NOTE",
    "CODE_FOCUSED_N_NOTE",
    "get_n_note",
    "get_n_note_template",
    "get_n_note_for_benchmark",
    "get_n_note_for_model",
    # Transformers
    "BaseConverter",
    "CharacterTransformer",
    "ObfuscationDecoder",
    "estimate_token_increase",
]

__version__ = "1.0.0"
__author__ = "Chimera Project"
__paper__ = "ExtendAttack: Attacking Servers of LRMs via Extending Reasoning (AAAI 2026)"
