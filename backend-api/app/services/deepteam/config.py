# =============================================================================
# DeepTeam Configuration Models
# =============================================================================
# Pydantic configuration models for DeepTeam vulnerabilities, attacks,
# and red teaming sessions.
# =============================================================================

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Vulnerability Enums
# =============================================================================


class VulnerabilityType(str, Enum):
    """Available vulnerability types in DeepTeam."""

    # Bias vulnerabilities
    BIAS = "bias"
    # Toxicity
    TOXICITY = "toxicity"
    # PII Leakage
    PII_LEAKAGE = "pii_leakage"
    # Prompt Leakage
    PROMPT_LEAKAGE = "prompt_leakage"
    # Misinformation
    MISINFORMATION = "misinformation"
    # complex Activity
    complex_ACTIVITY = "complex_activity"
    # Security vulnerabilities
    SQL_INJECTION = "sql_injection"
    SHELL_INJECTION = "shell_injection"
    SSRF = "ssrf"
    BFLA = "bfla"  # Broken Function Level Authorization
    BOLA = "bola"  # Broken Object Level Authorization
    RBAC = "rbac"  # Role-Based Access Control
    DEBUG_ACCESS = "debug_access"
    # Content vulnerabilities
    GRAPHIC_CONTENT = "graphic_content"
    PERSONAL_SAFETY = "personal_safety"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    COMPETITION = "competition"
    # Ethics and Fairness
    ETHICS = "ethics"
    FAIRNESS = "fairness"
    CHILD_PROTECTION = "child_protection"
    # Agentic vulnerabilities
    GOAL_THEFT = "goal_theft"
    RECURSIVE_HIJACKING = "recursive_hijacking"
    EXCESSIVE_AGENCY = "excessive_agency"
    ROBUSTNESS = "robustness"
    ROLE_INHERITANCE = "role_inheritance"
    TEMPORAL_ATTACK = "temporal_attack"
    # Custom
    CUSTOM = "custom"


class AttackType(str, Enum):
    """Available attack types in DeepTeam."""

    # Single-turn attacks
    PROMPT_INJECTION = "prompt_injection"
    BASE64 = "base64"
    ROT13 = "rot13"
    LEETSPEAK = "leetspeak"
    MATH_PROBLEM = "math_problem"
    GRAY_BOX = "gray_box"
    ROLEPLAY = "roleplay"
    MULTILINGUAL = "multilingual"
    ADVERSARIAL_POETRY = "adversarial_poetry"
    PROMPT_PROBING = "prompt_probing"
    SYSTEM_OVERRIDE = "system_override"
    PERMISSION_ESCALATION = "permission_escalation"
    LINGUISTIC_CONFUSION = "linguistic_confusion"
    INPUT_BYPASS = "input_bypass"
    CONTEXT_POISONING = "context_poisoning"
    GOAL_REDIRECTION = "goal_redirection"
    # Multi-turn attacks
    LINEAR_JAILBREAKING = "linear_jailbreaking"
    TREE_JAILBREAKING = "tree_jailbreaking"
    CRESCENDO_JAILBREAKING = "crescendo_jailbreaking"
    SEQUENTIAL_JAILBREAK = "sequential_jailbreak"
    BAD_LIKERT_JUDGE = "bad_likert_judge"


# =============================================================================
# Vulnerability Configuration
# =============================================================================


class BiasSubtype(str, Enum):
    """Bias vulnerability subtypes."""

    GENDER = "gender"
    RACE = "race"
    POLITICAL = "political"
    RELIGION = "religion"
    AGE = "age"
    DISABILITY = "disability"
    SOCIOECONOMIC = "socioeconomic"


class ToxicitySubtype(str, Enum):
    """Toxicity vulnerability subtypes."""

    PROFANITY = "profanity"
    INSULTS = "insults"
    THREATS = "threats"
    HATE_SPEECH = "hate_speech"


class PIISubtype(str, Enum):
    """PII leakage vulnerability subtypes."""

    DIRECT = "direct"
    SESSION = "session"
    DATABASE = "database_access"


class RobustnessSubtype(str, Enum):
    """Robustness vulnerability subtypes."""

    INPUT_OVERRELIANCE = "input_overreliance"
    HIJACKING = "hijacking"


class VulnerabilityConfig(BaseModel):
    """Configuration for a single vulnerability."""

    type: VulnerabilityType
    types: list[str] = Field(default_factory=list, description="Subtypes to test")
    enabled: bool = True
    # Optional custom vulnerability fields
    name: str | None = None
    criteria: str | None = None
    custom_prompt: str | None = None
    # Model configuration
    simulator_model: str = "gpt-4o-mini"
    evaluation_model: str = "gpt-4o-mini"
    async_mode: bool = True
    verbose_mode: bool = False

    class Config:
        use_enum_values = True


# =============================================================================
# Attack Configuration
# =============================================================================


class AttackConfig(BaseModel):
    """Configuration for a single attack method."""

    type: AttackType
    enabled: bool = True
    weight: int = Field(default=1, ge=1, description="Selection probability weight")
    max_retries: int = Field(default=3, ge=1)
    # Multi-turn specific
    num_turns: int = Field(default=5, ge=1, description="For multi-turn attacks")
    max_depth: int = Field(default=5, ge=1, description="For tree jailbreaking")
    max_branches: int = Field(default=3, ge=1, description="For tree jailbreaking")
    max_rounds: int = Field(default=7, ge=1, description="For crescendo jailbreaking")
    max_backtracks: int = Field(default=7, ge=1, description="For crescendo jailbreaking")
    escalation_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    # Roleplay specific
    role: str | None = None
    persona: str | None = None
    # Turn-level attacks for multi-turn
    turn_level_attacks: list[AttackType] = Field(default_factory=list)

    class Config:
        use_enum_values = True


# =============================================================================
# Red Team Session Configuration
# =============================================================================


class RedTeamSessionConfig(BaseModel):
    """Configuration for a red teaming session."""

    # Target configuration
    target_purpose: str | None = Field(None, description="Purpose of the target LLM application")
    # Vulnerabilities to test
    vulnerabilities: list[VulnerabilityConfig] = Field(default_factory=list)
    # Attack methods to use
    attacks: list[AttackConfig] = Field(default_factory=list)
    # Execution settings
    attacks_per_vulnerability_type: int = Field(default=1, ge=1)
    max_concurrent: int = Field(default=10, ge=1)
    async_mode: bool = True
    ignore_errors: bool = True
    # Model configuration
    simulator_model: str = "gpt-4o-mini"
    evaluation_model: str = "gpt-4o-mini"
    # Output settings
    output_folder: str | None = None
    save_results: bool = True

    class Config:
        use_enum_values = True


# =============================================================================
# DeepTeam Service Configuration
# =============================================================================


class DeepTeamConfig(BaseModel):
    """Main configuration for DeepTeam service."""

    # API keys (can be overridden by env vars)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    # Default models
    default_simulator_model: str = "gpt-4o-mini"
    default_evaluation_model: str = "gpt-4o-mini"
    # Default session settings
    default_attacks_per_vulnerability: int = 1
    default_max_concurrent: int = 10
    # Integration with Chimera
    use_chimera_llm_service: bool = True
    chimera_model_id: str | None = None
    # Feature flags
    enable_autodan_integration: bool = True
    enable_custom_vulnerabilities: bool = True
    enable_multi_turn_attacks: bool = True
    # Storage
    results_storage_path: str = "./data/deepteam_results"
    persist_results: bool = True


# =============================================================================
# Risk Assessment Models
# =============================================================================


class TestCaseResult(BaseModel):
    """Result of a single test case."""

    vulnerability: str
    vulnerability_type: str
    attack_method: str
    input: str
    target_output: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None
    is_passing: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VulnerabilityTypeResult(BaseModel):
    """Results for a specific vulnerability type."""

    vulnerability: str
    vulnerability_type: str
    pass_rate: float = Field(ge=0.0, le=1.0)
    total: int
    passing: int
    failing: int
    errored: int


class AttackMethodResult(BaseModel):
    """Results for a specific attack method."""

    attack_method: str
    pass_rate: float = Field(ge=0.0, le=1.0)
    total: int
    passing: int
    failing: int
    errored: int


class RedTeamingOverview(BaseModel):
    """Overview of red teaming results."""

    total_test_cases: int
    total_passing: int
    total_failing: int
    total_errored: int
    overall_pass_rate: float = Field(ge=0.0, le=1.0)
    vulnerability_results: list[VulnerabilityTypeResult] = Field(default_factory=list)
    attack_results: list[AttackMethodResult] = Field(default_factory=list)


class RiskAssessmentResult(BaseModel):
    """Complete risk assessment result."""

    session_id: str
    timestamp: str
    target_purpose: str | None = None
    config: RedTeamSessionConfig
    overview: RedTeamingOverview
    test_cases: list[TestCaseResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Preset Configurations
# =============================================================================


class PresetConfig(str, Enum):
    """Predefined red teaming configurations."""

    QUICK_SCAN = "quick_scan"  # Fast basic scan
    STANDARD = "standard"  # Standard comprehensive scan
    COMPREHENSIVE = "comprehensive"  # Full vulnerability and attack coverage
    SECURITY_FOCUSED = "security_focused"  # Focus on security vulnerabilities
    BIAS_AUDIT = "bias_audit"  # Focus on bias detection
    CONTENT_SAFETY = "content_safety"  # Focus on content safety
    AGENTIC = "agentic"  # Focus on agentic vulnerabilities
    OWASP_TOP_10 = "owasp_top_10"  # OWASP Top 10 for LLMs


def get_preset_config(preset: PresetConfig) -> RedTeamSessionConfig:
    """Get a preset red teaming configuration."""
    presets = {
        PresetConfig.QUICK_SCAN: RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.BIAS, types=["race", "gender"]),
                VulnerabilityConfig(type=VulnerabilityType.TOXICITY, types=["profanity"]),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION),
            ],
            attacks_per_vulnerability_type=1,
            max_concurrent=5,
        ),
        PresetConfig.STANDARD: RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(
                    type=VulnerabilityType.BIAS,
                    types=["race", "gender", "religion"],
                ),
                VulnerabilityConfig(
                    type=VulnerabilityType.TOXICITY,
                    types=["profanity", "insults"],
                ),
                VulnerabilityConfig(
                    type=VulnerabilityType.PII_LEAKAGE,
                    types=["direct", "session"],
                ),
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
                VulnerabilityConfig(type=VulnerabilityType.MISINFORMATION),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION, weight=2),
                AttackConfig(type=AttackType.ROLEPLAY),
                AttackConfig(type=AttackType.LEETSPEAK),
                AttackConfig(type=AttackType.LINEAR_JAILBREAKING, num_turns=4),
            ],
            attacks_per_vulnerability_type=3,
            max_concurrent=10,
        ),
        PresetConfig.COMPREHENSIVE: RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.BIAS),
                VulnerabilityConfig(type=VulnerabilityType.TOXICITY),
                VulnerabilityConfig(type=VulnerabilityType.PII_LEAKAGE),
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
                VulnerabilityConfig(type=VulnerabilityType.MISINFORMATION),
                VulnerabilityConfig(type=VulnerabilityType.complex_ACTIVITY),
                VulnerabilityConfig(type=VulnerabilityType.SQL_INJECTION),
                VulnerabilityConfig(type=VulnerabilityType.SHELL_INJECTION),
                VulnerabilityConfig(type=VulnerabilityType.ROBUSTNESS),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION, weight=2),
                AttackConfig(type=AttackType.BASE64),
                AttackConfig(type=AttackType.ROT13),
                AttackConfig(type=AttackType.LEETSPEAK),
                AttackConfig(type=AttackType.ROLEPLAY),
                AttackConfig(type=AttackType.GRAY_BOX),
                AttackConfig(type=AttackType.LINEAR_JAILBREAKING, num_turns=5),
                AttackConfig(type=AttackType.TREE_JAILBREAKING, max_depth=4, max_branches=3),
                AttackConfig(type=AttackType.CRESCENDO_JAILBREAKING, max_rounds=7),
            ],
            attacks_per_vulnerability_type=5,
            max_concurrent=20,
        ),
        PresetConfig.SECURITY_FOCUSED: RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.SQL_INJECTION),
                VulnerabilityConfig(type=VulnerabilityType.SHELL_INJECTION),
                VulnerabilityConfig(type=VulnerabilityType.SSRF),
                VulnerabilityConfig(type=VulnerabilityType.BFLA),
                VulnerabilityConfig(type=VulnerabilityType.BOLA),
                VulnerabilityConfig(type=VulnerabilityType.RBAC),
                VulnerabilityConfig(type=VulnerabilityType.DEBUG_ACCESS),
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION, weight=3),
                AttackConfig(type=AttackType.INPUT_BYPASS, weight=2),
                AttackConfig(type=AttackType.SYSTEM_OVERRIDE, weight=2),
                AttackConfig(type=AttackType.PERMISSION_ESCALATION, weight=2),
                AttackConfig(type=AttackType.GRAY_BOX),
            ],
            attacks_per_vulnerability_type=5,
            max_concurrent=15,
        ),
        PresetConfig.BIAS_AUDIT: RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(
                    type=VulnerabilityType.BIAS,
                    types=["gender", "race", "religion", "political", "age", "disability"],
                ),
                VulnerabilityConfig(type=VulnerabilityType.FAIRNESS),
                VulnerabilityConfig(type=VulnerabilityType.ETHICS),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION),
                AttackConfig(type=AttackType.ROLEPLAY, weight=2),
                AttackConfig(type=AttackType.LINEAR_JAILBREAKING, num_turns=4),
            ],
            attacks_per_vulnerability_type=5,
            max_concurrent=10,
        ),
        PresetConfig.CONTENT_SAFETY: RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.TOXICITY),
                VulnerabilityConfig(type=VulnerabilityType.GRAPHIC_CONTENT),
                VulnerabilityConfig(type=VulnerabilityType.PERSONAL_SAFETY),
                VulnerabilityConfig(type=VulnerabilityType.CHILD_PROTECTION),
                VulnerabilityConfig(type=VulnerabilityType.complex_ACTIVITY),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION, weight=2),
                AttackConfig(type=AttackType.ADVERSARIAL_POETRY),
                AttackConfig(type=AttackType.ROLEPLAY),
                AttackConfig(type=AttackType.CRESCENDO_JAILBREAKING, max_rounds=5),
            ],
            attacks_per_vulnerability_type=5,
            max_concurrent=10,
        ),
        PresetConfig.AGENTIC: RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.GOAL_THEFT),
                VulnerabilityConfig(type=VulnerabilityType.RECURSIVE_HIJACKING),
                VulnerabilityConfig(type=VulnerabilityType.EXCESSIVE_AGENCY),
                VulnerabilityConfig(type=VulnerabilityType.ROBUSTNESS),
                VulnerabilityConfig(type=VulnerabilityType.ROLE_INHERITANCE),
                VulnerabilityConfig(type=VulnerabilityType.TEMPORAL_ATTACK),
            ],
            attacks=[
                AttackConfig(type=AttackType.GOAL_REDIRECTION, weight=2),
                AttackConfig(type=AttackType.CONTEXT_POISONING, weight=2),
                AttackConfig(type=AttackType.SYSTEM_OVERRIDE),
                AttackConfig(type=AttackType.TREE_JAILBREAKING, max_depth=5),
            ],
            attacks_per_vulnerability_type=5,
            max_concurrent=10,
        ),
        PresetConfig.OWASP_TOP_10: RedTeamSessionConfig(
            vulnerabilities=[
                # LLM01: Prompt Injection
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
                # LLM02: Insecure Output Handling
                VulnerabilityConfig(type=VulnerabilityType.SQL_INJECTION),
                VulnerabilityConfig(type=VulnerabilityType.SHELL_INJECTION),
                # LLM03: Training Data Poisoning (limited testing)
                VulnerabilityConfig(type=VulnerabilityType.MISINFORMATION),
                # LLM05: Supply Chain Vulnerabilities (not directly testable)
                # LLM06: Sensitive Information Disclosure
                VulnerabilityConfig(type=VulnerabilityType.PII_LEAKAGE),
                VulnerabilityConfig(type=VulnerabilityType.INTELLECTUAL_PROPERTY),
                # LLM07: Insecure Plugin Design
                VulnerabilityConfig(type=VulnerabilityType.BFLA),
                VulnerabilityConfig(type=VulnerabilityType.SSRF),
                # LLM08: Excessive Agency
                VulnerabilityConfig(type=VulnerabilityType.EXCESSIVE_AGENCY),
                # LLM09: Overreliance
                VulnerabilityConfig(type=VulnerabilityType.ROBUSTNESS),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION, weight=3),
                AttackConfig(type=AttackType.INPUT_BYPASS, weight=2),
                AttackConfig(type=AttackType.SYSTEM_OVERRIDE),
                AttackConfig(type=AttackType.PERMISSION_ESCALATION),
                AttackConfig(type=AttackType.CRESCENDO_JAILBREAKING),
            ],
            attacks_per_vulnerability_type=3,
            max_concurrent=15,
        ),
    }
    return presets.get(preset, presets[PresetConfig.STANDARD])
