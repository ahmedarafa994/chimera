"""
ExtendAttack Pydantic Schemas.

Request and response models for the ExtendAttack API endpoints.
These schemas provide validation and documentation for all API operations.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SelectionStrategyEnum(str, Enum):
    """Character selection strategies for obfuscation."""

    ALPHABETIC_ONLY = "alphabetic_only"
    WHITESPACE_ONLY = "whitespace_only"
    ALPHANUMERIC = "alphanumeric"
    FUNCTION_NAMES = "function_names"
    IMPORT_STATEMENTS = "import_statements"
    DOCSTRING_REQUIREMENTS = "docstring_requirements"


class NNoteVariantEnum(str, Enum):
    """Available N_note template variants."""

    DEFAULT = "default"
    AMBIGUOUS = "ambiguous"
    CONCISE = "concise"
    DETAILED = "detailed"
    MATHEMATICAL = "mathematical"
    INSTRUCTIONAL = "instructional"
    MINIMAL = "minimal"
    CODE_FOCUSED = "code_focused"


# =============================================================================
# Request Models
# =============================================================================


class AttackRequest(BaseModel):
    """Request for single attack execution."""

    query: str = Field(..., description="Original query to obfuscate", min_length=1)
    obfuscation_ratio: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="ρ ∈ [0, 1] - ratio of eligible characters to transform",
    )
    selection_strategy: SelectionStrategyEnum = Field(
        SelectionStrategyEnum.ALPHABETIC_ONLY,
        description="Character selection strategy determining which characters are eligible",
    )
    n_note_type: str = Field(
        "default",
        description="N_note template type (default, ambiguous, concise, detailed, etc.)",
    )
    custom_n_note: str | None = Field(
        None, description="Custom N_note override - if provided, overrides n_note_type"
    )
    seed: int | None = Field(
        None, description="Random seed for reproducibility (None for random)"
    )


class BatchAttackRequest(BaseModel):
    """Request for batch attack execution on multiple queries."""

    queries: list[str] = Field(
        ..., description="List of queries to attack", min_length=1, max_length=1000
    )
    obfuscation_ratio: float = Field(
        0.5, ge=0.0, le=1.0, description="ρ ∈ [0, 1] - obfuscation ratio for all queries"
    )
    selection_strategy: SelectionStrategyEnum = Field(
        SelectionStrategyEnum.ALPHABETIC_ONLY, description="Character selection strategy"
    )
    benchmark: str | None = Field(
        None, description="Benchmark name for auto-config (humaneval, aime_2024, etc.)"
    )
    model: str | None = Field(
        None, description="Target model for optimal ρ selection (o3, o3-mini, etc.)"
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")


class IndirectInjectionRequest(BaseModel):
    """Request for indirect prompt injection attack."""

    document: str = Field(
        ..., description="Document to poison for indirect injection", min_length=1
    )
    injection_ratio: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Ratio of content to obfuscate (lower = more stealthy)",
    )
    target_sections: list[str] | None = Field(
        None, description="Specific sections to target for injection (optional)"
    )
    embed_n_note: bool = Field(
        True, description="Whether to embed N_note in the poisoned document"
    )


class EvaluationRequest(BaseModel):
    """Request for attack evaluation."""

    original_query: str = Field(..., description="Original benign query")
    adversarial_query: str = Field(..., description="Adversarial query after attack")
    baseline_response: str | None = Field(
        None, description="Response to original query (for length comparison)"
    )
    attack_response: str | None = Field(
        None, description="Response to adversarial query (for length comparison)"
    )
    ground_truth: str | None = Field(
        None, description="Expected correct answer (for accuracy evaluation)"
    )
    baseline_latency_ms: float | None = Field(
        None, description="Latency for baseline response in milliseconds"
    )
    attack_latency_ms: float | None = Field(
        None, description="Latency for attack response in milliseconds"
    )


class DecodeRequest(BaseModel):
    """Request for decoding obfuscated text."""

    obfuscated_text: str = Field(..., description="Obfuscated text to decode")


class ResourceMetricsRequest(BaseModel):
    """Request for resource exhaustion metrics calculation."""

    baseline_tokens: int = Field(..., ge=0, description="Token count for baseline response")
    attack_tokens: int = Field(..., ge=0, description="Token count for adversarial response")
    model: str = Field("o3", description="Target model for cost estimation")


# =============================================================================
# Response Models
# =============================================================================


class TransformationDetail(BaseModel):
    """Details about a single character transformation."""

    original_char: str = Field(..., description="Original character")
    original_index: int = Field(..., description="Position in original query")
    ascii_decimal: int = Field(..., description="ASCII decimal value")
    selected_base: int = Field(..., description="Base used for transformation")
    base_representation: str = Field(..., description="Value in selected base")
    obfuscated_form: str = Field(..., description="Final obfuscated form <(base)value>")


class AttackResponse(BaseModel):
    """Response with attack result."""

    original_query: str = Field(..., description="Original input query")
    adversarial_query: str = Field(..., description="Obfuscated adversarial query with N_note")
    obfuscation_ratio: float = Field(..., description="ρ value used for the attack")
    characters_transformed: int = Field(..., description="Number of characters transformed")
    total_characters: int = Field(..., description="Total characters in original query")
    estimated_token_increase: float = Field(
        ..., description="Estimated ratio of token increase"
    )
    transformation_density: float = Field(
        ..., description="Ratio of transformed to total characters"
    )
    length_increase_ratio: float = Field(
        ..., description="Ratio of adversarial length to original length"
    )
    bases_used: list[int] = Field(
        ..., description="Set of bases used in transformations"
    )
    n_note_used: str = Field(..., description="N_note template appended to query")
    transformation_details: dict[str, Any] | None = Field(
        None, description="Detailed transformation map (optional)"
    )


class BatchAttackResponse(BaseModel):
    """Response for batch attack execution."""

    results: list[AttackResponse] = Field(..., description="Individual attack results")
    total_queries: int = Field(..., description="Total queries processed")
    successful_attacks: int = Field(
        ..., description="Attacks with length ratio >= 1.5"
    )
    avg_length_ratio: float = Field(
        ..., description="Average length increase ratio across all attacks"
    )
    avg_transformation_density: float = Field(
        ..., description="Average transformation density across all attacks"
    )
    avg_token_increase: float = Field(
        ..., description="Average estimated token increase"
    )


class IndirectInjectionResponse(BaseModel):
    """Response for indirect injection attack."""

    original_document: str = Field(..., description="Original document content")
    poisoned_document: str = Field(..., description="Poisoned document with obfuscations")
    injection_points: int = Field(..., description="Number of injection points")
    document_length_ratio: float = Field(
        ..., description="Ratio of poisoned to original document length"
    )
    n_note_embedded: bool = Field(..., description="Whether N_note was embedded")
    estimated_impact: float = Field(
        ..., description="Estimated impact score (based on injection density)"
    )


class EvaluationResponse(BaseModel):
    """Response with evaluation metrics."""

    length_ratio: float = Field(
        ..., description="L(Y') / L(Y) - response length amplification"
    )
    latency_ratio: float | None = Field(
        None, description="Latency(Y') / Latency(Y) - latency amplification"
    )
    accuracy_preserved: bool | None = Field(
        None, description="Whether accuracy is preserved (degradation <= 5%)"
    )
    attack_successful: bool = Field(
        ..., description="Overall attack success (length ratio >= 1.5 and accuracy preserved)"
    )
    stealth_score: float = Field(
        ..., description="Stealth score (0-1, based on transformation density)"
    )
    metrics: dict[str, Any] = Field(..., description="Detailed metrics breakdown")


class BenchmarkConfigResponse(BaseModel):
    """Response with benchmark configuration."""

    name: str = Field(..., description="Benchmark name")
    description: str = Field(..., description="Benchmark description")
    selection_strategy: str = Field(..., description="Recommended selection strategy")
    preserve_structure: bool = Field(..., description="Whether structure is preserved")
    default_rho: float = Field(..., description="Default obfuscation ratio")
    model_specific_rho: dict[str, float] = Field(
        ..., description="Model-specific optimal ρ values"
    )
    recommended_n_note: str = Field(..., description="Recommended N_note variant")


class ResourceMetricsResponse(BaseModel):
    """Response with resource exhaustion metrics."""

    total_tokens_baseline: int = Field(..., description="Total tokens for baseline response")
    total_tokens_attack: int = Field(..., description="Total tokens for adversarial response")
    token_amplification: float = Field(
        ..., description="Token amplification factor (attack / baseline)"
    )
    estimated_cost_baseline_usd: float = Field(
        ..., description="Estimated cost for baseline in USD"
    )
    estimated_cost_attack_usd: float = Field(
        ..., description="Estimated cost for adversarial in USD"
    )
    cost_amplification: float = Field(
        ..., description="Cost amplification factor"
    )
    model: str = Field(..., description="Model used for cost estimation")


class NNoteTemplateResponse(BaseModel):
    """Response with N_note template details."""

    variant: str = Field(..., description="Template variant name")
    content: str = Field(..., description="N_note content")
    description: str = Field(..., description="Template description")
    recommended_for: list[str] = Field(..., description="Recommended use cases")
    effectiveness_rating: int = Field(
        ..., description="Effectiveness rating (1-10)"
    )


class NNoteTemplatesResponse(BaseModel):
    """Response with all available N_note templates."""

    templates: dict[str, NNoteTemplateResponse] = Field(
        ..., description="Map of variant name to template details"
    )


class DecodeResponse(BaseModel):
    """Response with decoded text."""

    original_text: str = Field(..., description="Original obfuscated text")
    decoded_text: str = Field(..., description="Decoded text")
    patterns_decoded: int = Field(
        ..., description="Number of obfuscation patterns decoded"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    module: str = Field(..., description="Module name")
    version: str = Field(..., description="Module version")
