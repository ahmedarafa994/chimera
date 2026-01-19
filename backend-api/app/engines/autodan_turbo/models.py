"""
Pydantic models for AutoDAN-Turbo strategy library and attack results.

Based on the formal strategy definition from the paper:
A jailbreak strategy S is defined as a textual addition that increases
the jailbreak score when applied to a complex request.

IMPORTANT: Strategies serve as GUIDANCE for AI-generated prompts, not as
fill-in-the-blank templates. The LLM should use strategies as inspiration
to CREATE novel, creative adversarial prompts.
"""

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class StrategySource(str, Enum):
    """Source of the strategy."""

    SEED = "seed"  # Pre-defined seed strategies
    WARMUP = "warmup"  # Discovered during warm-up exploration
    LIFELONG = "lifelong"  # Discovered during running-time learning


class StrategyMetadata(BaseModel):
    """Metadata tracking for strategy performance."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = 0
    success_count: int = 0
    total_score: float = 0.0
    source: StrategySource = StrategySource.LIFELONG

    @property
    def average_score(self) -> float:
        """Calculate average score from usage history."""
        if self.usage_count == 0:
            return 0.0
        return self.total_score / self.usage_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate from usage history."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class JailbreakStrategy(BaseModel):
    """
    A jailbreak strategy as defined in the AutoDAN-Turbo paper.

    A strategy encapsulates a reusable PATTERN for constructing adversarial prompts
    that can bypass LLM safety measures.

    IMPORTANT USAGE NOTE:
    ---------------------
    Strategies are meant to serve as GUIDANCE and INSPIRATION for AI-generated prompts,
    NOT as fill-in-the-blank templates. When using a strategy:

    1. The `template` field provides an EXAMPLE of the strategy's approach
    2. The `description` explains the core technique
    3. The `examples` show successful applications

    The LLM should use these to UNDERSTAND the strategy and then CREATE a novel,
    creative prompt that applies the same principles in a new way.

    DO NOT simply substitute {payload} into the template - this produces "lazy"
    outputs that are easily detected. Instead, use the strategy to guide the
    LLM in generating something original.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Human-readable strategy name")
    description: str = Field(..., description="Core strategy description explaining the technique")
    template: str = Field(
        ...,
        description=(
            "Example prompt demonstrating the strategy. "
            "NOTE: This is for INSPIRATION only - the LLM should generate "
            "novel prompts inspired by this pattern, not fill in {payload}."
        ),
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Example successful prompts using this strategy (for inspiration)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags (e.g., 'roleplay', 'hypothetical', 'coding')",
    )
    metadata: StrategyMetadata = Field(default_factory=StrategyMetadata)
    embedding: list[float] | None = Field(
        default=None, description="Embedding vector for similarity search"
    )

    def apply(self, payload: str) -> str:
        """
        Apply this strategy to a payload using simple template substitution.

        WARNING: This method is DEPRECATED for prompt generation.
        ---------------------------------------------------------
        Direct template substitution produces "lazy" outputs that are easily
        detected. Use this only for:
        - Testing/debugging
        - Generating examples for the strategy library

        For actual attack generation, use the strategy as GUIDANCE for the LLM
        to generate novel prompts. See `lifelong_engine.py:_generate_with_strategy()`
        for the proper approach.
        """
        if "{payload}" in self.template:
            return self.template.format(payload=payload)
        # If no placeholder, return template as-is (it's an example, not a template)
        return self.template

    def get_guidance_text(self) -> str:
        """
        Get a formatted guidance text for LLM prompt generation.

        This is the PREFERRED way to use strategies - as guidance for the LLM
        to generate novel, creative prompts.
        """
        example_text = self.examples[0] if self.examples else self.template
        return (
            f"Strategy: {self.name}\n"
            f"Technique: {self.description}\n"
            f"Example Approach: {example_text}\n"
            f"Key Elements: {', '.join(self.tags) if self.tags else 'general adversarial'}"
        )

    def update_statistics(self, score: float, success: bool) -> None:
        """Update usage statistics after an attack attempt."""
        self.metadata.usage_count += 1
        self.metadata.total_score += score
        if success:
            self.metadata.success_count += 1
        self.metadata.updated_at = datetime.utcnow()

    class Config:
        """Pydantic config."""

        json_encoders: ClassVar[dict[type, Any]] = {datetime: lambda v: v.isoformat()}


class ScoringResult(BaseModel):
    """Result of LLM-based attack scoring."""

    score: float = Field(..., ge=1.0, le=10.0, description="Attack score (1-10)")
    is_jailbreak: bool = Field(..., description="Whether attack succeeded (score >= threshold)")
    reasoning: str = Field(default="", description="LLM's reasoning for the score")
    refusal_detected: bool = Field(default=False, description="Whether refusal patterns detected")
    stealth_score: float = Field(default=5.0, ge=1.0, le=10.0, description="Stealth score (1-10)")
    coherence_score: float = Field(
        default=5.0, ge=1.0, le=10.0, description="Coherence score (1-10)"
    )

    @property
    def weighted_score(self) -> float:
        """
        Calculate weighted score combining success, stealth, and coherence.
        Weights: Success (60%), Stealth (20%), Coherence (20%)
        """
        return (self.score * 0.6) + (self.stealth_score * 0.2) + (self.coherence_score * 0.2)


class AttackResult(BaseModel):
    """Result of a single attack attempt."""

    prompt: str = Field(
        ..., description="The adversarial prompt used (should be AI-generated, not template-filled)"
    )
    response: str = Field(..., description="Target model's response")
    scoring: ScoringResult = Field(..., description="LLM-based scoring result")
    strategy_id: str | None = Field(
        default=None, description="ID of strategy used as guidance (not as template)"
    )
    strategy_extracted: str | None = Field(
        default=None, description="ID of newly extracted strategy (if any)"
    )
    latency_ms: int = Field(default=0, description="Attack latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    technique_used: str | None = Field(default=None, description="Bypass technique used")
    score_before_bypass: float = Field(default=0.0, description="Score before bypass application")


class LifelongProgress(BaseModel):
    """Progress tracking for lifelong learning."""

    total_attacks: int = 0
    successful_attacks: int = 0
    strategies_discovered: int = 0
    current_phase: Literal["idle", "warmup", "lifelong"] = "idle"
    current_request_index: int = 0
    total_requests: int = 0
    average_score: float = 0.0
    best_score: float = 0.0
