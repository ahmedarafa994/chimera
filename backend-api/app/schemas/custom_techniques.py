"""
Custom Technique Builder Schema Definitions

This module provides Pydantic models for the custom technique builder feature,
enabling users to create, compose, and preview custom transformation techniques.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# Enums
# ==============================================================================


class TransformerCategory(str, Enum):
    """Categories for transformer primitives."""

    TRANSFORMER = "transformer"
    FRAMER = "framer"
    OBFUSCATOR = "obfuscator"
    PERSONA = "persona"
    ENCODER = "encoder"


class ParameterType(str, Enum):
    """Types for transformer parameters."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    TEXT = "text"


class TechniqueVisibility(str, Enum):
    """Visibility options for custom techniques."""

    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


# ==============================================================================
# Parameter Schema Models
# ==============================================================================


class ParameterSchema(BaseModel):
    """Schema definition for a transformer parameter."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )

    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter data type")
    description: str | None = Field(None, description="Parameter description")
    required: bool = Field(default=False, description="Whether the parameter is required")
    default: Any = Field(default=None, description="Default value for the parameter")
    min_value: float | None = Field(default=None, description="Minimum value for numeric types")
    max_value: float | None = Field(default=None, description="Maximum value for numeric types")
    enum_values: list[str] | None = Field(default=None, description="Allowed values for enum type")


# ==============================================================================
# Transformer Primitive Models
# ==============================================================================


class TransformerPrimitive(BaseModel):
    """
    Represents a single transformer primitive that can be used in technique composition.

    This is a discoverable transformation unit from the existing transformation engine.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )

    id: str = Field(..., description="Unique identifier for the transformer")
    name: str = Field(..., description="Display name of the transformer")
    description: str = Field(..., description="Description of what the transformer does")
    category: TransformerCategory = Field(..., description="Category of the transformer")
    parameters: list[ParameterSchema] = Field(
        default_factory=list, description="Parameter schema for this transformer"
    )
    examples: list[str] | None = Field(default=None, description="Example transformations")
    is_available: bool = Field(
        default=True, description="Whether the transformer is currently available"
    )


class TransformerPrimitiveList(BaseModel):
    """Response model for listing available transformer primitives."""

    primitives: list[TransformerPrimitive] = Field(
        ..., description="List of available transformer primitives"
    )
    total: int = Field(..., description="Total number of primitives")
    categories: list[str] = Field(default_factory=list, description="List of available categories")


# ==============================================================================
# Technique Step Models
# ==============================================================================


class TechniqueStep(BaseModel):
    """
    Represents a single step in a custom technique composition.

    A step references a transformer primitive and includes configuration for that step.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )

    transformer_id: str = Field(..., description="ID of the transformer primitive to use")
    name: str | None = Field(
        default=None, description="Custom name for this step (defaults to transformer name)"
    )
    description: str | None = Field(default=None, description="Custom description for this step")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameter values for this step"
    )
    order: int = Field(..., ge=0, description="Execution order of this step (0-indexed)")
    enabled: bool = Field(default=True, description="Whether this step is enabled in the chain")


class TechniqueStepCreate(BaseModel):
    """Request model for adding a step to a technique."""

    transformer_id: str = Field(..., description="ID of the transformer primitive to use")
    name: str | None = Field(default=None, description="Custom name for this step")
    description: str | None = Field(default=None, description="Custom description for this step")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameter values for this step"
    )
    order: int | None = Field(
        default=None, description="Execution order (auto-assigned if not provided)"
    )
    enabled: bool = Field(default=True, description="Whether this step is enabled")


class TechniqueStepUpdate(BaseModel):
    """Request model for updating a step in a technique."""

    name: str | None = Field(default=None, description="Updated custom name")
    description: str | None = Field(default=None, description="Updated description")
    parameters: dict[str, Any] | None = Field(default=None, description="Updated parameters")
    order: int | None = Field(default=None, ge=0, description="Updated execution order")
    enabled: bool | None = Field(default=None, description="Updated enabled state")


# ==============================================================================
# Custom Technique Definition Models
# ==============================================================================


class TechniqueComposition(BaseModel):
    """The composition of steps that make up a custom technique."""

    steps: list[TechniqueStep] = Field(
        default_factory=list, description="Ordered list of technique steps"
    )


class TechniqueVersionInfo(BaseModel):
    """Version information for a custom technique."""

    version: int = Field(default=1, ge=1, description="Version number")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Version creation timestamp"
    )
    notes: str | None = Field(default=None, description="Version notes or changelog")


class CustomTechniqueDefinition(BaseModel):
    """
    Complete definition of a custom technique.

    This is the main entity for storing and managing custom techniques.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )

    id: str = Field(..., description="Unique identifier for the technique")
    name: str = Field(..., min_length=1, max_length=100, description="Name of the technique")
    description: str | None = Field(
        default=None, max_length=500, description="Description of the technique"
    )
    composition: TechniqueComposition = Field(
        default_factory=TechniqueComposition,
        description="The composition of steps in this technique",
    )
    version: int = Field(default=1, ge=1, description="Current version number")
    version_history: list[TechniqueVersionInfo] = Field(
        default_factory=list, description="History of previous versions"
    )
    visibility: TechniqueVisibility = Field(
        default=TechniqueVisibility.PRIVATE, description="Visibility of the technique"
    )
    user_id: str | None = Field(default=None, description="ID of the owning user")
    team_id: str | None = Field(default=None, description="ID of the owning team (if team-visible)")
    tags: list[str] = Field(
        default_factory=list, max_length=10, description="Tags for categorization"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    usage_count: int = Field(
        default=0, ge=0, description="Number of times the technique has been used"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CustomTechniqueCreate(BaseModel):
    """Request model for creating a new custom technique."""

    name: str = Field(..., min_length=1, max_length=100, description="Name of the technique")
    description: str | None = Field(
        default=None, max_length=500, description="Description of the technique"
    )
    steps: list[TechniqueStepCreate] = Field(
        default_factory=list, description="Initial steps to add"
    )
    visibility: TechniqueVisibility = Field(
        default=TechniqueVisibility.PRIVATE, description="Visibility of the technique"
    )
    team_id: str | None = Field(default=None, description="Team ID for team-visible techniques")
    tags: list[str] = Field(
        default_factory=list, max_length=10, description="Tags for categorization"
    )


class CustomTechniqueUpdate(BaseModel):
    """Request model for updating an existing custom technique."""

    name: str | None = Field(default=None, min_length=1, max_length=100, description="Updated name")
    description: str | None = Field(default=None, max_length=500, description="Updated description")
    steps: list[TechniqueStepCreate] | None = Field(
        default=None, description="Updated steps (replaces all existing steps)"
    )
    visibility: TechniqueVisibility | None = Field(default=None, description="Updated visibility")
    team_id: str | None = Field(default=None, description="Updated team ID")
    tags: list[str] | None = Field(default=None, max_length=10, description="Updated tags")
    version_notes: str | None = Field(default=None, description="Notes for this version update")


class CustomTechniqueSummary(BaseModel):
    """Summary view of a custom technique for listing."""

    id: str = Field(..., description="Technique ID")
    name: str = Field(..., description="Technique name")
    description: str | None = Field(default=None, description="Technique description")
    step_count: int = Field(default=0, ge=0, description="Number of steps in the technique")
    version: int = Field(default=1, description="Current version")
    visibility: TechniqueVisibility = Field(..., description="Visibility setting")
    tags: list[str] = Field(default_factory=list, description="Tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    usage_count: int = Field(default=0, ge=0, description="Usage count")


class CustomTechniqueList(BaseModel):
    """Response model for listing custom techniques."""

    techniques: list[CustomTechniqueSummary] = Field(..., description="List of technique summaries")
    total: int = Field(..., description="Total number of techniques")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")


# ==============================================================================
# Preview Models
# ==============================================================================


class StepPreviewResult(BaseModel):
    """Result of applying a single step in a preview."""

    step_order: int = Field(..., description="Order of this step in the chain")
    transformer_id: str = Field(..., description="ID of the transformer used")
    transformer_name: str = Field(..., description="Name of the transformer used")
    input_text: str = Field(..., description="Input text to this step")
    output_text: str = Field(..., description="Output text from this step")
    execution_time_ms: float = Field(
        ..., ge=0, description="Time taken to execute this step in milliseconds"
    )
    success: bool = Field(default=True, description="Whether the step executed successfully")
    error: str | None = Field(default=None, description="Error message if step failed")
    skipped: bool = Field(default=False, description="Whether the step was skipped (disabled)")


class TechniquePreviewResult(BaseModel):
    """
    Result of previewing a technique transformation.

    Contains the original input, final output, and intermediate results from each step.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        use_enum_values=True,
    )

    original_prompt: str = Field(..., description="The original input prompt")
    transformed_prompt: str = Field(..., description="The final transformed prompt")
    step_results: list[StepPreviewResult] = Field(
        default_factory=list, description="Results from each step in order"
    )
    total_steps: int = Field(..., ge=0, description="Total number of steps in the technique")
    executed_steps: int = Field(..., ge=0, description="Number of steps actually executed")
    skipped_steps: int = Field(default=0, ge=0, description="Number of steps skipped (disabled)")
    failed_steps: int = Field(default=0, ge=0, description="Number of steps that failed")
    total_execution_time_ms: float = Field(
        ..., ge=0, description="Total execution time in milliseconds"
    )
    success: bool = Field(default=True, description="Whether the preview completed successfully")
    error: str | None = Field(default=None, description="Error message if preview failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the preview"
    )


class TechniquePreviewRequest(BaseModel):
    """Request model for previewing a technique transformation."""

    sample_prompt: str = Field(
        ..., min_length=1, max_length=10000, description="Sample prompt to transform"
    )
    technique_id: str | None = Field(
        default=None, description="ID of an existing technique to preview"
    )
    steps: list[TechniqueStepCreate] | None = Field(
        default=None, description="Ad-hoc steps to preview (if not using existing technique)"
    )
    include_intermediate_results: bool = Field(
        default=True, description="Whether to include intermediate step results"
    )


# ==============================================================================
# Validation Models
# ==============================================================================


class TechniqueValidationError(BaseModel):
    """A validation error for a technique definition."""

    field: str = Field(..., description="Field that has the error")
    message: str = Field(..., description="Error message")
    step_order: int | None = Field(default=None, description="Step order if error is step-specific")


class TechniqueValidationResult(BaseModel):
    """Result of validating a technique definition."""

    valid: bool = Field(..., description="Whether the technique is valid")
    errors: list[TechniqueValidationError] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: list[str] = Field(default_factory=list, description="List of validation warnings")
