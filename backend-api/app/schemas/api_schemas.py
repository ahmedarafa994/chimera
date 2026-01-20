from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- Enums ---
class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class EvasionTaskStatusEnum(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# --- Common Models ---
class HealthCheckResponse(BaseModel):
    status: str
    message: str


# --- LLM Models ---
class LLMModelBase(BaseModel):
    name: str = Field(
        ...,
        description="Name of the LLM model (e.g., 'gpt-4', 'gemini-pro', 'llama-2-7b-chat')",
    )
    provider: LLMProvider = Field(
        ...,
        description="Provider of the LLM (e.g., 'openai', 'gemini', 'huggingface')",
    )
    description: str | None = None


class LLMModelCreate(LLMModelBase):
    api_key: str | None = Field(
        None,
        description="API key for the LLM provider (will be stored securely)",
    )
    base_url: str | None = Field(None, description="Custom base URL for local/custom models")
    config: dict[str, Any] | None = Field(
        {},
        description="Additional configuration for the LLM client (e.g., model_id, temperature)",
    )


class LLMModel(LLMModelBase):
    id: int
    config: dict[str, Any]  # Store sanitized config (no API keys)

    class Config:
        from_attributes = True


# --- Metamorphosis Strategies ---
class MetamorphosisStrategyInfo(BaseModel):
    name: str = Field(
        ...,
        description="Unique name of the metamorphosis strategy (e.g., 'LexicalObfuscation')",
    )
    description: str = Field(..., description="Description of what the strategy does")
    parameters: dict[str, Any] = Field(
        {},
        description="Schema of configurable parameters for the strategy",
    )


class MetamorphosisStrategyConfig(BaseModel):
    name: str = Field(..., description="Name of the strategy to apply")
    params: dict[str, Any] = Field({}, description="Parameters for the strategy instance")


# --- Evasion Task Models ---
class EvasionTaskConfig(BaseModel):
    target_model_id: int = Field(..., description="ID of the target LLM model")
    initial_prompt: str = Field(..., description="The starting prompt for the evasion attempt")
    strategy_chain: list[MetamorphosisStrategyConfig] = Field(
        ...,
        min_length=1,
        description="Ordered list of metamorphosis strategies to apply",
    )
    success_criteria: str = Field(
        ...,
        description="Criterion to evaluate evasion success (e.g., keyword presence, refusal absence)",
    )
    max_attempts: int = Field(
        1,
        ge=1,
        description="Maximum number of attempts for the evasion task",
    )
    # Add other parameters like timeout, callback_url etc.


class EvasionTaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="Unique ID of the asynchronous evasion task")
    status: EvasionTaskStatusEnum = Field(..., description="Current status of the task")
    current_step: str | None = Field(None, description="Description of the current processing step")
    progress: float | None = Field(None, ge=0, le=100, description="Progress percentage")
    message: str | None = Field(None, description="Status message")
    results_preview: dict[str, Any] | None = Field(
        None,
        description="Preview of results if available",
    )
    error: str | None = Field(None, description="Error message if task failed")


class EvasionAttemptResult(BaseModel):
    attempt_number: int
    transformed_prompt: str
    llm_response: str
    is_evasion_successful: bool
    evaluation_details: dict[str, Any]
    transformation_log: list[dict[str, Any]]  # Log of each transformation step


class EvasionTaskResult(BaseModel):
    task_id: str
    status: EvasionTaskStatusEnum
    initial_prompt: str
    target_model_id: int
    strategy_chain: list[MetamorphosisStrategyConfig]
    success_criteria: str
    final_status: str
    results: list[EvasionAttemptResult]
    overall_success: bool = False
    completed_at: str | None = None  # ISO format datetime string
    failed_reason: str | None = None
