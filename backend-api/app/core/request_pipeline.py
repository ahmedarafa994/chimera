"""
Request Pipeline Manager

Unified request pipeline management for AI operations.
Provides a structured way to:
- Define pipeline stages
- Execute stages in order
- Track context through the pipeline
- Handle errors and fallbacks
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PipelineStageStatus(Enum):
    """Status of a pipeline stage execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PipelineStage:
    """
    Represents a stage in the request pipeline.

    Attributes:
        name: Unique name for the stage
        handler: Async callable that processes the stage
        order: Execution order (lower = earlier)
        enabled: Whether the stage is enabled
        required: Whether stage failure should stop pipeline
        timeout_seconds: Optional timeout for stage execution
    """

    name: str
    handler: Callable
    order: int
    enabled: bool = True
    required: bool = True
    timeout_seconds: float | None = None

    def __lt__(self, other: "PipelineStage") -> bool:
        """Compare stages by order for sorting."""
        return self.order < other.order


@dataclass
class StageResult:
    """Result of executing a pipeline stage."""

    stage_name: str
    status: PipelineStageStatus
    duration_ms: float
    output: Any | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class AIRequestContext:
    """
    Context passed through the AI request pipeline.

    This object accumulates state as it passes through pipeline stages,
    allowing stages to access and modify shared context.
    """

    # Request identification
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)

    # Provider selection
    provider: str = ""
    model: str = ""
    failover_chain: str | None = None

    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    # Cost tracking
    cost: float = 0.0

    # Fallback tracking
    fallback_attempts: int = 0
    providers_tried: list[str] = field(default_factory=list)

    # Stage results
    stage_results: list[StageResult] = field(default_factory=list)

    # Custom data storage
    data: dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: list[str] = field(default_factory=list)

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        delta = datetime.utcnow() - self.start_time
        return delta.total_seconds() * 1000

    def add_stage_result(self, result: StageResult) -> None:
        """Add a stage execution result."""
        self.stage_results.append(result)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def set_provider(self, provider: str, model: str) -> None:
        """Set the selected provider and model."""
        self.provider = provider
        self.model = model
        if provider not in self.providers_tried:
            self.providers_tried.append(provider)

    def record_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        """Record token usage."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += reasoning_tokens
        self.cached_tokens += cached_tokens

    def record_fallback(self) -> None:
        """Record a fallback attempt."""
        self.fallback_attempts += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "start_time": self.start_time.isoformat(),
            "elapsed_ms": self.elapsed_ms(),
            "provider": self.provider,
            "model": self.model,
            "failover_chain": self.failover_chain,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cost": self.cost,
            "fallback_attempts": self.fallback_attempts,
            "providers_tried": self.providers_tried,
            "stage_results": [r.to_dict() for r in self.stage_results],
            "errors": self.errors,
        }


class RequestPipeline:
    """
    Manages request processing pipeline.

    The pipeline executes stages in order, passing context through each stage.
    Stages can modify the context, add results, and trigger early termination.

    Example:
        pipeline = RequestPipeline()

        pipeline.add_stage(PipelineStage(
            name="validation",
            handler=validate_request,
            order=10,
        ))

        pipeline.add_stage(PipelineStage(
            name="rate_limit",
            handler=check_rate_limit,
            order=20,
        ))

        pipeline.add_stage(PipelineStage(
            name="execute",
            handler=execute_request,
            order=100,
        ))

        result = await pipeline.execute(request, context)
    """

    def __init__(self, name: str = "default"):
        """
        Initialize request pipeline.

        Args:
            name: Pipeline name for logging and identification
        """
        self._name = name
        self._stages: list[PipelineStage] = []
        self._lock_stages = False

    @property
    def name(self) -> str:
        """Get pipeline name."""
        return self._name

    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add

        Raises:
            ValueError: If stage name already exists or pipeline is locked
        """
        if self._lock_stages:
            raise ValueError("Pipeline stages are locked")

        # Check for duplicate names
        for existing in self._stages:
            if existing.name == stage.name:
                raise ValueError(f"Stage '{stage.name}' already exists")

        self._stages.append(stage)
        self._stages.sort()  # Sort by order

        logger.debug(
            f"Added stage '{stage.name}' to pipeline '{self._name}' "
            f"at order {stage.order}"
        )

    def remove_stage(self, name: str) -> bool:
        """
        Remove a stage from the pipeline.

        Args:
            name: Name of stage to remove

        Returns:
            True if stage was removed, False if not found
        """
        if self._lock_stages:
            raise ValueError("Pipeline stages are locked")

        for i, stage in enumerate(self._stages):
            if stage.name == name:
                del self._stages[i]
                logger.debug(f"Removed stage '{name}' from pipeline")
                return True

        return False

    def enable_stage(self, name: str) -> bool:
        """Enable a stage by name."""
        for stage in self._stages:
            if stage.name == name:
                stage.enabled = True
                return True
        return False

    def disable_stage(self, name: str) -> bool:
        """Disable a stage by name."""
        for stage in self._stages:
            if stage.name == name:
                stage.enabled = False
                return True
        return False

    def lock(self) -> None:
        """Lock pipeline to prevent stage modifications."""
        self._lock_stages = True

    def unlock(self) -> None:
        """Unlock pipeline to allow stage modifications."""
        self._lock_stages = False

    async def execute(
        self,
        request: Any,
        context: AIRequestContext | None = None,
    ) -> dict[str, Any]:
        """
        Execute all pipeline stages in order.

        Args:
            request: Request object to process
            context: Optional context (created if not provided)

        Returns:
            Dict with execution results including:
                - success: bool
                - context: AIRequestContext dict
                - result: Final stage output
                - error: Error message if failed
        """
        if context is None:
            context = AIRequestContext()

        logger.debug(
            f"Starting pipeline '{self._name}' execution, "
            f"request_id={context.request_id}"
        )

        result = None
        pipeline_error = None

        for stage in self._stages:
            if not stage.enabled:
                context.add_stage_result(StageResult(
                    stage_name=stage.name,
                    status=PipelineStageStatus.SKIPPED,
                    duration_ms=0,
                ))
                continue

            stage_start = time.perf_counter()

            try:
                logger.debug(f"Executing stage '{stage.name}'")

                # Execute stage handler
                stage_output = await stage.handler(request, context)

                duration_ms = (time.perf_counter() - stage_start) * 1000

                context.add_stage_result(StageResult(
                    stage_name=stage.name,
                    status=PipelineStageStatus.COMPLETED,
                    duration_ms=duration_ms,
                    output=stage_output,
                ))

                # Store last successful output as result
                if stage_output is not None:
                    result = stage_output

                logger.debug(
                    f"Stage '{stage.name}' completed in {duration_ms:.2f}ms"
                )

            except Exception as e:
                duration_ms = (time.perf_counter() - stage_start) * 1000
                error_msg = str(e)

                context.add_stage_result(StageResult(
                    stage_name=stage.name,
                    status=PipelineStageStatus.FAILED,
                    duration_ms=duration_ms,
                    error=error_msg,
                ))

                context.add_error(f"Stage '{stage.name}' failed: {error_msg}")

                logger.warning(
                    f"Stage '{stage.name}' failed after {duration_ms:.2f}ms: "
                    f"{error_msg}"
                )

                if stage.required:
                    pipeline_error = error_msg
                    break

        # Build final result
        return {
            "success": pipeline_error is None,
            "context": context.to_dict(),
            "result": result,
            "error": pipeline_error,
            "total_duration_ms": context.elapsed_ms(),
        }

    def get_pipeline_status(self) -> dict[str, Any]:
        """
        Get status of all pipeline stages.

        Returns:
            Dict with pipeline status information
        """
        return {
            "name": self._name,
            "locked": self._lock_stages,
            "stage_count": len(self._stages),
            "stages": [
                {
                    "name": stage.name,
                    "order": stage.order,
                    "enabled": stage.enabled,
                    "required": stage.required,
                    "timeout_seconds": stage.timeout_seconds,
                }
                for stage in self._stages
            ],
        }

    def get_stage(self, name: str) -> PipelineStage | None:
        """Get a stage by name."""
        for stage in self._stages:
            if stage.name == name:
                return stage
        return None


class PipelineRegistry:
    """
    Registry for managing multiple pipelines.

    Provides centralized access to named pipelines.
    """

    _instance: Optional["PipelineRegistry"] = None
    _pipelines: dict[str, RequestPipeline] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._pipelines = {}
        return cls._instance

    def register(
        self,
        name: str,
        pipeline: RequestPipeline,
        overwrite: bool = False,
    ) -> None:
        """
        Register a pipeline.

        Args:
            name: Pipeline name
            pipeline: Pipeline instance
            overwrite: Whether to overwrite existing pipeline
        """
        if name in self._pipelines and not overwrite:
            raise ValueError(f"Pipeline '{name}' already registered")

        self._pipelines[name] = pipeline
        logger.info(f"Registered pipeline: {name}")

    def get(self, name: str) -> RequestPipeline | None:
        """Get a pipeline by name."""
        return self._pipelines.get(name)

    def list_pipelines(self) -> list[str]:
        """List all registered pipeline names."""
        return list(self._pipelines.keys())

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered pipelines."""
        return {
            name: pipeline.get_pipeline_status()
            for name, pipeline in self._pipelines.items()
        }


# Global pipeline registry
pipeline_registry = PipelineRegistry()


def get_pipeline_registry() -> PipelineRegistry:
    """Get the global pipeline registry."""
    return pipeline_registry


def create_default_ai_pipeline() -> RequestPipeline:
    """
    Create the default AI request pipeline.

    Returns a pipeline with standard stages for AI operations:
    1. Validation (order=10)
    2. Provider Selection (order=20)
    3. Rate Limiting (order=30)
    4. Fallback Setup (order=40)
    5. Execution (order=100)
    6. Cost Tracking (order=200)
    """
    pipeline = RequestPipeline(name="ai_request")

    # These handlers would be defined elsewhere and imported
    # Here we define placeholder handlers

    async def validation_stage(request: Any, context: AIRequestContext) -> None:
        """Validate the incoming request."""
        # Validation logic would go here
        pass

    async def provider_selection_stage(
        request: Any,
        context: AIRequestContext,
    ) -> None:
        """Select provider and model."""
        # Provider selection logic would go here
        # This would integrate with ProviderSelectionMiddleware
        pass

    async def rate_limit_stage(
        request: Any,
        context: AIRequestContext,
    ) -> None:
        """Check rate limits."""
        # Rate limit checking would go here
        pass

    async def fallback_setup_stage(
        request: Any,
        context: AIRequestContext,
    ) -> None:
        """Setup fallback chain."""
        # Fallback chain setup would go here
        pass

    async def cost_tracking_stage(
        request: Any,
        context: AIRequestContext,
    ) -> None:
        """Track costs after execution."""
        # Cost tracking would go here
        pass

    # Add stages
    pipeline.add_stage(PipelineStage(
        name="validation",
        handler=validation_stage,
        order=10,
        required=True,
    ))

    pipeline.add_stage(PipelineStage(
        name="provider_selection",
        handler=provider_selection_stage,
        order=20,
        required=True,
    ))

    pipeline.add_stage(PipelineStage(
        name="rate_limit",
        handler=rate_limit_stage,
        order=30,
        required=False,  # Don't fail on rate limit check errors
    ))

    pipeline.add_stage(PipelineStage(
        name="fallback_setup",
        handler=fallback_setup_stage,
        order=40,
        required=False,
    ))

    pipeline.add_stage(PipelineStage(
        name="cost_tracking",
        handler=cost_tracking_stage,
        order=200,
        required=False,  # Cost tracking failure shouldn't fail request
    ))

    return pipeline


def setup_default_pipelines() -> None:
    """
    Setup default pipelines and register them.

    Should be called during application startup.
    """
    # Create and register AI request pipeline
    ai_pipeline = create_default_ai_pipeline()
    pipeline_registry.register("ai_request", ai_pipeline)

    logger.info("Default pipelines registered")
