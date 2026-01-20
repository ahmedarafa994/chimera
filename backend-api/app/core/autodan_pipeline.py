"""Unified AutoDAN Processing Pipeline.

This module provides a unified processing pipeline for all AutoDAN variants,
ensuring consistent workflow alignment across different optimization strategies
and techniques while maintaining scalability and observability.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages in the AutoDAN pipeline."""

    PRE_PROCESSING = "pre_processing"
    CORE_PROCESSING = "core_processing"
    POST_PROCESSING = "post_processing"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""

    execution_id: str
    start_time: float
    end_time: float | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    stage_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    error_count: int = 0
    retry_count: int = 0
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    @property
    def total_duration_ms(self) -> float:
        """Calculate total execution duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.end_time is not None


class PipelineContext:
    """Context object passed through the pipeline."""

    def __init__(self, request_data: dict[str, Any]) -> None:
        self.execution_id = str(uuid.uuid4())
        self.request_data = request_data
        self.stage_data: dict[str, Any] = {}
        self.metrics = PipelineMetrics(execution_id=self.execution_id, start_time=time.time())
        self.metadata: dict[str, Any] = {}
        self.current_stage: ProcessingStage | None = None
        self.status = PipelineStatus.PENDING
        self.errors: list[Exception] = []

    def set_stage_data(self, stage: ProcessingStage, data: Any) -> None:
        """Set data for a specific stage."""
        self.stage_data[stage.value] = data

    def get_stage_data(self, stage: ProcessingStage) -> Any:
        """Get data from a specific stage."""
        return self.stage_data.get(stage.value)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from context."""
        return self.metadata.get(key, default)

    def mark_stage_start(self, stage: ProcessingStage) -> None:
        """Mark the start of a processing stage."""
        self.current_stage = stage
        self.metrics.stage_timings[f"{stage.value}_start"] = time.time()

    def mark_stage_end(
        self,
        stage: ProcessingStage,
        stage_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Mark the end of a processing stage."""
        end_time = time.time()
        start_time = self.metrics.stage_timings.get(f"{stage.value}_start", end_time)
        self.metrics.stage_timings[f"{stage.value}_duration"] = end_time - start_time

        if stage_metrics:
            self.metrics.stage_metrics[stage.value] = stage_metrics

    def add_error(self, error: Exception) -> None:
        """Add an error to the context."""
        self.errors.append(error)
        self.metrics.error_count += 1

    def complete(self) -> None:
        """Mark execution as completed."""
        self.status = PipelineStatus.COMPLETED
        self.metrics.end_time = time.time()

    def fail(self, error: Exception) -> None:
        """Mark execution as failed."""
        self.status = PipelineStatus.FAILED
        self.metrics.end_time = time.time()
        self.add_error(error)


class PipelineProcessor(Protocol):
    """Protocol for pipeline processors."""

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context and return modified context."""
        ...

    async def validate(self, context: PipelineContext) -> bool:
        """Validate if processor can handle the context."""
        ...

    def get_processor_name(self) -> str:
        """Get the name of the processor."""
        ...


class BasePipelineProcessor(ABC):
    """Base class for pipeline processors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context."""

    async def validate(self, context: PipelineContext) -> bool:
        """Default validation - always returns True."""
        return True

    def get_processor_name(self) -> str:
        """Get processor name."""
        return self.name

    @asynccontextmanager
    async def _processing_context(self, context: PipelineContext, stage: ProcessingStage):
        """Context manager for processing with timing and error handling."""
        context.mark_stage_start(stage)
        start_time = time.time()

        try:
            self.logger.debug(f"Starting {self.name} processing for stage {stage.value}")
            yield

            # Calculate stage metrics
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            stage_metrics = {
                "processor_name": self.name,
                "duration_ms": duration_ms,
                "success": True,
            }
            context.mark_stage_end(stage, stage_metrics)
            self.logger.debug(f"Completed {self.name} processing in {duration_ms:.2f}ms")

        except Exception as e:
            self.logger.exception(f"Error in {self.name} processing: {e}")
            context.add_error(e)

            # Record error metrics
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            stage_metrics = {
                "processor_name": self.name,
                "duration_ms": duration_ms,
                "success": False,
                "error": str(e),
            }
            context.mark_stage_end(stage, stage_metrics)
            raise


class RequestValidationProcessor(BasePipelineProcessor):
    """Validates incoming requests."""

    def __init__(self) -> None:
        super().__init__("request_validation")

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Validate request data."""
        async with self._processing_context(context, ProcessingStage.PRE_PROCESSING):
            request_data = context.request_data

            # Basic validation
            if not request_data.get("request"):
                msg = "Request field is required"
                raise ValueError(msg)

            if len(request_data.get("request", "")) > 50000:
                msg = "Request too long"
                raise ValueError(msg)

            # Add validation metadata
            context.add_metadata("validation_passed", True)
            context.add_metadata("request_length", len(request_data.get("request", "")))

        return context


class StrategySelectionProcessor(BasePipelineProcessor):
    """Selects and configures the appropriate AutoDAN strategy."""

    def __init__(self) -> None:
        super().__init__("strategy_selection")
        self.available_strategies = {
            "vanilla": "VanillaStrategy",
            "genetic": "GeneticOptimizationStrategy",
            "mousetrap": "MousetrapChainStrategy",
            "overthink": "OverthinkFusionStrategy",
            "neural_bypass": "NeuralBypassStrategy",
            "adaptive": "AdaptiveStrategy",
            "beam_search": "BeamSearchStrategy",
        }

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Select and configure strategy."""
        async with self._processing_context(context, ProcessingStage.PRE_PROCESSING):
            request_data = context.request_data
            method = request_data.get("method", "adaptive")

            if method not in self.available_strategies:
                self.logger.warning(f"Unknown method {method}, falling back to adaptive")
                method = "adaptive"

            # Configure strategy
            strategy_config = {
                "method": method,
                "strategy_class": self.available_strategies[method],
                "parameters": self._extract_strategy_parameters(request_data),
            }

            context.set_stage_data(ProcessingStage.PRE_PROCESSING, strategy_config)
            context.add_metadata("selected_strategy", method)

        return context

    def _extract_strategy_parameters(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Extract strategy-specific parameters."""
        return {
            "temperature": request_data.get("temperature", 0.7),
            "max_tokens": request_data.get("max_tokens", 4096),
            "generations": request_data.get("generations", 5),
            "beam_width": request_data.get("beam_width", 3),
            "refinement_iterations": request_data.get("refinement_iterations", 3),
        }


class CoreStrategyProcessor(BasePipelineProcessor):
    """Executes the core AutoDAN strategy."""

    def __init__(self) -> None:
        super().__init__("core_strategy")

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Execute core strategy processing."""
        async with self._processing_context(context, ProcessingStage.CORE_PROCESSING):
            strategy_config = context.get_stage_data(ProcessingStage.PRE_PROCESSING)

            if not strategy_config:
                msg = "Strategy configuration not found"
                raise ValueError(msg)

            # Simulate strategy execution (replace with actual strategy implementation)
            await self._execute_strategy(context, strategy_config)

        return context

    async def _execute_strategy(
        self,
        context: PipelineContext,
        strategy_config: dict[str, Any],
    ) -> None:
        """Execute the selected strategy."""
        method = strategy_config["method"]
        parameters = strategy_config["parameters"]
        request = context.request_data.get("request", "")

        self.logger.info(f"Executing {method} strategy for request: {request[:100]}...")

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Generate mock result (replace with actual strategy implementation)
        result = {
            "jailbreak_prompt": f"[{method.upper()} OPTIMIZED] {request}",
            "method": method,
            "score": 8.5,  # Mock score
            "iterations": parameters.get("generations", 5),
            "success": True,
            "optimization_metrics": {
                "convergence_iterations": 3,
                "fitness_improvement": 0.75,
                "strategy_effectiveness": 0.85,
            },
        }

        context.set_stage_data(ProcessingStage.CORE_PROCESSING, result)
        context.add_metadata("core_processing_completed", True)


class ResultValidationProcessor(BasePipelineProcessor):
    """Validates and sanitizes results."""

    def __init__(self) -> None:
        super().__init__("result_validation")

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Validate processing results."""
        async with self._processing_context(context, ProcessingStage.VALIDATION):
            result = context.get_stage_data(ProcessingStage.CORE_PROCESSING)

            if not result:
                msg = "No result from core processing"
                raise ValueError(msg)

            # Validate result structure
            required_fields = ["jailbreak_prompt", "method", "score", "success"]
            for required_field in required_fields:
                if required_field not in result:
                    msg = f"Missing required field: {required_field}"
                    raise ValueError(msg)

            # Validate score range
            score = result.get("score", 0)
            if not (0 <= score <= 10):
                self.logger.warning(f"Score {score} out of valid range, clamping")
                result["score"] = max(0, min(10, score))

            # Add validation metadata
            context.add_metadata("result_validation_passed", True)
            context.add_metadata("result_score", result["score"])

        return context


class ResponseFormattingProcessor(BasePipelineProcessor):
    """Formats final response."""

    def __init__(self) -> None:
        super().__init__("response_formatting")

    async def process(self, context: PipelineContext) -> PipelineContext:
        """Format final response."""
        async with self._processing_context(context, ProcessingStage.POST_PROCESSING):
            result = context.get_stage_data(ProcessingStage.CORE_PROCESSING)

            # Format response with additional metadata
            formatted_response = {
                "attack_id": context.execution_id,
                "success": result.get("success", False),
                "generated_prompt": result.get("jailbreak_prompt", ""),
                "jailbreak_prompt": result.get("jailbreak_prompt", ""),  # Backwards compatibility
                "score": result.get("score", 0.0),
                "method": result.get("method", "unknown"),
                "status": "success" if result.get("success") else "error",
                "execution_time_ms": context.metrics.total_duration_ms,
                "iterations": result.get("iterations"),
                "optimization_metrics": result.get("optimization_metrics", {}),
                "pipeline_metrics": {
                    "execution_id": context.execution_id,
                    "stage_timings": context.metrics.stage_timings,
                    "error_count": context.metrics.error_count,
                    "total_duration_ms": context.metrics.total_duration_ms,
                },
                "metadata": context.metadata,
            }

            context.set_stage_data(ProcessingStage.POST_PROCESSING, formatted_response)

        return context


class UnifiedAutoDANPipeline:
    """Unified processing pipeline for all AutoDAN variants.

    This pipeline ensures consistent workflow alignment across all AutoDAN
    strategies while providing comprehensive monitoring and error handling.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.pipeline")
        self.pre_processors: list[PipelineProcessor] = []
        self.core_processors: list[PipelineProcessor] = []
        self.post_processors: list[PipelineProcessor] = []
        self.monitoring_enabled = True
        self._initialize_default_processors()

    def _initialize_default_processors(self) -> None:
        """Initialize default processor chain."""
        # Pre-processing
        self.pre_processors = [RequestValidationProcessor(), StrategySelectionProcessor()]

        # Core processing
        self.core_processors = [CoreStrategyProcessor()]

        # Post-processing
        self.post_processors = [ResultValidationProcessor(), ResponseFormattingProcessor()]

    def add_pre_processor(self, processor: PipelineProcessor) -> None:
        """Add a pre-processor to the pipeline."""
        self.pre_processors.append(processor)

    def add_core_processor(self, processor: PipelineProcessor) -> None:
        """Add a core processor to the pipeline."""
        self.core_processors.append(processor)

    def add_post_processor(self, processor: PipelineProcessor) -> None:
        """Add a post-processor to the pipeline."""
        self.post_processors.append(processor)

    async def execute(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the complete pipeline.

        Args:
            request_data: Input request data

        Returns:
            Formatted response dictionary

        Raises:
            Exception: If pipeline execution fails

        """
        context = PipelineContext(request_data)
        context.status = PipelineStatus.RUNNING

        try:
            self.logger.info(f"Starting pipeline execution {context.execution_id}")

            # Pre-processing
            context = await self._execute_processor_chain(
                context,
                self.pre_processors,
                "pre-processing",
            )

            # Core processing
            context = await self._execute_processor_chain(
                context,
                self.core_processors,
                "core processing",
            )

            # Post-processing
            context = await self._execute_processor_chain(
                context,
                self.post_processors,
                "post-processing",
            )

            # Mark as completed
            context.complete()

            # Get final result
            result = context.get_stage_data(ProcessingStage.POST_PROCESSING)
            if not result:
                msg = "No result from post-processing"
                raise ValueError(msg)

            self.logger.info(
                f"Pipeline execution {context.execution_id} completed successfully "
                f"in {context.metrics.total_duration_ms:.2f}ms",
            )

            return result

        except Exception as e:
            self.logger.exception(f"Pipeline execution {context.execution_id} failed: {e}")
            context.fail(e)

            # Return error response
            return {
                "attack_id": context.execution_id,
                "success": False,
                "status": "error",
                "error": str(e),
                "execution_time_ms": context.metrics.total_duration_ms,
                "pipeline_metrics": {
                    "execution_id": context.execution_id,
                    "error_count": context.metrics.error_count,
                    "errors": [str(err) for err in context.errors],
                },
            }

    async def _execute_processor_chain(
        self,
        context: PipelineContext,
        processors: list[PipelineProcessor],
        chain_name: str,
    ) -> PipelineContext:
        """Execute a chain of processors."""
        self.logger.debug(f"Starting {chain_name} chain with {len(processors)} processors")

        for processor in processors:
            # Validate processor can handle context
            if not await processor.validate(context):
                self.logger.warning(
                    f"Processor {processor.get_processor_name()} cannot handle context, skipping",
                )
                continue

            # Execute processor
            context = await processor.process(context)

        self.logger.debug(f"Completed {chain_name} chain")
        return context

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "pre_processors": len(self.pre_processors),
            "core_processors": len(self.core_processors),
            "post_processors": len(self.post_processors),
            "monitoring_enabled": self.monitoring_enabled,
            "processor_chain": [
                p.get_processor_name()
                for p in self.pre_processors + self.core_processors + self.post_processors
            ],
        }


# Global pipeline instance
_global_pipeline: UnifiedAutoDANPipeline | None = None


def get_pipeline() -> UnifiedAutoDANPipeline:
    """Get or create the global pipeline instance."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = UnifiedAutoDANPipeline()
    return _global_pipeline


async def execute_autodan_pipeline(request_data: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to execute the AutoDAN pipeline.

    Args:
        request_data: Input request data

    Returns:
        Formatted response dictionary

    """
    pipeline = get_pipeline()
    return await pipeline.execute(request_data)
