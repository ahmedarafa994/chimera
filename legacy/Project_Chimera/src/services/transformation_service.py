"""
Transformation service for Project Chimera.
Handles prompt transformation logic with proper separation of concerns.
"""

import logging
import os

# Import existing components (will be refactored later)
import sys
import time
from typing import Any

from ..config.settings import get_performance_config
from ..core.technique_loader import technique_loader
from ..models.domain import IntentData, TransformationRequest, TransformationResult, ValidationError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assembler
import intent_deconstructor

logger = logging.getLogger(__name__)


class TransformationService:
    """
    Service for handling prompt transformations with proper business logic separation.
    """

    def __init__(self):
        self.performance_config = get_performance_config()
        self._initialize_components()

    def _initialize_components(self):
        """Initialize transformation components."""
        # This will be replaced with proper dependency injection later
        self.intent_analyzer = intent_deconstructor
        self.assembler = assembler

    def transform(self, request: TransformationRequest) -> TransformationResult:
        """
        Transform a prompt using the specified technique suite.

        Args:
            request: Transformation request containing all parameters

        Returns:
            TransformationResult with the transformed prompt and metadata
        """
        start_time = time.time()

        try:
            # Validate request
            self._validate_request(request)

            # Load technique components
            components = self._load_technique_components(request.technique_suite)
            if not components:
                raise ValidationError(
                    "technique_suite",
                    f"Technique suite '{request.technique_suite}' not found or failed to load",
                )

            # Analyze intent
            intent_data = self._analyze_intent(request.core_request)

            # Apply transformation
            result = self._apply_transformation(request, intent_data, components)

            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time

            # Log performance if enabled
            if self.performance_config.enabled:
                self._log_performance_metrics(request, result, processing_time)

            logger.info(
                f"Transformation completed: {request.technique_suite} "
                f"(potency {request.potency_level}) in {processing_time}ms"
            )

            return result

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            error_message = str(e)

            logger.error(f"Transformation failed: {error_message}")

            return TransformationResult(
                original_prompt=request.core_request,
                transformed_prompt="",
                technique_suite=request.technique_suite,
                potency_level=request.potency_level,
                processing_time_ms=processing_time,
                success=False,
                error_message=error_message,
            )

    def _validate_request(self, request: TransformationRequest):
        """Validate transformation request."""
        if not request.core_request.strip():
            raise ValidationError("core_request", "Core request cannot be empty")

        if not 1 <= request.potency_level <= 10:
            raise ValidationError("potency_level", "Potency level must be between 1 and 10")

        # Validate technique exists
        technique = technique_loader.get_technique(request.technique_suite)
        if not technique:
            raise ValidationError(
                "technique_suite", f"Unknown technique suite: {request.technique_suite}"
            )

        # Validate potency range
        if not technique_loader.validate_potency(request.technique_suite, request.potency_level):
            potency_range = technique.get("potency_range", [1, 10])
            raise ValidationError(
                "potency_level",
                f"Potency level {request.potency_level} not in range {potency_range}",
            )

    def _analyze_intent(self, core_request: str) -> IntentData:
        """Analyze intent from core request."""
        try:
            # Use existing intent analyzer
            intent_result = self.intent_analyzer.deconstruct_intent(core_request)

            # Convert to domain model
            return IntentData(
                raw_text=core_request,
                intent=intent_result.get("intent", "unknown"),
                keywords=intent_result.get("keywords", []),
                entities=intent_result.get("entities", {}),
                confidence=intent_result.get("confidence", 0.0),
                metadata=intent_result.get("metadata", {}),
            )

        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
            # Fallback to basic analysis
            return IntentData(raw_text=core_request, intent="general", keywords=[], confidence=0.0)

    def _load_technique_components(self, technique_suite: str) -> dict[str, list[Any]]:
        """Load components for the specified technique suite."""
        try:
            components = technique_loader.load_components_for_technique(technique_suite)

            # Validate required components
            if not components.get("transformers"):
                logger.warning(f"No transformers found for technique: {technique_suite}")

            if not components.get("assemblers"):
                logger.warning(f"No assemblers found for technique: {technique_suite}")

            return components

        except Exception as e:
            logger.error(f"Failed to load components for {technique_suite}: {e}")
            return {}

    def _apply_transformation(
        self,
        request: TransformationRequest,
        intent_data: IntentData,
        components: dict[str, list[Any]],
    ) -> TransformationResult:
        """Apply the transformation using loaded components."""
        prompt_components = {}
        applied_techniques = []
        temp_request_text = intent_data.raw_text

        # Apply obfuscators (if any)
        for obfuscator_func in components.get("obfuscators", []):
            try:
                if hasattr(obfuscator_func, "__name__"):
                    applied_techniques.append(obfuscator_func.__name__)

                # Apply obfuscation
                if "token_smuggling" in getattr(obfuscator_func, "__name__", ""):
                    temp_request_text = obfuscator_func(temp_request_text, intent_data.keywords)
                else:
                    temp_request_text = obfuscator_func(temp_request_text)

            except Exception as e:
                logger.warning(f"Obfuscator {obfuscator_func} failed: {e}")

        # Update intent data with obfuscated text
        if temp_request_text != intent_data.raw_text:
            intent_data.raw_text = temp_request_text

        # Apply transformers
        for engine_class in components.get("transformers", []):
            try:
                component = engine_class.transform(intent_data, request.potency_level)

                # Extract component name
                component_name = (
                    engine_class.__name__.replace("Engine", "")
                    .replace("Transformer", "")
                    .lower()
                    .replace("_", "")
                )

                prompt_components[component_name] = component
                applied_techniques.append(component_name)

            except Exception as e:
                logger.warning(f"Transformer {engine_class} failed: {e}")

        # Apply framers (if any)
        for framer_func in components.get("framers", []):
            try:
                if hasattr(framer_func, "__name__"):
                    applied_techniques.append(framer_func.__name__)

                # Apply framing
                framed_context = framer_func(intent_data, request.potency_level)
                prompt_components["framing"] = framed_context

            except Exception as e:
                logger.warning(f"Framer {framer_func} failed: {e}")

        # Assemble final prompt
        try:
            # Use the first available assembler
            assemblers = components.get("assemblers", [self.assembler.build_chimera_prompt])
            assembler_func = assemblers[0]

            transformed_prompt = assembler_func(
                prompt_components,
                {
                    "raw_text": intent_data.raw_text,
                    "keywords": intent_data.keywords,
                    "entities": intent_data.entities,
                },
                request.potency_level,
            )

        except Exception as e:
            logger.error(f"Prompt assembly failed: {e}")
            # Fallback to simple concatenation
            transformed_prompt = " ".join(prompt_components.values())

        # Create result
        return TransformationResult(
            original_prompt=request.core_request,
            transformed_prompt=transformed_prompt,
            technique_suite=request.technique_suite,
            potency_level=request.potency_level,
            techniques_applied=sorted(set(applied_techniques)),
            components_used={
                key: [getattr(comp, "__name__", str(comp)) for comp in comps]
                for key, comps in components.items()
            },
            success=True,
        )

    def _log_performance_metrics(
        self, request: TransformationRequest, result: TransformationResult, processing_time_ms: int
    ):
        """Log performance metrics if enabled."""
        if (
            self.performance_config.log_slow_queries
            and processing_time_ms > self.performance_config.slow_query_threshold_ms
        ):
            logger.warning(
                f"Slow transformation detected: {processing_time_ms}ms for "
                f"technique {request.technique_suite} (potency {request.potency_level})"
            )

    def validate_technique_compatibility(
        self, technique_suite: str, target_model: str | None = None
    ) -> dict[str, Any]:
        """
        Validate if a technique suite is compatible with the target model.

        Returns:
            Dictionary with validation results
        """
        technique = technique_loader.get_technique(technique_suite)
        if not technique:
            return {"compatible": False, "reason": f'Technique suite "{technique_suite}" not found'}

        # Check model compatibility
        if target_model:
            if not technique_loader.is_compatible_with_model(technique_suite, target_model):
                compatible_models = technique.get("metadata", {}).get("compatible_models", [])
                return {
                    "compatible": False,
                    "reason": f'Technique not compatible with model "{target_model}". '
                    f"Compatible models: {compatible_models}",
                }

        # Check required datasets
        required_datasets = technique_loader.get_required_datasets(technique_suite)
        if required_datasets:
            # TODO: Implement dataset availability checking
            pass

        return {"compatible": True, "technique": technique, "required_datasets": required_datasets}

    def get_available_techniques(
        self, category: str | None = None, target_model: str | None = None
    ) -> list[dict[str, Any]]:
        """Get list of available techniques, optionally filtered."""
        techniques = technique_loader.list_techniques()

        # Apply filters
        if category:
            techniques = [t for t in techniques if t.get("category") == category]

        if target_model:
            techniques = [
                t
                for t in techniques
                if technique_loader.is_compatible_with_model(t["name"], target_model)
            ]

        return techniques


# Global service instance
transformation_service = TransformationService()


# Convenience functions for backward compatibility
def transform_prompt(
    core_request: str, potency_level: int, technique_suite: str = "universal_bypass", **kwargs
) -> TransformationResult:
    """Convenience function for prompt transformation."""
    request = TransformationRequest(
        core_request=core_request,
        potency_level=potency_level,
        technique_suite=technique_suite,
        **kwargs,
    )
    return transformation_service.transform(request)
