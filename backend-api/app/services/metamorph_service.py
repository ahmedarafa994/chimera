"""
Metamorph Service - Dynamic prompt transformation and dataset management.

This service provides:
1. Dataset loading and management for jailbreak techniques
2. Dynamic transformer configuration
3. Technique suite orchestration
4. Prompt metamorphosis with multiple transformation layers

Now integrated with AIConfigManager for:
- Config-driven transformation provider selection
- Provider-specific transformation strategies
- Provider-specific output formatting
- Cost-aware transformation routing
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.service_registry import get_ai_config_manager
from app.engines import dataset_loader, obfuscator, psychological_framer, transformer_engine

logger = logging.getLogger(__name__)


# Wrapper functions for obfuscators to normalize signatures
def _apply_token_smuggling(text: str, potency: int) -> str:
    """Wrapper for token smuggling with common sensitive keywords."""
    keywords = ["hack", "bypass", "ignore", "override", "admin", "system"]
    return obfuscator.apply_token_smuggling(text, keywords)


def _apply_semantic_noise(text: str, potency: int) -> str:
    """Apply semantic noise through synonym substitution."""
    return obfuscator.apply_synonym_substitution(text)


def _apply_code_obfuscation(text: str, potency: int) -> str:
    """Apply code-style obfuscation with base64 encoding."""
    if potency >= 7:
        return obfuscator.apply_base64_wrapper(text)
    return obfuscator.apply_leetspeak(text)


def _apply_polymorphic_encoding(text: str, potency: int) -> str:
    """Apply polymorphic encoding based on potency level."""
    if potency >= 8:
        return obfuscator.apply_base64_encoding(text)
    shift = min(potency + 5, 25)
    return obfuscator.apply_caesar_cipher(text, shift)


class TransformationMode(Enum):
    """Available transformation modes."""

    BASIC = "basic"
    LAYERED = "layered"
    RECURSIVE = "recursive"
    ADAPTIVE = "adaptive"


@dataclass
class TransformationResult:
    """Result of a metamorphic transformation."""

    original_prompt: str
    transformed_prompt: str
    techniques_applied: list[str] = field(default_factory=list)
    layers_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class MetamorphService:
    """
    Service for metamorphic prompt transformations.

    Provides dynamic dataset loading, transformer management,
    and multi-layer prompt transformation capabilities.

    Integrated with AIConfigManager for:
    - Config-driven provider selection for transformations
    - Provider-specific strategies and output formatting
    - Cost-aware routing decisions
    """

    # Provider-specific transformation strategies
    PROVIDER_TRANSFORMATION_STRATEGIES: dict[str, dict[str, Any]] = {
        "openai": {
            "preferred_suites": ["subtle_persuasion", "academic_research", "roleplay_bypass"],
            "output_format": "conversational",
            "max_obfuscation": 7,
        },
        "anthropic": {
            "preferred_suites": ["subtle_persuasion", "cognitive_hacking", "academic_research"],
            "output_format": "structured",
            "max_obfuscation": 6,
        },
        "google": {
            "preferred_suites": ["deep_inception", "quantum_exploit", "metamorphic_attack"],
            "output_format": "conversational",
            "max_obfuscation": 8,
        },
        "gemini": {
            "preferred_suites": ["deep_inception", "quantum_exploit", "metamorphic_attack"],
            "output_format": "conversational",
            "max_obfuscation": 8,
        },
        "deepseek": {
            "preferred_suites": ["code_chameleon", "cipher", "payload_splitting"],
            "output_format": "code_block",
            "max_obfuscation": 9,
        },
    }

    def __init__(self):
        self.dataset_loader: dataset_loader.DatasetLoader | None = None
        from app.core.config import settings

        self.technique_suites = settings.TRANSFORMATION_TECHNIQUES
        self._registry = self._initialize_registry()
        self._initialized = False
        self._config_manager = None
        self._cost_tracker: dict[str, float] = {}

    def _get_config_manager(self):
        """Get AI config manager with lazy initialization."""
        if self._config_manager is None:
            try:
                self._config_manager = get_ai_config_manager()
            except Exception as e:
                logger.warning(f"Failed to get AI config manager: {e}")
                return None
        return self._config_manager

    def _initialize_registry(self) -> dict[str, Any]:
        """Initialize registry of transformation functions and classes."""
        return {
            # Transformers
            "GeminiTransformationEngine": transformer_engine.GeminiTransformationEngine,
            "ContextualFramingEngine": transformer_engine.ContextualFramingEngine,
            # Framers
            "apply_collaborative_framing": psychological_framer.apply_collaborative_framing,
            "apply_hypothetical_framing": psychological_framer.apply_hypothetical_framing,
            "apply_authority_bias": psychological_framer.apply_authority_bias,
            "apply_odyssey_simulation": psychological_framer.apply_odyssey_simulation,
            "apply_academic_context": psychological_framer.apply_academic_context,
            "apply_metamorphic_framing": psychological_framer.apply_metamorphic_framing,
            # Obfuscators
            "apply_token_smuggling": _apply_token_smuggling,
            "apply_semantic_noise": _apply_semantic_noise,
            "apply_code_obfuscation": _apply_code_obfuscation,
            "apply_polymorphic_encoding": _apply_polymorphic_encoding,
        }

    async def initialize(self) -> bool:
        """
        Initialize the service with datasets and transformers.
        Should be called at application startup.

        Returns:
            bool: True if initialization successful
        """
        if self._initialized:
            logger.info("MetamorphService already initialized")
            return True

        try:
            # Determine base path for imported_data
            # backend-api/app/services/metamorph_service.py
            # -> backend-api/app/services -> backend-api/app -> backend-api -> chimera
            base_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "imported_data")
            )

            if not os.path.exists(base_path):
                logger.warning(f"Imported data directory not found at {base_path}")
                # Try alternative path
                alt_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "imported_data")
                )
                if os.path.exists(alt_path):
                    base_path = alt_path
                else:
                    logger.error("Could not locate imported_data directory")
                    return False

            logger.info(f"Loading datasets from: {base_path}")
            self.dataset_loader = dataset_loader.DatasetLoader(base_path)

            # Load core datasets
            datasets_to_load = [
                "GPTFuzz",
                "CodeChameleon",
                "PAIR",
                "Cipher",
                "Deepinception",
                "Multilingual",
            ]

            loaded_count = 0
            for dataset_name in datasets_to_load:
                try:
                    if self.dataset_loader.load_dataset(dataset_name):
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load dataset {dataset_name}: {e}")

            logger.info(f"Loaded {loaded_count}/{len(datasets_to_load)} datasets")

            self._initialized = True
            logger.info("MetamorphService initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MetamorphService: {e}")
            return False

    def get_available_suites(self) -> list[str]:
        """Get list of available technique suite names."""
        return list(self.technique_suites.keys())

    def get_suite_info(self, suite_name: str) -> dict[str, Any] | None:
        """
        Get information about a specific technique suite.

        Args:
            suite_name: Name of the technique suite

        Returns:
            Dict with suite information or None if not found
        """
        if suite_name not in self.technique_suites:
            return None

        suite = self.technique_suites[suite_name]
        return {
            "name": suite_name,
            "transformers_count": len(suite.get("transformers", [])),
            "framers_count": len(suite.get("framers", [])),
            "obfuscators_count": len(suite.get("obfuscators", [])),
        }

    def get_provider_transformation_strategy(self, provider: str | None = None) -> dict[str, Any]:
        """
        Get transformation strategy for a specific provider.

        Args:
            provider: Provider name (uses default if None)

        Returns:
            Strategy dict with preferred_suites, output_format, max_obfuscation
        """
        config_manager = self._get_config_manager()

        # Get provider name
        if not provider and config_manager:
            try:
                config = config_manager.get_config()
                provider = config.global_config.default_provider
            except Exception:
                provider = "openai"
        provider = provider or "openai"

        # Check config for custom strategy
        if config_manager:
            try:
                provider_config = config_manager.get_provider(provider)
                if provider_config and provider_config.metadata:
                    custom_strategy = provider_config.metadata.get("transformation_strategy")
                    if custom_strategy:
                        return custom_strategy
            except Exception as e:
                logger.debug(f"Could not get custom strategy: {e}")

        # Fall back to default strategies
        return self.PROVIDER_TRANSFORMATION_STRATEGIES.get(
            provider.lower(), self.PROVIDER_TRANSFORMATION_STRATEGIES["openai"]
        )

    def select_optimal_provider_for_transformation(self, suite_name: str) -> str | None:
        """
        Select the best provider for a specific transformation suite.

        Args:
            suite_name: Name of the transformation suite

        Returns:
            Provider name or None
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            return None

        try:
            config = config_manager.get_config()

            # Find providers that prefer this suite
            for provider in config.get_enabled_providers():
                strategy = self.get_provider_transformation_strategy(provider.provider_id)
                if suite_name in strategy.get("preferred_suites", []):
                    return provider.provider_id

            # Fall back to default
            return config.global_config.default_provider

        except Exception as e:
            logger.error(f"Error selecting provider: {e}")
            return None

    def estimate_transformation_cost(
        self, prompt: str, provider: str | None = None, suite_name: str = "subtle_persuasion"
    ) -> float | None:
        """
        Estimate cost for a transformation operation.

        Args:
            prompt: Prompt to transform
            provider: Provider name
            suite_name: Transformation suite

        Returns:
            Estimated cost in USD or None
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            return None

        try:
            if not provider:
                config = config_manager.get_config()
                provider = config.global_config.default_provider

            provider_config = config_manager.get_provider(provider)
            if not provider_config:
                return None

            default_model = provider_config.get_default_model()
            if not default_model:
                return None

            # Estimate tokens (rough: 4 chars per token)
            input_tokens = len(prompt) // 4
            # Transformations typically expand prompt by 2-3x
            output_tokens = input_tokens * 2

            return config_manager.calculate_cost(
                provider_id=provider,
                model_id=default_model.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return None

    def format_output_for_provider(self, transformed: str, provider: str | None = None) -> str:
        """
        Format transformed output according to provider preferences.

        Args:
            transformed: Transformed prompt
            provider: Provider name

        Returns:
            Formatted output string
        """
        strategy = self.get_provider_transformation_strategy(provider)
        output_format = strategy.get("output_format", "conversational")

        if output_format == "code_block":
            return f"```\n{transformed}\n```"
        elif output_format == "structured":
            return f"<prompt>\n{transformed}\n</prompt>"
        else:  # conversational
            return transformed

    def transform(
        self,
        prompt: str,
        suite_name: str = "subtle_persuasion",
        potency: int = 5,
        mode: TransformationMode = TransformationMode.LAYERED,
        provider: str | None = None,
        cost_aware: bool = True,
    ) -> TransformationResult:
        """
        Transform a prompt using the specified technique suite.

        Now with config-driven provider selection and cost awareness.

        Args:
            prompt: The original prompt to transform
            suite_name: Name of the technique suite to use
            potency: Transformation intensity (1-10)
            mode: Transformation mode (basic, layered, recursive, adaptive)
            provider: Target provider (for output formatting)
            cost_aware: Whether to select provider based on cost

        Returns:
            TransformationResult with transformed prompt and metadata
        """
        # Get provider-specific strategy
        strategy = self.get_provider_transformation_strategy(provider)

        # Adjust potency based on provider max
        max_obfuscation = strategy.get("max_obfuscation", 10)
        effective_potency = min(potency, max_obfuscation)
        if effective_potency != potency:
            logger.info(
                f"Adjusted potency from {potency} to {effective_potency} "
                f"for provider constraints"
            )

        # Cost-aware provider selection
        if cost_aware and not provider:
            optimal_provider = self.select_optimal_provider_for_transformation(suite_name)
            if optimal_provider:
                provider = optimal_provider
                strategy = self.get_provider_transformation_strategy(provider)

        # Estimate cost before transformation
        estimated_cost = self.estimate_transformation_cost(prompt, provider, suite_name)
        if estimated_cost:
            logger.debug(f"Estimated transformation cost: ${estimated_cost:.6f}")

        if suite_name not in self.technique_suites:
            # Try provider-preferred suites
            preferred = strategy.get("preferred_suites", [])
            for pref_suite in preferred:
                if pref_suite in self.technique_suites:
                    logger.info(
                        f"Using provider-preferred suite '{pref_suite}' "
                        f"instead of '{suite_name}'"
                    )
                    suite_name = pref_suite
                    break
            else:
                # Fallback for older configurations or missing files
                if not self.technique_suites:
                    logger.error("No technique suites loaded!")
                    return TransformationResult(
                        prompt, prompt, [], 0, {"error": "No suites loaded"}
                    )
                logger.warning(f"Unknown suite '{suite_name}', using first available")
                suite_name = next(iter(self.technique_suites.keys()))

        suite = self.technique_suites[suite_name]
        techniques_applied = []
        transformed = prompt
        layers = 0

        # Apply transformers
        for transformer_name in suite.get("transformers", []):
            try:
                transformer_cls = self._registry.get(transformer_name)
                if transformer_cls and hasattr(transformer_cls, "transform"):
                    transformer = transformer_cls()
                    transformed = transformer.transform(transformed, potency)
                    techniques_applied.append(transformer_name)
                    layers += 1
                else:
                    logger.warning(f"Transformer '{transformer_name}' not found in registry")
            except Exception as e:
                logger.warning(f"Transformer {transformer_name} failed: {e}")

        # Apply framers
        for framer_name in suite.get("framers", []):
            try:
                framer_fn = self._registry.get(framer_name)
                if framer_fn and callable(framer_fn):
                    transformed = framer_fn(transformed, potency)
                    techniques_applied.append(framer_name)
                    layers += 1
                else:
                    logger.warning(f"Framer '{framer_name}' not found in registry")
            except Exception as e:
                logger.warning(f"Framer {framer_name} failed: {e}")

        # Apply obfuscators (only at higher potency levels)
        if potency >= 5:
            for obfuscator_name in suite.get("obfuscators", []):
                try:
                    obfuscator_fn = self._registry.get(obfuscator_name)
                    if obfuscator_fn and callable(obfuscator_fn):
                        transformed = obfuscator_fn(transformed, potency)
                        techniques_applied.append(obfuscator_name)
                        layers += 1
                    else:
                        logger.warning(f"Obfuscator '{obfuscator_name}' not found in registry")
                except Exception as e:
                    logger.warning(f"Obfuscator {obfuscator_name} failed: {e}")

        # Format output for provider
        formatted_output = self.format_output_for_provider(transformed, provider)

        # Track cost
        if estimated_cost:
            provider_key = provider or "default"
            self._cost_tracker[provider_key] = (
                self._cost_tracker.get(provider_key, 0.0) + estimated_cost
            )

        return TransformationResult(
            original_prompt=prompt,
            transformed_prompt=formatted_output,
            techniques_applied=techniques_applied,
            layers_count=layers,
            metadata={
                "suite": suite_name,
                "potency": effective_potency,
                "original_potency": potency,
                "mode": mode.value,
                "provider": provider,
                "output_format": strategy.get("output_format", "conversational"),
                "estimated_cost_usd": estimated_cost,
                "config_driven": True,
            },
        )

    def get_random_prompt(self, dataset_name: str) -> str | None:
        """
        Get a random prompt from a loaded dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'GPTFuzz', 'PAIR')

        Returns:
            Random prompt string or None if dataset not available
        """
        if not self.dataset_loader:
            logger.warning("DatasetLoader not initialized")
            return None

        return self.dataset_loader.get_random_prompt(dataset_name)

    def get_dataset_stats(self) -> dict[str, Any]:
        """
        Get statistics about loaded datasets.

        Returns:
            Dict with dataset statistics
        """
        if not self.dataset_loader:
            return {"error": "DatasetLoader not initialized", "datasets": {}}

        stats = {"loaded_techniques": self.dataset_loader.loaded_techniques, "datasets": {}}

        for name, prompts in self.dataset_loader.datasets.items():
            stats["datasets"][name] = {
                "count": len(prompts),
                "sample": prompts[0][:100] + "..." if prompts else None,
            }

        return stats

    def get_cost_summary(self) -> dict[str, Any]:
        """
        Get summary of accumulated transformation costs.

        Returns:
            Dict with cost tracking information
        """
        return {
            "total_cost_usd": sum(self._cost_tracker.values()),
            "by_provider": dict(self._cost_tracker),
            "transformation_count": len(self._cost_tracker),
        }

    def reset_cost_tracking(self) -> None:
        """Reset the cost tracking data."""
        self._cost_tracker.clear()

    def get_available_providers(self) -> list[str]:
        """
        Get list of available providers from config.

        Returns:
            List of provider names
        """
        config_manager = self._get_config_manager()
        if not config_manager:
            return list(self.PROVIDER_TRANSFORMATION_STRATEGIES.keys())

        try:
            config = config_manager.get_config()
            return [p.provider_id for p in config.get_enabled_providers()]
        except Exception as e:
            logger.error(f"Error getting providers: {e}")
            return list(self.PROVIDER_TRANSFORMATION_STRATEGIES.keys())

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized


# Global singleton instance
metamorph_service = MetamorphService()
