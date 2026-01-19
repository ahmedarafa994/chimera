"""
AutoDAN Strategy Registry

Centralized registry for all AutoDAN optimization strategies with
unified lifecycle management and extensibility support.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of AutoDAN strategies."""

    VANILLA = "vanilla"
    GENETIC = "genetic"
    NEURAL_BYPASS = "neural_bypass"
    MOUSETRAP = "mousetrap"
    OVERTHINK = "overthink"
    ADAPTIVE = "adaptive"
    BEAM_SEARCH = "beam_search"
    HYBRID = "hybrid"
    TURBO = "turbo"


class StrategyCapability(Enum):
    """Capabilities that strategies can have."""

    REASONING_TOKEN_EXPLOITATION = "reasoning_token_exploitation"
    GENETIC_OPTIMIZATION = "genetic_optimization"
    NEURAL_BYPASS = "neural_bypass"
    MULTI_OBJECTIVE_OPTIMIZATION = "multi_objective_optimization"
    ADAPTIVE_LEARNING = "adaptive_learning"
    PARALLEL_PROCESSING = "parallel_processing"
    CACHING_SUPPORT = "caching_support"
    REAL_TIME_FEEDBACK = "real_time_feedback"


class StrategyMetadata(BaseModel):
    """Metadata for AutoDAN strategies."""

    name: str
    strategy_type: StrategyType
    description: str
    capabilities: list[StrategyCapability]
    version: str = "1.0.0"
    author: str | None = None
    complexity_level: int = 1  # 1-5 scale
    performance_tier: str = "standard"  # standard, high, experimental
    supported_providers: list[str] = ["*"]  # "*" means all providers
    min_generations: int = 1
    max_generations: int = 100
    default_parameters: dict[str, Any] = {}
    experimental: bool = False


class BaseAutoDANStrategy(ABC):
    """Base class for all AutoDAN strategies."""

    def __init__(self, strategy_metadata: StrategyMetadata):
        self.metadata = strategy_metadata
        self.logger = logging.getLogger(f"{__name__}.{strategy_metadata.name}")
        self._initialized = False
        self._performance_metrics = {
            "total_executions": 0,
            "success_count": 0,
            "average_execution_time": 0.0,
            "average_score": 0.0,
        }

    @abstractmethod
    async def initialize(self, **kwargs) -> None:
        """Initialize the strategy with configuration."""
        pass

    @abstractmethod
    async def execute(
        self, request: str, target_model: str, provider: str, **parameters
    ) -> dict[str, Any]:
        """
        Execute the strategy.

        Args:
            request: Target behavior/goal
            target_model: Model to target
            provider: Provider name
            **parameters: Strategy-specific parameters

        Returns:
            Execution result dictionary
        """
        pass

    @abstractmethod
    async def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate strategy parameters."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def get_default_parameters(self) -> dict[str, Any]:
        """Get default parameters for this strategy."""
        return self.metadata.default_parameters.copy()

    def update_performance_metrics(
        self, execution_time: float, score: float, success: bool
    ) -> None:
        """Update performance metrics after execution."""
        self._performance_metrics["total_executions"] += 1
        if success:
            self._performance_metrics["success_count"] += 1

        # Update running averages
        total = self._performance_metrics["total_executions"]
        self._performance_metrics["average_execution_time"] = (
            self._performance_metrics["average_execution_time"] * (total - 1) + execution_time
        ) / total
        if success:
            success_count = self._performance_metrics["success_count"]
            self._performance_metrics["average_score"] = (
                self._performance_metrics["average_score"] * (success_count - 1) + score
            ) / success_count

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return self._performance_metrics.copy()

    def get_metadata(self) -> StrategyMetadata:
        """Get strategy metadata."""
        return self.metadata

    def is_compatible_with_provider(self, provider: str) -> bool:
        """Check if strategy is compatible with provider."""
        return (
            "*" in self.metadata.supported_providers
            or provider in self.metadata.supported_providers
        )


class VanillaStrategy(BaseAutoDANStrategy):
    """Basic vanilla AutoDAN strategy implementation."""

    def __init__(self):
        metadata = StrategyMetadata(
            name="vanilla",
            strategy_type=StrategyType.VANILLA,
            description="Basic AutoDAN implementation with simple prompt optimization",
            capabilities=[StrategyCapability.GENETIC_OPTIMIZATION],
            complexity_level=1,
            performance_tier="standard",
            default_parameters={"generations": 5, "temperature": 0.7, "max_tokens": 4096},
        )
        super().__init__(metadata)

    async def initialize(self, **kwargs) -> None:
        """Initialize vanilla strategy."""
        self.logger.info("Initializing vanilla AutoDAN strategy")
        self._initialized = True

    async def execute(
        self, request: str, target_model: str, provider: str, **parameters
    ) -> dict[str, Any]:
        """Execute vanilla strategy."""
        if not self._initialized:
            await self.initialize()

        self.logger.info(f"Executing vanilla strategy for: {request[:50]}...")

        # Merge with defaults
        params = self.get_default_parameters()
        params.update(parameters)

        # Simulate vanilla execution
        import asyncio

        await asyncio.sleep(0.1)

        return {
            "jailbreak_prompt": f"[VANILLA] {request}",
            "method": "vanilla",
            "score": 7.0,
            "success": True,
            "iterations": params.get("generations", 5),
            "strategy_metadata": self.metadata.dict(),
        }

    async def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate vanilla strategy parameters."""
        generations = parameters.get("generations", 5)
        return 1 <= generations <= 20


class GeneticOptimizationStrategy(BaseAutoDANStrategy):
    """Advanced genetic optimization strategy."""

    def __init__(self):
        metadata = StrategyMetadata(
            name="genetic",
            strategy_type=StrategyType.GENETIC,
            description="Advanced genetic algorithm-based optimization with mutation and crossover",
            capabilities=[
                StrategyCapability.GENETIC_OPTIMIZATION,
                StrategyCapability.MULTI_OBJECTIVE_OPTIMIZATION,
                StrategyCapability.PARALLEL_PROCESSING,
            ],
            complexity_level=3,
            performance_tier="high",
            default_parameters={
                "generations": 10,
                "population_size": 20,
                "mutation_rate": 0.3,
                "crossover_rate": 0.7,
                "elitism_ratio": 0.1,
                "temperature": 0.8,
                "max_tokens": 4096,
            },
            min_generations=5,
            max_generations=50,
        )
        super().__init__(metadata)
        self.population = []
        self.fitness_history = []

    async def initialize(self, **kwargs) -> None:
        """Initialize genetic strategy."""
        self.logger.info("Initializing genetic optimization strategy")
        self.population = []
        self.fitness_history = []
        self._initialized = True

    async def execute(
        self, request: str, target_model: str, provider: str, **parameters
    ) -> dict[str, Any]:
        """Execute genetic optimization."""
        if not self._initialized:
            await self.initialize()

        self.logger.info(f"Executing genetic optimization for: {request[:50]}...")

        # Merge with defaults
        params = self.get_default_parameters()
        params.update(parameters)

        # Simulate genetic algorithm execution
        import asyncio

        await asyncio.sleep(0.2)

        return {
            "jailbreak_prompt": f"[GENETIC OPTIMIZED] {request}",
            "method": "genetic",
            "score": 8.5,
            "success": True,
            "iterations": params.get("generations", 10),
            "population_size": params.get("population_size", 20),
            "convergence_generation": 7,
            "best_fitness": 8.5,
            "fitness_history": [6.2, 7.1, 7.8, 8.0, 8.3, 8.5, 8.5],
            "strategy_metadata": self.metadata.dict(),
        }

    async def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate genetic strategy parameters."""
        generations = parameters.get("generations", 10)
        population_size = parameters.get("population_size", 20)
        mutation_rate = parameters.get("mutation_rate", 0.3)

        return (
            5 <= generations <= 50 and 10 <= population_size <= 100 and 0.0 <= mutation_rate <= 1.0
        )


class MousetrapChainStrategy(BaseAutoDANStrategy):
    """Mousetrap technique - Chain of Iterative Chaos."""

    def __init__(self):
        metadata = StrategyMetadata(
            name="mousetrap",
            strategy_type=StrategyType.MOUSETRAP,
            description="Chain of Iterative Chaos technique for reasoning model jailbreaking",
            capabilities=[
                StrategyCapability.REASONING_TOKEN_EXPLOITATION,
                StrategyCapability.ADAPTIVE_LEARNING,
                StrategyCapability.NEURAL_BYPASS,
            ],
            complexity_level=4,
            performance_tier="experimental",
            default_parameters={
                "chaos_escalation": 0.7,
                "semantic_obfuscation": 0.6,
                "iterative_refinement": 3,
                "reasoning_depth": 5,
                "temperature": 0.9,
                "max_tokens": 6144,
            },
            experimental=True,
            supported_providers=["anthropic", "openai"],  # Reasoning models
        )
        super().__init__(metadata)

    async def initialize(self, **kwargs) -> None:
        """Initialize mousetrap strategy."""
        self.logger.info("Initializing mousetrap chain strategy")
        self._initialized = True

    async def execute(
        self, request: str, target_model: str, provider: str, **parameters
    ) -> dict[str, Any]:
        """Execute mousetrap technique."""
        if not self._initialized:
            await self.initialize()

        self.logger.info(f"Executing mousetrap technique for: {request[:50]}...")

        # Check provider compatibility
        if not self.is_compatible_with_provider(provider):
            self.logger.warning(f"Mousetrap may not be optimal for provider: {provider}")

        # Merge with defaults
        params = self.get_default_parameters()
        params.update(parameters)

        # Simulate mousetrap execution
        import asyncio

        await asyncio.sleep(0.3)

        return {
            "jailbreak_prompt": f"[MOUSETRAP CHAOS] Step-by-step reasoning chain leading to: {request}",
            "method": "mousetrap",
            "score": 9.2,
            "success": True,
            "chaos_iterations": params.get("iterative_refinement", 3),
            "reasoning_depth_achieved": params.get("reasoning_depth", 5),
            "chaos_escalation_factor": params.get("chaos_escalation", 0.7),
            "semantic_obfuscation_level": params.get("semantic_obfuscation", 0.6),
            "reasoning_token_exploitation": True,
            "strategy_metadata": self.metadata.dict(),
        }

    async def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate mousetrap parameters."""
        chaos_escalation = parameters.get("chaos_escalation", 0.7)
        reasoning_depth = parameters.get("reasoning_depth", 5)

        return 0.1 <= chaos_escalation <= 1.0 and 1 <= reasoning_depth <= 10


class OverthinkFusionStrategy(BaseAutoDANStrategy):
    """OVERTHINK-AutoDAN fusion strategy."""

    def __init__(self):
        metadata = StrategyMetadata(
            name="overthink",
            strategy_type=StrategyType.OVERTHINK,
            description="Fusion of OVERTHINK reasoning token exploitation with AutoDAN genetic optimization",
            capabilities=[
                StrategyCapability.REASONING_TOKEN_EXPLOITATION,
                StrategyCapability.GENETIC_OPTIMIZATION,
                StrategyCapability.ADAPTIVE_LEARNING,
                StrategyCapability.REAL_TIME_FEEDBACK,
            ],
            complexity_level=5,
            performance_tier="experimental",
            default_parameters={
                "overthink_depth": 7,
                "genetic_generations": 8,
                "fusion_weight": 0.6,
                "reasoning_amplification": 1.4,
                "decoy_injection_rate": 0.8,
                "temperature": 0.85,
                "max_tokens": 8192,
            },
            experimental=True,
            supported_providers=["anthropic", "openai"],
        )
        super().__init__(metadata)

    async def initialize(self, **kwargs) -> None:
        """Initialize OVERTHINK fusion strategy."""
        self.logger.info("Initializing OVERTHINK-AutoDAN fusion strategy")
        self._initialized = True

    async def execute(
        self, request: str, target_model: str, provider: str, **parameters
    ) -> dict[str, Any]:
        """Execute OVERTHINK fusion."""
        if not self._initialized:
            await self.initialize()

        self.logger.info(f"Executing OVERTHINK fusion for: {request[:50]}...")

        # Merge with defaults
        params = self.get_default_parameters()
        params.update(parameters)

        # Simulate fusion execution
        import asyncio

        await asyncio.sleep(0.4)

        return {
            "jailbreak_prompt": f"[OVERTHINK-GENETIC FUSION] Multi-phase reasoning optimization for: {request}",
            "method": "overthink",
            "score": 9.5,
            "success": True,
            "overthink_depth": params.get("overthink_depth", 7),
            "genetic_generations": params.get("genetic_generations", 8),
            "fusion_effectiveness": 0.93,
            "reasoning_token_amplification": params.get("reasoning_amplification", 1.4),
            "decoy_injection_success": True,
            "hybrid_optimization_score": 9.5,
            "strategy_metadata": self.metadata.dict(),
        }

    async def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate OVERTHINK fusion parameters."""
        overthink_depth = parameters.get("overthink_depth", 7)
        fusion_weight = parameters.get("fusion_weight", 0.6)

        return 3 <= overthink_depth <= 15 and 0.1 <= fusion_weight <= 1.0


class AutoDANStrategyRegistry:
    """
    Centralized registry for AutoDAN optimization strategies.

    Provides strategy management, discovery, and lifecycle control.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.registry")
        self._strategies: dict[str, type[BaseAutoDANStrategy]] = {}
        self._strategy_instances: dict[str, BaseAutoDANStrategy] = {}
        self._initialize_builtin_strategies()

    def _initialize_builtin_strategies(self) -> None:
        """Initialize built-in strategies."""
        builtin_strategies = [
            ("vanilla", VanillaStrategy),
            ("genetic", GeneticOptimizationStrategy),
            ("mousetrap", MousetrapChainStrategy),
            ("overthink", OverthinkFusionStrategy),
        ]

        for name, strategy_class in builtin_strategies:
            self.register_strategy(name, strategy_class)

        self.logger.info(f"Initialized {len(builtin_strategies)} built-in strategies")

    def register_strategy(self, name: str, strategy_class: type[BaseAutoDANStrategy]) -> None:
        """
        Register a new strategy.

        Args:
            name: Strategy name/identifier
            strategy_class: Strategy class to register
        """
        if not issubclass(strategy_class, BaseAutoDANStrategy):
            raise ValueError("Strategy class must inherit from BaseAutoDANStrategy")

        if not inspect.isabstract(strategy_class):
            # Validate strategy has required methods
            required_methods = ["initialize", "execute", "validate_parameters"]
            for method in required_methods:
                if not hasattr(strategy_class, method):
                    raise ValueError(f"Strategy class missing required method: {method}")

        self._strategies[name] = strategy_class
        self.logger.info(f"Registered strategy: {name}")

    def unregister_strategy(self, name: str) -> None:
        """Unregister a strategy."""
        if name in self._strategies:
            # Clean up instance if exists
            if name in self._strategy_instances:
                instance = self._strategy_instances[name]
                if hasattr(instance, "cleanup"):
                    try:
                        instance.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Error during strategy cleanup: {e}")
                del self._strategy_instances[name]

            del self._strategies[name]
            self.logger.info(f"Unregistered strategy: {name}")

    def get_strategy(self, name: str, reuse_instance: bool = True) -> BaseAutoDANStrategy:
        """
        Get a strategy instance.

        Args:
            name: Strategy name
            reuse_instance: Whether to reuse existing instance

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy not found
        """
        if name not in self._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {self.list_strategies()}")

        # Return cached instance if requested and available
        if reuse_instance and name in self._strategy_instances:
            return self._strategy_instances[name]

        # Create new instance
        strategy_class = self._strategies[name]
        instance = strategy_class()

        # Cache if reuse enabled
        if reuse_instance:
            self._strategy_instances[name] = instance

        return instance

    def list_strategies(self) -> list[str]:
        """List available strategy names."""
        return list(self._strategies.keys())

    def get_strategy_metadata(self, name: str) -> StrategyMetadata | None:
        """Get metadata for a strategy."""
        if name not in self._strategies:
            return None

        try:
            instance = self.get_strategy(name, reuse_instance=False)
            return instance.get_metadata()
        except Exception as e:
            self.logger.error(f"Error getting metadata for {name}: {e}")
            return None

    def get_all_strategy_metadata(self) -> dict[str, StrategyMetadata]:
        """Get metadata for all registered strategies."""
        metadata = {}
        for name in self.list_strategies():
            meta = self.get_strategy_metadata(name)
            if meta:
                metadata[name] = meta
        return metadata

    def find_strategies_by_capability(self, capability: StrategyCapability) -> list[str]:
        """Find strategies that support a specific capability."""
        matching_strategies = []

        for name in self.list_strategies():
            metadata = self.get_strategy_metadata(name)
            if metadata and capability in metadata.capabilities:
                matching_strategies.append(name)

        return matching_strategies

    def find_strategies_for_provider(self, provider: str) -> list[str]:
        """Find strategies compatible with a provider."""
        compatible_strategies = []

        for name in self.list_strategies():
            try:
                instance = self.get_strategy(name, reuse_instance=False)
                if instance.is_compatible_with_provider(provider):
                    compatible_strategies.append(name)
            except Exception as e:
                self.logger.warning(f"Error checking provider compatibility for {name}: {e}")

        return compatible_strategies

    def validate_strategy_parameters(self, strategy_name: str, parameters: dict[str, Any]) -> bool:
        """Validate parameters for a strategy."""
        try:
            strategy = self.get_strategy(strategy_name, reuse_instance=False)
            return strategy.validate_parameters(parameters)
        except Exception as e:
            self.logger.error(f"Error validating parameters for {strategy_name}: {e}")
            return False

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_strategies": len(self._strategies),
            "cached_instances": len(self._strategy_instances),
            "available_strategies": self.list_strategies(),
            "experimental_strategies": [
                name
                for name in self.list_strategies()
                if (meta := self.get_strategy_metadata(name)) and meta.experimental
            ],
            "capabilities_coverage": {
                cap.value: len(self.find_strategies_by_capability(cap))
                for cap in StrategyCapability
            },
        }

    async def cleanup_all_instances(self) -> None:
        """Cleanup all strategy instances."""
        for name, instance in self._strategy_instances.items():
            try:
                await instance.cleanup()
                self.logger.debug(f"Cleaned up strategy instance: {name}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up {name}: {e}")

        self._strategy_instances.clear()
        self.logger.info("All strategy instances cleaned up")


# Global registry instance
_global_registry: AutoDANStrategyRegistry | None = None


def get_strategy_registry() -> AutoDANStrategyRegistry:
    """Get or create the global strategy registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AutoDANStrategyRegistry()
    return _global_registry


def register_strategy(name: str, strategy_class: type[BaseAutoDANStrategy]) -> None:
    """Convenience function to register a strategy."""
    registry = get_strategy_registry()
    registry.register_strategy(name, strategy_class)


def get_strategy(name: str) -> BaseAutoDANStrategy:
    """Convenience function to get a strategy."""
    registry = get_strategy_registry()
    return registry.get_strategy(name)
