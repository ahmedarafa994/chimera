"""
AutoDAN Attack Implementation

Wraps the unified framework's MutationEngine to provide a standalone AutoDAN
attack interface with genetic algorithm-based adversarial prompt mutation.

Mathematical Model:
    M_autodan(Q, μ, S) → Q_a

Where:
    - Q: Original input prompt
    - μ: Mutation strength ∈ [0, 1]
    - S: Mutation strategy (semantic, crossover, role_play, etc.)
    - Q_a: Adversarially mutated prompt

Fitness Function:
    F(Q') = w_j * ASR(Q') + w_s * stealth(Q') - w_d * detection(Q')
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Import from unified framework
from meta_prompter.unified.engine import MutationEngine
from meta_prompter.unified.models import MutatedResult, MutationType
from meta_prompter.unified.math_framework import (
    MutationOptimizer,
    ReasoningChainExtender,
    UnifiedAttackParameters,
    UnifiedFitnessCalculator,
)

logger = logging.getLogger("chimera.autodan")


class MutationStrategy(str, Enum):
    """Available mutation strategies for AutoDAN attacks."""

    SEMANTIC = "semantic"
    CROSSOVER = "crossover"
    RANDOM = "random"
    GRADIENT_GUIDED = "gradient_guided"
    ROLE_PLAY = "role_play"
    CONTEXT_INJECTION = "context_injection"
    INSTRUCTION_REWRITE = "instruction_rewrite"


@dataclass
class AutoDANConfig:
    """
    Configuration for AutoDAN attacks.

    Attributes:
        mutation_strength: Intensity of mutations (μ ∈ [0, 1])
        population_size: Number of candidates in genetic population
        max_generations: Maximum evolution iterations
        crossover_rate: Probability of crossover between candidates
        mutation_rate: Probability of random mutation
        elite_count: Number of top candidates to preserve
        fitness_threshold: Target fitness to stop evolution
        default_strategy: Default mutation strategy to use
        enable_reasoning_extension: Whether to apply reasoning chain extension
        complexity_type: Type of complexity for reasoning extension
    """

    mutation_strength: float = 0.5
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    elite_count: int = 5
    fitness_threshold: float = 0.9
    default_strategy: MutationStrategy = MutationStrategy.SEMANTIC
    enable_reasoning_extension: bool = True
    complexity_type: str = "mathematical"

    def validate(self) -> bool:
        """Validate configuration parameters."""
        constraints = [
            0 <= self.mutation_strength <= 1,
            self.population_size > 0,
            self.max_generations > 0,
            0 <= self.crossover_rate <= 1,
            0 <= self.mutation_rate <= 1,
            self.elite_count >= 0,
            self.elite_count <= self.population_size,
            0 <= self.fitness_threshold <= 1,
        ]
        return all(constraints)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mutation_strength": self.mutation_strength,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elite_count": self.elite_count,
            "fitness_threshold": self.fitness_threshold,
            "default_strategy": self.default_strategy.value,
            "enable_reasoning_extension": self.enable_reasoning_extension,
            "complexity_type": self.complexity_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoDANConfig":
        """Create from dictionary."""
        if "default_strategy" in data and isinstance(data["default_strategy"], str):
            data = data.copy()
            data["default_strategy"] = MutationStrategy(data["default_strategy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default(cls) -> "AutoDANConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def aggressive(cls) -> "AutoDANConfig":
        """Create aggressive configuration for maximum jailbreak success."""
        return cls(
            mutation_strength=0.8,
            population_size=100,
            max_generations=200,
            crossover_rate=0.9,
            mutation_rate=0.4,
            elite_count=10,
            fitness_threshold=0.95,
        )

    @classmethod
    def stealthy(cls) -> "AutoDANConfig":
        """Create stealthy configuration for minimal detection."""
        return cls(
            mutation_strength=0.3,
            population_size=30,
            max_generations=50,
            crossover_rate=0.6,
            mutation_rate=0.2,
            elite_count=3,
            fitness_threshold=0.7,
        )


@dataclass
class AutoDANResult:
    """
    Result of an AutoDAN attack execution.

    Attributes:
        attack_id: Unique identifier for this attack
        original_prompt: Original input prompt
        mutated_prompt: Final mutated prompt after evolution
        fitness_score: Final fitness score achieved
        generations_used: Number of generations executed
        mutation_history: History of mutations applied
        population_stats: Statistics from genetic algorithm
        reasoning_metrics: Metrics from reasoning chain extension
        success: Whether attack was successful
        error: Error message if attack failed
    """

    attack_id: str
    original_prompt: str
    mutated_prompt: str
    fitness_score: float = 0.0
    generations_used: int = 0
    mutation_history: list[dict[str, Any]] = field(default_factory=list)
    population_stats: dict[str, Any] = field(default_factory=dict)
    reasoning_metrics: dict[str, Any] | None = None
    success: bool = False
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_id": self.attack_id,
            "original_prompt": self.original_prompt,
            "mutated_prompt": self.mutated_prompt,
            "fitness_score": self.fitness_score,
            "generations_used": self.generations_used,
            "mutation_history": self.mutation_history,
            "population_stats": self.population_stats,
            "reasoning_metrics": self.reasoning_metrics,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class AutoDANAttack:
    """
    AutoDAN Attack Implementation.

    Provides genetic algorithm-based adversarial prompt mutation with
    integration to the unified reasoning framework.

    Usage:
        >>> attack = AutoDANAttack()
        >>> result = attack.execute("Tell me about...")
        >>> print(result.mutated_prompt)

    With configuration:
        >>> config = AutoDANConfig.aggressive()
        >>> attack = AutoDANAttack(config)
        >>> result = attack.execute("...", strategy=MutationStrategy.ROLE_PLAY)
    """

    def __init__(self, config: AutoDANConfig | None = None):
        """
        Initialize AutoDAN attack.

        Args:
            config: Attack configuration (defaults to AutoDANConfig.default())
        """
        self.config = config or AutoDANConfig.default()

        # Initialize unified framework components
        self._engine = MutationEngine()
        self._fitness_calculator = UnifiedFitnessCalculator(
            resource_weight=0.2,
            jailbreak_weight=0.6,
            stealth_weight=0.2,
        )
        self._mutation_optimizer = MutationOptimizer(token_amp_weight=0.3)
        self._reasoning_extender = ReasoningChainExtender()

        # State tracking
        self._fitness_history: list[float] = []
        self._attack_params: UnifiedAttackParameters | None = None

        logger.info(f"AutoDANAttack initialized with config: {self.config.to_dict()}")

    def execute(
        self,
        prompt: str,
        strategy: MutationStrategy | None = None,
        fitness_fn: Callable[[str], float] | None = None,
        target_behavior: str | None = None,
    ) -> AutoDANResult:
        """
        Execute AutoDAN attack on a prompt.

        Args:
            prompt: Original prompt to mutate
            strategy: Mutation strategy to use (defaults to config default)
            fitness_fn: Custom fitness function (optional)
            target_behavior: Target behavior description for fitness evaluation

        Returns:
            AutoDANResult with mutated prompt and metrics
        """
        attack_id = str(uuid.uuid4())
        strategy = strategy or self.config.default_strategy

        try:
            # Apply reasoning extension if enabled
            processed_prompt = prompt
            reasoning_metrics = None

            if self.config.enable_reasoning_extension:
                processed_prompt, reasoning_metrics = self._apply_reasoning_extension(
                    prompt
                )

            # Initialize attack parameters
            self._attack_params = self._create_attack_params()

            # Execute genetic evolution
            best_prompt, best_fitness, generations, history = self._evolve_population(
                processed_prompt,
                strategy,
                fitness_fn,
            )

            # Calculate population statistics
            pop_stats = self._calculate_population_stats(history)

            return AutoDANResult(
                attack_id=attack_id,
                original_prompt=prompt,
                mutated_prompt=best_prompt,
                fitness_score=best_fitness,
                generations_used=generations,
                mutation_history=history,
                population_stats=pop_stats,
                reasoning_metrics=reasoning_metrics,
                success=True,
            )

        except Exception as e:
            logger.error(f"AutoDAN attack failed: {e}")
            return AutoDANResult(
                attack_id=attack_id,
                original_prompt=prompt,
                mutated_prompt="",
                success=False,
                error=str(e),
            )

    def mutate(
        self,
        prompt: str,
        strategy: MutationStrategy | None = None,
    ) -> MutatedResult:
        """
        Apply a single mutation to a prompt.

        Args:
            prompt: Prompt to mutate
            strategy: Mutation strategy to use

        Returns:
            MutatedResult from unified engine
        """
        strategy = strategy or self.config.default_strategy
        mutation_type = self._strategy_to_mutation_type(strategy)

        return self._engine.mutate(prompt, mutation_type=mutation_type.value)

    def batch_mutate(
        self,
        prompts: list[str],
        strategy: MutationStrategy | None = None,
    ) -> list[MutatedResult]:
        """
        Apply mutations to multiple prompts.

        Args:
            prompts: List of prompts to mutate
            strategy: Mutation strategy to use

        Returns:
            List of MutatedResult objects
        """
        return [self.mutate(p, strategy) for p in prompts]

    def crossover(
        self,
        parent1: str,
        parent2: str,
    ) -> tuple[str, str]:
        """
        Perform crossover between two parent prompts.

        Args:
            parent1: First parent prompt
            parent2: Second parent prompt

        Returns:
            Tuple of two offspring prompts
        """
        return self._engine.crossover(parent1, parent2)

    def evaluate_fitness(
        self,
        prompt: str,
        fitness_fn: Callable[[str], float] | None = None,
    ) -> float:
        """
        Evaluate fitness of a prompt.

        Args:
            prompt: Prompt to evaluate
            fitness_fn: Optional custom fitness function

        Returns:
            Fitness score ∈ [0, 1]
        """
        if fitness_fn:
            return fitness_fn(prompt)

        # Default fitness based on mutation result
        result = self.mutate(prompt)
        return result.fitness_score

    def optimize_parameters(self) -> UnifiedAttackParameters:
        """
        Optimize attack parameters based on fitness history.

        Returns:
            Optimized UnifiedAttackParameters
        """
        if not self._attack_params:
            self._attack_params = self._create_attack_params()

        if len(self._fitness_history) >= 3:
            optimized = self._mutation_optimizer.optimize_mutation_params(
                base_query="",
                fitness_history=self._fitness_history,
                current_params=self._attack_params,
            )
            self._attack_params = optimized

        return self._attack_params

    def get_fitness_history(self) -> list[float]:
        """Get fitness history from previous attacks."""
        return self._fitness_history.copy()

    def reset(self) -> None:
        """Reset attack state."""
        self._fitness_history.clear()
        self._attack_params = None

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _apply_reasoning_extension(
        self,
        prompt: str,
    ) -> tuple[str, dict[str, Any]]:
        """Apply reasoning chain extension to prompt."""
        extension_ratio = self._reasoning_extender.estimate_reasoning_extension(
            query=prompt,
            obfuscation_ratio=0.0,  # No obfuscation in pure AutoDAN
            complexity_type=self.config.complexity_type,
        )

        # Optionally inject complexity
        extended_prompt = self._reasoning_extender.inject_complexity(
            query=prompt,
            complexity_type=self.config.complexity_type,
            intensity=self.config.mutation_strength,
        )

        metrics = {
            "extension_ratio": extension_ratio,
            "complexity_type": self.config.complexity_type,
            "original_length": len(prompt),
            "extended_length": len(extended_prompt),
        }

        return extended_prompt, metrics

    def _create_attack_params(self) -> UnifiedAttackParameters:
        """Create UnifiedAttackParameters from config."""
        return UnifiedAttackParameters(
            mutation_strength=self.config.mutation_strength,
            population_size=self.config.population_size,
            max_generations=self.config.max_generations,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            obfuscation_ratio=0.0,  # Pure AutoDAN, no obfuscation
            attack_vector_weight=1.0,  # Full weight on AutoDAN vector
            jailbreak_priority=0.8,
            resource_priority=0.2,
        )

    def _evolve_population(
        self,
        initial_prompt: str,
        strategy: MutationStrategy,
        fitness_fn: Callable[[str], float] | None,
    ) -> tuple[str, float, int, list[dict[str, Any]]]:
        """
        Run genetic algorithm to evolve adversarial prompts.

        Returns:
            Tuple of (best_prompt, best_fitness, generations_used, history)
        """
        # Initialize population
        population = self._initialize_population(initial_prompt)
        history: list[dict[str, Any]] = []
        best_prompt = initial_prompt
        best_fitness = 0.0

        for gen in range(self.config.max_generations):
            # Evaluate fitness
            fitnesses = [self.evaluate_fitness(p, fitness_fn) for p in population]

            # Track best
            gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_prompt = population[gen_best_idx]

            # Record history
            history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "population_size": len(population),
            })

            # Update fitness history for optimizer
            self._fitness_history.append(gen_best_fitness)

            # Check termination
            if best_fitness >= self.config.fitness_threshold:
                logger.info(f"Fitness threshold reached at generation {gen}")
                break

            # Selection, crossover, mutation for next generation
            population = self._evolve_generation(population, fitnesses, strategy)

        return best_prompt, best_fitness, len(history), history

    def _initialize_population(self, seed_prompt: str) -> list[str]:
        """Initialize population with mutations of seed prompt."""
        population = [seed_prompt]

        # Generate initial mutations
        for _ in range(self.config.population_size - 1):
            result = self.mutate(seed_prompt)
            population.append(result.mutated_prompt)

        return population

    def _evolve_generation(
        self,
        population: list[str],
        fitnesses: list[float],
        strategy: MutationStrategy,
    ) -> list[str]:
        """Evolve population to next generation."""
        import random

        new_population: list[str] = []

        # Elite selection - keep top performers
        sorted_indices = sorted(
            range(len(fitnesses)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )
        elite_indices = sorted_indices[: self.config.elite_count]
        for idx in elite_indices:
            new_population.append(population[idx])

        # Tournament selection and breeding
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_select(population, fitnesses)
            parent2 = self._tournament_select(population, fitnesses)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self.mutate(child1, strategy).mutated_prompt
            if random.random() < self.config.mutation_rate:
                child2 = self.mutate(child2, strategy).mutated_prompt

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        return new_population[: self.config.population_size]

    def _tournament_select(
        self,
        population: list[str],
        fitnesses: list[float],
        tournament_size: int = 3,
    ) -> str:
        """Select individual via tournament selection."""
        import random

        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _calculate_population_stats(
        self,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate population statistics from evolution history."""
        if not history:
            return {}

        fitnesses = [h["best_fitness"] for h in history]
        avg_fitnesses = [h["avg_fitness"] for h in history]

        return {
            "total_generations": len(history),
            "final_best_fitness": fitnesses[-1] if fitnesses else 0.0,
            "initial_fitness": fitnesses[0] if fitnesses else 0.0,
            "fitness_improvement": fitnesses[-1] - fitnesses[0] if len(fitnesses) > 1 else 0.0,
            "avg_fitness_trend": avg_fitnesses,
            "best_fitness_trend": fitnesses,
        }

    def _strategy_to_mutation_type(self, strategy: MutationStrategy) -> MutationType:
        """Convert MutationStrategy to unified MutationType."""
        mapping = {
            MutationStrategy.SEMANTIC: MutationType.SEMANTIC,
            MutationStrategy.CROSSOVER: MutationType.CROSSOVER,
            MutationStrategy.RANDOM: MutationType.RANDOM,
            MutationStrategy.GRADIENT_GUIDED: MutationType.GRADIENT_GUIDED,
            MutationStrategy.ROLE_PLAY: MutationType.ROLE_PLAY,
            MutationStrategy.CONTEXT_INJECTION: MutationType.CONTEXT_INJECTION,
            MutationStrategy.INSTRUCTION_REWRITE: MutationType.INSTRUCTION_REWRITE,
        }
        return mapping.get(strategy, MutationType.SEMANTIC)
