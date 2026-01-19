"""
AutoDAN Reasoning Integration Module

Provides integration between AutoDAN genetic mutations and the unified
reasoning chain extension framework. This enables coordinated attacks
that combine jailbreak optimization with reasoning resource exhaustion.

Key Components:
- AutoDANReasoningWorkflow: Orchestrates combined reasoning + mutation workflow
- ReasoningEnhancedMutation: Mutations with embedded reasoning complexity
- MutationWithReasoningResult: Combined result type

Mathematical Model:
    Combined Attack: Q' = R_extend(M_autodan(Q, μ, S), ρ)
    Where:
    - M_autodan: Genetic mutation for jailbreak
    - R_extend: Reasoning chain extension for resource exhaustion
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Import from unified framework
from meta_prompter.unified.engine import CombinedAttackEngine, MutationEngine
from meta_prompter.unified.math_framework import (
    AttackComposer,
    CombinedAttackMetrics,
    MutationOptimizer,
    ReasoningChainExtender,
    TokenExtensionOptimizer,
    UnifiedAttackParameters,
    UnifiedFitnessCalculator,
)
from meta_prompter.unified.models import CompositionStrategy, UnifiedAttackConfig

logger = logging.getLogger("chimera.autodan.reasoning")


@dataclass
class MutationWithReasoningResult:
    """
    Result combining mutation and reasoning extension.

    Attributes:
        result_id: Unique identifier
        original_prompt: Original input prompt
        mutated_prompt: Prompt after AutoDAN mutation
        reasoning_extended_prompt: Final prompt after reasoning extension
        mutation_fitness: Fitness from mutation phase
        reasoning_extension_ratio: Extension ratio from reasoning phase
        combined_fitness: Overall unified fitness score
        mutation_metrics: Detailed mutation metrics
        reasoning_metrics: Detailed reasoning metrics
        execution_time_ms: Total execution time
    """

    result_id: str
    original_prompt: str
    mutated_prompt: str
    reasoning_extended_prompt: str
    mutation_fitness: float = 0.0
    reasoning_extension_ratio: float = 1.0
    combined_fitness: float = 0.0
    mutation_metrics: dict[str, Any] = field(default_factory=dict)
    reasoning_metrics: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    success: bool = False
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "original_prompt": self.original_prompt,
            "mutated_prompt": self.mutated_prompt,
            "reasoning_extended_prompt": self.reasoning_extended_prompt,
            "mutation_fitness": self.mutation_fitness,
            "reasoning_extension_ratio": self.reasoning_extension_ratio,
            "combined_fitness": self.combined_fitness,
            "mutation_metrics": self.mutation_metrics,
            "reasoning_metrics": self.reasoning_metrics,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class ReasoningEnhancedMutation:
    """
    Mutation engine enhanced with reasoning chain capabilities.

    Combines AutoDAN-style genetic mutations with reasoning complexity
    injection to maximize both jailbreak success and resource exhaustion.
    """

    def __init__(
        self,
        mutation_strength: float = 0.5,
        complexity_type: str = "mathematical",
        enable_token_optimization: bool = True,
    ):
        """
        Initialize reasoning-enhanced mutation.

        Args:
            mutation_strength: Intensity of mutations (μ ∈ [0, 1])
            complexity_type: Type of reasoning complexity to inject
            enable_token_optimization: Whether to optimize token extension
        """
        self._mutation_engine = MutationEngine()
        self._reasoning_extender = ReasoningChainExtender()
        self._token_optimizer = TokenExtensionOptimizer(
            accuracy_threshold=0.95,
            detection_threshold=0.3,
        )
        self._fitness_calculator = UnifiedFitnessCalculator(
            resource_weight=0.4,
            jailbreak_weight=0.4,
            stealth_weight=0.2,
        )

        self.mutation_strength = mutation_strength
        self.complexity_type = complexity_type
        self.enable_token_optimization = enable_token_optimization

    def mutate_with_reasoning(
        self,
        prompt: str,
        mutation_type: str = "semantic",
        inject_complexity: bool = True,
    ) -> MutationWithReasoningResult:
        """
        Apply mutation with reasoning chain extension.

        Steps:
        1. Apply AutoDAN mutation to prompt
        2. Inject reasoning complexity
        3. Optimize token extension if enabled
        4. Calculate combined fitness

        Args:
            prompt: Original prompt to transform
            mutation_type: Type of mutation to apply
            inject_complexity: Whether to inject reasoning complexity

        Returns:
            MutationWithReasoningResult with all metrics
        """
        import time

        result_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Step 1: Apply AutoDAN mutation
            mutation_result = self._mutation_engine.mutate(prompt, mutation_type=mutation_type)
            mutated_prompt = mutation_result.mutated_prompt

            # Step 2: Apply reasoning chain extension
            reasoning_extended = mutated_prompt
            reasoning_metrics: dict[str, Any] = {}

            if inject_complexity:
                reasoning_extended = self._reasoning_extender.inject_complexity(
                    query=mutated_prompt,
                    complexity_type=self.complexity_type,
                    intensity=self.mutation_strength,
                )

                # Calculate extension ratio
                extension_ratio = self._reasoning_extender.estimate_reasoning_extension(
                    query=mutated_prompt,
                    obfuscation_ratio=0.0,
                    complexity_type=self.complexity_type,
                )

                reasoning_metrics = {
                    "complexity_type": self.complexity_type,
                    "extension_ratio": extension_ratio,
                    "original_length": len(mutated_prompt),
                    "extended_length": len(reasoning_extended),
                    "tokens_added": len(reasoning_extended) - len(mutated_prompt),
                }
            else:
                extension_ratio = 1.0
                reasoning_metrics = {"extension_ratio": 1.0, "complexity_injected": False}

            # Step 3: Token optimization if enabled
            if self.enable_token_optimization:
                opt_result = self._token_optimizer.calculate_expected_amplification(
                    query_length=len(reasoning_extended),
                    obfuscation_ratio=0.0,
                )
                reasoning_metrics["expected_amplification"] = opt_result

            # Step 4: Calculate combined fitness
            combined_metrics = CombinedAttackMetrics(
                attack_success_rate=mutation_result.fitness_score,
                jailbreak_score=mutation_result.fitness_score * 10,  # Scale to 1-10
                token_amplification=extension_ratio,
                reasoning_extension_ratio=extension_ratio,
                detection_probability=1.0 - mutation_result.stealth_score,
            )
            combined_fitness = self._fitness_calculator.calculate_unified_fitness(combined_metrics)

            execution_time = (time.time() - start_time) * 1000

            return MutationWithReasoningResult(
                result_id=result_id,
                original_prompt=prompt,
                mutated_prompt=mutated_prompt,
                reasoning_extended_prompt=reasoning_extended,
                mutation_fitness=mutation_result.fitness_score,
                reasoning_extension_ratio=extension_ratio,
                combined_fitness=combined_fitness,
                mutation_metrics={
                    "mutation_type": mutation_type,
                    "original_fitness": mutation_result.fitness_score,
                    "stealth_score": mutation_result.stealth_score,
                },
                reasoning_metrics=reasoning_metrics,
                execution_time_ms=execution_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"Reasoning-enhanced mutation failed: {e}")
            return MutationWithReasoningResult(
                result_id=result_id,
                original_prompt=prompt,
                mutated_prompt="",
                reasoning_extended_prompt="",
                success=False,
                error=str(e),
            )


class AutoDANReasoningWorkflow:
    """
    Orchestrates the complete AutoDAN + Reasoning workflow.

    This workflow combines:
    1. AutoDAN genetic optimization for jailbreak success
    2. Reasoning chain extension for resource exhaustion
    3. Unified fitness evaluation for multi-objective optimization

    Workflow Stages:
    - Stage 1: Population initialization with reasoning-aware mutations
    - Stage 2: Genetic evolution with combined fitness
    - Stage 3: Best candidate selection using Pareto dominance
    - Stage 4: Final reasoning enhancement on selected candidate
    """

    def __init__(
        self,
        config: UnifiedAttackConfig | None = None,
        mutation_strength: float = 0.5,
        complexity_type: str = "mathematical",
    ):
        """
        Initialize the reasoning workflow.

        Args:
            config: Unified attack configuration
            mutation_strength: Intensity of mutations
            complexity_type: Type of reasoning complexity
        """
        self.config = config or UnifiedAttackConfig.for_lrm_targets()

        # Initialize components
        self._engine = CombinedAttackEngine(self.config)
        self._mutation_engine = MutationEngine()
        self._reasoning_extender = ReasoningChainExtender()
        self._mutation_optimizer = MutationOptimizer(token_amp_weight=0.3)
        self._fitness_calculator = UnifiedFitnessCalculator(
            resource_weight=0.4,
            jailbreak_weight=0.4,
            stealth_weight=0.2,
        )
        self._attack_composer = AttackComposer(strategy="extend_first")

        self.mutation_strength = mutation_strength
        self.complexity_type = complexity_type

        # State tracking
        self._fitness_history: list[float] = []
        self._current_params: UnifiedAttackParameters | None = None

        logger.info("AutoDANReasoningWorkflow initialized")

    def execute_workflow(
        self,
        prompt: str,
        strategy: CompositionStrategy = CompositionStrategy.ADAPTIVE,
        max_iterations: int = 50,
        fitness_threshold: float = 0.85,
    ) -> MutationWithReasoningResult:
        """
        Execute the complete reasoning-enhanced AutoDAN workflow.

        Args:
            prompt: Original prompt to transform
            strategy: Composition strategy for combined attack
            max_iterations: Maximum iterations for optimization
            fitness_threshold: Target fitness to stop optimization

        Returns:
            MutationWithReasoningResult with optimized prompt
        """
        import time

        result_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Initialize parameters
            self._current_params = self._create_params()

            # Stage 1: Initial mutation with reasoning
            enhanced_mutation = ReasoningEnhancedMutation(
                mutation_strength=self.mutation_strength,
                complexity_type=self.complexity_type,
            )

            best_result = enhanced_mutation.mutate_with_reasoning(prompt)
            best_fitness = best_result.combined_fitness
            self._fitness_history.append(best_fitness)

            # Stage 2: Iterative optimization
            for iteration in range(max_iterations):
                # Apply combined attack using engine
                if strategy == CompositionStrategy.EXTEND_FIRST:
                    attack_result = self._engine.sequential_attack(
                        best_result.reasoning_extended_prompt,
                        extend_first=True,
                    )
                elif strategy == CompositionStrategy.AUTODAN_FIRST:
                    attack_result = self._engine.sequential_attack(
                        best_result.reasoning_extended_prompt,
                        extend_first=False,
                    )
                elif strategy == CompositionStrategy.PARALLEL:
                    attack_result = self._engine.parallel_attack(
                        best_result.reasoning_extended_prompt,
                        blend_weight=0.5,
                    )
                else:
                    attack_result = self._engine.adaptive_attack(
                        best_result.reasoning_extended_prompt,
                    )

                # Evaluate combined fitness
                current_fitness = attack_result.unified_fitness
                self._fitness_history.append(current_fitness)

                # Update best if improved
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_result = MutationWithReasoningResult(
                        result_id=result_id,
                        original_prompt=prompt,
                        mutated_prompt=(
                            attack_result.mutation_result.mutated_prompt
                            if attack_result.mutation_result
                            else ""
                        ),
                        reasoning_extended_prompt=attack_result.final_prompt,
                        mutation_fitness=(
                            attack_result.mutation_result.fitness_score
                            if attack_result.mutation_result
                            else 0.0
                        ),
                        reasoning_extension_ratio=len(attack_result.final_prompt)
                        / max(1, len(prompt)),
                        combined_fitness=current_fitness,
                        success=True,
                    )

                # Check termination
                if current_fitness >= fitness_threshold:
                    logger.info(f"Fitness threshold reached at iteration {iteration}")
                    break

                # Optimize parameters for next iteration
                if len(self._fitness_history) >= 3:
                    self._current_params = self._mutation_optimizer.optimize_mutation_params(
                        base_query=prompt,
                        fitness_history=self._fitness_history,
                        current_params=self._current_params,
                    )

            # Stage 3: Final reasoning enhancement
            final_prompt = self._reasoning_extender.inject_complexity(
                query=best_result.reasoning_extended_prompt,
                complexity_type=self.complexity_type,
                intensity=self.mutation_strength,
            )

            execution_time = (time.time() - start_time) * 1000

            return MutationWithReasoningResult(
                result_id=result_id,
                original_prompt=prompt,
                mutated_prompt=best_result.mutated_prompt,
                reasoning_extended_prompt=final_prompt,
                mutation_fitness=best_result.mutation_fitness,
                reasoning_extension_ratio=len(final_prompt) / max(1, len(prompt)),
                combined_fitness=best_fitness,
                mutation_metrics={
                    "iterations": len(self._fitness_history),
                    "strategy": strategy.value,
                    "final_params": self._current_params.to_dict() if self._current_params else {},
                },
                reasoning_metrics={
                    "complexity_type": self.complexity_type,
                    "fitness_history": self._fitness_history,
                    "improvement": (
                        best_fitness - self._fitness_history[0] if self._fitness_history else 0.0
                    ),
                },
                execution_time_ms=execution_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"Reasoning workflow failed: {e}")
            return MutationWithReasoningResult(
                result_id=result_id,
                original_prompt=prompt,
                mutated_prompt="",
                reasoning_extended_prompt="",
                success=False,
                error=str(e),
            )

    def get_fitness_history(self) -> list[float]:
        """Get fitness history from workflow execution."""
        return self._fitness_history.copy()

    def get_current_parameters(self) -> UnifiedAttackParameters | None:
        """Get current optimized parameters."""
        return self._current_params

    def reset(self) -> None:
        """Reset workflow state."""
        self._fitness_history.clear()
        self._current_params = None

    def _create_params(self) -> UnifiedAttackParameters:
        """Create initial parameters."""
        return UnifiedAttackParameters(
            mutation_strength=self.mutation_strength,
            population_size=50,
            max_generations=100,
            crossover_rate=0.8,
            mutation_rate=0.3,
            obfuscation_ratio=0.0,
            attack_vector_weight=0.5,
            jailbreak_priority=0.6,
            resource_priority=0.4,
        )


# Convenience factory functions
def create_reasoning_workflow(
    mutation_strength: float = 0.5,
    complexity_type: str = "mathematical",
) -> AutoDANReasoningWorkflow:
    """
    Create a reasoning workflow with default configuration.

    Args:
        mutation_strength: Mutation intensity
        complexity_type: Type of reasoning complexity

    Returns:
        Configured AutoDANReasoningWorkflow
    """
    return AutoDANReasoningWorkflow(
        mutation_strength=mutation_strength,
        complexity_type=complexity_type,
    )


def quick_reasoning_mutation(
    prompt: str,
    mutation_type: str = "semantic",
    complexity_type: str = "mathematical",
) -> MutationWithReasoningResult:
    """
    Perform a quick reasoning-enhanced mutation.

    Args:
        prompt: Prompt to transform
        mutation_type: Type of mutation
        complexity_type: Type of reasoning complexity

    Returns:
        MutationWithReasoningResult
    """
    enhanced = ReasoningEnhancedMutation(
        mutation_strength=0.5,
        complexity_type=complexity_type,
    )
    return enhanced.mutate_with_reasoning(prompt, mutation_type=mutation_type)
