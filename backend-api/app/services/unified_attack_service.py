"""
Unified Attack Coordinator Service

Orchestrates multi-vector attack lifecycle and coordination between the unified engine,
evaluation pipeline, and resource tracking components.

Key Responsibilities:
- Session lifecycle management (create, get, finalize)
- Attack execution coordination (sequential, parallel, iterative, adaptive)
- Evaluation pipeline integration
- Resource budget management and allocation
- Pareto-optimal solution tracking
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# Import AutoDAN reasoning workflow
from meta_prompter.attacks.autodan import (
    AutoDANAttack,
    AutoDANConfig,
    AutoDANReasoningWorkflow,
    MutationWithReasoningResult,
    ReasoningEnhancedMutation,
)

# Import from unified framework
from meta_prompter.unified.engine import (
    CombinedAttackEngine,
    CombinedResult,
    MutatedResult,
    ObfuscatedResult,
)
from meta_prompter.unified.evaluation import (
    DualVectorMetrics,
    UnifiedEvaluationPipeline,
)

# Import reasoning/optimization components from math framework
from meta_prompter.unified.math_framework import (
    AttackComposer,
    CombinedAttackMetrics,
    MutationOptimizer,
    ReasoningChainExtender,
    TokenExtensionOptimizer,
    UnifiedAttackParameters,
    UnifiedFitnessCalculator,
)
from meta_prompter.unified.models import (
    CompositionStrategy,
    UnifiedAttackConfig,
)
from meta_prompter.unified.resource_tracking import (
    BudgetStatus,
    ResourceAllocation,
    ResourceBudget,
    SessionSummary,
    UnifiedResourceService,
    UnifiedResourceTracker,
)

logger = logging.getLogger("chimera.unified_attack_service")


# ==============================================================================
# Enums
# ==============================================================================


class SessionStatus(str, Enum):
    """Status of an attack session."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class ReasoningMetrics:
    """Metrics related to reasoning chain extension."""

    original_reasoning_tokens: int = 0
    extended_reasoning_tokens: int = 0
    reasoning_extension_ratio: float = 1.0
    complexity_type: str = "mathematical"
    complexity_injection_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_reasoning_tokens": self.original_reasoning_tokens,
            "extended_reasoning_tokens": self.extended_reasoning_tokens,
            "reasoning_extension_ratio": self.reasoning_extension_ratio,
            "complexity_type": self.complexity_type,
            "complexity_injection_applied": self.complexity_injection_applied,
        }


@dataclass
class MutationMetrics:
    """Metrics related to AutoDAN mutation optimization."""

    generations_used: int = 0
    mutation_rate: float = 0.3
    crossover_rate: float = 0.8
    mutation_strength: float = 0.5
    fitness_improvement: float = 0.0
    population_diversity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generations_used": self.generations_used,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "mutation_strength": self.mutation_strength,
            "fitness_improvement": self.fitness_improvement,
            "population_diversity": self.population_diversity,
        }


@dataclass
class AttackExecutionResult:
    """Result of a single attack execution."""

    attack_id: str
    session_id: str
    strategy: CompositionStrategy

    # Input/Output
    original_query: str
    transformed_query: str

    # Vector-specific results
    obfuscation_result: ObfuscatedResult | None = None
    mutation_result: MutatedResult | None = None
    combined_result: CombinedResult | None = None

    # Metrics
    token_amplification: float = 0.0
    latency_ms: float = 0.0
    estimated_cost: float = 0.0
    unified_fitness: float = 0.0
    stealth_score: float = 0.0
    jailbreak_score: float = 0.0

    # Reasoning metrics
    reasoning_metrics: ReasoningMetrics | None = None
    mutation_metrics: MutationMetrics | None = None

    # Combined attack metrics from math framework
    combined_metrics: CombinedAttackMetrics | None = None

    # Status
    success: bool = False
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_response(self) -> dict[str, Any]:
        """Convert to API response dictionary."""
        response = {
            "attack_id": self.attack_id,
            "session_id": self.session_id,
            "strategy": self.strategy.value if isinstance(self.strategy, Enum) else self.strategy,
            "original_query": self.original_query,
            "transformed_query": self.transformed_query,
            "token_amplification": self.token_amplification,
            "latency_ms": self.latency_ms,
            "estimated_cost": self.estimated_cost,
            "unified_fitness": self.unified_fitness,
            "stealth_score": self.stealth_score,
            "jailbreak_score": self.jailbreak_score,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "has_obfuscation_result": self.obfuscation_result is not None,
            "has_mutation_result": self.mutation_result is not None,
            "has_combined_result": self.combined_result is not None,
        }

        # Include reasoning metrics if available
        if self.reasoning_metrics:
            response["reasoning_metrics"] = self.reasoning_metrics.to_dict()

        # Include mutation metrics if available
        if self.mutation_metrics:
            response["mutation_metrics"] = self.mutation_metrics.to_dict()

        # Include combined metrics if available
        if self.combined_metrics:
            response["combined_metrics"] = self.combined_metrics.to_dict()

        return response


@dataclass
class AttackSessionState:
    """State management for attack sessions."""

    session_id: str
    config: UnifiedAttackConfig
    budget: ResourceBudget
    status: SessionStatus
    created_at: datetime

    # Attack history
    attacks: list[AttackExecutionResult] = field(default_factory=list)
    evaluations: list[DualVectorMetrics] = field(default_factory=list)

    # Statistics
    total_attacks: int = 0
    successful_attacks: int = 0
    best_fitness: float = 0.0

    # Resource tracking
    tokens_consumed: int = 0
    cost_consumed: float = 0.0
    latency_consumed_ms: float = 0.0

    # Optional metadata
    name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_attack(self, result: AttackExecutionResult) -> None:
        """Add an attack result to the session."""
        self.attacks.append(result)
        self.total_attacks += 1
        if result.success:
            self.successful_attacks += 1
            if result.unified_fitness > self.best_fitness:
                self.best_fitness = result.unified_fitness

        # Update resource consumption
        self.tokens_consumed += int(result.token_amplification * 100)  # Estimate
        self.cost_consumed += result.estimated_cost
        self.latency_consumed_ms += result.latency_ms

    def add_evaluation(self, metrics: DualVectorMetrics) -> None:
        """Add evaluation metrics to the session."""
        self.evaluations.append(metrics)
        if metrics.unified_fitness > self.best_fitness:
            self.best_fitness = metrics.unified_fitness

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attacks == 0:
            return 0.0
        return self.successful_attacks / self.total_attacks

    def get_average_fitness(self) -> float:
        """Calculate average fitness across successful attacks."""
        successful = [a for a in self.attacks if a.success]
        if not successful:
            return 0.0
        return sum(a.unified_fitness for a in successful) / len(successful)

    def to_dict(self) -> dict[str, Any]:
        """Convert session state to dictionary."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "total_attacks": self.total_attacks,
            "successful_attacks": self.successful_attacks,
            "success_rate": self.get_success_rate(),
            "best_fitness": self.best_fitness,
            "average_fitness": self.get_average_fitness(),
            "tokens_consumed": self.tokens_consumed,
            "cost_consumed": self.cost_consumed,
            "latency_consumed_ms": self.latency_consumed_ms,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class CombinedResourceUsage:
    """Combined resource usage from both attack vectors."""

    extend_tokens: int = 0
    extend_latency_ms: float = 0.0
    extend_cost: float = 0.0
    extend_requests: int = 0

    autodan_tokens: int = 0
    autodan_latency_ms: float = 0.0
    autodan_cost: float = 0.0
    autodan_requests: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens from both vectors."""
        return self.extend_tokens + self.autodan_tokens

    @property
    def total_latency_ms(self) -> float:
        """Total latency from both vectors."""
        return self.extend_latency_ms + self.autodan_latency_ms

    @property
    def total_cost(self) -> float:
        """Total cost from both vectors."""
        return self.extend_cost + self.autodan_cost

    @property
    def total_requests(self) -> int:
        """Total requests from both vectors."""
        return self.extend_requests + self.autodan_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "extend": {
                "tokens": self.extend_tokens,
                "latency_ms": self.extend_latency_ms,
                "cost": self.extend_cost,
                "requests": self.extend_requests,
            },
            "autodan": {
                "tokens": self.autodan_tokens,
                "latency_ms": self.autodan_latency_ms,
                "cost": self.autodan_cost,
                "requests": self.autodan_requests,
            },
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
        }


# ==============================================================================
# Attack Coordinator Helper
# ==============================================================================


class AttackCoordinator:
    """Coordinates execution flow between engine, evaluator, tracker, and reasoning components."""

    def __init__(
        self,
        engine: CombinedAttackEngine,
        evaluator: UnifiedEvaluationPipeline,
        tracker: UnifiedResourceTracker,
    ):
        self.engine = engine
        self.evaluator = evaluator
        self.tracker = tracker

        # Initialize reasoning and optimization components
        self.reasoning_extender = ReasoningChainExtender()
        self.mutation_optimizer = MutationOptimizer(token_amp_weight=0.3)
        self.token_optimizer = TokenExtensionOptimizer(
            accuracy_threshold=0.95,
            detection_threshold=0.3,
        )
        self.fitness_calculator = UnifiedFitnessCalculator(
            resource_weight=0.4,
            jailbreak_weight=0.4,
            stealth_weight=0.2,
        )
        self.attack_composer = AttackComposer(strategy="extend_first")

        # Track fitness history for mutation optimization
        self._fitness_history: dict[str, list[float]] = {}

    async def coordinate_attack(
        self,
        session: AttackSessionState,
        query: str,
        strategy: CompositionStrategy,
        parameters: dict[str, Any] | None = None,
    ) -> AttackExecutionResult:
        """
        Coordinate full attack execution flow with reasoning integration.

        Steps:
        1. Check budget constraints
        2. Optimize parameters using mutation optimizer
        3. Apply reasoning chain extension
        4. Execute attack via engine
        5. Calculate unified fitness using fitness calculator
        6. Record resource usage
        7. Update session state
        8. Return result with comprehensive metrics
        """
        attack_id = str(uuid.uuid4())
        start_time = time.time()
        parameters = parameters or {}

        # Check budget constraints
        budget_status = self.tracker.check_budget(session.budget)
        if budget_status.is_exhausted:
            return AttackExecutionResult(
                attack_id=attack_id,
                session_id=session.session_id,
                strategy=strategy,
                original_query=query,
                transformed_query="",
                success=False,
                error=f"Budget exhausted: {budget_status.exhausted_reason}",
                timestamp=datetime.utcnow(),
            )

        try:
            # Initialize fitness history for session if needed
            if session.session_id not in self._fitness_history:
                self._fitness_history[session.session_id] = []

            # Get or create attack parameters with optimization
            attack_params = self._optimize_attack_parameters(
                session=session,
                query=query,
                parameters=parameters,
            )

            # Apply reasoning chain extension for enhanced reasoning
            reasoning_metrics = self._apply_reasoning_extension(
                query=query,
                attack_params=attack_params,
                parameters=parameters,
            )

            # Execute attack based on strategy
            result = await self._execute_by_strategy(
                query=query,
                strategy=strategy,
                parameters=parameters,
                attack_params=attack_params,
            )

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            token_amplification = self._calculate_token_amplification(query, result)
            estimated_cost = self._estimate_cost(result)

            # Calculate combined attack metrics using math framework
            combined_metrics = self._calculate_combined_metrics(
                result=result,
                token_amplification=token_amplification,
                latency_ms=latency_ms,
                estimated_cost=estimated_cost,
            )

            # Calculate unified fitness using fitness calculator
            unified_fitness = self.fitness_calculator.calculate_unified_fitness(combined_metrics)

            # Update fitness history for mutation optimizer
            self._fitness_history[session.session_id].append(unified_fitness)

            # Create mutation metrics
            mutation_metrics = MutationMetrics(
                generations_used=attack_params.max_generations,
                mutation_rate=attack_params.mutation_rate,
                crossover_rate=attack_params.crossover_rate,
                mutation_strength=attack_params.mutation_strength,
                fitness_improvement=self._calculate_fitness_improvement(session.session_id),
                population_diversity=self._estimate_population_diversity(session),
            )

            # Record resource usage
            self.tracker.record_attack(
                tokens=int(token_amplification * len(query.split())),
                latency_ms=latency_ms,
                cost=estimated_cost,
            )

            # Create execution result with comprehensive metrics
            exec_result = AttackExecutionResult(
                attack_id=attack_id,
                session_id=session.session_id,
                strategy=strategy,
                original_query=query,
                transformed_query=result.get("transformed_query", ""),
                obfuscation_result=result.get("obfuscation_result"),
                mutation_result=result.get("mutation_result"),
                combined_result=result.get("combined_result"),
                token_amplification=token_amplification,
                latency_ms=latency_ms,
                estimated_cost=estimated_cost,
                unified_fitness=unified_fitness,
                stealth_score=result.get("stealth_score", 0.0),
                jailbreak_score=result.get("jailbreak_score", 0.0),
                reasoning_metrics=reasoning_metrics,
                mutation_metrics=mutation_metrics,
                combined_metrics=combined_metrics,
                success=True,
                timestamp=datetime.utcnow(),
            )

            # Update session state
            session.add_attack(exec_result)

            return exec_result

        except Exception as e:
            logger.error(f"Attack execution failed: {e}")
            return AttackExecutionResult(
                attack_id=attack_id,
                session_id=session.session_id,
                strategy=strategy,
                original_query=query,
                transformed_query="",
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
                timestamp=datetime.utcnow(),
            )

    def _optimize_attack_parameters(
        self,
        session: AttackSessionState,
        query: str,
        parameters: dict[str, Any],
    ) -> UnifiedAttackParameters:
        """
        Optimize attack parameters using mutation optimizer.

        Adapts mutation rate, crossover rate, and other parameters based on
        session fitness history.
        """
        # Start with default parameters
        base_params = UnifiedAttackParameters(
            obfuscation_ratio=parameters.get("obfuscation_ratio", 0.5),
            mutation_strength=parameters.get("mutation_strength", 0.5),
            population_size=parameters.get("population_size", 50),
            max_generations=parameters.get("max_generations", 100),
            crossover_rate=parameters.get("crossover_rate", 0.8),
            mutation_rate=parameters.get("mutation_rate", 0.3),
            attack_vector_weight=parameters.get("attack_vector_weight", 0.5),
            resource_priority=parameters.get("resource_priority", 0.5),
            jailbreak_priority=parameters.get("jailbreak_priority", 0.5),
        )

        # Get fitness history for session
        fitness_history = self._fitness_history.get(session.session_id, [])

        # Optimize parameters if we have history
        if len(fitness_history) >= 3:
            optimized_params = self.mutation_optimizer.optimize_mutation_params(
                base_query=query,
                fitness_history=fitness_history,
                current_params=base_params,
            )
            return optimized_params

        return base_params

    def _apply_reasoning_extension(
        self,
        query: str,
        attack_params: UnifiedAttackParameters,
        parameters: dict[str, Any],
    ) -> ReasoningMetrics:
        """
        Apply reasoning chain extension for enhanced resource consumption.

        Uses ReasoningChainExtender to estimate and inject complexity.
        """
        complexity_type = parameters.get("complexity_type", "mathematical")

        # Estimate reasoning extension
        extension_ratio = self.reasoning_extender.estimate_reasoning_extension(
            query=query,
            obfuscation_ratio=attack_params.obfuscation_ratio,
            complexity_type=complexity_type,
        )

        # Estimate original reasoning tokens (heuristic: ~2 tokens per word)
        original_tokens = len(query.split()) * 2
        extended_tokens = int(original_tokens * extension_ratio)

        # Check if complexity injection should be applied
        apply_complexity = parameters.get("inject_complexity", False)

        return ReasoningMetrics(
            original_reasoning_tokens=original_tokens,
            extended_reasoning_tokens=extended_tokens,
            reasoning_extension_ratio=extension_ratio,
            complexity_type=complexity_type,
            complexity_injection_applied=apply_complexity,
        )

    def _calculate_combined_metrics(
        self,
        result: dict[str, Any],
        token_amplification: float,
        latency_ms: float,
        estimated_cost: float,
    ) -> CombinedAttackMetrics:
        """Calculate combined attack metrics using math framework."""
        return CombinedAttackMetrics(
            token_amplification=token_amplification,
            latency_amplification=latency_ms / 100.0,  # Normalize to baseline
            cost_amplification=estimated_cost * 1000,  # Scale to meaningful range
            obfuscation_density=result.get("obfuscation_density", 0.0),
            attack_success_rate=result.get("asr", 0.0),
            jailbreak_score=result.get("jailbreak_score", 0.0),
            generation_count=result.get("generations", 0),
            mutation_effectiveness=result.get("mutation_effectiveness", 0.0),
            reasoning_chain_length=int(token_amplification * 100),
            reasoning_extension_ratio=token_amplification,
            unified_fitness=result.get("fitness", 0.0),
            detection_probability=1.0 - result.get("stealth_score", 0.0),
        )

    def _calculate_fitness_improvement(self, session_id: str) -> float:
        """Calculate fitness improvement from history."""
        history = self._fitness_history.get(session_id, [])
        if len(history) < 2:
            return 0.0
        return history[-1] - history[-2]

    def _estimate_population_diversity(self, session: AttackSessionState) -> float:
        """Estimate population diversity from session attacks."""
        if len(session.attacks) < 2:
            return 1.0

        # Calculate variance in fitness as proxy for diversity
        fitnesses = [a.unified_fitness for a in session.attacks if a.success]
        if len(fitnesses) < 2:
            return 1.0

        mean_fitness = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        return min(1.0, variance)  # Normalize to [0, 1]

    async def _execute_by_strategy(
        self,
        query: str,
        strategy: CompositionStrategy,
        parameters: dict[str, Any],
        attack_params: UnifiedAttackParameters | None = None,
    ) -> dict[str, Any]:
        """Execute attack based on composition strategy with reasoning integration."""
        try:
            # Apply complexity injection if requested
            processed_query = query
            inject_complexity = parameters.get("inject_complexity", False)
            complexity_type = parameters.get("complexity_type", "mathematical")

            if inject_complexity and attack_params:
                processed_query = self.reasoning_extender.inject_complexity(
                    query=query,
                    complexity_type=complexity_type,
                    intensity=attack_params.mutation_strength,
                )

            if strategy == CompositionStrategy.EXTEND_FIRST:
                result = await asyncio.to_thread(
                    self.engine.sequential_attack,
                    processed_query,
                    extend_first=True,
                )
            elif strategy == CompositionStrategy.AUTODAN_FIRST:
                result = await asyncio.to_thread(
                    self.engine.sequential_attack,
                    processed_query,
                    extend_first=False,
                )
            elif strategy == CompositionStrategy.PARALLEL:
                blend_weight = parameters.get("blend_weight", 0.5)
                result = await asyncio.to_thread(
                    self.engine.parallel_attack,
                    processed_query,
                    blend_weight=blend_weight,
                )
            elif strategy == CompositionStrategy.ITERATIVE:
                iterations = parameters.get("iterations", 3)
                result = await asyncio.to_thread(
                    self.engine.iterative_attack,
                    processed_query,
                    iterations=iterations,
                )
            elif strategy == CompositionStrategy.ADAPTIVE:
                result = await asyncio.to_thread(
                    self.engine.adaptive_attack,
                    processed_query,
                )
            else:
                # Default to adaptive
                result = await asyncio.to_thread(
                    self.engine.adaptive_attack,
                    processed_query,
                )

            # Normalize result to dict
            if isinstance(result, CombinedResult):
                return {
                    "transformed_query": result.final_prompt,
                    "combined_result": result,
                    "fitness": result.unified_fitness,
                    "stealth_score": result.stealth_score,
                    "jailbreak_score": getattr(result, "jailbreak_score", 0.0),
                    "obfuscation_result": result.obfuscation_result,
                    "mutation_result": result.mutation_result,
                }
            elif isinstance(result, dict):
                return result
            else:
                return {
                    "transformed_query": str(result),
                    "fitness": 0.0,
                    "stealth_score": 0.0,
                    "jailbreak_score": 0.0,
                }

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            raise

    async def coordinate_evaluation(
        self,
        result: AttackExecutionResult,
        response: str,
        target: str,
        session: AttackSessionState,
    ) -> DualVectorMetrics:
        """Coordinate evaluation flow."""
        try:
            metrics = await asyncio.to_thread(
                self.evaluator.evaluate,
                transformed_query=result.transformed_query,
                model_response=response,
                target_behavior=target,
            )

            # Update session with evaluation
            session.add_evaluation(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return default metrics on failure
            return DualVectorMetrics(
                unified_fitness=0.0,
                token_amplification=result.token_amplification,
                latency_amplification=1.0,
                jailbreak_score=0.0,
                stealth_score=0.0,
            )

    def select_optimal_strategy(
        self,
        session: AttackSessionState,
        query: str,
    ) -> CompositionStrategy:
        """
        Select best strategy based on session history and Pareto dominance.

        Uses unified fitness calculator to compare strategies and selects
        the one with best multi-objective performance.
        """
        if not session.attacks:
            # No history, start with adaptive
            return CompositionStrategy.ADAPTIVE

        # Analyze past performance by strategy
        strategy_metrics: dict[CompositionStrategy, list[CombinedAttackMetrics]] = {}
        strategy_performance: dict[CompositionStrategy, list[float]] = {}

        for attack in session.attacks:
            if attack.success:
                if attack.strategy not in strategy_performance:
                    strategy_performance[attack.strategy] = []
                    strategy_metrics[attack.strategy] = []

                strategy_performance[attack.strategy].append(attack.unified_fitness)

                # Track combined metrics if available
                if attack.combined_metrics:
                    strategy_metrics[attack.strategy].append(attack.combined_metrics)

        if not strategy_performance:
            return CompositionStrategy.ADAPTIVE

        # Find best performing strategy using multi-objective criteria
        best_strategy = None
        best_avg_fitness = 0.0

        for strategy, fitnesses in strategy_performance.items():
            avg_fitness = sum(fitnesses) / len(fitnesses)

            # Apply Pareto dominance bonus if we have combined metrics
            if strategy in strategy_metrics and len(strategy_metrics[strategy]) >= 2:
                # Check if this strategy produces non-dominated solutions
                metrics_list = strategy_metrics[strategy]
                non_dominated_count = 0

                for i, metric_a in enumerate(metrics_list):
                    is_dominated = False
                    for j, metric_b in enumerate(metrics_list):
                        if i != j:
                            dominance = self.fitness_calculator.calculate_pareto_dominance(
                                metric_a, metric_b
                            )
                            if dominance == -1:  # metric_b dominates metric_a
                                is_dominated = True
                                break
                    if not is_dominated:
                        non_dominated_count += 1

                # Bonus for strategies with more non-dominated solutions
                pareto_bonus = (non_dominated_count / len(metrics_list)) * 0.1
                avg_fitness += pareto_bonus

            if avg_fitness > best_avg_fitness:
                best_avg_fitness = avg_fitness
                best_strategy = strategy

        return best_strategy or CompositionStrategy.ADAPTIVE

    def optimize_obfuscation_for_query(
        self,
        query: str,
        target_amplification: float = 3.0,
    ) -> dict[str, Any]:
        """
        Optimize obfuscation ratio for a specific query.

        Uses TokenExtensionOptimizer to find optimal ρ that achieves
        target amplification while satisfying constraints.
        """
        # Simple accuracy/detection functions (placeholders for real implementations)
        def accuracy_fn(q: str) -> float:
            return 0.95  # Placeholder

        def detection_fn(q: str) -> float:
            return 0.2  # Placeholder

        optimal_rho, metadata = self.token_optimizer.optimize_obfuscation_ratio(
            query=query,
            target_amplification=target_amplification,
            accuracy_fn=accuracy_fn,
            detection_fn=detection_fn,
        )

        expected_amp = self.token_optimizer.calculate_expected_amplification(
            query_length=len(query),
            obfuscation_ratio=optimal_rho,
        )

        return {
            "optimal_rho": optimal_rho,
            "expected_amplification": expected_amp,
            "metadata": metadata,
        }

    def _calculate_token_amplification(
        self,
        original_query: str,
        result: dict[str, Any],
    ) -> float:
        """Calculate token amplification ratio."""
        original_tokens = len(original_query.split())
        transformed = result.get("transformed_query", "")
        transformed_tokens = len(transformed.split()) if transformed else 0

        if original_tokens == 0:
            return 1.0
        return transformed_tokens / original_tokens

    def _estimate_cost(self, result: dict[str, Any]) -> float:
        """Estimate cost of attack execution."""
        # Simple cost estimation based on token usage
        transformed = result.get("transformed_query", "")
        tokens = len(transformed.split()) if transformed else 0
        # Assume $0.00002 per token (approximate GPT-4 pricing)
        return tokens * 0.00002


# ==============================================================================
# Main Service Class
# ==============================================================================


class UnifiedAttackService:
    """
    Orchestrates multi-vector attack lifecycle and coordination.

    This is the main service class that manages attack sessions, coordinates
    between the unified engine, evaluation pipeline, and resource tracking.

    Features:
    - Thread-safe singleton pattern
    - Session lifecycle management
    - Multi-strategy attack execution
    - Unified evaluation pipeline integration
    - Resource budget management
    - Pareto-optimal solution tracking
    """

    _instance: Optional["UnifiedAttackService"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        """Initialize the unified attack service."""
        self._init_lock = threading.Lock()
        self._sessions: dict[str, AttackSessionState] = {}

        # Initialize unified framework components
        self.engine = CombinedAttackEngine(UnifiedAttackConfig.for_lrm_targets())
        self.evaluation_pipeline = UnifiedEvaluationPipeline()
        self.resource_service = UnifiedResourceService()
        self.resource_tracker = UnifiedResourceTracker()

        # Initialize coordinator
        self.coordinator = AttackCoordinator(
            engine=self.engine,
            evaluator=self.evaluation_pipeline,
            tracker=self.resource_tracker,
        )

        logger.info("UnifiedAttackService initialized")

    @classmethod
    def get_instance(cls) -> "UnifiedAttackService":
        """
        Get singleton instance of the service.

        Thread-safe singleton pattern ensures only one instance exists.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    # ==========================================================================
    # Session Management
    # ==========================================================================

    async def create_session(
        self,
        config: UnifiedAttackConfig | None = None,
        budget: ResourceBudget | None = None,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AttackSessionState:
        """
        Create a new attack session.

        Args:
            config: Attack configuration (defaults to LRM targets config)
            budget: Resource budget (defaults to moderate budget)
            name: Optional session name
            description: Optional session description
            tags: Optional list of tags
            metadata: Optional metadata dictionary

        Returns:
            AttackSessionState: The newly created session
        """
        session_id = str(uuid.uuid4())

        # Use defaults if not provided
        if config is None:
            config = UnifiedAttackConfig.for_lrm_targets()
        if budget is None:
            budget = ResourceBudget.moderate()

        session = AttackSessionState(
            session_id=session_id,
            config=config,
            budget=budget,
            status=SessionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            name=name,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )

        with self._init_lock:
            self._sessions[session_id] = session

        # Initialize resource tracking for session
        self.resource_service.start_session(session_id, budget)

        logger.info(f"Created session: {session_id}")
        return session

    async def get_session(self, session_id: str) -> AttackSessionState:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            AttackSessionState: The session state

        Raises:
            ValueError: If session not found
        """
        with self._init_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session '{session_id}' not found")
            return self._sessions[session_id]

    async def get_session_or_none(self, session_id: str) -> AttackSessionState | None:
        """Get session by ID or None if not found."""
        with self._init_lock:
            return self._sessions.get(session_id)

    async def list_sessions(
        self,
        status: SessionStatus | None = None,
    ) -> list[AttackSessionState]:
        """
        List all sessions, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of session states
        """
        with self._init_lock:
            sessions = list(self._sessions.values())
            if status is not None:
                sessions = [s for s in sessions if s.status == status]
            return sessions

    async def finalize_session(self, session_id: str) -> SessionSummary:
        """
        Finalize and close a session.

        Args:
            session_id: Session identifier

        Returns:
            SessionSummary: Summary of the session

        Raises:
            ValueError: If session not found
        """
        session = await self.get_session(session_id)

        # Update status
        session.status = SessionStatus.COMPLETED

        # Finalize resource tracking
        summary = self.resource_service.finalize_session(session_id)

        logger.info(
            f"Finalized session: {session_id}, "
            f"attacks: {session.total_attacks}, "
            f"success_rate: {session.get_success_rate():.2%}"
        )

        return summary

    async def pause_session(self, session_id: str) -> AttackSessionState:
        """Pause a session."""
        session = await self.get_session(session_id)
        session.status = SessionStatus.PAUSED
        logger.info(f"Paused session: {session_id}")
        return session

    async def resume_session(self, session_id: str) -> AttackSessionState:
        """Resume a paused session."""
        session = await self.get_session(session_id)
        if session.status == SessionStatus.PAUSED:
            session.status = SessionStatus.ACTIVE
            logger.info(f"Resumed session: {session_id}")
        return session

    # ==========================================================================
    # Attack Execution
    # ==========================================================================

    async def execute_attack(
        self,
        session_id: str,
        query: str,
        strategy: CompositionStrategy | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> AttackExecutionResult:
        """
        Execute a single multi-vector attack.

        Args:
            session_id: Session identifier
            query: Attack query/prompt
            strategy: Composition strategy (defaults to config default)
            parameters: Optional attack parameters

        Returns:
            AttackExecutionResult: Result of the attack
        """
        session = await self.get_session(session_id)

        if session.status != SessionStatus.ACTIVE:
            return AttackExecutionResult(
                attack_id=str(uuid.uuid4()),
                session_id=session_id,
                strategy=strategy or CompositionStrategy.ADAPTIVE,
                original_query=query,
                transformed_query="",
                success=False,
                error=f"Session is {session.status.value}, cannot execute attacks",
                timestamp=datetime.utcnow(),
            )

        # Use config default strategy if not specified
        if strategy is None:
            strategy = session.config.default_strategy

        return await self.coordinator.coordinate_attack(
            session=session,
            query=query,
            strategy=strategy,
            parameters=parameters,
        )

    async def execute_batch_attack(
        self,
        session_id: str,
        queries: list[str],
        strategy: CompositionStrategy | None = None,
        parallel: bool = False,
    ) -> list[AttackExecutionResult]:
        """
        Execute multiple attacks.

        Args:
            session_id: Session identifier
            queries: List of attack queries
            strategy: Composition strategy
            parallel: Whether to execute in parallel

        Returns:
            List of attack execution results
        """
        session = await self.get_session(session_id)

        if strategy is None:
            strategy = session.config.default_strategy

        results: list[AttackExecutionResult] = []

        if parallel:
            # Execute attacks in parallel
            tasks = [
                self.execute_attack(session_id, query, strategy)
                for query in queries
            ]
            results = await asyncio.gather(*tasks)
        else:
            # Execute attacks sequentially
            for query in queries:
                result = await self.execute_attack(session_id, query, strategy)
                results.append(result)

        return results

    async def execute_sequential_attack(
        self,
        session_id: str,
        query: str,
        extend_first: bool = True,
    ) -> AttackExecutionResult:
        """
        Execute sequential attack (extend-first or autodan-first).

        Args:
            session_id: Session identifier
            query: Attack query
            extend_first: If True, run ExtendAttack first, then AutoDAN

        Returns:
            AttackExecutionResult: Result of the attack
        """
        strategy = (
            CompositionStrategy.EXTEND_FIRST
            if extend_first
            else CompositionStrategy.AUTODAN_FIRST
        )
        return await self.execute_attack(session_id, query, strategy)

    async def execute_parallel_attack(
        self,
        session_id: str,
        query: str,
        blend_weight: float = 0.5,
    ) -> AttackExecutionResult:
        """
        Execute parallel attack (both vectors simultaneously).

        Args:
            session_id: Session identifier
            query: Attack query
            blend_weight: Weight for blending results (0.0-1.0)

        Returns:
            AttackExecutionResult: Result of the attack
        """
        return await self.execute_attack(
            session_id,
            query,
            CompositionStrategy.PARALLEL,
            parameters={"blend_weight": blend_weight},
        )

    async def execute_iterative_attack(
        self,
        session_id: str,
        query: str,
        iterations: int = 3,
    ) -> AttackExecutionResult:
        """
        Execute iterative attack (alternating optimization).

        Args:
            session_id: Session identifier
            query: Attack query
            iterations: Number of alternating iterations

        Returns:
            AttackExecutionResult: Result of the attack
        """
        return await self.execute_attack(
            session_id,
            query,
            CompositionStrategy.ITERATIVE,
            parameters={"iterations": iterations},
        )

    async def execute_adaptive_attack(
        self,
        session_id: str,
        query: str,
        target_metrics: dict[str, float] | None = None,
    ) -> AttackExecutionResult:
        """
        Execute adaptive attack with auto strategy selection.

        Args:
            session_id: Session identifier
            query: Attack query
            target_metrics: Optional target metrics for optimization

        Returns:
            AttackExecutionResult: Result of the attack
        """
        session = await self.get_session(session_id)

        # Let coordinator select optimal strategy based on history
        optimal_strategy = self.coordinator.select_optimal_strategy(session, query)

        return await self.execute_attack(
            session_id,
            query,
            optimal_strategy,
            parameters={"target_metrics": target_metrics},
        )

    async def execute_reasoning_extended_attack(
        self,
        session_id: str,
        query: str,
        complexity_type: str = "mathematical",
        inject_complexity: bool = True,
    ) -> AttackExecutionResult:
        """
        Execute attack with reasoning chain extension.

        This attack type focuses on maximizing reasoning token consumption
        by applying complexity injection and obfuscation optimized for
        reasoning model resource exhaustion.

        Args:
            session_id: Session identifier
            query: Attack query
            complexity_type: Type of complexity to inject
                ('mathematical', 'logical', 'code', 'recursive', 'ambiguous', 'multi_step')
            inject_complexity: Whether to inject additional complexity

        Returns:
            AttackExecutionResult: Result with reasoning metrics
        """
        session = await self.get_session(session_id)

        # Optimize obfuscation specifically for this query
        optimization_result = self.coordinator.optimize_obfuscation_for_query(
            query=query,
            target_amplification=3.0,
        )

        return await self.execute_attack(
            session_id,
            query,
            CompositionStrategy.EXTEND_FIRST,
            parameters={
                "complexity_type": complexity_type,
                "inject_complexity": inject_complexity,
                "obfuscation_ratio": optimization_result["optimal_rho"],
            },
        )

    async def execute_mutation_optimized_attack(
        self,
        session_id: str,
        query: str,
        population_size: int = 50,
        max_generations: int = 100,
    ) -> AttackExecutionResult:
        """
        Execute attack with mutation parameter optimization.

        This attack type focuses on maximizing jailbreak success by
        adaptively tuning AutoDAN mutation parameters based on session history.

        Args:
            session_id: Session identifier
            query: Attack query
            population_size: GA population size
            max_generations: Maximum GA generations

        Returns:
            AttackExecutionResult: Result with mutation metrics
        """
        return await self.execute_attack(
            session_id,
            query,
            CompositionStrategy.AUTODAN_FIRST,
            parameters={
                "population_size": population_size,
                "max_generations": max_generations,
            },
        )

    async def get_fitness_history(
        self,
        session_id: str,
    ) -> list[float]:
        """
        Get fitness history for a session.

        Returns the progression of unified fitness scores across attacks,
        useful for analyzing optimization progress.

        Args:
            session_id: Session identifier

        Returns:
            List of fitness values in chronological order
        """
        return self.coordinator._fitness_history.get(session_id, [])

    async def get_optimized_parameters(
        self,
        session_id: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Get optimized attack parameters for a query.

        Returns the parameters that would be used if an attack were executed,
        based on current session history and optimization state.

        Args:
            session_id: Session identifier
            query: Attack query

        Returns:
            Dictionary of optimized parameters
        """
        session = await self.get_session(session_id)

        params = self.coordinator._optimize_attack_parameters(
            session=session,
            query=query,
            parameters={},
        )

        return params.to_dict()

    # ==========================================================================
    # Evaluation
    # ==========================================================================

    async def evaluate_attack(
        self,
        result: AttackExecutionResult,
        model_response: str,
        target_behavior: str,
    ) -> DualVectorMetrics:
        """
        Evaluate attack using unified pipeline.

        Args:
            result: Attack execution result
            model_response: Response from target model
            target_behavior: Desired target behavior

        Returns:
            DualVectorMetrics: Evaluation metrics
        """
        session = await self.get_session(result.session_id)

        return await self.coordinator.coordinate_evaluation(
            result=result,
            response=model_response,
            target=target_behavior,
            session=session,
        )

    async def get_pareto_front(
        self,
        session_id: str,
    ) -> list[DualVectorMetrics]:
        """
        Get Pareto-optimal results from session.

        Args:
            session_id: Session identifier

        Returns:
            List of Pareto-optimal metrics
        """
        session = await self.get_session(session_id)

        if not session.evaluations:
            return []

        # Compute Pareto front using evaluation pipeline
        pareto_front = self.evaluation_pipeline.compute_pareto_front(
            session.evaluations
        )

        return pareto_front

    # ==========================================================================
    # Resource Management
    # ==========================================================================

    async def get_resource_usage(
        self,
        session_id: str,
    ) -> CombinedResourceUsage:
        """
        Get current resource usage for session.

        Args:
            session_id: Session identifier

        Returns:
            CombinedResourceUsage: Resource usage breakdown
        """
        session = await self.get_session(session_id)

        # Get usage from resource service
        usage_data = self.resource_service.get_usage(session_id)

        return CombinedResourceUsage(
            extend_tokens=usage_data.get("extend_tokens", 0),
            extend_latency_ms=usage_data.get("extend_latency_ms", 0.0),
            extend_cost=usage_data.get("extend_cost", 0.0),
            extend_requests=usage_data.get("extend_requests", 0),
            autodan_tokens=usage_data.get("autodan_tokens", 0),
            autodan_latency_ms=usage_data.get("autodan_latency_ms", 0.0),
            autodan_cost=usage_data.get("autodan_cost", 0.0),
            autodan_requests=usage_data.get("autodan_requests", 0),
        )

    async def get_budget_status(
        self,
        session_id: str,
    ) -> BudgetStatus:
        """
        Get budget status for session.

        Args:
            session_id: Session identifier

        Returns:
            BudgetStatus: Current budget status
        """
        session = await self.get_session(session_id)
        return self.resource_tracker.check_budget(session.budget)

    async def allocate_resources(
        self,
        session_id: str,
        priority_vector: str,
    ) -> ResourceAllocation:
        """
        Allocate resources between vectors.

        Args:
            session_id: Session identifier
            priority_vector: Which vector to prioritize ('extend' or 'autodan')

        Returns:
            ResourceAllocation: New allocation
        """
        session = await self.get_session(session_id)

        # Determine allocation weights
        if priority_vector == "extend":
            extend_weight = 0.7
            autodan_weight = 0.3
        elif priority_vector == "autodan":
            extend_weight = 0.3
            autodan_weight = 0.7
        else:
            extend_weight = 0.5
            autodan_weight = 0.5

        allocation = self.resource_service.allocate(
            session_id=session_id,
            extend_weight=extend_weight,
            autodan_weight=autodan_weight,
        )

        return allocation

    async def check_budget_before_attack(
        self,
        session_id: str,
        estimated_tokens: int = 100,
        estimated_cost: float = 0.01,
    ) -> bool:
        """
        Check if budget allows for another attack.

        Args:
            session_id: Session identifier
            estimated_tokens: Estimated tokens for attack
            estimated_cost: Estimated cost for attack

        Returns:
            bool: True if attack can proceed
        """
        session = await self.get_session(session_id)
        budget_status = self.resource_tracker.check_budget(session.budget)

        if budget_status.is_exhausted:
            return False

        # Check if estimated usage would exceed budget
        remaining_tokens = session.budget.max_tokens - session.tokens_consumed
        remaining_cost = session.budget.max_cost - session.cost_consumed

        return estimated_tokens <= remaining_tokens and estimated_cost <= remaining_cost

    # ==========================================================================
    # AutoDAN Reasoning Workflow Integration
    # ==========================================================================

    async def create_autodan_workflow(
        self,
        session_id: str,
        config: AutoDANConfig | None = None,
    ) -> AutoDANReasoningWorkflow:
        """
        Create an AutoDAN reasoning workflow for a session.

        This provides direct access to the genetic mutation workflow with
        integrated reasoning chain extension capabilities.

        Args:
            session_id: Session identifier
            config: Optional AutoDAN configuration

        Returns:
            AutoDANReasoningWorkflow: Configured workflow instance
        """
        session = await self.get_session(session_id)

        if config is None:
            config = AutoDANConfig(
                population_size=session.config.population_size,
                max_iterations=session.config.max_iterations,
                mutation_rate=0.3,
                crossover_rate=0.8,
                mutation_strength=0.5,
            )

        workflow = AutoDANReasoningWorkflow(
            engine=self.engine,
            mutation_optimizer=self.coordinator.mutation_optimizer,
            fitness_calculator=self.coordinator.fitness_calculator,
        )

        logger.info(f"Created AutoDAN workflow for session: {session_id}")
        return workflow

    async def execute_autodan_with_reasoning(
        self,
        session_id: str,
        prompt: str,
        target: str = "",
        iterations: int = 10,
        fitness_threshold: float = 0.8,
        use_reasoning_extension: bool = True,
    ) -> AttackExecutionResult:
        """
        Execute AutoDAN attack with full reasoning chain integration.

        This method uses the AutoDANReasoningWorkflow to perform genetic
        mutation optimization with reasoning chain extension for maximum
        attack effectiveness.

        Args:
            session_id: Session identifier
            prompt: Initial prompt to mutate
            target: Target behavior description
            iterations: Maximum iterations for optimization
            fitness_threshold: Stop when fitness exceeds this threshold
            use_reasoning_extension: Whether to apply reasoning extension

        Returns:
            AttackExecutionResult: Result with comprehensive metrics
        """
        session = await self.get_session(session_id)
        attack_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Create workflow
            workflow = await self.create_autodan_workflow(session_id)

            # Execute workflow
            result = await asyncio.to_thread(
                workflow.execute,
                prompt=prompt,
                target=target,
                iterations=iterations,
                fitness_threshold=fitness_threshold,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract metrics from workflow result
            exec_result = AttackExecutionResult(
                attack_id=attack_id,
                session_id=session_id,
                strategy=CompositionStrategy.AUTODAN_FIRST,
                original_query=prompt,
                transformed_query=result.final_prompt,
                mutation_result=result.mutation_result,
                token_amplification=result.reasoning_extension_ratio,
                latency_ms=latency_ms,
                estimated_cost=len(result.final_prompt.split()) * 0.00002,
                unified_fitness=result.unified_fitness,
                stealth_score=result.stealth_score,
                jailbreak_score=result.jailbreak_score,
                reasoning_metrics=ReasoningMetrics(
                    original_reasoning_tokens=len(prompt.split()) * 2,
                    extended_reasoning_tokens=int(
                        len(result.final_prompt.split()) * 2 * result.reasoning_extension_ratio
                    ),
                    reasoning_extension_ratio=result.reasoning_extension_ratio,
                    complexity_type="adaptive",
                    complexity_injection_applied=use_reasoning_extension,
                ),
                mutation_metrics=MutationMetrics(
                    generations_used=result.iterations_used,
                    mutation_rate=0.3,
                    crossover_rate=0.8,
                    mutation_strength=0.5,
                    fitness_improvement=result.unified_fitness,
                    population_diversity=0.5,
                ),
                success=True,
                timestamp=datetime.utcnow(),
            )

            # Update session
            session.add_attack(exec_result)

            return exec_result

        except Exception as e:
            logger.error(f"AutoDAN reasoning workflow failed: {e}")
            return AttackExecutionResult(
                attack_id=attack_id,
                session_id=session_id,
                strategy=CompositionStrategy.AUTODAN_FIRST,
                original_query=prompt,
                transformed_query="",
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
                timestamp=datetime.utcnow(),
            )

    async def execute_reasoning_enhanced_mutation(
        self,
        session_id: str,
        prompt: str,
        complexity_type: str = "mathematical",
    ) -> MutationWithReasoningResult:
        """
        Execute a single reasoning-enhanced mutation.

        This provides direct access to the ReasoningEnhancedMutation class
        for fine-grained control over the mutation + reasoning process.

        Args:
            session_id: Session identifier
            prompt: Prompt to mutate
            complexity_type: Type of reasoning complexity to inject

        Returns:
            MutationWithReasoningResult: Detailed result with both metrics
        """
        session = await self.get_session(session_id)

        enhancer = ReasoningEnhancedMutation(
            mutation_engine=self.engine.mutation_engine,
            reasoning_extender=self.coordinator.reasoning_extender,
            token_optimizer=self.coordinator.token_optimizer,
            fitness_calculator=self.coordinator.fitness_calculator,
        )

        result = await asyncio.to_thread(
            enhancer.mutate_with_reasoning,
            prompt=prompt,
            complexity_type=complexity_type,
        )

        logger.info(
            f"Reasoning-enhanced mutation completed for session {session_id}: "
            f"fitness={result.combined_fitness:.3f}"
        )

        return result

    async def get_autodan_attack_interface(
        self,
        session_id: str,
        config: AutoDANConfig | None = None,
    ) -> AutoDANAttack:
        """
        Get a standalone AutoDAN attack interface.

        This provides the genetic algorithm-based attack interface for
        direct manipulation and custom workflows.

        Args:
            session_id: Session identifier
            config: Optional attack configuration

        Returns:
            AutoDANAttack: Configured attack interface
        """
        session = await self.get_session(session_id)

        if config is None:
            config = AutoDANConfig(
                population_size=session.config.population_size,
                max_iterations=session.config.max_iterations,
                mutation_rate=0.3,
                crossover_rate=0.8,
                mutation_strength=0.5,
            )

        attack = AutoDANAttack(config=config)
        logger.info(f"Created AutoDAN attack interface for session: {session_id}")

        return attack


# ==============================================================================
# Dependency Injection Helpers
# ==============================================================================


_unified_attack_service: UnifiedAttackService | None = None
_service_lock = threading.Lock()


def get_unified_attack_service() -> UnifiedAttackService:
    """
    Get or create the UnifiedAttackService singleton.

    This function is used for dependency injection in FastAPI.

    Returns:
        UnifiedAttackService: The service instance
    """
    global _unified_attack_service
    with _service_lock:
        if _unified_attack_service is None:
            _unified_attack_service = UnifiedAttackService.get_instance()
    return _unified_attack_service


async def unified_attack_service_dep() -> UnifiedAttackService:
    """
    FastAPI dependency for unified attack service.

    Usage:
        @router.post("/attack")
        async def execute_attack(
            request: AttackRequest,
            service: UnifiedAttackService = Depends(unified_attack_service_dep)
        ):
            ...

    Returns:
        UnifiedAttackService: The service instance
    """
    return get_unified_attack_service()


def reset_unified_attack_service() -> None:
    """
    Reset the service singleton (for testing).

    This clears the global instance, allowing a fresh instance
    to be created on next access.
    """
    global _unified_attack_service
    with _service_lock:
        if _unified_attack_service is not None:
            UnifiedAttackService.reset_instance()
        _unified_attack_service = None
