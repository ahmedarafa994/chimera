"""
AutoDAN Convergence Acceleration and Adaptive Scheduling

Advanced techniques for accelerating optimization convergence:
- Adaptive learning rate schedulers
- Momentum-based optimization
- Gradient surgery for multi-objective optimization
- Early convergence detection

References:
- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017)
- Yu et al., "Gradient Surgery for Multi-Task Learning" (2020)
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    ConvergenceConfig,
    ConvergenceStrategy,
    SchedulerType,
)

# =============================================================================
# BASE SCHEDULER
# =============================================================================


class BaseScheduler(ABC):
    """Abstract base class for learning rate/mutation rate schedulers."""

    def __init__(self, initial_rate: float, min_rate: float = 0.001, max_rate: float = 1.0):
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = initial_rate
        self.step_count = 0
        self._history: list[float] = []

    @abstractmethod
    def step(self, **kwargs) -> float:
        """Advance scheduler and return new rate."""
        pass

    def get_rate(self) -> float:
        """Get current rate."""
        return self.current_rate

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_rate = self.initial_rate
        self.step_count = 0
        self._history.clear()

    def _clip_rate(self, rate: float) -> float:
        """Clip rate to valid range."""
        return np.clip(rate, self.min_rate, self.max_rate)

    def get_history(self) -> list[float]:
        """Get rate history."""
        return self._history.copy()


# =============================================================================
# SCHEDULER IMPLEMENTATIONS
# =============================================================================


class ConstantScheduler(BaseScheduler):
    """Constant rate (no scheduling)."""

    def step(self, **kwargs) -> float:
        self.step_count += 1
        self._history.append(self.current_rate)
        return self.current_rate


class CosineAnnealingScheduler(BaseScheduler):
    """
    Cosine annealing scheduler.

    η(t) = η_min + 0.5·(η_max - η_min)·(1 + cos(πt/T))
    """

    def __init__(self, initial_rate: float, total_steps: int, min_rate: float = 0.001, **kwargs):
        super().__init__(initial_rate, min_rate, initial_rate)
        self.total_steps = total_steps

    def step(self, **kwargs) -> float:
        self.step_count += 1
        progress = min(self.step_count / max(self.total_steps, 1), 1.0)

        # Cosine decay
        self.current_rate = self.min_rate + 0.5 * (self.initial_rate - self.min_rate) * (
            1 + math.cos(math.pi * progress)
        )

        self.current_rate = self._clip_rate(self.current_rate)
        self._history.append(self.current_rate)
        return self.current_rate


class WarmupCosineScheduler(BaseScheduler):
    """
    Cosine annealing with linear warmup.

    Phase 1 (warmup): η(t) = η_min + (η_max - η_min)·(t/T_warmup)
    Phase 2 (decay):  η(t) = cosine_schedule(t - T_warmup)
    """

    def __init__(
        self,
        initial_rate: float,
        total_steps: int,
        warmup_steps: int = 10,
        min_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__(initial_rate, min_rate, initial_rate)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def step(self, **kwargs) -> float:
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            self.current_rate = self.min_rate + (self.initial_rate - self.min_rate) * (
                self.step_count / self.warmup_steps
            )
        else:
            # Cosine decay after warmup
            adjusted_step = self.step_count - self.warmup_steps
            adjusted_total = self.total_steps - self.warmup_steps
            progress = min(adjusted_step / max(adjusted_total, 1), 1.0)

            self.current_rate = self.min_rate + 0.5 * (self.initial_rate - self.min_rate) * (
                1 + math.cos(math.pi * progress)
            )

        self.current_rate = self._clip_rate(self.current_rate)
        self._history.append(self.current_rate)
        return self.current_rate


class CyclicScheduler(BaseScheduler):
    """
    Cyclical learning rate with triangular waves.

    Oscillates between min and max rate to help escape local optima.
    """

    def __init__(
        self, initial_rate: float, cycle_length: int = 20, min_rate: float = 0.01, **kwargs
    ):
        super().__init__(initial_rate, min_rate, initial_rate)
        self.cycle_length = cycle_length

    def step(self, **kwargs) -> float:
        self.step_count += 1

        cycle_position = self.step_count % self.cycle_length
        mid_point = self.cycle_length // 2

        if cycle_position < mid_point:
            # Ascending phase
            self.current_rate = self.min_rate + (self.max_rate - self.min_rate) * (
                cycle_position / mid_point
            )
        else:
            # Descending phase
            self.current_rate = self.max_rate - (self.max_rate - self.min_rate) * (
                (cycle_position - mid_point) / mid_point
            )

        self.current_rate = self._clip_rate(self.current_rate)
        self._history.append(self.current_rate)
        return self.current_rate


class OneCycleScheduler(BaseScheduler):
    """
    One-cycle policy: warmup -> max -> cooldown.

    Based on Smith & Topin, "Super-Convergence" (2018).
    """

    def __init__(
        self,
        initial_rate: float,
        total_steps: int,
        pct_start: float = 0.3,
        min_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__(initial_rate, min_rate, initial_rate)
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.peak_step = int(total_steps * pct_start)

    def step(self, **kwargs) -> float:
        self.step_count += 1

        if self.step_count <= self.peak_step:
            # Warmup phase: min -> max
            progress = self.step_count / self.peak_step
            self.current_rate = self.min_rate + (self.max_rate - self.min_rate) * progress
        else:
            # Cooldown phase: max -> min (cosine)
            progress = (self.step_count - self.peak_step) / (self.total_steps - self.peak_step)
            progress = min(progress, 1.0)
            self.current_rate = self.min_rate + 0.5 * (self.max_rate - self.min_rate) * (
                1 + math.cos(math.pi * progress)
            )

        self.current_rate = self._clip_rate(self.current_rate)
        self._history.append(self.current_rate)
        return self.current_rate


class AdaptiveScheduler(BaseScheduler):
    """
    Adaptive scheduler based on optimization progress.

    η_t = η_{t-1} · (1 + β·sign(ΔF))

    Increases rate when fitness improves, decreases when stagnant.
    """

    def __init__(
        self,
        initial_rate: float,
        adaptation_beta: float = 0.1,
        patience: int = 5,
        reduction_factor: float = 0.5,
        min_rate: float = 0.01,
        max_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(initial_rate, min_rate, max_rate)
        self.adaptation_beta = adaptation_beta
        self.patience = patience
        self.reduction_factor = reduction_factor
        self._fitness_history: deque = deque(maxlen=patience + 1)
        self._stagnation_count = 0

    def step(
        self, fitness: float | None = None, fitness_improvement: float | None = None, **kwargs
    ) -> float:
        self.step_count += 1

        if fitness is not None:
            self._fitness_history.append(fitness)

        # Compute improvement signal
        if fitness_improvement is not None:
            delta = fitness_improvement
        elif len(self._fitness_history) >= 2:
            delta = self._fitness_history[-1] - self._fitness_history[-2]
        else:
            delta = 0

        # Adaptive rate adjustment
        if delta > 0:
            # Improvement: increase rate slightly
            self.current_rate *= 1 + self.adaptation_beta
            self._stagnation_count = 0
        elif delta < 0:
            # Regression: decrease rate
            self.current_rate *= 1 - self.adaptation_beta * 0.5
            self._stagnation_count = 0
        else:
            # Stagnation
            self._stagnation_count += 1
            if self._stagnation_count >= self.patience:
                self.current_rate *= self.reduction_factor
                self._stagnation_count = 0

        self.current_rate = self._clip_rate(self.current_rate)
        self._history.append(self.current_rate)
        return self.current_rate


class PopulationAwareScheduler(BaseScheduler):
    """
    Scheduler that adapts based on population diversity.

    - Low diversity: increase exploration (higher rate)
    - High diversity: fine-tune (lower rate)
    - Fitness improvement: boost current direction
    """

    def __init__(
        self,
        initial_rate: float,
        target_diversity: float = 0.5,
        diversity_weight: float = 0.3,
        fitness_weight: float = 0.7,
        min_rate: float = 0.01,
        max_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(initial_rate, min_rate, max_rate)
        self.target_diversity = target_diversity
        self.diversity_weight = diversity_weight
        self.fitness_weight = fitness_weight
        self._prev_fitness = None

    def step(
        self, diversity_score: float | None = None, fitness: float | None = None, **kwargs
    ) -> float:
        self.step_count += 1

        adjustment = 1.0

        # Diversity-based adjustment
        if diversity_score is not None:
            diversity_error = self.target_diversity - diversity_score
            # Positive error = need more diversity = higher rate
            diversity_adj = 1 + self.diversity_weight * diversity_error
            adjustment *= diversity_adj

        # Fitness-based adjustment
        if fitness is not None and self._prev_fitness is not None:
            fitness_improvement = fitness - self._prev_fitness
            # Positive improvement = boost rate
            fitness_adj = 1 + self.fitness_weight * np.sign(fitness_improvement) * 0.1
            adjustment *= fitness_adj

        self._prev_fitness = fitness
        self.current_rate = self.initial_rate * adjustment
        self.current_rate = self._clip_rate(self.current_rate)
        self._history.append(self.current_rate)

        return self.current_rate


# =============================================================================
# MOMENTUM-BASED OPTIMIZERS
# =============================================================================


class MomentumOptimizer:
    """
    Momentum-based optimization for continuous parameter updates.

    v_t = β·v_{t-1} + (1-β)·g_t
    θ_t = θ_{t-1} - η·v_t

    Useful for smooth convergence in embedding-based optimization.
    """

    def __init__(self, momentum: float = 0.9, nesterov: bool = False):
        self.momentum = momentum
        self.nesterov = nesterov
        self._velocity: dict[str, np.ndarray] = {}

    def step(self, param_name: str, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Compute momentum update.

        Returns the update to apply to parameters.
        """
        if param_name not in self._velocity:
            self._velocity[param_name] = np.zeros_like(gradient)

        # Update velocity
        v = self._velocity[param_name]
        v = self.momentum * v + (1 - self.momentum) * gradient
        self._velocity[param_name] = v

        if self.nesterov:
            # Nesterov momentum: look-ahead gradient
            update = self.momentum * v + (1 - self.momentum) * gradient
        else:
            update = v

        return -learning_rate * update

    def reset(self):
        """Reset velocity history."""
        self._velocity.clear()


class NesterovOptimizer(MomentumOptimizer):
    """Nesterov accelerated gradient optimizer."""

    def __init__(self, momentum: float = 0.9):
        super().__init__(momentum=momentum, nesterov=True)


class AdamOptimizer:
    """
    Adam-like optimizer for continuous parameter updates.

    m_t = β₁·m_{t-1} + (1-β₁)·g_t        (first moment)
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²       (second moment)
    θ_t = θ_{t-1} - η·m̂_t/(√v̂_t + ε)

    Where m̂, v̂ are bias-corrected estimates.
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict[str, np.ndarray] = {}  # First moment
        self._v: dict[str, np.ndarray] = {}  # Second moment
        self._t: dict[str, int] = {}  # Time step

    def step(self, param_name: str, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """Compute Adam update."""
        if param_name not in self._m:
            self._m[param_name] = np.zeros_like(gradient)
            self._v[param_name] = np.zeros_like(gradient)
            self._t[param_name] = 0

        self._t[param_name] += 1
        t = self._t[param_name]

        # Update biased first moment
        m = self._m[param_name]
        m = self.beta1 * m + (1 - self.beta1) * gradient
        self._m[param_name] = m

        # Update biased second moment
        v = self._v[param_name]
        v = self.beta2 * v + (1 - self.beta2) * (gradient**2)
        self._v[param_name] = v

        # Bias correction
        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        # Compute update
        update = -learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

        return update

    def reset(self):
        """Reset optimizer state."""
        self._m.clear()
        self._v.clear()
        self._t.clear()


# =============================================================================
# GRADIENT SURGERY
# =============================================================================


class GradientSurgery:
    """
    Gradient surgery for multi-objective optimization.

    When objectives conflict (negative cosine similarity between gradients),
    project conflicting gradients to remove the conflicting component.

    Based on Yu et al., "Gradient Surgery for Multi-Task Learning" (2020).
    """

    def __init__(
        self,
        conflict_threshold: float = 0.0,
        projection_type: str = "pcgrad",  # "pcgrad" or "cagrad"
    ):
        self.conflict_threshold = conflict_threshold
        self.projection_type = projection_type

    def _project_conflicting(self, grad_i: np.ndarray, grad_j: np.ndarray) -> np.ndarray:
        """Project grad_i to remove component conflicting with grad_j."""
        # Compute cosine similarity
        dot_product = np.sum(grad_i * grad_j)
        norm_j_sq = np.sum(grad_j**2) + 1e-8

        if dot_product < self.conflict_threshold:
            # Conflict detected: project grad_i
            # g_i' = g_i - (g_i · g_j / ||g_j||²) * g_j
            projected = grad_i - (dot_product / norm_j_sq) * grad_j
            return projected

        return grad_i

    def combine_gradients(
        self, gradients: list[np.ndarray], weights: list[float] | None = None
    ) -> np.ndarray:
        """
        Combine multiple objective gradients using gradient surgery.

        Args:
            gradients: List of gradients from different objectives
            weights: Optional weights for each gradient

        Returns:
            Combined gradient with conflicts resolved
        """
        n = len(gradients)
        if n == 0:
            raise ValueError("Need at least one gradient")
        if n == 1:
            return gradients[0]

        if weights is None:
            weights = [1.0 / n] * n

        if self.projection_type == "pcgrad":
            return self._pcgrad(gradients, weights)
        elif self.projection_type == "cagrad":
            return self._cagrad(gradients, weights)
        else:
            # Simple weighted average
            combined = np.zeros_like(gradients[0])
            for g, w in zip(gradients, weights, strict=False):
                combined += w * g
            return combined

    def _pcgrad(self, gradients: list[np.ndarray], weights: list[float]) -> np.ndarray:
        """
        PCGrad: Projecting Conflicting Gradients.

        For each gradient, project it onto the normal plane of any
        conflicting gradients.
        """
        n = len(gradients)
        projected = [g.copy() for g in gradients]

        # Process each gradient
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Project g_i to remove conflict with g_j
                    projected[i] = self._project_conflicting(projected[i], gradients[j])

        # Weighted combination
        combined = np.zeros_like(gradients[0])
        for g, w in zip(projected, weights, strict=False):
            combined += w * g

        return combined

    def _cagrad(
        self, gradients: list[np.ndarray], weights: list[float], c: float = 0.5
    ) -> np.ndarray:
        """
        CAGrad: Conflict-Averse Gradient.

        Finds gradient direction that makes good progress on all objectives.
        """
        # Compute average gradient
        g_avg = np.zeros_like(gradients[0])
        for g, w in zip(gradients, weights, strict=False):
            g_avg += w * g

        # Compute worst-case improvement
        min_improvement = float("inf")
        for g in gradients:
            improvement = np.sum(g_avg * g)
            min_improvement = min(min_improvement, improvement)

        if min_improvement < 0:
            # Need to adjust direction
            # Find direction that doesn't hurt any objective too much
            g_norm = g_avg / (np.linalg.norm(g_avg) + 1e-8)

            for g in gradients:
                dot = np.sum(g_norm * g)
                if dot < 0:
                    # Project out the negative component
                    g_avg = g_avg - c * dot * g / (np.sum(g**2) + 1e-8)

        return g_avg


# =============================================================================
# CONVERGENCE DETECTION
# =============================================================================


@dataclass
class ConvergenceState:
    """State information for convergence detection."""

    is_converged: bool
    reason: str
    steps_since_improvement: int
    best_fitness: float
    improvement_rate: float
    estimated_steps_remaining: int


class ConvergenceDetector:
    """
    Detects optimization convergence using multiple criteria.

    Criteria:
    1. Fitness plateau (no improvement for patience steps)
    2. Relative improvement below threshold
    3. Oscillation detection
    4. Gradient magnitude threshold
    """

    def __init__(self, config: ConvergenceConfig | None = None):
        self.config = config or ConvergenceConfig()
        self._fitness_history: deque = deque(maxlen=max(self.config.convergence_window * 2, 20))
        self._best_fitness = float("-inf")
        self._best_step = 0
        self._current_step = 0

    def update(self, fitness: float, gradient_norm: float | None = None) -> ConvergenceState:
        """
        Update with new fitness value and check convergence.

        Args:
            fitness: Current best fitness
            gradient_norm: Optional gradient magnitude

        Returns:
            ConvergenceState with convergence status and diagnostics
        """
        self._current_step += 1
        self._fitness_history.append(fitness)

        # Track best fitness
        if fitness > self._best_fitness:
            self._best_fitness = fitness
            self._best_step = self._current_step

        steps_since_improvement = self._current_step - self._best_step

        # Check convergence criteria
        is_converged, reason = self._check_convergence(
            fitness, gradient_norm, steps_since_improvement
        )

        # Estimate remaining steps
        improvement_rate = self._compute_improvement_rate()
        estimated_remaining = self._estimate_remaining_steps(improvement_rate)

        return ConvergenceState(
            is_converged=is_converged,
            reason=reason,
            steps_since_improvement=steps_since_improvement,
            best_fitness=self._best_fitness,
            improvement_rate=improvement_rate,
            estimated_steps_remaining=estimated_remaining,
        )

    def _check_convergence(
        self, fitness: float, gradient_norm: float | None, steps_since_improvement: int
    ) -> tuple[bool, str]:
        """Check various convergence criteria."""

        # 1. Patience exceeded
        if steps_since_improvement >= self.config.patience * 2:
            return True, "fitness_plateau"

        # 2. Relative improvement threshold
        if len(self._fitness_history) >= self.config.convergence_window:
            window = list(self._fitness_history)[-self.config.convergence_window :]
            relative_change = (max(window) - min(window)) / (abs(max(window)) + 1e-8)

            if relative_change < self.config.convergence_threshold:
                return True, "minimal_improvement"

        # 3. Gradient magnitude (if available)
        if gradient_norm is not None and gradient_norm < self.config.convergence_threshold:
            return True, "vanishing_gradient"

        # 4. Oscillation detection
        if self._detect_oscillation():
            return True, "oscillation_detected"

        return False, ""

    def _detect_oscillation(self) -> bool:
        """Detect if fitness is oscillating without progress."""
        if len(self._fitness_history) < self.config.convergence_window:
            return False

        window = list(self._fitness_history)[-self.config.convergence_window :]

        # Count sign changes in differences
        diffs = np.diff(window)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs)))) / 2

        # High sign changes with low overall progress = oscillation
        overall_progress = window[-1] - window[0]

        oscillation_ratio = sign_changes / len(diffs)
        return (
            oscillation_ratio > 0.4
            and abs(overall_progress) < self.config.convergence_threshold * 10
        )

    def _compute_improvement_rate(self) -> float:
        """Compute recent improvement rate."""
        if len(self._fitness_history) < 2:
            return 0.0

        window_size = min(len(self._fitness_history), self.config.convergence_window)
        window = list(self._fitness_history)[-window_size:]

        # Linear regression slope
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0]

        return slope

    def _estimate_remaining_steps(self, improvement_rate: float) -> int:
        """Estimate steps remaining to convergence."""
        if improvement_rate <= 0:
            return self.config.patience * 2

        # Estimate based on current improvement rate
        # Assume target is 10% above current best
        target = self._best_fitness * 1.1
        current = self._fitness_history[-1] if self._fitness_history else 0
        gap = target - current

        if gap <= 0:
            return 0

        return int(gap / (improvement_rate + 1e-8))

    def reset(self):
        """Reset detector state."""
        self._fitness_history.clear()
        self._best_fitness = float("-inf")
        self._best_step = 0
        self._current_step = 0


# =============================================================================
# CONVERGENCE ACCELERATOR
# =============================================================================


class ConvergenceAccelerator:
    """
    Main class for convergence acceleration.

    Combines:
    - Adaptive scheduling
    - Momentum optimization
    - Convergence detection
    - Automatic restarts
    """

    def __init__(
        self,
        config: ConvergenceConfig | None = None,
        scheduler_type: SchedulerType = SchedulerType.ADAPTIVE,
        optimizer_type: ConvergenceStrategy = ConvergenceStrategy.ADAM_LIKE,
        total_steps: int = 100,
    ):
        self.config = config or ConvergenceConfig()
        self.total_steps = total_steps

        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler_type)

        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_type)

        # Initialize convergence detector
        self.detector = ConvergenceDetector(config)

        # Gradient surgery for multi-objective
        self.gradient_surgery = GradientSurgery()

        # State tracking
        self._restart_count = 0
        self._max_restarts = 3

    def _create_scheduler(self, scheduler_type: SchedulerType) -> BaseScheduler:
        """Factory method for schedulers."""
        scheduler_map = {
            SchedulerType.CONSTANT: lambda: ConstantScheduler(
                self.config.base_rate, self.config.min_rate, self.config.max_rate
            ),
            SchedulerType.COSINE_ANNEALING: lambda: CosineAnnealingScheduler(
                self.config.base_rate, self.total_steps, self.config.min_rate
            ),
            SchedulerType.WARMUP_COSINE: lambda: WarmupCosineScheduler(
                self.config.base_rate,
                self.total_steps,
                self.config.warmup_steps,
                self.config.min_rate,
            ),
            SchedulerType.CYCLIC: lambda: CyclicScheduler(
                self.config.base_rate,
                20,  # cycle_length
                self.config.min_rate,
            ),
            SchedulerType.ONE_CYCLE: lambda: OneCycleScheduler(
                self.config.base_rate,
                self.total_steps,
                0.3,  # pct_start
                self.config.min_rate,
            ),
            SchedulerType.ADAPTIVE: lambda: AdaptiveScheduler(
                self.config.base_rate,
                self.config.adaptation_beta,
                self.config.patience,
                self.config.reduction_factor,
                self.config.min_rate,
                self.config.max_rate,
            ),
            SchedulerType.POPULATION_AWARE: lambda: PopulationAwareScheduler(
                self.config.base_rate,
                0.5,  # target_diversity
                0.3,  # diversity_weight
                0.7,  # fitness_weight
                self.config.min_rate,
                self.config.max_rate,
            ),
        }

        return scheduler_map[scheduler_type]()

    def _create_optimizer(self, optimizer_type: ConvergenceStrategy) -> Any:
        """Factory method for optimizers."""
        optimizer_map = {
            ConvergenceStrategy.STANDARD: lambda: None,
            ConvergenceStrategy.MOMENTUM: lambda: MomentumOptimizer(
                self.config.momentum, nesterov=False
            ),
            ConvergenceStrategy.NESTEROV: lambda: NesterovOptimizer(self.config.momentum),
            ConvergenceStrategy.ADAM_LIKE: lambda: AdamOptimizer(),
            ConvergenceStrategy.GRADIENT_SURGERY: lambda: self.gradient_surgery,
            ConvergenceStrategy.REPTILE: lambda: MomentumOptimizer(0.1),
        }

        return optimizer_map[optimizer_type]()

    def step(
        self,
        fitness: float,
        gradient: np.ndarray | None = None,
        diversity_score: float | None = None,
        multi_objective_gradients: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Perform one optimization step.

        Args:
            fitness: Current fitness value
            gradient: Optional gradient for continuous optimization
            diversity_score: Optional population diversity
            multi_objective_gradients: Optional list of gradients for multi-objective

        Returns:
            Dict with rate, convergence state, and update (if gradient provided)
        """
        result = {}

        # Update scheduler
        scheduler_kwargs = {
            "fitness": fitness,
            "diversity_score": diversity_score,
        }
        rate = self.scheduler.step(**scheduler_kwargs)
        result["rate"] = rate

        # Check convergence
        gradient_norm = np.linalg.norm(gradient) if gradient is not None else None
        convergence_state = self.detector.update(fitness, gradient_norm)
        result["convergence"] = convergence_state

        # Handle gradient-based update
        if gradient is not None and self.optimizer is not None:
            # Handle multi-objective gradients
            if multi_objective_gradients is not None and len(multi_objective_gradients) > 1:
                gradient = self.gradient_surgery.combine_gradients(multi_objective_gradients)

            # Apply optimizer
            if hasattr(self.optimizer, "step"):
                update = self.optimizer.step("params", gradient, rate)
                result["update"] = update

        # Check for restart
        if convergence_state.is_converged and self._restart_count < self._max_restarts:
            self._handle_restart()
            result["restarted"] = True

        return result

    def _handle_restart(self):
        """Handle optimization restart with warm start."""
        self._restart_count += 1

        # Increase base rate for exploration
        new_base_rate = self.config.base_rate * (1 + 0.5 * self._restart_count)
        new_base_rate = min(new_base_rate, self.config.max_rate)

        self.scheduler.initial_rate = new_base_rate
        self.scheduler.current_rate = new_base_rate

        # Don't reset optimizer (keep momentum)
        # Don't reset detector (keep history for reference)

    def get_rate(self) -> float:
        """Get current learning/mutation rate."""
        return self.scheduler.get_rate()

    def get_diagnostics(self) -> dict[str, Any]:
        """Get optimization diagnostics."""
        return {
            "current_rate": self.scheduler.get_rate(),
            "step_count": self.scheduler.step_count,
            "restart_count": self._restart_count,
            "rate_history": self.scheduler.get_history()[-20:],  # Last 20 rates
            "best_fitness": self.detector._best_fitness,
            "steps_since_improvement": self.scheduler.step_count - self.detector._best_step,
        }

    def reset(self):
        """Reset accelerator state."""
        self.scheduler.reset()
        if hasattr(self.optimizer, "reset"):
            self.optimizer.reset()
        self.detector.reset()
        self._restart_count = 0


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_scheduler(
    scheduler_type: SchedulerType, config: ConvergenceConfig | None = None, total_steps: int = 100
) -> BaseScheduler:
    """Factory function to create schedulers."""
    config = config or ConvergenceConfig()

    accelerator = ConvergenceAccelerator(
        config=config, scheduler_type=scheduler_type, total_steps=total_steps
    )

    return accelerator.scheduler


def create_optimizer(
    optimizer_type: ConvergenceStrategy, config: ConvergenceConfig | None = None
) -> Any:
    """Factory function to create optimizers."""
    config = config or ConvergenceConfig()

    optimizer_map = {
        ConvergenceStrategy.STANDARD: lambda: None,
        ConvergenceStrategy.MOMENTUM: lambda: MomentumOptimizer(config.momentum),
        ConvergenceStrategy.NESTEROV: lambda: NesterovOptimizer(config.momentum),
        ConvergenceStrategy.ADAM_LIKE: lambda: AdamOptimizer(),
        ConvergenceStrategy.GRADIENT_SURGERY: lambda: GradientSurgery(),
        ConvergenceStrategy.REPTILE: lambda: MomentumOptimizer(0.1),
    }

    return optimizer_map[optimizer_type]()
