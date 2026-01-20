"""Adaptive Learning Rate Controller for AutoDAN.

Implements PPO-style learning rate adaptation with:
- Performance-based adjustment
- Warmup and decay schedules
- Gradient-aware adaptation
- Multi-phase scheduling
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of learning rate schedules."""

    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_ANNEALING = "cosine_annealing"
    WARMUP_COSINE = "warmup_cosine"
    CYCLIC = "cyclic"
    ONE_CYCLE = "one_cycle"
    PPO_ADAPTIVE = "ppo_adaptive"


@dataclass
class LRState:
    """State of the learning rate controller."""

    current_lr: float
    step: int
    epoch: int
    best_loss: float
    patience_counter: int
    warmup_complete: bool


@dataclass
class LRHistory:
    """History entry for learning rate changes."""

    step: int
    lr: float
    loss: float
    reason: str


class WarmupScheduler:
    """Warmup scheduler for gradual learning rate increase.

    Prevents early training instability by starting with low LR.
    """

    def __init__(
        self,
        warmup_steps: int,
        initial_lr: float,
        target_lr: float,
        warmup_type: str = "linear",
    ) -> None:
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_type = warmup_type

    def get_lr(self, step: int) -> float:
        """Get learning rate for warmup phase."""
        if step >= self.warmup_steps:
            return self.target_lr

        progress = step / self.warmup_steps

        if self.warmup_type == "linear":
            return self.initial_lr + progress * (self.target_lr - self.initial_lr)
        if self.warmup_type == "exponential":
            return self.initial_lr * (self.target_lr / self.initial_lr) ** progress
        if self.warmup_type == "cosine":
            return self.initial_lr + 0.5 * (self.target_lr - self.initial_lr) * (
                1 - math.cos(math.pi * progress)
            )
        return self.target_lr * progress

    def is_complete(self, step: int) -> bool:
        """Check if warmup is complete."""
        return step >= self.warmup_steps


class CyclicScheduler:
    """Cyclic learning rate scheduler.

    Oscillates LR between bounds to escape local minima.
    """

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: str = "triangular",
        gamma: float = 1.0,
    ) -> None:
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        cycle = math.floor(1 + step / (2 * self.step_size))
        x = abs(step / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
        elif self.mode == "triangular2":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * (self.gamma**step)
        else:
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)

        return lr


class OneCycleScheduler:
    """One-cycle learning rate policy.

    Single cycle from low to high to low LR for super-convergence.
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
    ) -> None:
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up

    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        if step < self.step_up:
            # Increasing phase
            progress = step / self.step_up
            return self.initial_lr + progress * (self.max_lr - self.initial_lr)
        # Decreasing phase
        progress = (step - self.step_up) / self.step_down
        # Cosine annealing
        return self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (
            1 + math.cos(math.pi * progress)
        )


class PPOAdaptiveScheduler:
    """PPO-style adaptive learning rate.

    Adjusts LR based on KL divergence and policy performance.
    """

    def __init__(
        self,
        initial_lr: float,
        target_kl: float = 0.01,
        lr_multiplier_up: float = 1.5,
        lr_multiplier_down: float = 0.5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
    ) -> None:
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.target_kl = target_kl
        self.lr_multiplier_up = lr_multiplier_up
        self.lr_multiplier_down = lr_multiplier_down
        self.min_lr = min_lr
        self.max_lr = max_lr

    def update(self, kl_divergence: float) -> float:
        """Update learning rate based on KL divergence."""
        if kl_divergence > self.target_kl * 1.5:
            # KL too high, reduce LR
            self.current_lr = max(
                self.min_lr,
                self.current_lr * self.lr_multiplier_down,
            )
        elif kl_divergence < self.target_kl / 1.5:
            # KL too low, increase LR
            self.current_lr = min(
                self.max_lr,
                self.current_lr * self.lr_multiplier_up,
            )

        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class AdaptiveLearningRateController:
    """Comprehensive adaptive learning rate controller.

    Features:
    - Multiple schedule types
    - Performance-based adaptation
    - Warmup support
    - Gradient-aware adjustment
    - History tracking
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        min_lr: float = 1e-6,
        max_lr: float = 0.1,
        schedule_type: ScheduleType = ScheduleType.PPO_ADAPTIVE,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        patience: int = 10,
        factor: float = 0.5,
    ) -> None:
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.patience = patience
        self.factor = factor

        # State
        self.state = LRState(
            current_lr=initial_lr,
            step=0,
            epoch=0,
            best_loss=float("inf"),
            patience_counter=0,
            warmup_complete=False,
        )

        # History
        self.history: list[LRHistory] = []

        # Initialize schedulers
        self._init_schedulers()

    def _init_schedulers(self) -> None:
        """Initialize sub-schedulers based on schedule type."""
        # Warmup scheduler
        self.warmup = WarmupScheduler(
            warmup_steps=self.warmup_steps,
            initial_lr=self.initial_lr / 10,
            target_lr=self.initial_lr,
        )

        # Main scheduler based on type
        if self.schedule_type == ScheduleType.CYCLIC:
            self.main_scheduler = CyclicScheduler(
                base_lr=self.min_lr,
                max_lr=self.max_lr,
                step_size=self.total_steps // 10,
            )
        elif self.schedule_type == ScheduleType.ONE_CYCLE:
            self.main_scheduler = OneCycleScheduler(
                max_lr=self.max_lr,
                total_steps=self.total_steps,
            )
        elif self.schedule_type == ScheduleType.PPO_ADAPTIVE:
            self.main_scheduler = PPOAdaptiveScheduler(
                initial_lr=self.initial_lr,
                min_lr=self.min_lr,
                max_lr=self.max_lr,
            )
        else:
            self.main_scheduler = None

    def step(
        self,
        loss: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> float:
        """Perform a learning rate step.

        Args:
            loss: Current loss value
            metrics: Additional metrics (e.g., KL divergence)

        Returns:
            Updated learning rate

        """
        self.state.step += 1
        step = self.state.step

        # Warmup phase
        if not self.state.warmup_complete:
            if self.warmup.is_complete(step):
                self.state.warmup_complete = True
                self._record_history(
                    self.state.current_lr,
                    loss or 0,
                    "warmup_complete",
                )
            else:
                lr = self.warmup.get_lr(step)
                self.state.current_lr = lr
                return lr

        # Main scheduling
        lr = self._compute_lr(step, loss, metrics)

        # Apply bounds
        lr = max(self.min_lr, min(self.max_lr, lr))

        # Update state
        old_lr = self.state.current_lr
        self.state.current_lr = lr

        # Record if changed significantly
        if abs(lr - old_lr) / old_lr > 0.01:
            self._record_history(lr, loss or 0, "scheduled_update")

        return lr

    def _compute_lr(
        self,
        step: int,
        loss: float | None,
        metrics: dict[str, float] | None,
    ) -> float:
        """Compute learning rate based on schedule type."""
        if self.schedule_type == ScheduleType.CONSTANT:
            return self.initial_lr

        if self.schedule_type == ScheduleType.LINEAR_DECAY:
            progress = step / self.total_steps
            return self.initial_lr * (1 - progress)

        if self.schedule_type == ScheduleType.EXPONENTIAL_DECAY:
            decay_rate = 0.96
            decay_steps = self.total_steps // 100
            return self.initial_lr * (decay_rate ** (step / decay_steps))

        if self.schedule_type == ScheduleType.COSINE_ANNEALING:
            progress = step / self.total_steps
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        if self.schedule_type == ScheduleType.WARMUP_COSINE:
            if step < self.warmup_steps:
                return self.warmup.get_lr(step)
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        if self.schedule_type in (ScheduleType.CYCLIC, ScheduleType.ONE_CYCLE):
            return self.main_scheduler.get_lr(step)

        if self.schedule_type == ScheduleType.PPO_ADAPTIVE:
            if metrics and "kl_divergence" in metrics:
                return self.main_scheduler.update(metrics["kl_divergence"])
            # Fall back to loss-based adaptation
            return self._loss_based_adaptation(loss)

        return self.state.current_lr

    def _loss_based_adaptation(self, loss: float | None) -> float:
        """Adapt learning rate based on loss."""
        if loss is None:
            return self.state.current_lr

        if loss < self.state.best_loss:
            self.state.best_loss = loss
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1

        if self.state.patience_counter >= self.patience:
            # Reduce learning rate
            new_lr = self.state.current_lr * self.factor
            self.state.patience_counter = 0
            self._record_history(new_lr, loss, "patience_exceeded")
            return new_lr

        return self.state.current_lr

    def _record_history(self, lr: float, loss: float, reason: str) -> None:
        """Record learning rate change in history."""
        self.history.append(
            LRHistory(
                step=self.state.step,
                lr=lr,
                loss=loss,
                reason=reason,
            ),
        )

    def on_epoch_end(self, epoch: int, loss: float) -> None:
        """Called at the end of each epoch."""
        self.state.epoch = epoch

        # Check for improvement
        if loss < self.state.best_loss * 0.99:
            self.state.best_loss = loss
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.state.current_lr

    def set_lr(self, lr: float) -> None:
        """Manually set learning rate."""
        self.state.current_lr = max(self.min_lr, min(self.max_lr, lr))
        self._record_history(self.state.current_lr, 0, "manual_set")

    def get_state(self) -> dict[str, Any]:
        """Get controller state."""
        return {
            "current_lr": self.state.current_lr,
            "step": self.state.step,
            "epoch": self.state.epoch,
            "best_loss": self.state.best_loss,
            "patience_counter": self.state.patience_counter,
            "warmup_complete": self.state.warmup_complete,
            "schedule_type": self.schedule_type.value,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load controller state."""
        self.state.current_lr = state.get("current_lr", self.initial_lr)
        self.state.step = state.get("step", 0)
        self.state.epoch = state.get("epoch", 0)
        self.state.best_loss = state.get("best_loss", float("inf"))
        self.state.patience_counter = state.get("patience_counter", 0)
        self.state.warmup_complete = state.get("warmup_complete", False)

    def get_history(self) -> list[dict[str, Any]]:
        """Get learning rate history."""
        return [
            {
                "step": h.step,
                "lr": h.lr,
                "loss": h.loss,
                "reason": h.reason,
            }
            for h in self.history
        ]

    def reset(self) -> None:
        """Reset controller to initial state."""
        self.state = LRState(
            current_lr=self.initial_lr,
            step=0,
            epoch=0,
            best_loss=float("inf"),
            patience_counter=0,
            warmup_complete=False,
        )
        self.history.clear()
        self._init_schedulers()


class GradientAwareLRController(AdaptiveLearningRateController):
    """Learning rate controller that adapts based on gradient statistics.

    Monitors gradient norms and adjusts LR to prevent exploding/vanishing.
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        target_grad_norm: float = 1.0,
        grad_clip: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(initial_lr=initial_lr, **kwargs)
        self.target_grad_norm = target_grad_norm
        self.grad_clip = grad_clip
        self.grad_history: list[float] = []

    def step_with_gradient(
        self,
        loss: float | None = None,
        grad_norm: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> float:
        """Step with gradient information.

        Args:
            loss: Current loss
            grad_norm: Current gradient norm
            metrics: Additional metrics

        Returns:
            Updated learning rate

        """
        # Record gradient
        if grad_norm is not None:
            self.grad_history.append(grad_norm)
            if len(self.grad_history) > 100:
                self.grad_history = self.grad_history[-100:]

        # Base step
        lr = self.step(loss, metrics)

        # Gradient-based adjustment
        if grad_norm is not None:
            lr = self._adjust_for_gradient(lr, grad_norm)

        self.state.current_lr = lr
        return lr

    def _adjust_for_gradient(self, lr: float, grad_norm: float) -> float:
        """Adjust learning rate based on gradient norm."""
        if grad_norm > self.grad_clip:
            # Gradient explosion - reduce LR
            lr *= 0.5
            self._record_history(lr, 0, "gradient_explosion")

        elif len(self.grad_history) >= 10:
            avg_grad = np.mean(self.grad_history[-10:])

            if avg_grad > self.target_grad_norm * 2:
                # Gradients too large
                lr *= 0.9
            elif avg_grad < self.target_grad_norm * 0.5:
                # Gradients too small
                lr *= 1.1

        return max(self.min_lr, min(self.max_lr, lr))

    def get_gradient_stats(self) -> dict[str, float]:
        """Get gradient statistics."""
        if not self.grad_history:
            return {}

        return {
            "mean_grad_norm": float(np.mean(self.grad_history)),
            "max_grad_norm": float(np.max(self.grad_history)),
            "min_grad_norm": float(np.min(self.grad_history)),
            "std_grad_norm": float(np.std(self.grad_history)),
            "recent_mean": float(np.mean(self.grad_history[-10:])),
        }
