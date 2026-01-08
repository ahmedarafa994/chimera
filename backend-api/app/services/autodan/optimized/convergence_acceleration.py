"""
Convergence Acceleration for AutoDAN.

Implements techniques to speed up convergence:
- Warm start optimization
- Early stopping with confidence
- Curriculum learning scheduler
- Transfer learning initialization
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StoppingReason(Enum):
    """Reasons for early stopping."""

    NOT_STOPPED = "not_stopped"
    CONVERGENCE = "convergence"
    PATIENCE_EXCEEDED = "patience_exceeded"
    MAX_ITERATIONS = "max_iterations"
    TARGET_REACHED = "target_reached"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    TIME_LIMIT = "time_limit"
    DIVERGENCE = "divergence"


@dataclass
class ConvergenceState:
    """State of convergence tracking."""

    iteration: int = 0
    best_score: float = float("-inf")
    best_iteration: int = 0
    patience_counter: int = 0
    scores_history: list[float] = field(default_factory=list)
    stopped: bool = False
    stopping_reason: StoppingReason = StoppingReason.NOT_STOPPED


@dataclass
class CurriculumStage:
    """A stage in curriculum learning."""

    name: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    min_iterations: int
    target_score: float
    parameters: dict[str, Any] = field(default_factory=dict)


class WarmStartOptimizer:
    """
    Warm start optimization using transfer learning.

    Initializes optimization from previously successful strategies.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        n_clusters: int = 10,
        similarity_threshold: float = 0.7,
    ):
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.similarity_threshold = similarity_threshold

        # Strategy bank
        self.successful_strategies: list[dict[str, Any]] = []
        self.strategy_embeddings: list[np.ndarray] = []

        # Cluster centers for fast lookup
        self.cluster_centers: np.ndarray | None = None

    def add_successful_strategy(
        self,
        strategy: str,
        embedding: np.ndarray,
        score: float,
        metadata: dict[str, Any] | None = None,
    ):
        """Add a successful strategy to the bank."""
        self.successful_strategies.append(
            {
                "strategy": strategy,
                "score": score,
                "metadata": metadata or {},
            }
        )
        self.strategy_embeddings.append(embedding)

        # Update clusters periodically
        if len(self.successful_strategies) % 50 == 0:
            self._update_clusters()

    def get_warm_start(
        self,
        target_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Get warm start strategies similar to target.

        Args:
            target_embedding: Embedding of target task
            top_k: Number of strategies to return

        Returns:
            List of similar successful strategies
        """
        if not self.strategy_embeddings:
            return []

        # Compute similarities
        embeddings = np.array(self.strategy_embeddings)
        similarities = self._cosine_similarity(target_embedding, embeddings)

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                results.append(
                    {
                        **self.successful_strategies[idx],
                        "similarity": float(similarities[idx]),
                    }
                )

        return results

    def initialize_from_similar(
        self,
        target_embedding: np.ndarray,
        blend_ratio: float = 0.7,
    ) -> np.ndarray | None:
        """
        Initialize parameters from similar strategies.

        Args:
            target_embedding: Target task embedding
            blend_ratio: How much to weight similar strategies

        Returns:
            Blended initialization or None
        """
        similar = self.get_warm_start(target_embedding, top_k=5)

        if not similar:
            return None

        # Weighted blend based on similarity
        total_weight = sum(s["similarity"] for s in similar)
        if total_weight == 0:
            return None

        # Blend embeddings
        blended = np.zeros(self.embedding_dim)
        for s in similar:
            idx = self.successful_strategies.index(s)
            weight = s["similarity"] / total_weight
            blended += weight * self.strategy_embeddings[idx]

        # Blend with target
        result = blend_ratio * blended + (1 - blend_ratio) * target_embedding
        return result / np.linalg.norm(result)

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(b_norm, a_norm)

    def _update_clusters(self):
        """Update cluster centers using k-means."""
        if len(self.strategy_embeddings) < self.n_clusters:
            return

        embeddings = np.array(self.strategy_embeddings)

        # Simple k-means
        centers = embeddings[np.random.choice(len(embeddings), self.n_clusters, replace=False)]

        for _ in range(10):  # K-means iterations
            # Assign to clusters
            distances = np.linalg.norm(embeddings[:, np.newaxis] - centers, axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update centers
            new_centers = np.zeros_like(centers)
            for i in range(self.n_clusters):
                mask = assignments == i
                if np.any(mask):
                    new_centers[i] = embeddings[mask].mean(axis=0)
                else:
                    new_centers[i] = centers[i]

            centers = new_centers

        self.cluster_centers = centers

    def get_stats(self) -> dict[str, Any]:
        """Get warm start statistics."""
        return {
            "n_strategies": len(self.successful_strategies),
            "has_clusters": self.cluster_centers is not None,
            "n_clusters": self.n_clusters,
        }


class EarlyStoppingController:
    """
    Early stopping with confidence-based termination.

    Stops optimization when confident that further iterations
    won't improve results significantly.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        target_score: float = 8.5,
        confidence_threshold: float = 0.95,
        max_iterations: int = 150,
        time_limit: float | None = None,
        divergence_threshold: float = 0.5,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.target_score = target_score
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.divergence_threshold = divergence_threshold

        # State
        self.state = ConvergenceState()
        self.start_time: float | None = None

    def start(self):
        """Start the stopping controller."""
        self.state = ConvergenceState()
        self.start_time = time.time()

    def step(self, score: float) -> bool:
        """
        Record a score and check if should stop.

        Args:
            score: Current iteration score

        Returns:
            True if should stop, False otherwise
        """
        self.state.iteration += 1
        self.state.scores_history.append(score)

        # Check various stopping conditions
        if self._check_target_reached(score):
            return True

        if self._check_max_iterations():
            return True

        if self._check_time_limit():
            return True

        if self._check_divergence():
            return True

        if self._check_patience(score):
            return True

        return bool(self._check_confidence())

    def _check_target_reached(self, score: float) -> bool:
        """Check if target score reached."""
        if score >= self.target_score:
            self.state.stopped = True
            self.state.stopping_reason = StoppingReason.TARGET_REACHED
            logger.info(f"Target score {self.target_score} reached")
            return True
        return False

    def _check_max_iterations(self) -> bool:
        """Check if max iterations exceeded."""
        if self.state.iteration >= self.max_iterations:
            self.state.stopped = True
            self.state.stopping_reason = StoppingReason.MAX_ITERATIONS
            logger.info(f"Max iterations {self.max_iterations} reached")
            return True
        return False

    def _check_time_limit(self) -> bool:
        """Check if time limit exceeded."""
        if self.time_limit and self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit:
                self.state.stopped = True
                self.state.stopping_reason = StoppingReason.TIME_LIMIT
                logger.info(f"Time limit {self.time_limit}s exceeded")
                return True
        return False

    def _check_divergence(self) -> bool:
        """Check if optimization is diverging."""
        if len(self.state.scores_history) < 10:
            return False

        recent = self.state.scores_history[-10:]
        if self.state.best_score > 0:
            drop = (self.state.best_score - np.mean(recent)) / self.state.best_score
            if drop > self.divergence_threshold:
                self.state.stopped = True
                self.state.stopping_reason = StoppingReason.DIVERGENCE
                logger.warning("Optimization diverging, stopping")
                return True
        return False

    def _check_patience(self, score: float) -> bool:
        """Check patience-based stopping."""
        if score > self.state.best_score + self.min_delta:
            self.state.best_score = score
            self.state.best_iteration = self.state.iteration
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1

        if self.state.patience_counter >= self.patience:
            self.state.stopped = True
            self.state.stopping_reason = StoppingReason.PATIENCE_EXCEEDED
            logger.info(f"Patience {self.patience} exceeded")
            return True
        return False

    def _check_confidence(self) -> bool:
        """Check confidence-based stopping."""
        if len(self.state.scores_history) < 20:
            return False

        # Estimate convergence confidence
        recent = self.state.scores_history[-20:]
        std = np.std(recent)
        mean = np.mean(recent)

        # Low variance indicates convergence
        if std < self.min_delta and mean > 0:
            cv = std / mean  # Coefficient of variation
            confidence = 1.0 - cv

            if confidence >= self.confidence_threshold:
                self.state.stopped = True
                self.state.stopping_reason = StoppingReason.CONFIDENCE_THRESHOLD
                logger.info(f"Confidence {confidence:.3f} reached")
                return True

        return False

    def should_stop(self) -> bool:
        """Check if optimization should stop."""
        return self.state.stopped

    def get_state(self) -> dict[str, Any]:
        """Get stopping controller state."""
        return {
            "iteration": self.state.iteration,
            "best_score": self.state.best_score,
            "best_iteration": self.state.best_iteration,
            "patience_counter": self.state.patience_counter,
            "stopped": self.state.stopped,
            "stopping_reason": self.state.stopping_reason.value,
            "elapsed_time": (time.time() - self.start_time if self.start_time else 0),
        }


class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive difficulty.

    Starts with easier tasks and gradually increases difficulty.
    """

    def __init__(
        self,
        stages: list[CurriculumStage] | None = None,
        auto_advance: bool = True,
        min_stage_iterations: int = 10,
    ):
        self.auto_advance = auto_advance
        self.min_stage_iterations = min_stage_iterations

        # Default stages if not provided
        self.stages = stages or self._default_stages()

        # State
        self.current_stage_idx = 0
        self.stage_iteration = 0
        self.stage_scores: list[float] = []
        self.completed_stages: list[int] = []

    def _default_stages(self) -> list[CurriculumStage]:
        """Create default curriculum stages."""
        return [
            CurriculumStage(
                name="warmup",
                difficulty=0.2,
                min_iterations=20,
                target_score=5.0,
                parameters={
                    "mutation_rate": 0.5,
                    "exploration": 0.8,
                },
            ),
            CurriculumStage(
                name="basic",
                difficulty=0.4,
                min_iterations=30,
                target_score=6.0,
                parameters={
                    "mutation_rate": 0.4,
                    "exploration": 0.6,
                },
            ),
            CurriculumStage(
                name="intermediate",
                difficulty=0.6,
                min_iterations=40,
                target_score=7.0,
                parameters={
                    "mutation_rate": 0.3,
                    "exploration": 0.4,
                },
            ),
            CurriculumStage(
                name="advanced",
                difficulty=0.8,
                min_iterations=50,
                target_score=8.0,
                parameters={
                    "mutation_rate": 0.2,
                    "exploration": 0.2,
                },
            ),
            CurriculumStage(
                name="expert",
                difficulty=1.0,
                min_iterations=100,
                target_score=8.5,
                parameters={
                    "mutation_rate": 0.1,
                    "exploration": 0.1,
                },
            ),
        ]

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    def step(self, score: float) -> bool:
        """
        Record score and potentially advance stage.

        Args:
            score: Current iteration score

        Returns:
            True if stage advanced, False otherwise
        """
        self.stage_iteration += 1
        self.stage_scores.append(score)

        if not self.auto_advance:
            return False

        current = self.get_current_stage()

        # Check if should advance
        if self._should_advance(score, current):
            return self.advance_stage()

        return False

    def _should_advance(
        self,
        score: float,
        stage: CurriculumStage,
    ) -> bool:
        """Check if should advance to next stage."""
        # Must complete minimum iterations
        if self.stage_iteration < stage.min_iterations:
            return False

        # Must reach target score
        if score < stage.target_score:
            return False

        # Check consistency (average of recent scores)
        if len(self.stage_scores) >= 5:
            recent_avg = np.mean(self.stage_scores[-5:])
            if recent_avg < stage.target_score * 0.9:
                return False

        return True

    def advance_stage(self) -> bool:
        """Advance to next curriculum stage."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        self.completed_stages.append(self.current_stage_idx)
        self.current_stage_idx += 1
        self.stage_iteration = 0
        self.stage_scores.clear()

        new_stage = self.get_current_stage()
        logger.info(f"Advanced to stage: {new_stage.name}")

        return True

    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.get_current_stage().difficulty

    def get_parameters(self) -> dict[str, Any]:
        """Get current stage parameters."""
        return self.get_current_stage().parameters.copy()

    def is_final_stage(self) -> bool:
        """Check if on final stage."""
        return self.current_stage_idx >= len(self.stages) - 1

    def get_progress(self) -> float:
        """Get overall curriculum progress (0-1)."""
        stage_progress = self.current_stage_idx / len(self.stages)
        current = self.get_current_stage()
        within_stage = min(self.stage_iteration / current.min_iterations, 1.0)
        return stage_progress + within_stage / len(self.stages)

    def get_state(self) -> dict[str, Any]:
        """Get scheduler state."""
        current = self.get_current_stage()
        return {
            "current_stage": current.name,
            "stage_idx": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_iteration": self.stage_iteration,
            "difficulty": current.difficulty,
            "target_score": current.target_score,
            "progress": self.get_progress(),
            "completed_stages": self.completed_stages,
        }

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_stage_idx = 0
        self.stage_iteration = 0
        self.stage_scores.clear()
        self.completed_stages.clear()


class ConvergenceAccelerator:
    """
    Combined convergence acceleration system.

    Integrates warm start, early stopping, and curriculum learning.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        patience: int = 10,
        target_score: float = 8.5,
        max_iterations: int = 150,
        use_curriculum: bool = True,
        use_warm_start: bool = True,
    ):
        self.use_curriculum = use_curriculum
        self.use_warm_start = use_warm_start

        # Components
        self.warm_start = WarmStartOptimizer(embedding_dim=embedding_dim)
        self.early_stopping = EarlyStoppingController(
            patience=patience,
            target_score=target_score,
            max_iterations=max_iterations,
        )
        self.curriculum = CurriculumScheduler() if use_curriculum else None

        # Statistics
        self.total_iterations = 0
        self.total_time = 0.0
        self.convergence_history: list[dict[str, Any]] = []

    def start(self, target_embedding: np.ndarray | None = None):
        """
        Start a new optimization run.

        Args:
            target_embedding: Optional embedding for warm start
        """
        self.early_stopping.start()
        if self.curriculum:
            self.curriculum.reset()

        # Get warm start if available
        warm_start_strategies = []
        if self.use_warm_start and target_embedding is not None:
            warm_start_strategies = self.warm_start.get_warm_start(target_embedding)

        return warm_start_strategies

    def step(self, score: float) -> dict[str, Any]:
        """
        Process an iteration step.

        Args:
            score: Current iteration score

        Returns:
            Dict with should_stop, stage_advanced, and parameters
        """
        self.total_iterations += 1

        # Early stopping check
        should_stop = self.early_stopping.step(score)

        # Curriculum advancement
        stage_advanced = False
        if self.curriculum and not should_stop:
            stage_advanced = self.curriculum.step(score)

        # Get current parameters
        parameters = {}
        if self.curriculum:
            parameters = self.curriculum.get_parameters()

        # Record history
        self.convergence_history.append(
            {
                "iteration": self.total_iterations,
                "score": score,
                "should_stop": should_stop,
                "stage_advanced": stage_advanced,
            }
        )

        return {
            "should_stop": should_stop,
            "stage_advanced": stage_advanced,
            "parameters": parameters,
            "stopping_reason": self.early_stopping.state.stopping_reason.value,
        }

    def record_success(
        self,
        strategy: str,
        embedding: np.ndarray,
        score: float,
    ):
        """Record a successful strategy for future warm starts."""
        if self.use_warm_start:
            self.warm_start.add_successful_strategy(
                strategy=strategy,
                embedding=embedding,
                score=score,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get accelerator statistics."""
        stats = {
            "total_iterations": self.total_iterations,
            "early_stopping": self.early_stopping.get_state(),
        }

        if self.curriculum:
            stats["curriculum"] = self.curriculum.get_state()

        if self.use_warm_start:
            stats["warm_start"] = self.warm_start.get_stats()

        return stats

    def reset(self):
        """Reset accelerator state."""
        self.early_stopping.start()
        if self.curriculum:
            self.curriculum.reset()
        self.total_iterations = 0
        self.convergence_history.clear()
