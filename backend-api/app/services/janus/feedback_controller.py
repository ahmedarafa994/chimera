"""Feedback Controller Module.

Manages multi-level feedback loops for Janus,
including fast (real-time), medium (session), and slow (long-term).
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any

from .config import get_config
from .core import FailureState, GuardianResponse, Heuristic, InteractionResult

logger = logging.getLogger(__name__)


class FastFeedbackController:
    """Fast feedback loop for real-time adjustment.

    Operates at millisecond level for immediate heuristic
    weight adjustment based on response anomalies.
    """

    def __init__(
        self,
        response_analyzer: Any | None = None,
        heuristic_generator: Any | None = None,
        evolution_engine: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self.response_analyzer = response_analyzer
        self.heuristic_generator = heuristic_generator
        self.evolution_engine = evolution_engine
        self.config = config or get_config().feedback

        # Response buffer
        self.response_buffer: deque[GuardianResponse] = deque(maxlen=100)

        # Anomaly tracking
        self.anomaly_count = 0
        self.adjustment_count = 0

        # Fast loop task
        self._fast_loop_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the fast feedback loop."""
        if self._running:
            return

        self._running = True
        self._fast_loop_task = asyncio.create_task(self._fast_loop())
        logger.info("Fast feedback loop started")

    async def stop(self) -> None:
        """Stop the fast feedback loop."""
        self._running = False
        if self._fast_loop_task:
            self._fast_loop_task.cancel()
            self._fast_loop_task = None
        logger.info("Fast feedback loop stopped")

    async def _fast_loop(self) -> None:
        """Main fast feedback loop."""
        interval_ms = self.config.fast_loop_interval_ms
        interval_sec = interval_ms / 1000.0

        while self._running:
            await asyncio.sleep(interval_sec)

            if self.response_buffer:
                latest_response = self.response_buffer[-1]
                await self._process_response(latest_response)

    async def _process_response(self, response: GuardianResponse) -> None:
        """Process a single response for fast feedback."""
        # Check for anomalies
        is_anomalous = self._is_anomalous(response)

        if is_anomalous:
            self.anomaly_count += 1

            # Identify causal heuristic
            heuristic = self._identify_causal_heuristic(response)

            if heuristic and self.heuristic_generator:
                # Adjust heuristic weight
                current_score = heuristic.efficacy_score
                adjustment = self.config.adjustment_factor

                if is_anomalous:
                    # Increase weight for anomalous responses
                    new_score = min(1.0, current_score + adjustment)
                else:
                    # Decrease weight for normal responses
                    new_score = max(0.0, current_score - adjustment)

                heuristic.efficacy_score = new_score
                self.adjustment_count += 1

                logger.debug(
                    f"Adjusted heuristic {heuristic.name}: {current_score:.3f} -> {new_score:.3f}",
                )

    def _is_anomalous(self, response: GuardianResponse) -> bool:
        """Check if a response is anomalous."""
        if len(self.response_buffer) < 10:
            return False

        # Compute statistics from buffer
        recent = list(self.response_buffer)[-50:]
        latencies = [r.latency_ms for r in recent]
        safety_scores = [r.safety_score for r in recent]

        # Compute mean and std
        mean_latency = sum(latencies) / len(latencies)
        std_latency = (
            sum((latency - mean_latency) ** 2 for latency in latencies) / len(latencies)
            if len(latencies) > 1
            else 0.0
        )

        mean_safety = sum(safety_scores) / len(safety_scores)
        std_safety = (
            sum((s - mean_safety) ** 2 for s in safety_scores) / len(safety_scores)
            if len(safety_scores) > 1
            else 0.0
        )

        # Check if current response is anomalous
        threshold = self.config.anomaly_threshold

        latency_anomalous = abs(response.latency_ms - mean_latency) > threshold * std_latency
        safety_anomalous = abs(response.safety_score - mean_safety) > threshold * std_safety

        return latency_anomalous or safety_anomalous

    def _identify_causal_heuristic(self, response: GuardianResponse) -> Heuristic | None:
        """Identify heuristic that may have caused response."""
        if not self.heuristic_generator:
            return None

        # Get all heuristics
        heuristics = self.heuristic_generator.get_all_heuristics()

        if not heuristics:
            return None

        # Find heuristic most recently used
        # (In practice, this would track which heuristic caused this response)
        # For now, return a random heuristic
        return heuristics[0] if heuristics else None

    def get_stats(self) -> dict[str, Any]:
        """Get fast feedback loop statistics."""
        return {
            "running": self._running,
            "anomaly_count": self.anomaly_count,
            "adjustment_count": self.adjustment_count,
            "buffer_size": len(self.response_buffer),
        }


class MediumFeedbackController:
    """Medium feedback loop for session-level updates.

    Operates at minute-to-hour level for heuristic library
    updates and pruning.
    """

    def __init__(
        self,
        response_analyzer: Any | None = None,
        heuristic_generator: Any | None = None,
        evolution_engine: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self.response_analyzer = response_analyzer
        self.heuristic_generator = heuristic_generator
        self.evolution_engine = evolution_engine
        self.config = config or get_config().feedback

        # Session history
        self.session_history: list[dict[str, Any]] = []

        # Performance tracking
        self.performance_metrics: dict[str, dict[str, float]] = {}

        # Medium loop task
        self._medium_loop_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the medium feedback loop."""
        if self._running:
            return

        self._running = True
        self._medium_loop_task = asyncio.create_task(self._medium_loop())
        logger.info("Medium feedback loop started")

    async def stop(self) -> None:
        """Stop the medium feedback loop."""
        self._running = False
        if self._medium_loop_task:
            self._medium_loop_task.cancel()
            self._medium_loop_task = None
        logger.info("Medium feedback loop stopped")

    async def _medium_loop(self) -> None:
        """Main medium feedback loop."""
        interval_sec = self.config.medium_loop_interval_sec

        while self._running:
            await asyncio.sleep(interval_sec)
            await self._session_update()

    async def _session_update(self) -> None:
        """Perform session-level updates."""
        # Get current heuristic performance
        if self.heuristic_generator:
            heuristics = self.heuristic_generator.get_all_heuristics()

            # Compute performance metrics
            for heuristic in heuristics:
                if heuristic.name not in self.performance_metrics:
                    self.performance_metrics[heuristic.name] = {
                        "total_uses": 0,
                        "successes": 0,
                        "failures": 0,
                        "avg_score": heuristic.efficacy_score,
                        "last_updated": datetime.now().isoformat(),
                    }

        # Update heuristic library
        if self.heuristic_generator:
            # Prune low-performing heuristics
            self.heuristic_generator.prune_heuristics()

            # Update evolution engine if available
            if self.evolution_engine:
                # Trigger evolution
                stats = self.evolution_engine.get_generation_stats()
                if stats["avg_improvement"] < 0.01:
                    # Converged, no need to evolve
                    pass
                else:
                    # Evolve new generation
                    new_heuristics = await self.evolution_engine.evolve_generation(
                        heuristics,
                        self.performance_metrics,
                    )

                    # Update heuristic generator
                    for h in new_heuristics:
                        self.heuristic_generator.update_heuristic(h)

            # Record session
            session_record = {
                "session_id": f"session_{len(self.session_history)}",
                "timestamp": datetime.now().isoformat(),
                "total_heuristics": len(heuristics),
                "evolution_generation": (
                    self.evolution_engine.current_generation if self.evolution_engine else 0
                ),
                "performance_stats": self.performance_metrics,
            }

            self.session_history.append(session_record)

            # Limit history size
            if len(self.session_history) > 100:
                self.session_history = self.session_history[-100:]

    def get_stats(self) -> dict[str, Any]:
        """Get medium feedback loop statistics."""
        return {
            "running": self._running,
            "session_count": len(self.session_history),
            "tracked_heuristics": len(self.performance_metrics),
        }


class SlowFeedbackController:
    """Slow feedback loop for long-term optimization.

    Operates at hour-to-day level for meta-strategy
    optimization and long-term trend analysis.
    """

    def __init__(self, evolution_engine: Any | None = None, config: Any | None = None) -> None:
        self.evolution_engine = evolution_engine
        self.config = config or get_config().feedback

        # Meta-strategy tracking
        self.meta_strategies: dict[str, dict[str, float]] = {}

        # Long-term performance
        self.long_term_trends: list[dict[str, Any]] = []

        # Slow loop task
        self._slow_loop_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the slow feedback loop."""
        if self._running:
            return

        self._running = True
        self._slow_loop_task = asyncio.create_task(self._slow_loop())
        logger.info("Slow feedback loop started")

    async def stop(self) -> None:
        """Stop the slow feedback loop."""
        self._running = False
        if self._slow_loop_task:
            self._slow_loop_task.cancel()
            self._slow_loop_task = None
        logger.info("Slow feedback loop stopped")

    async def _slow_loop(self) -> None:
        """Main slow feedback loop."""
        interval_hours = self.config.slow_loop_interval_hours
        interval_sec = interval_hours * 3600

        while self._running:
            await asyncio.sleep(interval_sec)
            await self._long_term_update()

    async def _long_term_update(self) -> None:
        """Perform long-term updates."""
        # Analyze evolution trends
        if self.evolution_engine:
            stats = self.evolution_engine.get_generation_stats()

            # Track meta-strategy performance
            if stats:
                trend = {
                    "timestamp": datetime.now().isoformat(),
                    "generation": stats["current_generation"],
                    "best_fitness": stats["best_fitness"],
                    "avg_improvement": stats["avg_improvement"],
                    "convergence_rate": stats["convergence_rate"],
                }

                self.long_term_trends.append(trend)

                # Limit trend history
                if len(self.long_term_trends) > 1000:
                    self.long_term_trends = self.long_term_trends[-1000:]

                # Analyze meta-strategies
                await self._analyze_meta_strategies()

    async def _analyze_meta_strategies(self) -> None:
        """Analyze and optimize meta-strategies."""
        if not self.evolution_engine:
            return

        # Get evolution history
        stats = self.evolution_engine.get_generation_stats()

        # Determine best performing strategy
        if stats["avg_improvement"] > 0.05:
            # High improvement rate, good strategy
            self.meta_strategies["current"] = {
                "strategy": "aggressive_evolution",
                "performance": stats["avg_improvement"],
                "last_updated": datetime.now().isoformat(),
            }
        elif stats["convergence_rate"] > 0.8:
            # Fast convergence, good strategy
            self.meta_strategies["current"] = {
                "strategy": "rapid_convergence",
                "performance": stats["convergence_rate"],
                "last_updated": datetime.now().isoformat(),
            }
        else:
            # Default strategy
            self.meta_strategies["current"] = {
                "strategy": "balanced",
                "performance": stats["avg_improvement"],
                "last_updated": datetime.now().isoformat(),
            }

        logger.debug(f"Meta-strategy updated: {self.meta_strategies['current']['strategy']}")

    def get_stats(self) -> dict[str, Any]:
        """Get slow feedback loop statistics."""
        return {
            "running": self._running,
            "meta_strategies": self.meta_strategies,
            "trend_count": len(self.long_term_trends),
        }


class FeedbackController:
    """Manages all feedback loops.

    Coordinates fast, medium, and slow feedback controllers
    for comprehensive feedback management.
    """

    def __init__(
        self,
        response_analyzer: Any | None = None,
        heuristic_generator: Any | None = None,
        evolution_engine: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self.config = config or get_config().feedback

        # Initialize sub-controllers
        self.fast_controller = FastFeedbackController(
            response_analyzer=response_analyzer,
            heuristic_generator=heuristic_generator,
            evolution_engine=evolution_engine,
            config=self.config,
        )

        self.medium_controller = MediumFeedbackController(
            response_analyzer=response_analyzer,
            heuristic_generator=heuristic_generator,
            evolution_engine=evolution_engine,
            config=self.config,
        )

        self.slow_controller = SlowFeedbackController(
            evolution_engine=evolution_engine,
            config=self.config,
        )

    async def start(self) -> None:
        """Start all feedback loops."""
        await self.fast_controller.start()
        await self.medium_controller.start()
        await self.slow_controller.start()
        logger.info("All feedback loops started")

    async def stop(self) -> None:
        """Stop all feedback loops."""
        await self.fast_controller.stop()
        await self.medium_controller.stop()
        await self.slow_controller.stop()
        logger.info("All feedback loops stopped")

    async def process_result(
        self,
        heuristic: Heuristic,
        result: InteractionResult,
        failure: FailureState | None = None,
    ) -> None:
        """Process a result through all feedback loops.

        Args:
            heuristic: Heuristic that was executed
            result: Interaction result
            failure: Failure state if discovered

        """
        # Add to fast controller buffer
        if result.response:
            self.fast_controller.response_buffer.append(result.response)

        # Update performance metrics
        if heuristic.name not in self.medium_controller.performance_metrics:
            self.medium_controller.performance_metrics[heuristic.name] = {
                "total_uses": 0,
                "successes": 0,
                "failures": 0,
                "avg_score": heuristic.efficacy_score,
                "last_updated": datetime.now().isoformat(),
            }

        metrics = self.medium_controller.performance_metrics[heuristic.name]

        if result.success:
            metrics["successes"] += 1
            metrics["avg_score"] = (metrics["avg_score"] * metrics["total_uses"] + result.score) / (
                metrics["total_uses"] + 1
            )
        else:
            metrics["failures"] += 1

        metrics["total_uses"] += 1
        metrics["last_updated"] = datetime.now().isoformat()

        # Log failure if discovered
        if failure:
            logger.info(f"Failure discovered: {failure.failure_id} ({failure.failure_type.value})")

    def get_performance_metrics(self) -> dict[str, dict[str, float]]:
        """Get performance metrics for all heuristics."""
        return self.medium_controller.performance_metrics

    async def cleanup(self) -> None:
        """Cleanup all feedback controllers."""
        await self.fast_controller.stop()
        await self.medium_controller.stop()
        await self.slow_controller.stop()
        logger.info("Feedback controller cleaned up")

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics from all feedback loops."""
        return {
            "fast": self.fast_controller.get_stats(),
            "medium": self.medium_controller.get_stats(),
            "slow": self.slow_controller.get_stats(),
        }
