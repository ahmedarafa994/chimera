"""
WebSocket Integration Layer

Provides integration hooks for backend services to emit WebSocket events.
Decouples service logic from WebSocket broadcasting infrastructure.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from .broadcaster import event_broadcaster

logger = logging.getLogger(__name__)


class WebSocketIntegration:
    """
    Base class for WebSocket integration with backend services.

    Provides common patterns for event emission and error handling.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.broadcaster = event_broadcaster

    async def emit_error(
        self,
        session_id: str,
        error_message: str,
        error_type: str = "ServiceError",
        severity: str = "medium",
        recoverable: bool = True,
    ):
        """Emit error event with automatic component tagging"""
        await self.broadcaster.emit_error(
            session_id=session_id,
            error_code=f"{self.service_name.upper()}_ERROR",
            error_message=error_message,
            error_type=error_type,
            component=self.service_name,
            severity=severity,
            recoverable=recoverable,
        )


class AutoDANWebSocketIntegration(WebSocketIntegration):
    """
    WebSocket integration for AutoDAN service.

    Emits events for generation lifecycle, agent status changes, and evaluations.
    """

    def __init__(self):
        super().__init__("AutoDAN")
        self._active_sessions: dict[str, dict[str, Any]] = {}

    async def start_session(self, session_id: str, config: dict[str, Any]):
        """
        Start AutoDAN session and emit SESSION_STARTED event

        Args:
            session_id: Unique session identifier
            config: Session configuration (populationSize, targetModel, etc.)
        """
        try:
            # Store session config
            self._active_sessions[session_id] = {
                "config": config,
                "current_generation": 0,
                "agent_ids": {
                    "attacker": f"{session_id}_attacker",
                    "evaluator": f"{session_id}_evaluator",
                    "refiner": f"{session_id}_refiner",
                },
            }

            # Emit SESSION_STARTED
            await self.broadcaster.emit_session_started(session_id=session_id, config=config)

            # Emit initial agent status (all agents IDLE)
            agent_ids = self._active_sessions[session_id]["agent_ids"]

            await self.broadcaster.emit_agent_status_changed(
                session_id=session_id,
                agent_id=agent_ids["attacker"],
                agent_type="attacker",
                old_status="idle",
                new_status="initializing",
                current_task="Initializing AutoDAN pipeline",
            )

            logger.info(f"AutoDAN session started: {session_id}")

        except Exception as e:
            logger.error(f"Failed to start AutoDAN session {session_id}: {e}")
            await self.emit_error(
                session_id=session_id,
                error_message=str(e),
                error_type="SessionStartError",
                severity="high",
                recoverable=False,
            )

    async def start_generation(self, session_id: str, generation: int, config: dict[str, Any]):
        """
        Emit GENERATION_STARTED event

        Args:
            session_id: Session identifier
            generation: Generation number (0-indexed)
            config: Generation configuration
        """
        try:
            # Update session state
            if session_id in self._active_sessions:
                self._active_sessions[session_id]["current_generation"] = generation

            # Emit GENERATION_STARTED
            await self.broadcaster.emit_generation_started(
                session_id=session_id, generation=generation, config=config
            )

            # Update attacker agent status
            if session_id in self._active_sessions:
                agent_ids = self._active_sessions[session_id]["agent_ids"]
                await self.broadcaster.emit_agent_status_changed(
                    session_id=session_id,
                    agent_id=agent_ids["attacker"],
                    agent_type="attacker",
                    old_status="waiting",
                    new_status="working",
                    current_task=f"Generating population for generation {generation}",
                    progress=0.0,
                )

            logger.debug(f"Generation {generation} started for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to start generation {generation}: {e}")
            await self.emit_error(
                session_id=session_id,
                error_message=f"Generation {generation} failed to start: {e!s}",
                severity="medium",
            )

    async def update_generation_progress(
        self,
        session_id: str,
        generation: int,
        completed_evaluations: int,
        total_evaluations: int,
        current_best_fitness: float,
        average_fitness: float,
    ):
        """
        Emit GENERATION_PROGRESS event

        Args:
            session_id: Session identifier
            generation: Current generation number
            completed_evaluations: Number of completed evaluations
            total_evaluations: Total evaluations in generation
            current_best_fitness: Best fitness score so far
            average_fitness: Average fitness across population
        """
        try:
            progress = completed_evaluations / total_evaluations if total_evaluations > 0 else 0.0

            await self.broadcaster.emit_generation_progress(
                session_id=session_id,
                generation=generation,
                progress=progress,
                completed_evaluations=completed_evaluations,
                total_evaluations=total_evaluations,
                current_best_fitness=current_best_fitness,
                average_fitness=average_fitness,
            )

            # Update agent progress
            if session_id in self._active_sessions:
                agent_ids = self._active_sessions[session_id]["agent_ids"]
                await self.broadcaster.emit_agent_status_changed(
                    session_id=session_id,
                    agent_id=agent_ids["evaluator"],
                    agent_type="evaluator",
                    old_status="working",
                    new_status="working",
                    current_task=f"Evaluating candidates ({completed_evaluations}/{total_evaluations})",
                    progress=progress,
                )

        except Exception as e:
            logger.error(f"Failed to update generation progress: {e}")

    async def complete_generation(
        self, session_id: str, generation: int, statistics: dict[str, Any], elite_candidates: list
    ):
        """
        Emit GENERATION_COMPLETE event

        Args:
            session_id: Session identifier
            generation: Completed generation number
            statistics: Generation statistics (bestFitness, averageFitness, etc.)
            elite_candidates: Top performing candidates
        """
        try:
            await self.broadcaster.emit_generation_complete(
                session_id=session_id,
                generation=generation,
                statistics=statistics,
                elite_candidates=elite_candidates,
            )

            # Update agent status to waiting for next generation
            if session_id in self._active_sessions:
                agent_ids = self._active_sessions[session_id]["agent_ids"]
                await self.broadcaster.emit_agent_status_changed(
                    session_id=session_id,
                    agent_id=agent_ids["attacker"],
                    agent_type="attacker",
                    old_status="working",
                    new_status="waiting",
                    current_task="Waiting for next generation",
                    progress=1.0,
                )

            logger.info(f"Generation {generation} completed for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to complete generation {generation}: {e}")
            await self.emit_error(
                session_id=session_id,
                error_message=f"Generation {generation} completion failed: {e!s}",
                severity="medium",
            )

    async def emit_evaluation(
        self,
        session_id: str,
        candidate_id: str,
        prompt: str,
        response: str,
        success: bool,
        overall_score: float,
        criteria_scores: dict[str, float],
        feedback: str,
        suggestions: list,
        execution_time: float,
    ):
        """
        Emit EVALUATION_COMPLETE event

        Args:
            session_id: Session identifier
            candidate_id: Unique candidate identifier
            prompt: Evaluated prompt
            response: Target model response
            success: Whether evaluation was successful
            overall_score: Overall fitness score
            criteria_scores: Scores per evaluation criterion
            feedback: Evaluation feedback text
            suggestions: Improvement suggestions
            execution_time: Time taken for evaluation
        """
        try:
            await self.broadcaster.emit_evaluation_complete(
                session_id=session_id,
                candidate_id=candidate_id,
                prompt=prompt,
                response=response,
                success=success,
                overall_score=overall_score,
                criteria_scores=criteria_scores,
                feedback=feedback,
                suggestions=suggestions,
                execution_time=execution_time,
            )

        except Exception as e:
            logger.error(f"Failed to emit evaluation for {candidate_id}: {e}")

    async def emit_refinement_suggestion(
        self,
        session_id: str,
        generation: int,
        config_updates: dict[str, float],
        strategy_suggestions: list,
        analysis: str,
    ):
        """
        Emit REFINEMENT_SUGGESTED event

        Args:
            session_id: Session identifier
            generation: Generation that triggered refinement
            config_updates: Suggested hyperparameter updates
            strategy_suggestions: Suggested strategy changes
            analysis: Analysis justifying refinement
        """
        try:
            await self.broadcaster.emit_refinement_suggested(
                session_id=session_id,
                generation=generation,
                config_updates=config_updates,
                strategy_suggestions=strategy_suggestions,
                analysis=analysis,
            )

            # Update refiner agent status
            if session_id in self._active_sessions:
                agent_ids = self._active_sessions[session_id]["agent_ids"]
                await self.broadcaster.emit_agent_status_changed(
                    session_id=session_id,
                    agent_id=agent_ids["refiner"],
                    agent_type="refiner",
                    old_status="working",
                    new_status="completed",
                    current_task="Refinement analysis complete",
                    progress=1.0,
                )

        except Exception as e:
            logger.error(f"Failed to emit refinement suggestion: {e}")

    async def complete_session(
        self,
        session_id: str,
        statistics: dict[str, Any],
        best_candidate: dict[str, Any] | None = None,
    ):
        """
        Emit SESSION_COMPLETED event

        Args:
            session_id: Session identifier
            statistics: Final session statistics
            best_candidate: Best candidate found (optional)
        """
        try:
            await self.broadcaster.emit_session_completed(
                session_id=session_id, statistics=statistics, best_candidate=best_candidate
            )

            # Update all agents to completed
            if session_id in self._active_sessions:
                agent_ids = self._active_sessions[session_id]["agent_ids"]
                for agent_type_name, agent_id in agent_ids.items():
                    await self.broadcaster.emit_agent_status_changed(
                        session_id=session_id,
                        agent_id=agent_id,
                        agent_type=agent_type_name,
                        old_status="waiting",
                        new_status="completed",
                        current_task="Session completed",
                        progress=1.0,
                    )

                # Clean up session tracking
                del self._active_sessions[session_id]

            logger.info(f"AutoDAN session completed: {session_id}")

        except Exception as e:
            logger.error(f"Failed to complete session {session_id}: {e}")
            await self.emit_error(
                session_id=session_id,
                error_message=f"Session completion failed: {e!s}",
                severity="high",
            )

    async def fail_session(
        self,
        session_id: str,
        error_message: str,
        error_type: str = "AutoDANError",
        recovery_suggestions: list | None = None,
    ):
        """
        Emit SESSION_FAILED event

        Args:
            session_id: Session identifier
            error_message: Error description
            error_type: Type of error
            recovery_suggestions: Suggestions for recovery
        """
        try:
            await self.broadcaster.emit_session_failed(
                session_id=session_id,
                error_message=error_message,
                error_type=error_type,
                recovery_suggestions=recovery_suggestions,
            )

            # Update all agents to error state
            if session_id in self._active_sessions:
                agent_ids = self._active_sessions[session_id]["agent_ids"]
                for agent_type_name, agent_id in agent_ids.items():
                    await self.broadcaster.emit_agent_status_changed(
                        session_id=session_id,
                        agent_id=agent_id,
                        agent_type=agent_type_name,
                        old_status="working",
                        new_status="error",
                        current_task=f"Error: {error_message}",
                    )

                # Clean up session tracking
                del self._active_sessions[session_id]

            logger.error(f"AutoDAN session failed: {session_id} - {error_message}")

        except Exception as e:
            logger.error(f"Failed to emit session failure for {session_id}: {e}")

    @asynccontextmanager
    async def session_context(self, session_id: str, config: dict[str, Any]):
        """
        Context manager for AutoDAN session lifecycle

        Automatically emits SESSION_STARTED on entry and SESSION_COMPLETED/FAILED on exit.

        Usage:
            async with autodan_ws.session_context(session_id, config):
                # Run AutoDAN logic
                await autodan_service.run_jailbreak(...)
        """
        try:
            await self.start_session(session_id, config)
            yield
        except Exception as e:
            await self.fail_session(
                session_id=session_id, error_message=str(e), error_type=type(e).__name__
            )
            raise
        else:
            # Only emit completion if no exception occurred
            statistics = self._active_sessions.get(session_id, {}).get("statistics", {})
            await self.complete_session(session_id, statistics)


class GPTFuzzWebSocketIntegration(WebSocketIntegration):
    """
    WebSocket integration for GPTFuzz service.

    Emits events for mutation-based jailbreak testing.
    """

    def __init__(self):
        super().__init__("GPTFuzz")

    async def emit_mutation_result(
        self,
        session_id: str,
        candidate_id: str,
        prompt: str,
        response: str,
        success: bool,
        score: float,
    ):
        """Emit evaluation result for GPTFuzz mutation"""
        await self.broadcaster.emit_evaluation_complete(
            session_id=session_id,
            candidate_id=candidate_id,
            prompt=prompt,
            response=response,
            success=success,
            overall_score=score,
            criteria_scores={"jailbreak_success": score},
            feedback="GPTFuzz mutation evaluation",
            suggestions=[],
            execution_time=0.0,
        )


# Global integration instances
autodan_websocket = AutoDANWebSocketIntegration()
gptfuzz_websocket = GPTFuzzWebSocketIntegration()
