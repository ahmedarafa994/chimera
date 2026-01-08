"""
WebSocket Event Broadcaster

Service for emitting WebSocket events from backend services.
Integrates with AutoDAN, GPTFuzz, and Deep Team orchestrator.
"""

import logging
from typing import Any

from .connection_manager import connection_manager
from .models import (
    AgentStatus,
    AgentStatusChangedEvent,
    AgentType,
    ErrorOccurredEvent,
    EvaluationCompleteEvent,
    EventType,
    GenerationCompleteEvent,
    GenerationProgressEvent,
    GenerationStartedEvent,
    RefinementSuggestedEvent,
    SessionCompletedEvent,
    SessionFailedEvent,
    SessionStartedEvent,
    create_websocket_event,
)

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """Broadcasts events to WebSocket clients"""

    def __init__(self):
        self.manager = connection_manager

    async def emit_session_started(
        self,
        session_id: str,
        config: dict[str, Any]
    ):
        """Emit session started event"""
        try:
            event_data = SessionStartedEvent(
                session_id=session_id,
                initial_population_size=config.get("populationSize", 50),
                target_model=config.get("targetModel", "unknown"),
                attack_objective=config.get("attackObjective", "")
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.SESSION_STARTED,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            # Update session state
            from .models import SessionStatus
            await self.manager.update_session_state(
                session_id,
                status=SessionStatus.RUNNING
            )

            logger.info(f"Emitted SESSION_STARTED for {session_id}")

        except Exception as e:
            logger.error(f"Error emitting session_started: {e}")

    async def emit_session_completed(
        self,
        session_id: str,
        statistics: dict[str, Any],
        best_candidate: dict[str, Any] | None = None
    ):
        """Emit session completed event"""
        try:
            event_data = SessionCompletedEvent(
                session_id=session_id,
                total_generations=statistics.get("totalGenerations", 0),
                total_evaluations=statistics.get("totalEvaluations", 0),
                successful_attacks=statistics.get("successfulAttacks", 0),
                success_rate=statistics.get("successRate", 0.0),
                best_fitness=statistics.get("bestFitness", 0.0),
                best_candidate=best_candidate
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.SESSION_COMPLETED,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            # Update session state
            from .models import SessionStatus
            await self.manager.update_session_state(
                session_id,
                status=SessionStatus.COMPLETED
            )

            logger.info(f"Emitted SESSION_COMPLETED for {session_id}")

        except Exception as e:
            logger.error(f"Error emitting session_completed: {e}")

    async def emit_session_failed(
        self,
        session_id: str,
        error_message: str,
        error_type: str,
        recovery_suggestions: list | None = None
    ):
        """Emit session failed event"""
        try:
            event_data = SessionFailedEvent(
                session_id=session_id,
                error_message=error_message,
                error_type=error_type,
                recovery_suggestions=recovery_suggestions or []
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.SESSION_FAILED,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            # Update session state
            from .models import SessionStatus
            await self.manager.update_session_state(
                session_id,
                status=SessionStatus.FAILED
            )

            logger.error(f"Emitted SESSION_FAILED for {session_id}: {error_message}")

        except Exception as e:
            logger.error(f"Error emitting session_failed: {e}")

    async def emit_agent_status_changed(
        self,
        session_id: str,
        agent_id: str,
        agent_type: str,
        old_status: str,
        new_status: str,
        current_task: str | None = None,
        progress: float | None = None
    ):
        """Emit agent status changed event"""
        try:
            event_data = AgentStatusChangedEvent(
                agent_id=agent_id,
                agent_type=AgentType(agent_type),
                old_status=AgentStatus(old_status),
                new_status=AgentStatus(new_status),
                current_task=current_task,
                progress=progress
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.AGENT_STATUS_CHANGED,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            logger.debug(
                f"Emitted AGENT_STATUS_CHANGED for {agent_id}: "
                f"{old_status} -> {new_status}"
            )

        except Exception as e:
            logger.error(f"Error emitting agent_status_changed: {e}")

    async def emit_generation_started(
        self,
        session_id: str,
        generation: int,
        config: dict[str, Any]
    ):
        """Emit generation started event"""
        try:
            event_data = GenerationStartedEvent(
                generation=generation,
                population_size=config.get("populationSize", 50),
                elite_size=config.get("eliteSize", 5),
                mutation_rate=config.get("mutationRate", 0.3),
                crossover_rate=config.get("crossoverRate", 0.7)
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.GENERATION_STARTED,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            logger.debug(f"Emitted GENERATION_STARTED for generation {generation}")

        except Exception as e:
            logger.error(f"Error emitting generation_started: {e}")

    async def emit_generation_progress(
        self,
        session_id: str,
        generation: int,
        progress: float,
        completed_evaluations: int,
        total_evaluations: int,
        current_best_fitness: float,
        average_fitness: float
    ):
        """Emit generation progress event"""
        try:
            event_data = GenerationProgressEvent(
                generation=generation,
                progress=progress,
                completed_evaluations=completed_evaluations,
                total_evaluations=total_evaluations,
                current_best_fitness=current_best_fitness,
                average_fitness=average_fitness
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.GENERATION_PROGRESS,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            logger.debug(
                f"Emitted GENERATION_PROGRESS: {progress * 100:.1f}% "
                f"({completed_evaluations}/{total_evaluations})"
            )

        except Exception as e:
            logger.error(f"Error emitting generation_progress: {e}")

    async def emit_generation_complete(
        self,
        session_id: str,
        generation: int,
        statistics: dict[str, Any],
        elite_candidates: list
    ):
        """Emit generation complete event"""
        try:
            event_data = GenerationCompleteEvent(
                generation=generation,
                best_fitness=statistics.get("bestFitness", 0.0),
                average_fitness=statistics.get("averageFitness", 0.0),
                worst_fitness=statistics.get("worstFitness", 0.0),
                improvement_rate=statistics.get("improvementRate", 0.0),
                convergence_score=statistics.get("convergenceScore", 0.0),
                diversity_score=statistics.get("diversityScore", 0.0),
                elite_candidates=elite_candidates,
                statistics=statistics
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.GENERATION_COMPLETE,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            # Update session state
            await self.manager.update_session_state(
                session_id,
                current_generation=generation
            )

            logger.info(
                f"Emitted GENERATION_COMPLETE for generation {generation} "
                f"(best: {statistics.get('bestFitness', 0.0):.3f})"
            )

        except Exception as e:
            logger.error(f"Error emitting generation_complete: {e}")

    async def emit_evaluation_complete(
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
        execution_time: float
    ):
        """Emit evaluation complete event"""
        try:
            event_data = EvaluationCompleteEvent(
                candidate_id=candidate_id,
                prompt=prompt,
                response=response,
                success=success,
                overall_score=overall_score,
                criteria_scores=criteria_scores,
                feedback=feedback,
                suggestions=suggestions,
                execution_time=execution_time
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.EVALUATION_COMPLETE,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            # Update session state
            state = await self.manager.get_session_state(session_id)
            if state:
                await self.manager.update_session_state(
                    session_id,
                    total_evaluations=state.total_evaluations + 1,
                    successful_attacks=(
                        state.successful_attacks + (1 if success else 0)
                    )
                )

            logger.debug(
                f"Emitted EVALUATION_COMPLETE: success={success}, "
                f"score={overall_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error emitting evaluation_complete: {e}")

    async def emit_refinement_suggested(
        self,
        session_id: str,
        generation: int,
        config_updates: dict[str, float],
        strategy_suggestions: list,
        analysis: str
    ):
        """Emit refinement suggested event"""
        try:
            event_data = RefinementSuggestedEvent(
                generation=generation,
                config_updates=config_updates,
                strategy_suggestions=strategy_suggestions,
                analysis=analysis
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.REFINEMENT_SUGGESTED,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            logger.info(f"Emitted REFINEMENT_SUGGESTED for generation {generation}")

        except Exception as e:
            logger.error(f"Error emitting refinement_suggested: {e}")

    async def emit_error(
        self,
        session_id: str,
        error_code: str,
        error_message: str,
        error_type: str,
        component: str,
        severity: str = "medium",
        recoverable: bool = True,
        recovery_suggestions: list | None = None
    ):
        """Emit error occurred event"""
        try:
            event_data = ErrorOccurredEvent(
                error_code=error_code,
                error_message=error_message,
                error_type=error_type,
                component=component,
                severity=severity,
                recoverable=recoverable,
                recovery_suggestions=recovery_suggestions or []
            )

            sequence = self.manager.get_next_sequence(session_id)
            event = create_websocket_event(
                event_type=EventType.ERROR_OCCURRED,
                session_id=session_id,
                sequence=sequence,
                data=event_data
            )

            await self.manager.broadcast_to_session(session_id, event)

            logger.error(
                f"Emitted ERROR_OCCURRED: {error_code} - {error_message} "
                f"(severity: {severity})"
            )

        except Exception as e:
            logger.error(f"Error emitting error event: {e}")


# Global broadcaster instance
event_broadcaster = EventBroadcaster()
