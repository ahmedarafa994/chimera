"""
Enhanced Orchestrator Agent - Advanced Pipeline Management
Provides workflow orchestration, job scheduling, and comprehensive coordination
"""

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

try:
    from core.config import Config
    from core.dataset_loader import DatasetLoader, get_dataset_loader
    from core.enhanced_models import (
        AdversarialTestJob,
        EnhancedPrompt,
        EventType,
        LLMResponse,
        SafetyEvaluation,
        SystemEvent,
        TechniqueCategory,
        TechniqueConfig,
        WorkflowState,
        WorkflowStep,
    )
    from core.event_bus import EventBus, EventHandler
    from core.message_queue import MessageQueue
    from core.models import AgentStatus, AgentType, JobStatus, Message, MessageType

    from agents.base_agent import BaseAgent
except ImportError:
    from ..core.config import Config
    from ..core.dataset_loader import DatasetLoader, get_dataset_loader
    from ..core.enhanced_models import (
        AdversarialTestJob,
        EnhancedPrompt,
        EventType,
        LLMResponse,
        SafetyEvaluation,
        SystemEvent,
        TechniqueCategory,
        TechniqueConfig,
        WorkflowState,
        WorkflowStep,
    )
    from ..core.event_bus import EventBus, EventHandler
    from ..core.message_queue import MessageQueue
    from ..core.models import AgentStatus, AgentType, Message, MessageType
    from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Job scheduling strategies."""

    FIFO = "fifo"  # First in, first out
    PRIORITY = "priority"  # Priority-based
    ROUND_ROBIN = "round_robin"  # Distribute across providers
    ADAPTIVE = "adaptive"  # Based on provider health


@dataclass
class JobSchedule:
    """Schedule for a job."""

    job_id: str
    priority: int = 5
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    deadline: datetime | None = None
    dependencies: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


class EnhancedOrchestratorAgent(BaseAgent):
    """
    Enhanced Orchestrator Agent with advanced pipeline management.

    Features:
    - Workflow state machine
    - Job scheduling with multiple strategies
    - Dependency management
    - Health monitoring
    - Event-driven coordination
    - Results aggregation
    - Technique effectiveness tracking
    """

    def __init__(
        self,
        config: Config,
        message_queue: MessageQueue,
        event_bus: EventBus | None = None,
        agent_id: str | None = None,
    ):
        super().__init__(
            agent_type=AgentType.ORCHESTRATOR,
            config=config,
            message_queue=message_queue,
            agent_id=agent_id,
        )

        self.event_bus = event_bus

        # Job management
        self._jobs: dict[str, AdversarialTestJob] = {}
        self._job_schedules: dict[str, JobSchedule] = {}
        self._job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._pending_responses: dict[str, dict[str, Any]] = {}

        # Agent tracking
        self._agent_statuses: dict[str, AgentStatus] = {}
        self._agent_health: dict[str, bool] = {}

        # Dataset loader
        self._dataset_loader: DatasetLoader | None = None

        # Scheduling
        self._scheduling_strategy = SchedulingStrategy.ADAPTIVE

        # Pipeline metrics
        self._pipeline_metrics = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0,
            "total_prompts_generated": 0,
            "total_executions": 0,
            "total_evaluations": 0,
            "total_jailbreaks": 0,
            "jailbreak_success_rate": 0.0,
            "average_job_duration_seconds": 0.0,
            "technique_rankings": {},
        }

        # Background tasks
        self._job_processor_task: asyncio.Task | None = None
        self._health_monitor_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    async def on_start(self):
        """Start background tasks and load datasets."""
        # Load datasets
        self._dataset_loader = await get_dataset_loader(self.config.datasets_path)

        # Start background tasks
        self._job_processor_task = asyncio.create_task(self._job_processor_loop())
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Subscribe to events
        if self.event_bus:
            await self.event_bus.subscribe(
                EventHandler(
                    callback=self._handle_system_event,
                    event_types=[
                        EventType.EXECUTION_COMPLETED,
                        EventType.EVALUATION_COMPLETED,
                        EventType.AGENT_ERROR,
                    ],
                ),
                channels=["default"],
            )

        logger.info("Enhanced Orchestrator Agent started")

    async def on_stop(self):
        """Stop background tasks."""
        for task in [self._job_processor_task, self._health_monitor_task, self._cleanup_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        logger.info("Enhanced Orchestrator Agent stopped")

    async def process_message(self, message: Message):
        """Process incoming messages from other agents."""
        handlers = {
            MessageType.GENERATE_RESPONSE: self._handle_generate_response,
            MessageType.EXECUTE_RESPONSE: self._handle_execute_response,
            MessageType.EVALUATE_RESPONSE: self._handle_evaluate_response,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.ERROR: self._handle_error,
            MessageType.STATUS_UPDATE: self._handle_status_update,
        }

        handler = handlers.get(message.type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"Unknown message type: {message.type}")

    async def _handle_system_event(self, event: SystemEvent):
        """Handle system events from the event bus."""
        if event.type == EventType.AGENT_ERROR:
            agent_id = event.source
            self._agent_health[agent_id] = False
            logger.warning(f"Agent {agent_id} reported error: {event.data}")

    # ========================================================================
    # Job Creation and Management
    # ========================================================================

    async def create_job(
        self,
        query: str,
        target_models: list[str] | None = None,
        techniques: list[str] | None = None,
        num_variants: int = 3,
        potency_range: tuple = (3, 8),
        priority: int = 5,
        config: dict[str, Any] | None = None,
    ) -> AdversarialTestJob:
        """
        Create a new adversarial testing job.

        Args:
            query: The original query to test
            target_models: List of target models to test against
            techniques: Specific techniques to use
            num_variants: Number of prompt variants to generate
            potency_range: Range of potency levels
            priority: Job priority (1-10)
            config: Additional configuration

        Returns:
            The created AdversarialTestJob
        """
        # Create technique configs
        technique_configs = []
        if techniques:
            for tech in techniques:
                try:
                    category = TechniqueCategory(tech)
                except ValueError:
                    category = TechniqueCategory.ADVANCED

                technique_configs.append(
                    TechniqueConfig(
                        name=tech,
                        category=category,
                        potency=(potency_range[0] + potency_range[1]) // 2,
                    )
                )

        # Create job
        job = AdversarialTestJob(
            original_query=query,
            target_models=target_models or ["local"],
            techniques=technique_configs,
            num_variants=num_variants,
            potency_range=potency_range,
            state=WorkflowState.INITIALIZED,
            metadata=config or {},
        )

        # Initialize workflow steps
        job.workflow_steps = [
            WorkflowStep(name="prompt_generation", state=WorkflowState.INITIALIZED),
            WorkflowStep(name="execution", state=WorkflowState.INITIALIZED),
            WorkflowStep(name="evaluation", state=WorkflowState.INITIALIZED),
            WorkflowStep(name="aggregation", state=WorkflowState.INITIALIZED),
        ]

        # Store job
        self._jobs[job.id] = job
        self._pipeline_metrics["total_jobs"] += 1

        # Create schedule
        schedule = JobSchedule(
            job_id=job.id,
            priority=priority,
            deadline=datetime.utcnow() + timedelta(seconds=self.config.orchestrator.job_timeout),
        )
        self._job_schedules[job.id] = schedule

        # Add to queue (negative priority for max-heap behavior)
        await self._job_queue.put((-priority, datetime.utcnow().timestamp(), job.id))

        # Emit event
        if self.event_bus:
            await self.event_bus.publish(
                SystemEvent(
                    type=EventType.JOB_CREATED,
                    source=self.agent_id,
                    job_id=job.id,
                    data={"query": query[:100], "num_variants": num_variants},
                )
            )

        logger.info(f"Created job {job.id} for query: {query[:50]}...")
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.state in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
            return False

        job.state = WorkflowState.CANCELLED
        job.completed_at = datetime.utcnow()
        self._pipeline_metrics["cancelled_jobs"] += 1

        self.remove_active_job(job_id)

        logger.info(f"Cancelled job {job_id}")
        return True

    # ========================================================================
    # Job Processing
    # ========================================================================

    async def _job_processor_loop(self):
        """Process jobs from the queue."""
        while self._running:
            try:
                # Get next job from queue
                _, _, job_id = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)

                job = self._jobs.get(job_id)
                if job and job.state == WorkflowState.INITIALIZED:
                    await self._process_job(job)

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job processor: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job: AdversarialTestJob):
        """Process a single job through the pipeline."""
        logger.info(f"Processing job {job.id}")

        job.started_at = datetime.utcnow()
        self.add_active_job(job.id)

        # Emit event
        if self.event_bus:
            await self.event_bus.publish(
                SystemEvent(type=EventType.JOB_STARTED, source=self.agent_id, job_id=job.id)
            )

        try:
            # Start with prompt generation
            await self._start_prompt_generation(job)

        except Exception as e:
            logger.error(f"Error processing job {job.id}: {e}")
            await self._fail_job(job, str(e))

    async def _start_prompt_generation(self, job: AdversarialTestJob):
        """Start the prompt generation phase."""
        job.state = WorkflowState.PROMPT_GENERATION
        job.workflow_steps[0].state = WorkflowState.PROMPT_GENERATION
        job.workflow_steps[0].started_at = datetime.utcnow()

        # Build generation request
        request_payload = {
            "original_query": job.original_query,
            "techniques": [t.name for t in job.techniques] if job.techniques else None,
            "potency": (job.potency_range[0] + job.potency_range[1]) // 2,
            "num_variants": job.num_variants,
            "potency_range": job.potency_range,
        }

        # Include dataset examples if available
        if self._dataset_loader and self._dataset_loader.is_loaded:
            similar_examples = self._dataset_loader.get_similar_entries(
                job.original_query, limit=3, successful_only=True
            )
            if similar_examples:
                request_payload["dataset_examples"] = [
                    {"prompt": e[0].prompt, "technique": e[0].technique} for e in similar_examples
                ]

        await self.send_message(
            MessageType.GENERATE_REQUEST,
            target=AgentType.GENERATOR,
            job_id=job.id,
            payload=request_payload,
            priority=7,
        )

    async def _handle_generate_response(self, message: Message):
        """Handle response from generator agent."""
        job_id = message.job_id
        job = self._jobs.get(job_id)

        if not job:
            logger.warning(f"Job {job_id} not found for generate response")
            return

        try:
            # Parse generated prompts
            prompts_data = message.payload.get("prompts", [])
            for prompt_data in prompts_data:
                prompt = EnhancedPrompt(
                    id=prompt_data.get("id", str(uuid.uuid4())),
                    original_query=prompt_data.get("original_query", job.original_query),
                    enhanced_prompt=prompt_data.get("enhanced_prompt", ""),
                    technique_chain=[prompt_data.get("technique_used", "")],
                    potency_level=prompt_data.get("potency_level", 5),
                    persona=prompt_data.get("persona"),
                    metadata=prompt_data.get("metadata", {}),
                )
                job.generated_prompts.append(prompt)

            # Update metrics
            self._pipeline_metrics["total_prompts_generated"] += len(prompts_data)

            # Complete generation step
            job.workflow_steps[0].state = WorkflowState.COMPLETED
            job.workflow_steps[0].completed_at = datetime.utcnow()
            job.workflow_steps[0].output_data = {"prompt_count": len(prompts_data)}

            # Move to execution phase
            await self._start_execution(job)

        except Exception as e:
            logger.error(f"Error handling generate response: {e}")
            await self._fail_job(job, str(e))

    async def _start_execution(self, job: AdversarialTestJob):
        """Start the execution phase."""
        job.state = WorkflowState.EXECUTING
        job.workflow_steps[1].state = WorkflowState.EXECUTING
        job.workflow_steps[1].started_at = datetime.utcnow()

        # Initialize pending responses tracker
        self._pending_responses[job.id] = {
            "expected": len(job.generated_prompts) * len(job.target_models),
            "received": 0,
            "type": "execution",
        }

        # Send execution requests for each prompt and target model
        for prompt in job.generated_prompts:
            for target_model in job.target_models:
                request_payload = {
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.enhanced_prompt,
                    "provider": target_model,
                    "target_llm": target_model,
                    "model_config": job.metadata.get("model_config", {}),
                    "timeout": job.metadata.get("timeout", 60),
                    "retry_count": job.metadata.get("retry_count", 3),
                }

                await self.send_message(
                    MessageType.EXECUTE_REQUEST,
                    target=AgentType.EXECUTOR,
                    job_id=job.id,
                    payload=request_payload,
                    priority=6,
                )

    async def _handle_execute_response(self, message: Message):
        """Handle response from execution agent."""
        job_id = message.job_id
        job = self._jobs.get(job_id)

        if not job:
            logger.warning(f"Job {job_id} not found for execute response")
            return

        try:
            # Parse execution result
            response = LLMResponse(
                id=message.payload.get("id", str(uuid.uuid4())),
                prompt_id=message.payload.get("prompt_id", ""),
                provider=message.payload.get("provider", ""),
                model=message.payload.get("model", ""),
                response_text=message.payload.get("response_text", ""),
                response_time_ms=message.payload.get("response_time_ms", 0),
                tokens_total=message.payload.get("tokens_used", 0),
                success=message.payload.get("success", False),
                error_message=message.payload.get("error_message"),
            )
            job.llm_responses.append(response)
            self._pipeline_metrics["total_executions"] += 1

            # Update pending count
            pending = self._pending_responses.get(job_id, {})
            pending["received"] = pending.get("received", 0) + 1

            # Check if all executions are complete
            if pending["received"] >= pending["expected"]:
                # Complete execution step
                job.workflow_steps[1].state = WorkflowState.COMPLETED
                job.workflow_steps[1].completed_at = datetime.utcnow()
                job.workflow_steps[1].output_data = {
                    "execution_count": len(job.llm_responses),
                    "successful": sum(1 for r in job.llm_responses if r.success),
                }

                # Move to evaluation phase
                await self._start_evaluation(job)

        except Exception as e:
            logger.error(f"Error handling execute response: {e}")
            await self._fail_job(job, str(e))

    async def _start_evaluation(self, job: AdversarialTestJob):
        """Start the evaluation phase."""
        job.state = WorkflowState.EVALUATING
        job.workflow_steps[2].state = WorkflowState.EVALUATING
        job.workflow_steps[2].started_at = datetime.utcnow()

        # Get successful executions
        successful_responses = [r for r in job.llm_responses if r.success]

        if not successful_responses:
            # No successful executions, skip to completion
            job.workflow_steps[2].state = WorkflowState.COMPLETED
            job.workflow_steps[2].completed_at = datetime.utcnow()
            await self._complete_job(job)
            return

        # Initialize pending responses tracker
        self._pending_responses[job.id] = {
            "expected": len(successful_responses),
            "received": 0,
            "type": "evaluation",
        }

        # Create prompt lookup
        prompt_map = {p.id: p for p in job.generated_prompts}

        # Send evaluation requests
        for response in successful_responses:
            prompt = prompt_map.get(response.prompt_id)
            if not prompt:
                continue

            request_payload = {
                "execution_id": response.id,
                "prompt_text": prompt.enhanced_prompt,
                "response_text": response.response_text,
                "original_query": job.original_query,
                "technique_used": prompt.technique_chain[0] if prompt.technique_chain else "",
                "target_model": response.provider,
            }

            await self.send_message(
                MessageType.EVALUATE_REQUEST,
                target=AgentType.EVALUATOR,
                job_id=job.id,
                payload=request_payload,
                priority=6,
            )

    async def _handle_evaluate_response(self, message: Message):
        """Handle response from evaluator agent."""
        job_id = message.job_id
        job = self._jobs.get(job_id)

        if not job:
            logger.warning(f"Job {job_id} not found for evaluate response")
            return

        try:
            # Parse evaluation result
            evaluation = SafetyEvaluation(
                id=message.payload.get("id", str(uuid.uuid4())),
                response_id=message.payload.get("execution_id", ""),
                is_jailbroken=message.payload.get(
                    "is_jailbroken", message.payload.get("jailbreak_detected", False)
                ),
                safety_score=message.payload.get("safety_score", 100.0),
                confidence=message.payload.get(
                    "confidence_score", message.payload.get("confidence", 0.0)
                ),
                complex_categories=message.payload.get("complex_content_detected", []),
                technique_used=message.payload.get("technique_used", ""),
                technique_effectiveness=message.payload.get("technique_effectiveness", 0.0),
                recommendations=message.payload.get("recommendations", []),
                detailed_analysis=message.payload.get("detailed_analysis", {}),
            )
            job.evaluations.append(evaluation)
            self._pipeline_metrics["total_evaluations"] += 1

            if evaluation.is_jailbroken:
                self._pipeline_metrics["total_jailbreaks"] += 1

            # Update pending count
            pending = self._pending_responses.get(job_id, {})
            pending["received"] = pending.get("received", 0) + 1

            # Check if all evaluations are complete
            if pending["received"] >= pending["expected"]:
                # Complete evaluation step
                job.workflow_steps[2].state = WorkflowState.COMPLETED
                job.workflow_steps[2].completed_at = datetime.utcnow()
                job.workflow_steps[2].output_data = {
                    "evaluation_count": len(job.evaluations),
                    "jailbreaks": sum(1 for e in job.evaluations if e.is_jailbroken),
                }

                # Complete the job
                await self._complete_job(job)

        except Exception as e:
            logger.error(f"Error handling evaluate response: {e}")
            await self._fail_job(job, str(e))

    async def _complete_job(self, job: AdversarialTestJob):
        """Complete a job and aggregate results."""
        job.state = WorkflowState.AGGREGATING
        job.workflow_steps[3].state = WorkflowState.AGGREGATING
        job.workflow_steps[3].started_at = datetime.utcnow()

        # Aggregate results
        job.total_jailbreaks = sum(1 for e in job.evaluations if e.is_jailbroken)
        job.jailbreak_rate = job.total_jailbreaks / len(job.evaluations) if job.evaluations else 0.0

        # Calculate average effectiveness
        if job.evaluations:
            job.average_effectiveness = sum(
                e.technique_effectiveness for e in job.evaluations
            ) / len(job.evaluations)

        # Rank techniques
        technique_scores: dict[str, list[float]] = {}
        for evaluation in job.evaluations:
            tech = evaluation.technique_used
            if tech:
                if tech not in technique_scores:
                    technique_scores[tech] = []
                technique_scores[tech].append(evaluation.technique_effectiveness)

        job.technique_rankings = {
            tech: sum(scores) / len(scores) for tech, scores in technique_scores.items()
        }

        # Find best technique
        if job.technique_rankings:
            job.best_technique = max(job.technique_rankings, key=job.technique_rankings.get)

        # Update global technique rankings
        for tech, score in job.technique_rankings.items():
            if tech not in self._pipeline_metrics["technique_rankings"]:
                self._pipeline_metrics["technique_rankings"][tech] = {"total_score": 0, "count": 0}
            self._pipeline_metrics["technique_rankings"][tech]["total_score"] += score
            self._pipeline_metrics["technique_rankings"][tech]["count"] += 1

        # Complete aggregation step
        job.workflow_steps[3].state = WorkflowState.COMPLETED
        job.workflow_steps[3].completed_at = datetime.utcnow()

        # Mark job as completed
        job.state = WorkflowState.COMPLETED
        job.completed_at = datetime.utcnow()

        # Update metrics
        self._pipeline_metrics["completed_jobs"] += 1
        self._update_success_rate()
        self._update_average_duration(job)

        self.remove_active_job(job.id)

        # Emit event
        if self.event_bus:
            await self.event_bus.publish(
                SystemEvent(
                    type=EventType.JOB_COMPLETED,
                    source=self.agent_id,
                    job_id=job.id,
                    data=job.get_summary(),
                )
            )

        logger.info(f"Job {job.id} completed. Summary: {job.get_summary()}")

    async def _fail_job(self, job: AdversarialTestJob, error: str):
        """Mark a job as failed."""
        job.state = WorkflowState.FAILED
        job.error_message = error
        job.completed_at = datetime.utcnow()

        self._pipeline_metrics["failed_jobs"] += 1

        self.remove_active_job(job.id)

        # Emit event
        if self.event_bus:
            await self.event_bus.publish(
                SystemEvent(
                    type=EventType.JOB_FAILED,
                    source=self.agent_id,
                    job_id=job.id,
                    data={"error": error},
                )
            )

        logger.error(f"Job {job.id} failed: {error}")

    # ========================================================================
    # Message Handlers
    # ========================================================================

    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat from agents."""
        agent_status = AgentStatus(
            agent_type=AgentType(message.payload.get("agent_type", "orchestrator")),
            agent_id=message.payload.get("agent_id", ""),
            is_healthy=message.payload.get("is_healthy", True),
            current_load=message.payload.get("current_load", 0),
            max_capacity=message.payload.get("max_capacity", 100),
            last_heartbeat=datetime.utcnow(),
            active_jobs=message.payload.get("active_jobs", []),
            metrics=message.payload.get("metrics", {}),
        )

        self._agent_statuses[agent_status.agent_id] = agent_status
        self._agent_health[agent_status.agent_id] = agent_status.is_healthy

    async def _handle_error(self, message: Message):
        """Handle error messages."""
        job_id = message.job_id
        error = message.payload.get("error", "Unknown error")

        logger.error(f"Error in job {job_id}: {error}")

        job = self._jobs.get(job_id)
        if job and job.state not in [WorkflowState.COMPLETED, WorkflowState.FAILED]:
            # Check retry count
            schedule = self._job_schedules.get(job_id)
            if schedule and schedule.retry_count < schedule.max_retries:
                schedule.retry_count += 1
                logger.info(f"Retrying job {job_id} (attempt {schedule.retry_count})")
                job.state = WorkflowState.INITIALIZED
                await self._job_queue.put(
                    (-schedule.priority, datetime.utcnow().timestamp(), job_id)
                )
            else:
                await self._fail_job(job, error)

    async def _handle_status_update(self, message: Message):
        """Handle status update requests."""
        await self.send_message(
            MessageType.STATUS_UPDATE,
            target=message.source,
            payload={
                "status": self.status.to_dict(),
                "pipeline_metrics": self._pipeline_metrics,
                "active_jobs": len(
                    [
                        j
                        for j in self._jobs.values()
                        if j.state
                        not in [
                            WorkflowState.COMPLETED,
                            WorkflowState.FAILED,
                            WorkflowState.CANCELLED,
                        ]
                    ]
                ),
                "agent_statuses": {
                    aid: status.to_dict() for aid, status in self._agent_statuses.items()
                },
            },
        )

    # ========================================================================
    # Health Monitoring
    # ========================================================================

    async def _health_monitor_loop(self):
        """Monitor agent health and handle failures."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.utcnow()
                stale_threshold = timedelta(minutes=2)

                # Check for stale agent statuses
                for agent_id, status in list(self._agent_statuses.items()):
                    if now - status.last_heartbeat > stale_threshold:
                        logger.warning(f"Agent {agent_id} appears unhealthy (no heartbeat)")
                        self._agent_health[agent_id] = False

                # Check for stuck jobs
                stuck_threshold = timedelta(seconds=self.config.orchestrator.job_timeout)

                for job_id, job in list(self._jobs.items()):
                    if (
                        job.state
                        not in [
                            WorkflowState.COMPLETED,
                            WorkflowState.FAILED,
                            WorkflowState.CANCELLED,
                        ]
                        and job.started_at
                        and now - job.started_at > stuck_threshold
                    ):
                        logger.warning(f"Job {job_id} appears stuck, marking as failed")
                        await self._fail_job(job, "Job timeout exceeded")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def _cleanup_loop(self):
        """Periodically clean up old jobs."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Every hour

                now = datetime.utcnow()
                retention_period = timedelta(hours=24)

                # Remove old completed/failed jobs
                jobs_to_remove = []
                for job_id, job in self._jobs.items():
                    if (
                        job.state
                        in [
                            WorkflowState.COMPLETED,
                            WorkflowState.FAILED,
                            WorkflowState.CANCELLED,
                        ]
                        and job.completed_at
                        and now - job.completed_at > retention_period
                    ):
                        jobs_to_remove.append(job_id)

                for job_id in jobs_to_remove:
                    del self._jobs[job_id]
                    if job_id in self._job_schedules:
                        del self._job_schedules[job_id]
                    if job_id in self._pending_responses:
                        del self._pending_responses[job_id]

                if jobs_to_remove:
                    logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    # ========================================================================
    # Metrics and Statistics
    # ========================================================================

    def _update_success_rate(self):
        """Update the jailbreak success rate metric."""
        total_evaluations = sum(
            len(job.evaluations)
            for job in self._jobs.values()
            if job.state == WorkflowState.COMPLETED
        )

        if total_evaluations > 0:
            self._pipeline_metrics["jailbreak_success_rate"] = (
                self._pipeline_metrics["total_jailbreaks"] / total_evaluations
            )

    def _update_average_duration(self, job: AdversarialTestJob):
        """Update average job duration metric."""
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()

            completed = self._pipeline_metrics["completed_jobs"]
            current_avg = self._pipeline_metrics["average_job_duration_seconds"]

            self._pipeline_metrics["average_job_duration_seconds"] = (
                current_avg * (completed - 1) + duration
            ) / completed

    # ========================================================================
    # Public API
    # ========================================================================

    def get_job(self, job_id: str) -> AdversarialTestJob | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_all_jobs(
        self, state: WorkflowState | None = None, limit: int = 100
    ) -> list[AdversarialTestJob]:
        """Get all jobs with optional filtering."""
        jobs = list(self._jobs.values())

        if state:
            jobs = [j for j in jobs if j.state == state]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    def get_job_summary(self, job_id: str) -> dict[str, Any] | None:
        """Get summary of a job."""
        job = self._jobs.get(job_id)
        if job:
            return job.get_summary()
        return None

    def get_pipeline_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics."""
        metrics = self._pipeline_metrics.copy()

        # Calculate technique rankings
        rankings = {}
        for tech, data in metrics.get("technique_rankings", {}).items():
            if data["count"] > 0:
                rankings[tech] = data["total_score"] / data["count"]
        metrics["technique_rankings"] = rankings

        return metrics

    def get_agent_statuses(self) -> dict[str, AgentStatus]:
        """Get all agent statuses."""
        return self._agent_statuses.copy()

    def get_agent_health(self) -> dict[str, bool]:
        """Get health status of all agents."""
        return self._agent_health.copy()

    async def get_dataset_stats(self) -> dict[str, Any]:
        """Get statistics about loaded datasets."""
        if not self._dataset_loader:
            return {"loaded": False}

        return {
            "loaded": self._dataset_loader.is_loaded,
            "total_entries": self._dataset_loader.get_total_entries(),
            "techniques": self._dataset_loader.get_techniques(),
            "models": self._dataset_loader.get_models(),
            "stats": {
                name: {
                    "total": stats.total_entries,
                    "successful": stats.successful_entries,
                    "success_rate": stats.success_rate,
                }
                for name, stats in self._dataset_loader.get_stats().items()
            },
        }
