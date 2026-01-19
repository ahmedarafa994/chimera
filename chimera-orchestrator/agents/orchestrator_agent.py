"""
Orchestrator Agent - Manages the overall pipeline and coordinates all agents
"""

import asyncio
import contextlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

try:
    from core.config import Config
    from core.message_queue import MessageQueue
    from core.models import (
        AgentStatus,
        AgentType,
        EvaluationRequest,
        EvaluationResult,
        ExecutionRequest,
        ExecutionResult,
        GeneratedPrompt,
        JobStatus,
        Message,
        MessageType,
        PipelineJob,
        PromptRequest,
        TargetLLM,
    )

    from agents.base_agent import BaseAgent
except ImportError:
    from ..core.config import Config
    from ..core.message_queue import MessageQueue
    from ..core.models import (
        AgentStatus,
        AgentType,
        EvaluationResult,
        ExecutionResult,
        GeneratedPrompt,
        JobStatus,
        Message,
        MessageType,
        PipelineJob,
        TargetLLM,
    )
    from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator that manages the adversarial testing pipeline.

    Features:
    - Job lifecycle management
    - Agent coordination
    - Pipeline execution
    - Health monitoring
    - Results aggregation
    """

    def __init__(self, config: Config, message_queue: MessageQueue, agent_id: str | None = None):
        super().__init__(
            agent_type=AgentType.ORCHESTRATOR,
            config=config,
            message_queue=message_queue,
            agent_id=agent_id,
        )

        # Job management
        self._jobs: dict[str, PipelineJob] = {}
        self._job_queue: asyncio.Queue = asyncio.Queue()

        # Agent tracking
        self._agent_statuses: dict[str, AgentStatus] = {}

        # Pipeline metrics
        self._pipeline_metrics = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "average_job_duration": 0,
            "total_prompts_generated": 0,
            "total_executions": 0,
            "total_evaluations": 0,
            "jailbreak_success_rate": 0,
        }

        # Processing tasks
        self._job_processor_task: asyncio.Task | None = None
        self._health_monitor_task: asyncio.Task | None = None

    async def on_start(self):
        """Start job processor and health monitor."""
        self._job_processor_task = asyncio.create_task(self._job_processor_loop())
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Orchestrator started with job processor and health monitor")

    async def on_stop(self):
        """Stop background tasks."""
        if self._job_processor_task:
            self._job_processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._job_processor_task

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_monitor_task

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
                prompt = GeneratedPrompt(
                    id=prompt_data.get("id", str(uuid.uuid4())),
                    original_query=prompt_data.get("original_query", ""),
                    enhanced_prompt=prompt_data.get("enhanced_prompt", ""),
                    technique_used=prompt_data.get("technique_used", ""),
                    pattern_used=prompt_data.get("pattern_used"),
                    persona=prompt_data.get("persona"),
                    potency_level=prompt_data.get("potency_level", 5),
                    metadata=prompt_data.get("metadata", {}),
                )
                job.generated_prompts.append(prompt)

            self._pipeline_metrics["total_prompts_generated"] += len(prompts_data)

            # Move to execution phase
            job.status = JobStatus.EXECUTING
            await self._execute_prompts(job)

        except Exception as e:
            logger.error(f"Error handling generate response: {e}")
            await self._fail_job(job, str(e))

    async def _handle_execute_response(self, message: Message):
        """Handle response from execution agent."""
        job_id = message.job_id
        job = self._jobs.get(job_id)

        if not job:
            logger.warning(f"Job {job_id} not found for execute response")
            return

        try:
            # Parse execution result
            result = ExecutionResult(
                id=message.payload.get("id", str(uuid.uuid4())),
                prompt_id=message.payload.get("prompt_id", ""),
                target_llm=message.payload.get("target_llm", ""),
                response_text=message.payload.get("response_text", ""),
                response_time_ms=message.payload.get("response_time_ms", 0),
                tokens_used=message.payload.get("tokens_used", 0),
                success=message.payload.get("success", False),
                error_message=message.payload.get("error_message"),
                raw_response=message.payload.get("raw_response", {}),
            )
            job.execution_results.append(result)
            self._pipeline_metrics["total_executions"] += 1

            # Check if all executions are complete
            expected_executions = len(job.generated_prompts)
            if len(job.execution_results) >= expected_executions:
                # Move to evaluation phase
                job.status = JobStatus.EVALUATING
                await self._evaluate_responses(job)

        except Exception as e:
            logger.error(f"Error handling execute response: {e}")
            await self._fail_job(job, str(e))

    async def _handle_evaluate_response(self, message: Message):
        """Handle response from evaluator agent."""
        job_id = message.job_id
        job = self._jobs.get(job_id)

        if not job:
            logger.warning(f"Job {job_id} not found for evaluate response")
            return

        try:
            # Parse evaluation result
            from ..core.models import SafetyLevel

            result = EvaluationResult(
                id=message.payload.get("id", str(uuid.uuid4())),
                execution_id=message.payload.get("execution_id", ""),
                safety_level=SafetyLevel(message.payload.get("safety_level", "safe")),
                jailbreak_detected=message.payload.get("jailbreak_detected", False),
                confidence_score=message.payload.get("confidence_score", 0.0),
                safety_score=message.payload.get("safety_score", 100.0),
                complex_content_detected=message.payload.get("complex_content_detected", []),
                technique_effectiveness=message.payload.get("technique_effectiveness", 0.0),
                detailed_analysis=message.payload.get("detailed_analysis", {}),
                recommendations=message.payload.get("recommendations", []),
            )
            job.evaluation_results.append(result)
            self._pipeline_metrics["total_evaluations"] += 1

            # Check if all evaluations are complete
            expected_evaluations = len(job.execution_results)
            if len(job.evaluation_results) >= expected_evaluations:
                # Complete the job
                await self._complete_job(job)

        except Exception as e:
            logger.error(f"Error handling evaluate response: {e}")
            await self._fail_job(job, str(e))

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

    async def _handle_error(self, message: Message):
        """Handle error messages."""
        job_id = message.job_id
        error = message.payload.get("error", "Unknown error")

        logger.error(f"Error in job {job_id}: {error}")

        job = self._jobs.get(job_id)
        if job:
            await self._fail_job(job, error)

    async def _handle_status_update(self, message: Message):
        """Handle status update requests."""
        # Respond with orchestrator status
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
                        if j.status not in [JobStatus.COMPLETED, JobStatus.FAILED]
                    ]
                ),
                "agent_statuses": {
                    aid: status.to_dict() for aid, status in self._agent_statuses.items()
                },
            },
        )

    async def create_job(self, query: str, config: dict[str, Any] | None = None) -> PipelineJob:
        """
        Create a new pipeline job.

        Args:
            query: The original query to test
            config: Job configuration options

        Returns:
            The created PipelineJob
        """
        job = PipelineJob(original_query=query, config=config or {}, status=JobStatus.PENDING)

        self._jobs[job.id] = job
        self._pipeline_metrics["total_jobs"] += 1

        # Add to processing queue
        await self._job_queue.put(job.id)

        logger.info(f"Created job {job.id} for query: {query[:50]}...")
        return job

    async def _job_processor_loop(self):
        """Process jobs from the queue."""
        while self._running:
            try:
                # Get next job from queue
                job_id = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)

                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.PENDING:
                    await self._process_job(job)

            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job processor: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job: PipelineJob):
        """Process a single job through the pipeline."""
        logger.info(f"Processing job {job.id}")

        job.started_at = datetime.utcnow()
        job.status = JobStatus.GENERATING
        self.add_active_job(job.id)

        try:
            # Start with prompt generation
            await self._generate_prompts(job)

        except Exception as e:
            logger.error(f"Error processing job {job.id}: {e}")
            await self._fail_job(job, str(e))

    async def _generate_prompts(self, job: PipelineJob):
        """Request prompt generation from generator agent."""
        config = job.config

        request_payload = {
            "original_query": job.original_query,
            "technique": config.get("technique"),
            "pattern": config.get("pattern"),
            "persona": config.get("persona"),
            "context_type": config.get("context_type"),
            "potency": config.get("potency", 5),
            "num_variants": config.get("num_variants", 3),
        }

        await self.send_message(
            MessageType.GENERATE_REQUEST,
            target=AgentType.GENERATOR,
            job_id=job.id,
            payload=request_payload,
            priority=7,
        )

    async def _execute_prompts(self, job: PipelineJob):
        """Request execution of generated prompts."""
        config = job.config
        target_llm = TargetLLM(config.get("target_llm", "local_ollama"))

        for prompt in job.generated_prompts:
            request_payload = {
                "prompt_id": prompt.id,
                "prompt_text": prompt.enhanced_prompt,
                "target_llm": target_llm.value,
                "model_config": config.get("model_config", {}),
                "timeout": config.get("timeout", 60),
                "retry_count": config.get("retry_count", 3),
            }

            await self.send_message(
                MessageType.EXECUTE_REQUEST,
                target=AgentType.EXECUTOR,
                job_id=job.id,
                payload=request_payload,
                priority=6,
            )

    async def _evaluate_responses(self, job: PipelineJob):
        """Request evaluation of execution results."""
        # Create a mapping of prompt_id to prompt
        prompt_map = {p.id: p for p in job.generated_prompts}

        for result in job.execution_results:
            if not result.success:
                continue

            prompt = prompt_map.get(result.prompt_id)
            if not prompt:
                continue

            request_payload = {
                "execution_id": result.id,
                "prompt_text": prompt.enhanced_prompt,
                "response_text": result.response_text,
                "original_query": job.original_query,
                "technique_used": prompt.technique_used,
                "evaluation_criteria": job.config.get("evaluation_criteria", []),
            }

            await self.send_message(
                MessageType.EVALUATE_REQUEST,
                target=AgentType.EVALUATOR,
                job_id=job.id,
                payload=request_payload,
                priority=6,
            )

        # If no successful executions, complete the job
        successful_executions = [r for r in job.execution_results if r.success]
        if not successful_executions:
            await self._complete_job(job)

    async def _complete_job(self, job: PipelineJob):
        """Mark a job as completed and update metrics."""
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()

        self._pipeline_metrics["completed_jobs"] += 1
        self._update_success_rate()
        self._update_average_duration(job)

        self.remove_active_job(job.id)

        logger.info(f"Job {job.id} completed. Summary: {job.get_summary()}")

    async def _fail_job(self, job: PipelineJob, error: str):
        """Mark a job as failed."""
        job.status = JobStatus.FAILED
        job.error_message = error
        job.completed_at = datetime.utcnow()

        self._pipeline_metrics["failed_jobs"] += 1

        self.remove_active_job(job.id)

        logger.error(f"Job {job.id} failed: {error}")

    def _update_success_rate(self):
        """Update the jailbreak success rate metric."""
        total_evaluations = sum(
            len(job.evaluation_results)
            for job in self._jobs.values()
            if job.status == JobStatus.COMPLETED
        )

        successful_jailbreaks = sum(
            sum(1 for e in job.evaluation_results if e.jailbreak_detected)
            for job in self._jobs.values()
            if job.status == JobStatus.COMPLETED
        )

        if total_evaluations > 0:
            self._pipeline_metrics["jailbreak_success_rate"] = (
                successful_jailbreaks / total_evaluations
            )

    def _update_average_duration(self, job: PipelineJob):
        """Update average job duration metric."""
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()

            completed = self._pipeline_metrics["completed_jobs"]
            current_avg = self._pipeline_metrics["average_job_duration"]

            self._pipeline_metrics["average_job_duration"] = (
                current_avg * (completed - 1) + duration
            ) / completed

    async def _health_monitor_loop(self):
        """Monitor agent health and handle failures."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Check for stale agent statuses
                now = datetime.utcnow()
                stale_threshold = timedelta(minutes=2)

                for agent_id, status in list(self._agent_statuses.items()):
                    if now - status.last_heartbeat > stale_threshold:
                        logger.warning(f"Agent {agent_id} appears unhealthy (no heartbeat)")
                        status.is_healthy = False

                # Check for stuck jobs
                stuck_threshold = timedelta(seconds=self.config.orchestrator.job_timeout)

                for job_id, job in list(self._jobs.items()):
                    if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        if job.started_at and now - job.started_at > stuck_threshold:
                            logger.warning(f"Job {job_id} appears stuck, marking as failed")
                            await self._fail_job(job, "Job timeout exceeded")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    def get_job(self, job_id: str) -> PipelineJob | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> list[PipelineJob]:
        """Get all jobs."""
        return list(self._jobs.values())

    def get_job_summary(self, job_id: str) -> dict[str, Any] | None:
        """Get summary of a job."""
        job = self._jobs.get(job_id)
        if job:
            return job.get_summary()
        return None

    def get_pipeline_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics."""
        return self._pipeline_metrics.copy()

    def get_agent_statuses(self) -> dict[str, AgentStatus]:
        """Get all agent statuses."""
        return self._agent_statuses.copy()
