"""
Base Agent class for all Chimera agents
"""

import asyncio
import contextlib
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

# Use try/except for flexible imports
try:
    from core.config import Config
    from core.message_queue import MessageQueue
    from core.models import AgentStatus, AgentType, Message, MessageType
except ImportError:
    from ..core.config import Config
    from ..core.message_queue import MessageQueue
    from ..core.models import AgentStatus, AgentType, Message, MessageType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Chimera system.

    Provides common functionality for:
    - Message queue communication
    - Health monitoring
    - Lifecycle management
    - Metrics collection
    """

    def __init__(
        self,
        agent_type: AgentType,
        config: Config,
        message_queue: MessageQueue,
        agent_id: str | None = None,
    ):
        self.agent_type = agent_type
        self.agent_id = agent_id or f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        self.config = config
        self.message_queue = message_queue

        # State
        self._running = False
        self._paused = False
        self._active_jobs: list[str] = []
        self._metrics: dict[str, Any] = {
            "messages_processed": 0,
            "messages_sent": 0,
            "errors": 0,
            "start_time": None,
            "last_activity": None,
        }

        # Tasks
        self._main_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Check if the agent is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if the agent is paused."""
        return self._paused

    @property
    def status(self) -> AgentStatus:
        """Get the current status of the agent."""
        return AgentStatus(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            is_healthy=self._running and not self._paused,
            current_load=len(self._active_jobs),
            max_capacity=self.config.orchestrator.max_concurrent_jobs,
            last_heartbeat=datetime.utcnow(),
            active_jobs=self._active_jobs.copy(),
            metrics=self._metrics.copy(),
        )

    async def start(self):
        """Start the agent."""
        if self._running:
            logger.warning(f"Agent {self.agent_id} is already running")
            return

        logger.info(f"Starting agent {self.agent_id}")
        self._running = True
        self._metrics["start_time"] = datetime.utcnow()

        # Start main processing loop
        self._main_task = asyncio.create_task(self._run_loop())

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        await self.on_start()
        logger.info(f"Agent {self.agent_id} started")

    async def stop(self):
        """Stop the agent."""
        if not self._running:
            return

        logger.info(f"Stopping agent {self.agent_id}")
        self._running = False

        # Cancel tasks
        if self._main_task:
            self._main_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main_task

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        await self.on_stop()
        logger.info(f"Agent {self.agent_id} stopped")

    async def pause(self):
        """Pause the agent."""
        self._paused = True
        logger.info(f"Agent {self.agent_id} paused")

    async def resume(self):
        """Resume the agent."""
        self._paused = False
        logger.info(f"Agent {self.agent_id} resumed")

    async def _run_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # Consume messages from queue
                message = await self.message_queue.consume(self.agent_type, timeout=1.0)

                if message:
                    await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id} run loop: {e}")
                self._metrics["errors"] += 1
                await asyncio.sleep(1)

    async def _handle_message(self, message: Message):
        """Handle an incoming message."""
        try:
            self._metrics["messages_processed"] += 1
            self._metrics["last_activity"] = datetime.utcnow()

            # Route to appropriate handler
            if message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            else:
                await self.process_message(message)

        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}")
            self._metrics["errors"] += 1

            # Send error response
            await self.send_message(
                MessageType.ERROR,
                target=message.source,
                job_id=message.job_id,
                payload={"error": str(e), "original_message_id": message.id},
            )

    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat message."""
        # Respond with status
        await self.send_message(
            MessageType.HEARTBEAT, target=message.source, payload=self.status.to_dict()
        )

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds

                await self.send_message(
                    MessageType.HEARTBEAT,
                    target=AgentType.ORCHESTRATOR,
                    payload=self.status.to_dict(),
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def send_message(
        self,
        message_type: MessageType,
        target: AgentType | None = None,
        job_id: str = "",
        payload: dict[str, Any] | None = None,
        priority: int = 5,
    ) -> bool:
        """
        Send a message to the queue.

        Args:
            message_type: Type of message
            target: Target agent (None for broadcast)
            job_id: Associated job ID
            payload: Message payload
            priority: Message priority (1-10)

        Returns:
            True if sent successfully
        """
        message = Message(
            type=message_type,
            source=self.agent_type,
            target=target,
            job_id=job_id,
            payload=payload or {},
            priority=priority,
        )

        success = await self.message_queue.publish(message)
        if success:
            self._metrics["messages_sent"] += 1
        return success

    def add_active_job(self, job_id: str):
        """Add a job to the active jobs list."""
        if job_id not in self._active_jobs:
            self._active_jobs.append(job_id)

    def remove_active_job(self, job_id: str):
        """Remove a job from the active jobs list."""
        if job_id in self._active_jobs:
            self._active_jobs.remove(job_id)

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    async def process_message(self, message: Message):
        """
        Process an incoming message.
        Must be implemented by subclasses.
        """
        pass

    async def on_start(self):
        """Called when the agent starts. Override for custom initialization."""
        pass

    async def on_stop(self):
        """Called when the agent stops. Override for custom cleanup."""
        pass
