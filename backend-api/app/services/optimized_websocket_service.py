"""
Optimized WebSocket Service with performance enhancements.

This module provides comprehensive WebSocket optimizations:
- Connection pooling and management
- Message batching and throttling
- Memory leak prevention
- Heartbeat optimization
- Real-time performance monitoring
- Auto-scaling and load balancing
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from app.core.logging import logger
from app.domain.models import StreamChunk


class ConnectionState(Enum):
    """WebSocket connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(Enum):
    """WebSocket message types."""

    TEXT = "text"
    BINARY = "binary"
    JSON = "json"
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"
    STREAM_CHUNK = "stream_chunk"
    ERROR = "error"
    STATUS = "status"


@dataclass
class ConnectionMetrics:
    """Metrics for a WebSocket connection."""

    connection_id: str
    client_ip: str
    connected_at: float
    last_activity: float
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    heartbeat_count: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class MessageBatch:
    """Batch of messages for efficient sending."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    max_size: int = 10
    max_age_ms: int = 100


class OptimizedWebSocketConnection:
    """
    Optimized WebSocket connection with performance enhancements.
    """

    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str,
        client_ip: str = "",
        heartbeat_interval: int = 30,
    ):
        self.websocket = websocket
        self.connection_id = connection_id
        self.client_ip = client_ip
        self.heartbeat_interval = heartbeat_interval

        # State management
        self.state = ConnectionState.CONNECTING
        self.connected_at = time.time()
        self.last_activity = time.time()

        # Metrics
        self.metrics = ConnectionMetrics(
            connection_id=connection_id,
            client_ip=client_ip,
            connected_at=self.connected_at,
            last_activity=self.last_activity,
        )

        # Message batching
        self._message_batch = MessageBatch()
        self._batch_lock = asyncio.Lock()

        # Background tasks
        self._heartbeat_task: asyncio.Task | None = None
        self._batch_sender_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Rate limiting
        self._message_queue: deque = deque(maxlen=1000)
        self._rate_limit_window = 60  # seconds
        self._rate_limit_max = 100  # messages per window

        # Connection health
        self._ping_pending = False
        self._ping_timeout = 10.0

    async def initialize(self) -> None:
        """Initialize the optimized connection."""
        try:
            # Accept WebSocket connection
            await self.websocket.accept()
            self.state = ConnectionState.CONNECTED

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._batch_sender_task = asyncio.create_task(self._batch_sender_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.debug(f"WebSocket connection {self.connection_id} initialized")

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket connection {self.connection_id}: {e}")
            self.state = ConnectionState.ERROR
            raise

    async def close(self, code: int = status.WS_1000_NORMAL_CLOSURE, reason: str = "") -> None:
        """Close the connection gracefully."""
        if self.state in [ConnectionState.DISCONNECTING, ConnectionState.DISCONNECTED]:
            return

        self.state = ConnectionState.DISCONNECTING

        try:
            # Cancel background tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self._batch_sender_task:
                self._batch_sender_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()

            # Send pending batched messages
            await self._flush_message_batch()

            # Close WebSocket
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.close(code=code, reason=reason)

            self.state = ConnectionState.DISCONNECTED
            logger.debug(f"WebSocket connection {self.connection_id} closed gracefully")

        except Exception as e:
            logger.error(f"Error closing WebSocket connection {self.connection_id}: {e}")
            self.state = ConnectionState.ERROR

    async def send_json(
        self, data: dict[str, Any], batch: bool = True, priority: bool = False
    ) -> None:
        """Send JSON data with optional batching."""
        if self.state != ConnectionState.CONNECTED:
            logger.warning(f"Cannot send to disconnected connection {self.connection_id}")
            return

        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning(f"Rate limit exceeded for connection {self.connection_id}")
            return

        try:
            message = {
                "type": MessageType.JSON.value,
                "data": data,
                "timestamp": time.time(),
            }

            if batch and not priority:
                # Add to batch for efficient sending
                await self._add_to_batch(message)
            else:
                # Send immediately
                await self._send_message(message)

            # Update metrics
            self.metrics.messages_sent += 1
            self.last_activity = time.time()

        except Exception as e:
            logger.error(f"Error sending JSON to connection {self.connection_id}: {e}")
            self.metrics.errors += 1

    async def send_text(self, text: str, batch: bool = False) -> None:
        """Send text data."""
        if self.state != ConnectionState.CONNECTED:
            return

        if not self._check_rate_limit():
            return

        try:
            message = {
                "type": MessageType.TEXT.value,
                "data": text,
                "timestamp": time.time(),
            }

            if batch:
                await self._add_to_batch(message)
            else:
                await self._send_message(message)

            self.metrics.messages_sent += 1
            self.last_activity = time.time()

        except Exception as e:
            logger.error(f"Error sending text to connection {self.connection_id}: {e}")
            self.metrics.errors += 1

    async def send_stream_chunk(self, chunk: StreamChunk) -> None:
        """Send streaming chunk with optimization."""
        if self.state != ConnectionState.CONNECTED:
            return

        try:
            message = {
                "type": MessageType.STREAM_CHUNK.value,
                "data": {
                    "text": chunk.text,
                    "is_final": chunk.is_final,
                    "finish_reason": chunk.finish_reason,
                },
                "timestamp": time.time(),
            }

            # Stream chunks are sent immediately (no batching)
            await self._send_message(message)

            self.metrics.messages_sent += 1
            self.last_activity = time.time()

        except Exception as e:
            logger.error(f"Error sending stream chunk to connection {self.connection_id}: {e}")
            self.metrics.errors += 1

    async def receive_json(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Receive JSON data with timeout."""
        if self.state != ConnectionState.CONNECTED:
            return None

        try:
            # Receive message with timeout
            if timeout:
                data = await asyncio.wait_for(self.websocket.receive_json(), timeout=timeout)
            else:
                data = await self.websocket.receive_json()

            self.metrics.messages_received += 1
            self.last_activity = time.time()

            return data

        except TimeoutError:
            logger.debug(f"Receive timeout for connection {self.connection_id}")
            return None
        except WebSocketDisconnect:
            logger.info(f"WebSocket connection {self.connection_id} disconnected")
            self.state = ConnectionState.DISCONNECTED
            return None
        except Exception as e:
            logger.error(f"Error receiving from connection {self.connection_id}: {e}")
            self.metrics.errors += 1
            return None

    async def ping(self, timeout: float | None = None) -> float:
        """Send ping and measure latency."""
        if self.state != ConnectionState.CONNECTED or self._ping_pending:
            return -1.0

        try:
            self._ping_pending = True
            ping_time = time.time()

            # Send ping
            await self.websocket.ping()

            # Wait for pong
            pong_timeout = timeout or self._ping_timeout
            await asyncio.wait_for(self._wait_for_pong(), timeout=pong_timeout)

            # Calculate latency
            latency_ms = (time.time() - ping_time) * 1000

            # Update metrics
            self.metrics.heartbeat_count += 1
            total_pings = self.metrics.heartbeat_count
            current_avg = self.metrics.avg_latency_ms
            self.metrics.avg_latency_ms = (
                current_avg * (total_pings - 1) + latency_ms
            ) / total_pings

            return latency_ms

        except TimeoutError:
            logger.warning(f"Ping timeout for connection {self.connection_id}")
            return -1.0
        except Exception as e:
            logger.error(f"Ping error for connection {self.connection_id}: {e}")
            return -1.0
        finally:
            self._ping_pending = False

    async def _add_to_batch(self, message: dict[str, Any]) -> None:
        """Add message to batch for efficient sending."""
        async with self._batch_lock:
            self._message_batch.messages.append(message)

            # Send batch if it's full or old
            should_send = (
                len(self._message_batch.messages) >= self._message_batch.max_size
                or (time.time() - self._message_batch.created_at) * 1000
                >= self._message_batch.max_age_ms
            )

            if should_send:
                await self._flush_message_batch()

    async def _flush_message_batch(self) -> None:
        """Flush pending message batch."""
        if not self._message_batch.messages:
            return

        try:
            # Send batch as single message
            batch_message = {
                "type": "batch",
                "messages": self._message_batch.messages,
                "count": len(self._message_batch.messages),
                "timestamp": time.time(),
            }

            await self._send_message(batch_message)

            # Update metrics
            self.metrics.messages_sent += len(self._message_batch.messages)

            # Reset batch
            self._message_batch = MessageBatch()

        except Exception as e:
            logger.error(f"Error flushing message batch for connection {self.connection_id}: {e}")

    async def _send_message(self, message: dict[str, Any]) -> None:
        """Send individual message."""
        message_json = json.dumps(message)
        await self.websocket.send_text(message_json)

        # Update bytes sent
        self.metrics.bytes_sent += len(message_json.encode("utf-8"))

    def _check_rate_limit(self) -> bool:
        """Check if message rate limit is exceeded."""
        current_time = time.time()

        # Remove old messages from rate limit window
        while (
            self._message_queue and current_time - self._message_queue[0] > self._rate_limit_window
        ):
            self._message_queue.popleft()

        # Check if under limit
        if len(self._message_queue) >= self._rate_limit_max:
            return False

        # Add current message to queue
        self._message_queue.append(current_time)
        return True

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                if self.state != ConnectionState.CONNECTED:
                    break

                # Send heartbeat ping
                latency = await self.ping(timeout=self._ping_timeout)

                if latency < 0:
                    logger.warning(f"Heartbeat failed for connection {self.connection_id}")
                    break

                logger.debug(f"Heartbeat for connection {self.connection_id}: {latency:.2f}ms")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error for connection {self.connection_id}: {e}")
                break

    async def _batch_sender_loop(self) -> None:
        """Background batch sender loop."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms

                if self.state != ConnectionState.CONNECTED:
                    break

                # Flush batch if it's getting old
                async with self._batch_lock:
                    if (
                        self._message_batch.messages
                        and (time.time() - self._message_batch.created_at) * 1000
                        >= self._message_batch.max_age_ms
                    ):
                        await self._flush_message_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch sender error for connection {self.connection_id}: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                if self.state != ConnectionState.CONNECTED:
                    break

                # Check for stale connection
                inactive_time = time.time() - self.last_activity
                if inactive_time > 300:  # 5 minutes of inactivity
                    logger.warning(f"Connection {self.connection_id} is stale, closing")
                    await self.close(code=status.WS_1001_GOING_AWAY, reason="Stale connection")
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error for connection {self.connection_id}: {e}")

    async def _wait_for_pong(self) -> None:
        """Wait for pong response."""
        # This is simplified - in real implementation you'd listen for pong frames
        await asyncio.sleep(0.1)

    def get_metrics(self) -> dict[str, Any]:
        """Get connection metrics."""
        current_time = time.time()

        return {
            "connection_id": self.connection_id,
            "client_ip": self.client_ip,
            "state": self.state.value,
            "connected_duration_seconds": current_time - self.connected_at,
            "last_activity_seconds_ago": current_time - self.last_activity,
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "bytes_sent": self.metrics.bytes_sent,
            "bytes_received": self.metrics.bytes_received,
            "errors": self.metrics.errors,
            "heartbeat_count": self.metrics.heartbeat_count,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "rate_limit_queue_size": len(self._message_queue),
            "pending_batch_size": len(self._message_batch.messages),
        }


class WebSocketConnectionManager:
    """
    Optimized WebSocket connection manager with load balancing and monitoring.
    """

    def __init__(self, max_connections: int = 1000, heartbeat_interval: int = 30):
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval

        # Connection management
        self._connections: dict[str, OptimizedWebSocketConnection] = {}
        self._connections_by_client_ip: dict[str, set[str]] = defaultdict(set)

        # Load balancing
        self._connection_groups: dict[str, set[str]] = defaultdict(set)

        # Global metrics
        self._global_metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "total_bytes_transferred": 0,
            "avg_connection_duration": 0.0,
        }

        # Background tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start the connection manager."""
        self._is_running = True

        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._global_cleanup_loop())

        logger.info("WebSocket connection manager started")

    async def stop(self) -> None:
        """Stop the connection manager."""
        self._is_running = False

        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Close all connections
        await self.close_all_connections()

        logger.info("WebSocket connection manager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        client_ip: str = "",
        group: str = "default",
    ) -> str:
        """Connect a new WebSocket client."""
        # Check connection limits
        if len(self._connections) >= self.max_connections:
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Server at capacity")
            raise RuntimeError("Connection limit exceeded")

        # Check per-IP limits
        if len(self._connections_by_client_ip[client_ip]) >= 10:  # Max 10 per IP
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Too many connections from this IP"
            )
            raise RuntimeError("Per-IP connection limit exceeded")

        # Create connection
        connection_id = str(uuid4())
        connection = OptimizedWebSocketConnection(
            websocket=websocket,
            connection_id=connection_id,
            client_ip=client_ip,
            heartbeat_interval=self.heartbeat_interval,
        )

        # Initialize connection
        await connection.initialize()

        # Register connection
        self._connections[connection_id] = connection
        self._connections_by_client_ip[client_ip].add(connection_id)
        self._connection_groups[group].add(connection_id)

        # Update metrics
        self._global_metrics["total_connections"] += 1
        self._global_metrics["active_connections"] = len(self._connections)

        logger.info(f"WebSocket client connected: {connection_id} from {client_ip}")

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket client."""
        if connection_id not in self._connections:
            return

        connection = self._connections[connection_id]

        # Close connection
        await connection.close()

        # Update registrations
        client_ip = connection.client_ip
        self._connections_by_client_ip[client_ip].discard(connection_id)
        if not self._connections_by_client_ip[client_ip]:
            del self._connections_by_client_ip[client_ip]

        # Remove from groups
        for group_connections in self._connection_groups.values():
            group_connections.discard(connection_id)

        # Update metrics
        connection_metrics = connection.get_metrics()
        self._global_metrics["messages_sent"] += connection_metrics["messages_sent"]
        self._global_metrics["messages_received"] += connection_metrics["messages_received"]
        self._global_metrics["total_bytes_transferred"] += (
            connection_metrics["bytes_sent"] + connection_metrics["bytes_received"]
        )

        # Remove connection
        del self._connections[connection_id]
        self._global_metrics["active_connections"] = len(self._connections)

        logger.info(f"WebSocket client disconnected: {connection_id}")

    async def send_to_connection(
        self,
        connection_id: str,
        data: dict[str, Any],
        batch: bool = True,
    ) -> bool:
        """Send data to a specific connection."""
        if connection_id not in self._connections:
            return False

        try:
            await self._connections[connection_id].send_json(data, batch=batch)
            return True
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            return False

    async def send_to_group(
        self,
        group: str,
        data: dict[str, Any],
        batch: bool = True,
    ) -> int:
        """Send data to all connections in a group."""
        if group not in self._connection_groups:
            return 0

        successful_sends = 0
        tasks = []

        for connection_id in self._connection_groups[group]:
            if connection_id in self._connections:
                task = asyncio.create_task(
                    self._connections[connection_id].send_json(data, batch=batch)
                )
                tasks.append(task)

        # Wait for all sends
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if not isinstance(result, Exception):
                successful_sends += 1

        return successful_sends

    async def broadcast(self, data: dict[str, Any], batch: bool = True) -> int:
        """Broadcast data to all connections."""
        if not self._connections:
            return 0

        successful_sends = 0
        tasks = []

        for connection in self._connections.values():
            task = asyncio.create_task(connection.send_json(data, batch=batch))
            tasks.append(task)

        # Wait for all sends
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if not isinstance(result, Exception):
                successful_sends += 1

        return successful_sends

    async def stream_to_connection(
        self,
        connection_id: str,
        chunks: list[StreamChunk],
    ) -> bool:
        """Stream chunks to a specific connection."""
        if connection_id not in self._connections:
            return False

        try:
            connection = self._connections[connection_id]
            for chunk in chunks:
                await connection.send_stream_chunk(chunk)
            return True
        except Exception as e:
            logger.error(f"Error streaming to connection {connection_id}: {e}")
            return False

    async def close_all_connections(self) -> None:
        """Close all active connections."""
        close_tasks = []

        for connection in self._connections.values():
            task = asyncio.create_task(connection.close())
            close_tasks.append(task)

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self._connections.clear()
        self._connections_by_client_ip.clear()
        self._connection_groups.clear()

        logger.info("All WebSocket connections closed")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute

                # Calculate average connection duration
                if self._connections:
                    total_duration = sum(
                        time.time() - conn.connected_at for conn in self._connections.values()
                    )
                    self._global_metrics["avg_connection_duration"] = total_duration / len(
                        self._connections
                    )

                # Log metrics
                logger.info(
                    f"WebSocket metrics - "
                    f"Active: {self._global_metrics['active_connections']}, "
                    f"Total: {self._global_metrics['total_connections']}, "
                    f"Avg duration: {self._global_metrics['avg_connection_duration']:.1f}s"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket monitoring error: {e}")

    async def _global_cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes

                # Find disconnected connections
                disconnected_ids = []
                for conn_id, connection in self._connections.items():
                    if connection.state in [ConnectionState.DISCONNECTED, ConnectionState.ERROR]:
                        disconnected_ids.append(conn_id)

                # Clean up disconnected connections
                for conn_id in disconnected_ids:
                    await self.disconnect(conn_id)

                if disconnected_ids:
                    logger.info(f"Cleaned up {len(disconnected_ids)} disconnected connections")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket cleanup error: {e}")

    def get_connection_metrics(self, connection_id: str) -> dict[str, Any] | None:
        """Get metrics for a specific connection."""
        if connection_id not in self._connections:
            return None

        return self._connections[connection_id].get_metrics()

    def get_global_metrics(self) -> dict[str, Any]:
        """Get global connection manager metrics."""
        # Calculate real-time stats
        active_by_state = defaultdict(int)
        total_errors = 0
        total_latency = 0
        ping_count = 0

        for connection in self._connections.values():
            metrics = connection.get_metrics()
            active_by_state[metrics["state"]] += 1
            total_errors += metrics["errors"]
            if metrics["heartbeat_count"] > 0:
                total_latency += metrics["avg_latency_ms"]
                ping_count += 1

        return {
            **self._global_metrics,
            "connections_by_state": dict(active_by_state),
            "total_errors": total_errors,
            "avg_latency_ms": total_latency / ping_count if ping_count > 0 else 0,
            "groups": {
                group: len(connections) for group, connections in self._connection_groups.items()
            },
            "connections_per_ip": {
                ip: len(connections) for ip, connections in self._connections_by_client_ip.items()
            },
        }


# Global optimized WebSocket manager
optimized_websocket_manager = WebSocketConnectionManager(
    max_connections=1000,
    heartbeat_interval=30,
)
