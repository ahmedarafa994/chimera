"""
Data Pipeline Streaming WebSocket Endpoints

Real-time WebSocket endpoints for delivering live metrics and pipeline events
to frontend clients.

Endpoints:
- WS /api/v1/pipeline/streaming/metrics - Real-time metrics stream
- WS /api/v1/pipeline/streaming/events - Live event stream
- WS /api/v1/pipeline/streaming/health - Pipeline health monitoring
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.services.data_pipeline.realtime_metrics import AggregationType, MetricType
from app.services.data_pipeline.streaming_pipeline import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Pipeline Streaming"])


# Connection manager for WebSocket clients
class ConnectionManager:
    """Manages WebSocket connections for broadcasting"""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}
        self.client_metadata: dict[WebSocket, dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, channel: str, client_id: str) -> None:
        """Connect a WebSocket client to a channel"""
        await websocket.accept()

        if channel not in self.active_connections:
            self.active_connections[channel] = set()

        self.active_connections[channel].add(websocket)
        self.client_metadata[websocket] = {
            "client_id": client_id,
            "channel": channel,
            "connected_at": datetime.utcnow(),
        }

        logger.info(f"Client {client_id} connected to channel {channel}")

    def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client"""
        metadata = self.client_metadata.get(websocket, {})
        channel = metadata.get("channel")
        client_id = metadata.get("client_id", "unknown")

        if channel and channel in self.active_connections:
            self.active_connections[channel].discard(websocket)

        self.client_metadata.pop(websocket, None)
        logger.info(f"Client {client_id} disconnected from channel {channel}")

    async def send_personal(self, message: dict, websocket: WebSocket) -> None:
        """Send a message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, channel: str, message: dict) -> None:
        """Broadcast a message to all clients in a channel"""
        if channel not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def get_client_count(self, channel: str | None = None) -> int:
        """Get count of connected clients"""
        if channel:
            return len(self.active_connections.get(channel, set()))
        return sum(len(conns) for conns in self.active_connections.values())


# Global connection manager
manager = ConnectionManager()


@router.websocket("/pipeline/streaming/metrics")
async def metrics_stream(
    websocket: WebSocket,
    provider: str = Query(..., description="LLM provider to stream metrics for"),
    metric_type: str = Query(
        "latency", description="Type of metric (latency, tokens, errors, requests)"
    ),
    interval_seconds: int = Query(5, description="Update interval in seconds"),
):
    """
    Real-time metrics stream for a provider

    Streams metrics updates at the specified interval.

    Example:
        ws://localhost:8001/api/v1/pipeline/streaming/metrics?provider=google&metric_type=latency&interval_seconds=5
    """
    client_id = f"metrics-{provider}-{metric_type}-{datetime.utcnow().timestamp()}"
    channel = "metrics"

    await manager.connect(websocket, channel, client_id)

    try:
        pipeline = await get_pipeline()

        # Send initial connection message
        await manager.send_personal(
            {
                "type": "connected",
                "client_id": client_id,
                "provider": provider,
                "metric_type": metric_type,
                "interval_seconds": interval_seconds,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )

        # Map metric type string to enum
        metric_type_map = {
            "latency": MetricType.LATENCY,
            "tokens": MetricType.TOKEN_USAGE,
            "errors": MetricType.ERROR_RATE,
            "requests": MetricType.REQUEST_COUNT,
        }
        metric_type_enum = metric_type_map.get(metric_type, MetricType.LATENCY)

        # Stream metrics at interval
        while True:
            try:
                # Get metrics data
                metrics_data = await pipeline.get_metrics(
                    provider=provider,
                    metric_type=metric_type_enum,
                    start_ts=f"-{interval_seconds * 2}m",  # Get 2x interval window
                    aggregation_type=AggregationType.AVG,
                    bucket_size_ms=interval_seconds * 1000,
                )

                # Send metrics update
                await manager.send_personal(
                    {
                        "type": "metrics_update",
                        "provider": provider,
                        "metric_type": metric_type,
                        "data": metrics_data,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in metrics stream: {e}")
                await manager.send_personal(
                    {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )
                await asyncio.sleep(interval_seconds)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


@router.websocket("/pipeline/streaming/events")
async def events_stream(
    websocket: WebSocket,
    event_types: str = Query(
        "all", description="Comma-separated event types (llm,transformation,jailbreak)"
    ),
    client_id: str | None = Query(None, description="Optional client identifier"),
):
    """
    Live event stream from the pipeline

    Streams events as they are published to Redis Streams.

    Example:
        ws://localhost:8001/api/v1/pipeline/streaming/events?event_types=llm,transformation
    """
    client_id = client_id or f"events-{datetime.utcnow().timestamp()}"
    channel = "events"

    await manager.connect(websocket, channel, client_id)

    try:
        pipeline = await get_pipeline()

        await manager.send_personal(
            {
                "type": "connected",
                "client_id": client_id,
                "event_types": event_types,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )

        # Parse event types
        types = (
            [t.strip() for t in event_types.split(",")]
            if event_types != "all"
            else ["llm", "transformation", "jailbreak"]
        )

        # Set custom analytics handler to forward events
        async def event_forwarder(entry):
            """Forward events to WebSocket client"""
            try:
                event_type = entry.event_type

                # Filter by event types
                if (
                    "all" not in types
                    and event_type.replace("_interaction", "").replace("_experiment", "")
                    not in types
                ):
                    return

                await manager.send_personal(
                    {
                        "type": "event",
                        "event_type": event_type,
                        "event_id": entry.entry_id,
                        "timestamp": entry.timestamp.isoformat(),
                        "data": entry.data,
                        "metadata": entry.metadata,
                    },
                    websocket,
                )

            except Exception as e:
                logger.error(f"Error forwarding event: {e}")

        # Register handler
        pipeline.set_analytics_handler(event_forwarder)

        # Keep connection alive with heartbeat
        while True:
            try:
                await asyncio.sleep(30)
                await manager.send_personal(
                    {
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in event stream: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


@router.websocket("/pipeline/streaming/health")
async def health_stream(
    websocket: WebSocket,
    interval_seconds: int = Query(10, description="Health check interval in seconds"),
    client_id: str | None = Query(None, description="Optional client identifier"),
):
    """
    Pipeline health monitoring stream

    Streams health status and pipeline statistics at the specified interval.

    Example:
        ws://localhost:8001/api/v1/pipeline/streaming/health?interval_seconds=10
    """
    client_id = client_id or f"health-{datetime.utcnow().timestamp()}"
    channel = "health"

    await manager.connect(websocket, channel, client_id)

    try:
        pipeline = await get_pipeline()

        await manager.send_personal(
            {
                "type": "connected",
                "client_id": client_id,
                "interval_seconds": interval_seconds,
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket,
        )

        while True:
            try:
                # Get health status
                health = await pipeline.health_check()

                # Get pipeline stats
                stats = await pipeline.get_pipeline_stats()

                # Send health update
                await manager.send_personal(
                    {
                        "type": "health_update",
                        "health": health,
                        "stats": stats,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )

                await asyncio.sleep(interval_seconds)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in health stream: {e}")
                await manager.send_personal(
                    {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )
                await asyncio.sleep(interval_seconds)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


# REST endpoints for streaming status
class StreamingStatusResponse(BaseModel):
    """Response model for streaming status"""

    active_connections: int
    channels: dict[str, int]
    pipeline_running: bool
    timestamp: str


@router.get("/pipeline/streaming/status", response_model=StreamingStatusResponse)
async def get_streaming_status():
    """
    Get current streaming status

    Returns information about active WebSocket connections and pipeline status.
    """
    try:
        pipeline = await get_pipeline()

        # Get connection counts per channel
        channels = {}
        for channel_name in ["metrics", "events", "health"]:
            channels[channel_name] = manager.get_client_count(channel_name)

        return StreamingStatusResponse(
            active_connections=manager.get_client_count(),
            channels=channels,
            pipeline_running=pipeline._is_running,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to get streaming status: {e}")
        raise


class BroadcastRequest(BaseModel):
    """Request model for broadcasting messages"""

    channel: str
    message: dict[str, Any]


@router.post("/pipeline/streaming/broadcast")
async def broadcast_message(request: BroadcastRequest):
    """
    Broadcast a message to all clients in a channel

    Admin endpoint for sending system-wide notifications.
    """
    try:
        await manager.broadcast(
            request.channel,
            {
                **request.message,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        return {
            "success": True,
            "channel": request.channel,
            "client_count": manager.get_client_count(request.channel),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to broadcast message: {e}")
        raise
