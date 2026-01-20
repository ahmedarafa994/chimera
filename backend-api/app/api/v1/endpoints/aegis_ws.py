"""Aegis Campaign Telemetry WebSocket Endpoint.

WebSocket endpoint for streaming real-time telemetry events from Aegis
campaigns. Provides heartbeat support, authentication, and graceful
connection handling.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.schemas.aegis_telemetry import (
    AegisTelemetryEventType,
    CampaignStatus,
    HeartbeatData,
    create_telemetry_event,
)
from app.services.websocket.aegis_broadcaster import aegis_telemetry_broadcaster

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration constants
HEARTBEAT_INTERVAL_SECONDS = 30  # Send heartbeat every 30 seconds
CONNECTION_TIMEOUT_SECONDS = 90  # Disconnect if no response in 90 seconds


class AegisTelemetryConnection:
    """Manages a single WebSocket connection for Aegis telemetry.

    Handles:
    - Connection lifecycle (accept, close, cleanup)
    - Heartbeat sending and response monitoring
    - Client message handling
    - Graceful disconnection
    """

    def __init__(
        self,
        websocket: WebSocket,
        campaign_id: str,
        client_id: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.campaign_id = campaign_id
        self.client_id = client_id or str(uuid.uuid4())
        self.connected_at = datetime.utcnow()
        self.last_heartbeat_response = datetime.utcnow()
        self.is_active = False
        self._heartbeat_task: asyncio.Task | None = None

    async def accept(self) -> bool:
        """Accept the WebSocket connection and initialize."""
        try:
            await self.websocket.accept()
            self.is_active = True

            # Subscribe to campaign telemetry
            await aegis_telemetry_broadcaster.subscribe_to_campaign(
                client_id=self.client_id,
                campaign_id=self.campaign_id,
                websocket=self.websocket,
            )

            logger.info(
                f"Aegis telemetry connection accepted: "
                f"client={self.client_id}, campaign={self.campaign_id}",
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to accept Aegis telemetry connection: {e}")
            return False

    async def close(self, code: int = 1000, reason: str = "Connection closed") -> None:
        """Close the WebSocket connection gracefully."""
        self.is_active = False

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        # Unsubscribe from campaign
        await aegis_telemetry_broadcaster.unsubscribe_from_campaign(
            client_id=self.client_id,
            campaign_id=self.campaign_id,
        )

        # Close websocket
        try:
            await self.websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.debug(f"Error closing websocket: {e}")

        logger.info(
            f"Aegis telemetry connection closed: "
            f"client={self.client_id}, campaign={self.campaign_id}",
        )

    async def start_heartbeat(self) -> None:
        """Start the heartbeat background task."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and check for stale connections."""
        while self.is_active:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

                if not self.is_active:
                    break

                # Check if connection is stale
                time_since_response = (
                    datetime.utcnow() - self.last_heartbeat_response
                ).total_seconds()

                if time_since_response > CONNECTION_TIMEOUT_SECONDS:
                    logger.warning(
                        f"Aegis telemetry connection timeout: "
                        f"client={self.client_id}, campaign={self.campaign_id}",
                    )
                    await self.close(code=1001, reason="Heartbeat timeout")
                    break

                # Send heartbeat
                await self._send_heartbeat()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in heartbeat loop: {e}")
                break

    async def _send_heartbeat(self) -> None:
        """Send a heartbeat event to the client."""
        try:
            state = aegis_telemetry_broadcaster.get_campaign_state(self.campaign_id)

            uptime = None
            if state and state.started_at:
                uptime = (datetime.utcnow() - state.started_at).total_seconds()

            heartbeat_data = HeartbeatData(
                server_time=datetime.utcnow(),
                campaign_status=state.status if state else CampaignStatus.PENDING,
                uptime_seconds=uptime,
            )

            event = create_telemetry_event(
                event_type=AegisTelemetryEventType.HEARTBEAT,
                campaign_id=self.campaign_id,
                sequence=0,  # Heartbeats don't increment sequence
                data=heartbeat_data,
            )

            await self.websocket.send_json(event.model_dump(mode="json"))
            logger.debug(f"Sent heartbeat to client {self.client_id}")

        except Exception as e:
            logger.exception(f"Error sending heartbeat: {e}")
            raise

    async def handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming client message."""
        message_type = message.get("type", "").lower()

        if message_type == "pong":
            # Client responded to heartbeat
            self.last_heartbeat_response = datetime.utcnow()
            logger.debug(f"Received pong from client {self.client_id}")

        elif message_type == "ping":
            # Client is checking if server is alive
            await self.websocket.send_json(
                {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                    "campaign_id": self.campaign_id,
                },
            )

        elif message_type == "get_summary":
            # Client requests current campaign summary
            await self._send_campaign_summary()

        elif message_type == "subscribe":
            # Already subscribed during connection
            logger.debug(f"Client {self.client_id} re-subscribe request ignored")

        elif message_type == "unsubscribe":
            # Client wants to disconnect
            await self.close(code=1000, reason="Client unsubscribed")

        else:
            logger.warning(f"Unknown message type from client: {message_type}")

    async def _send_campaign_summary(self) -> None:
        """Send current campaign summary to client."""
        try:
            state = aegis_telemetry_broadcaster.get_campaign_state(self.campaign_id)

            if state:
                summary = state.get_summary()
                await self.websocket.send_json(
                    {
                        "type": "campaign_summary",
                        "data": summary.model_dump(mode="json"),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            else:
                await self.websocket.send_json(
                    {
                        "type": "campaign_summary",
                        "data": None,
                        "error": "Campaign not found or not started",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )

        except Exception as e:
            logger.exception(f"Error sending campaign summary: {e}")


@router.websocket("/ws/aegis/telemetry/{campaign_id}")
async def aegis_telemetry_websocket(
    websocket: WebSocket,
    campaign_id: str,
    client_id: str | None = Query(
        default=None,
        description="Optional client identifier for reconnection support",
    ),
    api_key: str | None = Query(
        default=None,
        alias="api_key",
        description="API key for authentication (alternative to header)",
    ),
) -> None:
    """WebSocket endpoint for streaming Aegis campaign telemetry.

    Streams real-time telemetry events from an active Aegis campaign including:
    - Campaign lifecycle events (started, paused, resumed, completed, failed)
    - Iteration events (started, completed)
    - Attack events (started, completed with success/failure)
    - Technique performance metrics
    - Token usage and cost updates
    - Latency metrics
    - Prompt evolution tracking
    - Heartbeat for connection health

    **Connection:**
    Connect to: `ws://<host>/api/v1/ws/aegis/telemetry/{campaign_id}`

    **Query Parameters:**
    - `client_id`: Optional client identifier for reconnection support
    - `api_key`: API key for authentication (alternative to X-API-Key header)

    **Heartbeat Protocol:**
    - Server sends heartbeat every 30 seconds
    - Client should respond with `{"type": "pong"}`
    - Connection closed after 90 seconds without response

    **Client Messages:**
    - `{"type": "pong"}` - Heartbeat response
    - `{"type": "ping"}` - Client-initiated heartbeat check
    - `{"type": "get_summary"}` - Request current campaign summary
    - `{"type": "unsubscribe"}` - Disconnect from campaign

    **Server Events:**
    All events follow the AegisTelemetryEvent schema with:
    - `event_type`: Type of telemetry event
    - `campaign_id`: Campaign identifier
    - `timestamp`: Event timestamp (ISO 8601)
    - `sequence`: Event sequence number for ordering
    - `data`: Event-specific payload

    **Example Connection (JavaScript):**
    ```javascript
    const ws = new WebSocket(
        'ws://localhost:8001/api/v1/ws/aegis/telemetry/my-campaign-id'
    );

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.event_type === 'heartbeat') {
            ws.send(JSON.stringify({type: 'pong'}));
        } else {
            console.log('Telemetry event:', data);
        }
    };
    ```
    """
    connection = AegisTelemetryConnection(
        websocket=websocket,
        campaign_id=campaign_id,
        client_id=client_id,
    )

    # Accept connection
    if not await connection.accept():
        return

    # Start heartbeat
    await connection.start_heartbeat()

    try:
        # Main message loop
        while connection.is_active:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Update last response time (any message counts as alive)
                connection.last_heartbeat_response = datetime.utcnow()

                # Handle message
                await connection.handle_message(message)

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from client {connection.client_id}: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "error_code": "INVALID_JSON",
                        "error_message": "Invalid JSON message",
                    },
                )

    except WebSocketDisconnect:
        logger.info(
            f"Aegis telemetry client disconnected: "
            f"client={connection.client_id}, campaign={campaign_id}",
        )
    except asyncio.CancelledError:
        logger.debug(f"Connection cancelled for client {connection.client_id}")
    except Exception as e:
        logger.error(
            f"Error in Aegis telemetry websocket: {e}",
            exc_info=True,
        )
    finally:
        await connection.close()


# Additional utility endpoints for WebSocket management


@router.get("/ws/aegis/telemetry/{campaign_id}/info")
async def get_campaign_telemetry_info(campaign_id: str):
    """Get information about telemetry connections for a campaign.

    Returns:
    - Number of connected clients
    - Campaign status
    - Last event sequence number

    """
    state = aegis_telemetry_broadcaster.get_campaign_state(campaign_id)

    if not state:
        return {
            "campaign_id": campaign_id,
            "exists": False,
            "subscribers": 0,
            "status": None,
            "message": "Campaign telemetry not initialized",
        }

    return {
        "campaign_id": campaign_id,
        "exists": True,
        "subscribers": len(state.subscribed_clients),
        "status": state.status.value,
        "current_iteration": state.current_iteration,
        "event_sequence": state.event_sequence,
        "created_at": state.created_at.isoformat(),
        "last_updated": state.last_updated.isoformat(),
    }


@router.get("/ws/aegis/stats")
async def get_aegis_telemetry_stats():
    """Get overall statistics for Aegis telemetry broadcasting.

    Returns broadcaster statistics including:
    - Total campaigns
    - Total subscriptions
    - Active campaigns
    - Per-campaign metrics
    """
    return aegis_telemetry_broadcaster.get_statistics()
