"""
WebSocket Server for Model Selection

Provides real-time updates for model selection changes via WebSocket.

Features:
- Broadcasts SELECTION_CHANGED events when user changes provider/model
- Sends PROVIDER_STATUS updates for health changes
- Handles connection management
- Integrates with GlobalModelSelectionState subscription system

Message Format:
{
    "type": "SELECTION_CHANGED",
    "data": {
        "user_id": "user123",
        "provider": "openai",
        "model": "gpt-4o",
        "timestamp": "2026-01-07T12:30:00Z"
    }
}
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.services.global_model_selection_state import (
    get_global_model_selection_state,
    Selection,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Message Models
# ============================================================================


class WebSocketMessage(BaseModel):
    """Base WebSocket message"""
    type: str
    data: dict


class SelectionChangedData(BaseModel):
    """Data for SELECTION_CHANGED event"""
    user_id: str
    provider: str
    model: str
    timestamp: str


class ProviderStatusData(BaseModel):
    """Data for PROVIDER_STATUS event"""
    provider_id: str
    is_available: bool
    health_score: float


class ModelValidationData(BaseModel):
    """Data for MODEL_VALIDATION event"""
    provider: str
    model: str
    is_valid: bool
    error_message: str | None = None


# ============================================================================
# Connection Manager
# ============================================================================


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts.

    Allows multiple clients to connect and receive real-time updates
    about model selection changes.
    """

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._subscription_ids: Dict[str, str] = {}  # WebSocket -> subscription_id

    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Register a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            user_id: User identifier
        """
        await websocket.accept()

        async with self._lock:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = set()

            self.active_connections[user_id].add(websocket)

        # Subscribe to selection changes for this user
        state = get_global_model_selection_state()
        subscription_id = state.subscribe_to_changes(
            user_id,
            lambda selection: asyncio.create_task(
                self._broadcast_selection_change(user_id, selection)
            )
        )

        self._subscription_ids[id(websocket)] = subscription_id

        logger.info(f"WebSocket connected for user {user_id}, total connections: {len(self.active_connections.get(user_id, []))}")

    async def disconnect(self, websocket: WebSocket, user_id: str):
        """
        Unregister a WebSocket connection.

        Args:
            websocket: WebSocket connection
            user_id: User identifier
        """
        async with self._lock:
            if user_id in self.active_connections:
                self.active_connections[user_id].discard(websocket)

                # Remove user entry if no more connections
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]

        # Unsubscribe from selection changes
        ws_id = id(websocket)
        if ws_id in self._subscription_ids:
            state = get_global_model_selection_state()
            state.unsubscribe(self._subscription_ids[ws_id])
            del self._subscription_ids[ws_id]

        logger.info(f"WebSocket disconnected for user {user_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: Message dictionary
            websocket: Target WebSocket
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def broadcast_to_user(self, message: dict, user_id: str):
        """
        Broadcast a message to all connections for a specific user.

        Args:
            message: Message dictionary
            user_id: Target user
        """
        if user_id not in self.active_connections:
            return

        # Create copy of connections to avoid modification during iteration
        connections = list(self.active_connections[user_id])

        for connection in connections:
            try:
                await self.send_personal_message(message, connection)
            except Exception as e:
                logger.error(f"Failed to broadcast to user {user_id}: {e}")
                # Connection is dead, remove it
                await self.disconnect(connection, user_id)

    async def _broadcast_selection_change(self, user_id: str, selection: Selection):
        """
        Broadcast selection change event to all user connections.

        This is called automatically by the subscription system when
        the user's selection changes.

        Args:
            user_id: User whose selection changed
            selection: New selection
        """
        message = {
            "type": "SELECTION_CHANGED",
            "data": {
                "user_id": selection.user_id,
                "provider": selection.provider_id,
                "model": selection.model_id,
                "timestamp": selection.updated_at.isoformat()
            }
        }

        await self.broadcast_to_user(message, user_id)
        logger.debug(f"Broadcasted selection change to user {user_id}")

    async def broadcast_provider_status(self, provider_id: str, is_available: bool, health_score: float):
        """
        Broadcast provider status update to all connected clients.

        Args:
            provider_id: Provider ID
            is_available: Availability status
            health_score: Health score (0.0 to 1.0)
        """
        message = {
            "type": "PROVIDER_STATUS",
            "data": {
                "provider_id": provider_id,
                "is_available": is_available,
                "health_score": health_score,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        # Broadcast to all users
        for user_id in list(self.active_connections.keys()):
            await self.broadcast_to_user(message, user_id)

    async def send_validation_result(
        self,
        websocket: WebSocket,
        provider: str,
        model: str,
        is_valid: bool,
        error_message: str | None = None
    ):
        """
        Send validation result to a specific connection.

        Args:
            websocket: Target WebSocket
            provider: Provider ID
            model: Model ID
            is_valid: Validation result
            error_message: Error message if invalid
        """
        message = {
            "type": "MODEL_VALIDATION",
            "data": {
                "provider": provider,
                "model": model,
                "is_valid": is_valid,
                "error_message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        await self.send_personal_message(message, websocket)


# Global connection manager instance
connection_manager = ConnectionManager()


# ============================================================================
# WebSocket Endpoint
# ============================================================================


async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time model selection updates.

    Args:
        websocket: WebSocket connection
        user_id: User identifier (from query param or auth)

    Message types handled:
    - PING: Keep-alive ping (responds with PONG)
    - VALIDATE: Validate provider/model selection

    Message types sent:
    - SELECTION_CHANGED: User's selection changed
    - PROVIDER_STATUS: Provider availability/health changed
    - MODEL_VALIDATION: Validation result
    - PONG: Response to PING
    """
    await connection_manager.connect(websocket, user_id)

    try:
        # Send initial selection
        state = get_global_model_selection_state()
        selection = await state.get_selection(user_id)

        await connection_manager.send_personal_message(
            {
                "type": "SELECTION_CHANGED",
                "data": {
                    "user_id": selection.user_id,
                    "provider": selection.provider_id,
                    "model": selection.model_id,
                    "timestamp": selection.updated_at.isoformat()
                }
            },
            websocket
        )

        # Listen for client messages
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "PING":
                    # Respond to ping
                    await connection_manager.send_personal_message(
                        {"type": "PONG", "data": {}},
                        websocket
                    )

                elif message_type == "VALIDATE":
                    # Validate provider/model selection
                    msg_data = message.get("data", {})
                    provider = msg_data.get("provider")
                    model = msg_data.get("model")

                    if provider and model:
                        is_valid = await state.validate_selection(provider, model)

                        await connection_manager.send_validation_result(
                            websocket,
                            provider,
                            model,
                            is_valid,
                            None if is_valid else f"Model '{model}' not valid for provider '{provider}'"
                        )

                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}", exc_info=True)
    finally:
        await connection_manager.disconnect(websocket, user_id)


# Convenience function to get the connection manager
def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return connection_manager
