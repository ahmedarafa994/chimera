"""WebSocket Connection Manager.

Manages WebSocket connections, session subscriptions, and event broadcasting.
Handles connection lifecycle, heartbeat, and client state management.
"""

import asyncio
import contextlib
import logging
from collections import defaultdict
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from .models import (
    ConnectionAckEvent,
    ConnectionState,
    EventType,
    HeartbeatEvent,
    SessionState,
    WebSocketEvent,
    create_websocket_event,
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and event broadcasting."""

    def __init__(self) -> None:
        # Active connections: client_id -> WebSocket
        self.active_connections: dict[str, WebSocket] = {}

        # Connection state: client_id -> ConnectionState
        self.connection_states: dict[str, ConnectionState] = {}

        # Session subscriptions: session_id -> Set[client_id]
        self.session_subscriptions: dict[str, set[str]] = defaultdict(set)

        # Session state: session_id -> SessionState
        self.session_states: dict[str, SessionState] = {}

        # Heartbeat task
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_interval = 30  # seconds
        self._heartbeat_timeout = 90  # seconds (3 missed heartbeats)

    async def start(self) -> None:
        """Start the connection manager and background tasks."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("WebSocket connection manager started")

    async def stop(self) -> None:
        """Stop the connection manager and close all connections."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        # Close all active connections
        for client_id in list(self.active_connections.keys()):
            await self.disconnect(client_id)

        logger.info("WebSocket connection manager stopped")

    async def connect(self, websocket: WebSocket, client_id: str, session_id: str) -> bool:
        """Accept a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier
            session_id: Session to subscribe to

        Returns:
            True if connection successful, False otherwise

        """
        try:
            # Accept the WebSocket connection
            await websocket.accept()

            # Store connection
            self.active_connections[client_id] = websocket

            # Initialize connection state
            self.connection_states[client_id] = ConnectionState(
                client_id=client_id,
                session_id=session_id,
            )

            # Subscribe to session
            await self.subscribe_to_session(client_id, session_id)

            # Send connection acknowledgment
            ack_event = ConnectionAckEvent(
                session_id=session_id,
                client_id=client_id,
                supported_events=[event.value for event in EventType],
            )

            await self._send_to_client(
                client_id,
                create_websocket_event(
                    event_type=EventType.CONNECTION_ACK,
                    session_id=session_id,
                    sequence=0,
                    data=ack_event,
                ),
            )

            logger.info(f"Client {client_id} connected to session {session_id}")
            return True

        except Exception as e:
            logger.exception(f"Failed to connect client {client_id}: {e}")
            return False

    async def disconnect(self, client_id: str) -> None:
        """Disconnect a client.

        Args:
            client_id: Client identifier

        """
        try:
            # Get connection state
            state = self.connection_states.get(client_id)
            if state:
                # Unsubscribe from session
                await self.unsubscribe_from_session(client_id, state.session_id)

            # Close WebSocket connection
            websocket = self.active_connections.get(client_id)
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket for {client_id}: {e}")

            # Clean up state
            self.active_connections.pop(client_id, None)
            self.connection_states.pop(client_id, None)

            logger.info(f"Client {client_id} disconnected")

        except Exception as e:
            logger.exception(f"Error disconnecting client {client_id}: {e}")

    async def subscribe_to_session(self, client_id: str, session_id: str) -> None:
        """Subscribe a client to session events.

        Args:
            client_id: Client identifier
            session_id: Session identifier

        """
        self.session_subscriptions[session_id].add(client_id)

        # Initialize session state if not exists
        if session_id not in self.session_states:
            from .models import SessionStatus

            self.session_states[session_id] = SessionState(
                session_id=session_id,
                status=SessionStatus.PENDING,
            )

        # Add client to session's connected clients
        session_state = self.session_states[session_id]
        if client_id not in session_state.connected_clients:
            session_state.connected_clients.append(client_id)

        logger.info(f"Client {client_id} subscribed to session {session_id}")

    async def unsubscribe_from_session(self, client_id: str, session_id: str) -> None:
        """Unsubscribe a client from session events.

        Args:
            client_id: Client identifier
            session_id: Session identifier

        """
        self.session_subscriptions[session_id].discard(client_id)

        # Remove from session state
        if session_id in self.session_states:
            session_state = self.session_states[session_id]
            if client_id in session_state.connected_clients:
                session_state.connected_clients.remove(client_id)

        logger.info(f"Client {client_id} unsubscribed from session {session_id}")

    async def broadcast_to_session(
        self,
        session_id: str,
        event: WebSocketEvent,
        exclude_client: str | None = None,
    ) -> None:
        """Broadcast an event to all clients subscribed to a session.

        Args:
            session_id: Session identifier
            event: Event to broadcast
            exclude_client: Optional client ID to exclude from broadcast

        """
        client_ids = self.session_subscriptions.get(session_id, set())

        if not client_ids:
            logger.debug(f"No clients subscribed to session {session_id}")
            return

        # Send to all subscribed clients
        send_tasks = []
        for client_id in client_ids:
            if client_id != exclude_client:
                send_tasks.append(self._send_to_client(client_id, event))

        # Execute sends concurrently
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Log any errors
            for client_id, result in zip(client_ids, results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Failed to send event to client {client_id}: {result}")

    async def send_to_client(self, client_id: str, event: WebSocketEvent) -> None:
        """Send an event to a specific client.

        Args:
            client_id: Client identifier
            event: Event to send

        """
        await self._send_to_client(client_id, event)

    async def _send_to_client(self, client_id: str, event: WebSocketEvent) -> None:
        """Internal method to send event to client with error handling.

        Args:
            client_id: Client identifier
            event: Event to send

        """
        websocket = self.active_connections.get(client_id)
        if not websocket:
            logger.warning(f"Client {client_id} not found in active connections")
            return

        try:
            # Update connection state
            state = self.connection_states.get(client_id)
            if state:
                state.last_event_sequence = event.sequence

            # Send event as JSON
            await websocket.send_json(event.dict())

            logger.debug(
                f"Sent {event.event_type} to client {client_id} (seq: {event.sequence})",
            )

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected during send")
            await self.disconnect(client_id)

        except Exception as e:
            logger.exception(f"Error sending to client {client_id}: {e}")
            await self.disconnect(client_id)

    async def update_heartbeat(self, client_id: str) -> None:
        """Update last heartbeat timestamp for a client.

        Args:
            client_id: Client identifier

        """
        state = self.connection_states.get(client_id)
        if state:
            state.last_heartbeat = datetime.utcnow()
            state.is_alive = True

    async def get_session_state(self, session_id: str) -> SessionState | None:
        """Get session state.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if exists, None otherwise

        """
        return self.session_states.get(session_id)

    async def update_session_state(self, session_id: str, **updates) -> None:
        """Update session state fields.

        Args:
            session_id: Session identifier
            **updates: Field updates

        """
        state = self.session_states.get(session_id)
        if state:
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            state.last_updated = datetime.utcnow()

    def get_next_sequence(self, session_id: str) -> int:
        """Get next event sequence number for a session.

        Args:
            session_id: Session identifier

        Returns:
            Next sequence number

        """
        state = self.session_states.get(session_id)
        if state:
            return state.increment_sequence()
        return 0

    async def _heartbeat_loop(self) -> None:
        """Background task to send heartbeats and detect stale connections."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)

                current_time = datetime.utcnow()
                stale_clients = []

                # Check all connections
                for client_id, state in list(self.connection_states.items()):
                    # Check if connection is stale
                    time_since_heartbeat = (current_time - state.last_heartbeat).total_seconds()

                    if time_since_heartbeat > self._heartbeat_timeout:
                        logger.warning(
                            f"Client {client_id} heartbeat timeout ({time_since_heartbeat}s)",
                        )
                        stale_clients.append(client_id)
                        continue

                    # Send heartbeat
                    try:
                        heartbeat = HeartbeatEvent(
                            uptime=int((current_time - state.connected_at).total_seconds()),
                        )

                        await self._send_to_client(
                            client_id,
                            create_websocket_event(
                                event_type=EventType.HEARTBEAT,
                                session_id=state.session_id,
                                sequence=0,  # Heartbeats don't increment sequence
                                data=heartbeat,
                            ),
                        )
                    except Exception as e:
                        logger.exception(f"Error sending heartbeat to {client_id}: {e}")
                        stale_clients.append(client_id)

                # Disconnect stale clients
                for client_id in stale_clients:
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in heartbeat loop: {e}")

    def get_statistics(self) -> dict:
        """Get connection manager statistics."""
        return {
            "total_connections": len(self.active_connections),
            "total_sessions": len(self.session_states),
            "total_subscriptions": sum(
                len(clients) for clients in self.session_subscriptions.values()
            ),
            "active_sessions": sum(
                1
                for state in self.session_states.values()
                if state.status.value in ["running", "initializing"]
            ),
        }


# Global connection manager instance
connection_manager = ConnectionManager()
