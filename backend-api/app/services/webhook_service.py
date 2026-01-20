"""Webhook Service - External integration via HTTP callbacks.

Provides:
- Webhook registration and management
- Secure delivery with HMAC signatures
- Retry with exponential backoff
- Delivery status tracking
- Event filtering

Usage:
    from app.services.webhook_service import webhook_service, WebhookConfig

    # Register a webhook
    webhook_id = webhook_service.register(WebhookConfig(
        url="https://example.com/webhook",
        secret="my-secret-key",
        events=[WebhookEventType.GENERATION_COMPLETED],
    ))

    # Dispatch an event
    await webhook_service.dispatch(
        WebhookEventType.GENERATION_COMPLETED,
        {"prompt_id": "abc123", "result": "..."}
    )
"""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Try to import httpx, fall back gracefully
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available, webhook delivery will be disabled")


class WebhookEventType(str, Enum):
    """Event types that can trigger webhooks."""

    # Generation events
    GENERATION_STARTED = "generation.started"
    GENERATION_COMPLETED = "generation.completed"
    GENERATION_FAILED = "generation.failed"

    # Transformation events
    TRANSFORMATION_STARTED = "transformation.started"
    TRANSFORMATION_COMPLETED = "transformation.completed"
    TRANSFORMATION_FAILED = "transformation.failed"

    # Session events
    SESSION_CREATED = "session.created"
    SESSION_EXPIRED = "session.expired"

    # Error events
    ERROR_OCCURRED = "error.occurred"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"

    # Health events
    HEALTH_STATUS_CHANGED = "health.status_changed"


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookConfig:
    """Webhook configuration.

    Attributes:
        url: Endpoint URL to deliver webhooks to
        secret: Secret key for HMAC signature
        events: List of event types to subscribe to
        webhook_id: Unique identifier (auto-generated)
        active: Whether webhook is active
        created_at: ISO timestamp of creation
        description: Optional description
        headers: Additional headers to include

    """

    url: str
    secret: str
    events: list[WebhookEventType]
    webhook_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    description: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes secret)."""
        return {
            "webhook_id": self.webhook_id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "active": self.active,
            "created_at": self.created_at,
            "description": self.description,
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt.

    Attributes:
        delivery_id: Unique delivery identifier
        webhook_id: Associated webhook ID
        event_type: Type of event
        payload: Event payload
        status: Delivery status
        attempts: Number of delivery attempts
        last_attempt: ISO timestamp of last attempt
        next_attempt: ISO timestamp of next retry (if scheduled)
        response_code: HTTP response code (if received)
        response_body: HTTP response body (if received)
        error: Error message (if failed)

    """

    webhook_id: str
    event_type: WebhookEventType
    payload: dict[str, Any]
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_attempt: str | None = None
    next_attempt: str | None = None
    response_code: int | None = None
    response_body: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "delivery_id": self.delivery_id,
            "webhook_id": self.webhook_id,
            "event_type": self.event_type.value,
            "status": self.status.value,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "last_attempt": self.last_attempt,
            "response_code": self.response_code,
            "error": self.error,
        }


@dataclass
class WebhookServiceStats:
    """Service statistics."""

    total_webhooks: int = 0
    active_webhooks: int = 0
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    pending_deliveries: int = 0


class WebhookService:
    """Webhook registry and delivery service.

    Features:
    - Webhook registration and management
    - Secure HMAC-SHA256 signatures
    - Exponential backoff retry
    - Delivery tracking and history
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delays: list[float] | None = None,
        timeout: float | None = None,  # No timeout by default
        max_deliveries_history: int = 1000,
    ) -> None:
        self._webhooks: dict[str, WebhookConfig] = {}
        self._deliveries: dict[str, WebhookDelivery] = {}
        self._max_retries = max_retries
        self._retry_delays = retry_delays or [10, 60, 300]  # seconds
        self._timeout = timeout
        self._max_deliveries = max_deliveries_history
        self._stats = WebhookServiceStats()

    def register(self, config: WebhookConfig) -> str:
        """Register a new webhook.

        Args:
            config: Webhook configuration

        Returns:
            Webhook ID

        """
        self._webhooks[config.webhook_id] = config
        self._stats.total_webhooks += 1
        if config.active:
            self._stats.active_webhooks += 1

        logger.info(
            f"Registered webhook {config.webhook_id} for events: {[e.value for e in config.events]}",
        )
        return config.webhook_id

    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook.

        Returns True if webhook was found and removed.
        """
        webhook = self._webhooks.pop(webhook_id, None)
        if webhook:
            if webhook.active:
                self._stats.active_webhooks -= 1
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False

    def get_webhook(self, webhook_id: str) -> WebhookConfig | None:
        """Get webhook by ID."""
        return self._webhooks.get(webhook_id)

    def list_webhooks(self, event_type: WebhookEventType | None = None) -> list[WebhookConfig]:
        """List all webhooks, optionally filtered by event type."""
        webhooks = list(self._webhooks.values())
        if event_type:
            webhooks = [w for w in webhooks if event_type in w.events]
        return webhooks

    def update_webhook(
        self,
        webhook_id: str,
        active: bool | None = None,
        events: list[WebhookEventType] | None = None,
        description: str | None = None,
    ) -> bool:
        """Update webhook configuration.

        Returns True if webhook was found and updated.
        """
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return False

        if active is not None:
            was_active = webhook.active
            webhook.active = active
            if active and not was_active:
                self._stats.active_webhooks += 1
            elif not active and was_active:
                self._stats.active_webhooks -= 1

        if events is not None:
            webhook.events = events

        if description is not None:
            webhook.description = description

        logger.info(f"Updated webhook {webhook_id}")
        return True

    async def dispatch(
        self,
        event_type: WebhookEventType,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> list[str]:
        """Dispatch an event to all subscribed webhooks.

        Args:
            event_type: Type of event
            payload: Event payload
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of delivery IDs

        """
        delivery_ids = []

        for webhook in self._webhooks.values():
            if not webhook.active:
                continue
            if event_type not in webhook.events:
                continue

            delivery = WebhookDelivery(
                webhook_id=webhook.webhook_id,
                event_type=event_type,
                payload={
                    **payload,
                    "correlation_id": correlation_id,
                    "event_type": event_type.value,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
            )

            self._deliveries[delivery.delivery_id] = delivery
            self._stats.total_deliveries += 1
            self._stats.pending_deliveries += 1
            delivery_ids.append(delivery.delivery_id)

            # Fire and forget delivery
            asyncio.create_task(self._deliver(webhook, delivery))

        return delivery_ids

    async def _deliver(
        self,
        webhook: WebhookConfig,
        delivery: WebhookDelivery,
    ) -> None:
        """Deliver a webhook with retry logic."""
        if not HTTPX_AVAILABLE:
            delivery.status = DeliveryStatus.FAILED
            delivery.error = "httpx not available"
            self._stats.pending_deliveries -= 1
            self._stats.failed_deliveries += 1
            return

        delivery.status = DeliveryStatus.IN_PROGRESS

        payload_json = json.dumps(delivery.payload, default=str)
        signature = self._create_signature(webhook.secret, payload_json)

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": f"sha256={signature}",
            "X-Webhook-ID": webhook.webhook_id,
            "X-Delivery-ID": delivery.delivery_id,
            "X-Event-Type": delivery.event_type.value,
            **webhook.headers,
        }

        async with httpx.AsyncClient() as client:
            for attempt in range(self._max_retries + 1):
                delivery.attempts = attempt + 1
                delivery.last_attempt = datetime.utcnow().isoformat() + "Z"

                try:
                    response = await client.post(
                        webhook.url,
                        content=payload_json,
                        headers=headers,
                        timeout=self._timeout,
                    )

                    delivery.response_code = response.status_code
                    delivery.response_body = response.text[:1000]  # Limit stored response

                    if response.is_success:
                        delivery.status = DeliveryStatus.DELIVERED
                        self._stats.pending_deliveries -= 1
                        self._stats.successful_deliveries += 1
                        logger.debug(f"Webhook delivered: {delivery.delivery_id} to {webhook.url}")
                        self._cleanup_deliveries()
                        return

                    # Non-success response
                    delivery.error = f"HTTP {response.status_code}"

                except TimeoutError:
                    delivery.error = "Request timed out"
                except Exception as e:
                    delivery.error = str(e)
                    logger.warning(f"Webhook delivery failed: {delivery.delivery_id}: {e}")

                # Check if we should retry
                if attempt < self._max_retries:
                    delivery.status = DeliveryStatus.RETRYING
                    delay = self._retry_delays[min(attempt, len(self._retry_delays) - 1)]
                    delivery.next_attempt = datetime.utcnow().isoformat() + "Z"
                    logger.info(
                        f"Retrying webhook {delivery.delivery_id} in {delay}s "
                        f"(attempt {attempt + 1}/{self._max_retries + 1})",
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        delivery.status = DeliveryStatus.FAILED
        self._stats.pending_deliveries -= 1
        self._stats.failed_deliveries += 1
        logger.warning(f"Webhook delivery failed permanently: {delivery.delivery_id}")
        self._cleanup_deliveries()

    def _create_signature(self, secret: str, payload: str) -> str:
        """Create HMAC-SHA256 signature for payload."""
        return hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def verify_signature(
        self,
        secret: str,
        payload: str,
        signature: str,
    ) -> bool:
        """Verify a webhook signature.

        Useful for receiving webhooks from other services.
        """
        expected = self._create_signature(secret, payload)
        # Remove "sha256=" prefix if present
        signature = signature.removeprefix("sha256=")
        return hmac.compare_digest(expected, signature)

    def get_delivery(self, delivery_id: str) -> WebhookDelivery | None:
        """Get delivery by ID."""
        return self._deliveries.get(delivery_id)

    def get_deliveries(
        self,
        webhook_id: str | None = None,
        status: DeliveryStatus | None = None,
        limit: int = 50,
    ) -> list[WebhookDelivery]:
        """Get delivery history with optional filtering."""
        deliveries = list(self._deliveries.values())

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        if status:
            deliveries = [d for d in deliveries if d.status == status]

        return deliveries[-limit:]

    async def retry_delivery(self, delivery_id: str) -> bool:
        """Manually retry a failed delivery.

        Returns True if retry was initiated.
        """
        delivery = self._deliveries.get(delivery_id)
        if not delivery or delivery.status != DeliveryStatus.FAILED:
            return False

        webhook = self._webhooks.get(delivery.webhook_id)
        if not webhook or not webhook.active:
            return False

        # Reset delivery status
        delivery.status = DeliveryStatus.PENDING
        delivery.attempts = 0
        delivery.error = None
        self._stats.pending_deliveries += 1
        self._stats.failed_deliveries -= 1

        asyncio.create_task(self._deliver(webhook, delivery))
        return True

    def _cleanup_deliveries(self) -> None:
        """Clean up old deliveries to limit memory usage."""
        if len(self._deliveries) <= self._max_deliveries:
            return

        # Keep only recent deliveries
        deliveries = sorted(
            self._deliveries.values(),
            key=lambda d: d.created_at,
            reverse=True,
        )

        self._deliveries = {d.delivery_id: d for d in deliveries[: self._max_deliveries]}

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "total_webhooks": self._stats.total_webhooks,
            "active_webhooks": self._stats.active_webhooks,
            "total_deliveries": self._stats.total_deliveries,
            "successful_deliveries": self._stats.successful_deliveries,
            "failed_deliveries": self._stats.failed_deliveries,
            "pending_deliveries": self._stats.pending_deliveries,
            "delivery_history_size": len(self._deliveries),
        }


# Global webhook service instance
webhook_service = WebhookService()


def get_webhook_service() -> WebhookService:
    """Get the global webhook service instance."""
    return webhook_service
