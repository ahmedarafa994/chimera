"""
Service Registry - Central registry for all services in the Chimera ecosystem
Provides service registration, discovery, health tracking, and load balancing
"""

import asyncio
import contextlib
import hashlib
import json
import logging

# Helper: cryptographically secure pseudo-floats for security-sensitive choices
import secrets
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def _secure_random() -> float:
    """Cryptographically secure float in [0,1)."""
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Types of services that can be registered."""

    LLM_PROVIDER = "llm_provider"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    API_GATEWAY = "api_gateway"
    AUTHENTICATION = "authentication"
    STORAGE = "storage"
    TRANSFORMATION = "transformation"
    MONITORING = "monitoring"
    AGENT = "agent"
    EXTERNAL_API = "external_api"
    MICROSERVICE = "microservice"
    CLOUD_SERVICE = "cloud_service"


class ServiceStatus(Enum):
    """Status of a registered service."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for service selection."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    LATENCY_BASED = "latency_based"


@dataclass
class ServiceEndpoint:
    """Endpoint configuration for a service."""

    protocol: str  # http, https, grpc, ws, wss, amqp, redis
    host: str
    port: int
    path: str = ""

    @property
    def url(self) -> str:
        """Get the full URL for this endpoint."""
        base = f"{self.protocol}://{self.host}:{self.port}"
        if self.path:
            return f"{base}/{self.path.lstrip('/')}"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceEndpoint":
        return cls(
            protocol=data["protocol"],
            host=data["host"],
            port=data["port"],
            path=data.get("path", ""),
        )


@dataclass
class ServiceHealth:
    """Health information for a service."""

    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: datetime | None = None
    last_healthy: datetime | None = None
    consecutive_failures: int = 0
    latency_ms: float = 0.0
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_healthy": self.last_healthy.isoformat() if self.last_healthy else None,
            "consecutive_failures": self.consecutive_failures,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
            "details": self.details,
        }


@dataclass
class ServiceDefinition:
    """Complete definition of a service."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: ServiceType = ServiceType.MICROSERVICE
    version: str = "1.0.0"
    description: str = ""

    # Endpoints
    endpoints: list[ServiceEndpoint] = field(default_factory=list)

    # Health
    health: ServiceHealth = field(default_factory=ServiceHealth)
    health_check_url: str | None = None
    health_check_interval: int = 30  # seconds

    # Load balancing
    weight: int = 100
    max_connections: int = 100
    current_connections: int = 0

    # Metadata
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    capabilities: set[str] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)

    # Timestamps
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    # Authentication
    requires_auth: bool = False
    auth_type: str | None = None  # api_key, oauth, jwt, basic

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "version": self.version,
            "description": self.description,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "health": self.health.to_dict(),
            "health_check_url": self.health_check_url,
            "health_check_interval": self.health_check_interval,
            "weight": self.weight,
            "max_connections": self.max_connections,
            "current_connections": self.current_connections,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "registered_at": self.registered_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "requires_auth": self.requires_auth,
            "auth_type": self.auth_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceDefinition":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            type=ServiceType(data.get("type", "microservice")),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            endpoints=[ServiceEndpoint.from_dict(e) for e in data.get("endpoints", [])],
            health_check_url=data.get("health_check_url"),
            health_check_interval=data.get("health_check_interval", 30),
            weight=data.get("weight", 100),
            max_connections=data.get("max_connections", 100),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
            capabilities=set(data.get("capabilities", [])),
            dependencies=set(data.get("dependencies", [])),
            requires_auth=data.get("requires_auth", False),
            auth_type=data.get("auth_type"),
        )

    @property
    def primary_endpoint(self) -> ServiceEndpoint | None:
        """Get the primary endpoint for this service."""
        return self.endpoints[0] if self.endpoints else None

    def get_fingerprint(self) -> str:
        """Generate a unique fingerprint for this service configuration."""
        data = f"{self.name}:{self.version}:{[e.url for e in self.endpoints]}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class ServiceRegistryBackend(ABC):
    """Abstract backend for service registry storage."""

    @abstractmethod
    async def register(self, service: ServiceDefinition) -> bool:
        """Register a service."""
        pass

    @abstractmethod
    async def deregister(self, service_id: str) -> bool:
        """Deregister a service."""
        pass

    @abstractmethod
    async def get(self, service_id: str) -> ServiceDefinition | None:
        """Get a service by ID."""
        pass

    @abstractmethod
    async def get_all(self) -> list[ServiceDefinition]:
        """Get all registered services."""
        pass

    @abstractmethod
    async def update(self, service: ServiceDefinition) -> bool:
        """Update a service definition."""
        pass

    @abstractmethod
    async def find_by_type(self, service_type: ServiceType) -> list[ServiceDefinition]:
        """Find services by type."""
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> list[ServiceDefinition]:
        """Find services by name."""
        pass


class InMemoryRegistryBackend(ServiceRegistryBackend):
    """In-memory implementation of service registry backend."""

    def __init__(self):
        self._services: dict[str, ServiceDefinition] = {}
        self._lock = asyncio.Lock()

    async def register(self, service: ServiceDefinition) -> bool:
        async with self._lock:
            self._services[service.id] = service
            return True

    async def deregister(self, service_id: str) -> bool:
        async with self._lock:
            if service_id in self._services:
                del self._services[service_id]
                return True
            return False

    async def get(self, service_id: str) -> ServiceDefinition | None:
        return self._services.get(service_id)

    async def get_all(self) -> list[ServiceDefinition]:
        return list(self._services.values())

    async def update(self, service: ServiceDefinition) -> bool:
        async with self._lock:
            if service.id in self._services:
                service.last_updated = datetime.utcnow()
                self._services[service.id] = service
                return True
            return False

    async def find_by_type(self, service_type: ServiceType) -> list[ServiceDefinition]:
        return [s for s in self._services.values() if s.type == service_type]

    async def find_by_name(self, name: str) -> list[ServiceDefinition]:
        return [s for s in self._services.values() if s.name == name]


class RedisRegistryBackend(ServiceRegistryBackend):
    """Redis-backed implementation of service registry backend."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", prefix: str = "chimera:registry:"
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as redis

            self._redis = redis.from_url(self.redis_url)
        return self._redis

    async def register(self, service: ServiceDefinition) -> bool:
        try:
            redis = await self._get_redis()
            key = f"{self.prefix}service:{service.id}"
            await redis.set(key, json.dumps(service.to_dict()))

            # Add to type index
            type_key = f"{self.prefix}type:{service.type.value}"
            await redis.sadd(type_key, service.id)

            # Add to name index
            name_key = f"{self.prefix}name:{service.name}"
            await redis.sadd(name_key, service.id)

            return True
        except Exception as e:
            logger.error(f"Failed to register service in Redis: {e}")
            return False

    async def deregister(self, service_id: str) -> bool:
        try:
            redis = await self._get_redis()
            service = await self.get(service_id)
            if not service:
                return False

            key = f"{self.prefix}service:{service_id}"
            await redis.delete(key)

            # Remove from indices
            type_key = f"{self.prefix}type:{service.type.value}"
            await redis.srem(type_key, service_id)

            name_key = f"{self.prefix}name:{service.name}"
            await redis.srem(name_key, service_id)

            return True
        except Exception as e:
            logger.error(f"Failed to deregister service from Redis: {e}")
            return False

    async def get(self, service_id: str) -> ServiceDefinition | None:
        try:
            redis = await self._get_redis()
            key = f"{self.prefix}service:{service_id}"
            data = await redis.get(key)
            if data:
                return ServiceDefinition.from_dict(json.loads(data))
            return None
        except Exception as e:
            logger.error(f"Failed to get service from Redis: {e}")
            return None

    async def get_all(self) -> list[ServiceDefinition]:
        try:
            redis = await self._get_redis()
            keys = await redis.keys(f"{self.prefix}service:*")
            services = []
            for key in keys:
                data = await redis.get(key)
                if data:
                    services.append(ServiceDefinition.from_dict(json.loads(data)))
            return services
        except Exception as e:
            logger.error(f"Failed to get all services from Redis: {e}")
            return []

    async def update(self, service: ServiceDefinition) -> bool:
        service.last_updated = datetime.utcnow()
        return await self.register(service)

    async def find_by_type(self, service_type: ServiceType) -> list[ServiceDefinition]:
        try:
            redis = await self._get_redis()
            type_key = f"{self.prefix}type:{service_type.value}"
            service_ids = await redis.smembers(type_key)

            services = []
            for sid in service_ids:
                service = await self.get(sid.decode() if isinstance(sid, bytes) else sid)
                if service:
                    services.append(service)
            return services
        except Exception as e:
            logger.error(f"Failed to find services by type: {e}")
            return []

    async def find_by_name(self, name: str) -> list[ServiceDefinition]:
        try:
            redis = await self._get_redis()
            name_key = f"{self.prefix}name:{name}"
            service_ids = await redis.smembers(name_key)

            services = []
            for sid in service_ids:
                service = await self.get(sid.decode() if isinstance(sid, bytes) else sid)
                if service:
                    services.append(service)
            return services
        except Exception as e:
            logger.error(f"Failed to find services by name: {e}")
            return []


class ServiceRegistry:
    """
    Central service registry for the Chimera ecosystem.

    Features:
    - Service registration and deregistration
    - Health monitoring
    - Load balancing
    - Service discovery
    - Event notifications
    """

    def __init__(
        self,
        backend: ServiceRegistryBackend | None = None,
        health_check_enabled: bool = True,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ):
        self._backend = backend or InMemoryRegistryBackend()
        self._health_check_enabled = health_check_enabled
        self._load_balancing_strategy = load_balancing_strategy

        # Health checking
        self._health_check_task: asyncio.Task | None = None
        self._health_checkers: dict[str, Callable] = {}

        # Load balancing state
        self._round_robin_index: dict[str, int] = {}

        # Event callbacks
        self._on_register_callbacks: list[Callable] = []
        self._on_deregister_callbacks: list[Callable] = []
        self._on_health_change_callbacks: list[Callable] = []

        self._running = False

    async def start(self):
        """Start the service registry."""
        self._running = True
        if self._health_check_enabled:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Service registry started")

    async def stop(self):
        """Stop the service registry."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
        logger.info("Service registry stopped")

    async def register(
        self, name: str, service_type: ServiceType, endpoints: list[ServiceEndpoint], **kwargs
    ) -> ServiceDefinition:
        """
        Register a new service.

        Args:
            name: Service name
            service_type: Type of service
            endpoints: List of service endpoints
            **kwargs: Additional service configuration

        Returns:
            The registered ServiceDefinition
        """
        service = ServiceDefinition(name=name, type=service_type, endpoints=endpoints, **kwargs)

        await self._backend.register(service)

        # Notify callbacks
        for callback in self._on_register_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(service)
                else:
                    callback(service)
            except Exception as e:
                logger.error(f"Error in register callback: {e}")

        logger.info(f"Registered service: {name} ({service.id})")
        return service

    async def register_service(self, service: ServiceDefinition) -> bool:
        """Register an existing service definition."""
        success = await self._backend.register(service)
        if success:
            for callback in self._on_register_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(service)
                    else:
                        callback(service)
                except Exception as e:
                    logger.error(f"Error in register callback: {e}")
        return success

    async def deregister(self, service_id: str) -> bool:
        """Deregister a service."""
        service = await self._backend.get(service_id)
        if not service:
            return False

        success = await self._backend.deregister(service_id)

        if success:
            # Notify callbacks
            for callback in self._on_deregister_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(service)
                    else:
                        callback(service)
                except Exception as e:
                    logger.error(f"Error in deregister callback: {e}")

            logger.info(f"Deregistered service: {service.name} ({service_id})")

        return success

    async def get_service(self, service_id: str) -> ServiceDefinition | None:
        """Get a service by ID."""
        return await self._backend.get(service_id)

    async def get_all_services(self) -> list[ServiceDefinition]:
        """Get all registered services."""
        return await self._backend.get_all()

    async def find_services(
        self,
        service_type: ServiceType | None = None,
        name: str | None = None,
        tags: set[str] | None = None,
        capabilities: set[str] | None = None,
        healthy_only: bool = True,
    ) -> list[ServiceDefinition]:
        """
        Find services matching criteria.

        Args:
            service_type: Filter by service type
            name: Filter by name
            tags: Filter by tags (must have all)
            capabilities: Filter by capabilities (must have all)
            healthy_only: Only return healthy services

        Returns:
            List of matching services
        """
        if service_type:
            services = await self._backend.find_by_type(service_type)
        elif name:
            services = await self._backend.find_by_name(name)
        else:
            services = await self._backend.get_all()

        # Apply filters
        if tags:
            services = [s for s in services if tags.issubset(s.tags)]

        if capabilities:
            services = [s for s in services if capabilities.issubset(s.capabilities)]

        if healthy_only:
            services = [s for s in services if s.health.status == ServiceStatus.HEALTHY]

        return services

    async def select_service(
        self, service_type: ServiceType | None = None, name: str | None = None, **kwargs
    ) -> ServiceDefinition | None:
        """
        Select a single service using load balancing.

        Args:
            service_type: Filter by service type
            name: Filter by name
            **kwargs: Additional filter criteria

        Returns:
            Selected service or None
        """
        services = await self.find_services(service_type=service_type, name=name, **kwargs)

        if not services:
            return None

        return self._select_with_load_balancing(services, name or str(service_type))

    def _select_with_load_balancing(
        self, services: list[ServiceDefinition], key: str
    ) -> ServiceDefinition:
        """Select a service using the configured load balancing strategy."""
        if self._load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(services, key)
        elif self._load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(services)
        elif self._load_balancing_strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted_select(services)
        elif self._load_balancing_strategy == LoadBalancingStrategy.LATENCY_BASED:
            return self._latency_based_select(services)
        else:
            import secrets

            return secrets.choice(services)

    def _round_robin_select(self, services: list[ServiceDefinition], key: str) -> ServiceDefinition:
        """Round-robin selection."""
        if key not in self._round_robin_index:
            self._round_robin_index[key] = 0

        index = self._round_robin_index[key] % len(services)
        self._round_robin_index[key] = index + 1

        return services[index]

    def _least_connections_select(self, services: list[ServiceDefinition]) -> ServiceDefinition:
        """Select service with least connections."""
        return min(services, key=lambda s: s.current_connections)

    def _weighted_select(self, services: list[ServiceDefinition]) -> ServiceDefinition:
        """Weighted random selection."""

        total_weight = sum(s.weight for s in services)
        r = _secure_uniform(0, total_weight)

        cumulative = 0
        for service in services:
            cumulative += service.weight
            if r <= cumulative:
                return service

        return services[-1]

    def _latency_based_select(self, services: list[ServiceDefinition]) -> ServiceDefinition:
        """Select service with lowest latency."""
        return min(services, key=lambda s: s.health.latency_ms)

    async def update_health(
        self,
        service_id: str,
        status: ServiceStatus,
        latency_ms: float = 0.0,
        error_message: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Update the health status of a service."""
        service = await self._backend.get(service_id)
        if not service:
            return

        old_status = service.health.status

        service.health.status = status
        service.health.last_check = datetime.utcnow()
        service.health.latency_ms = latency_ms
        service.health.error_message = error_message

        if details:
            service.health.details = details

        if status == ServiceStatus.HEALTHY:
            service.health.last_healthy = datetime.utcnow()
            service.health.consecutive_failures = 0
        else:
            service.health.consecutive_failures += 1

        await self._backend.update(service)

        # Notify on status change
        if old_status != status:
            for callback in self._on_health_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(service, old_status, status)
                    else:
                        callback(service, old_status, status)
                except Exception as e:
                    logger.error(f"Error in health change callback: {e}")

    def register_health_checker(
        self, service_type: ServiceType, checker: Callable[[ServiceDefinition], bool]
    ):
        """Register a custom health checker for a service type."""
        self._health_checkers[service_type.value] = checker

    async def _health_check_loop(self):
        """Background task for health checking."""
        while self._running:
            try:
                services = await self._backend.get_all()

                for service in services:
                    if service.health_check_url:
                        await self._check_service_health(service)

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)

    async def _check_service_health(self, service: ServiceDefinition):
        """Check the health of a single service."""
        import aiohttp

        start_time = asyncio.get_event_loop().time()

        try:
            # Use custom checker if available
            if service.type.value in self._health_checkers:
                checker = self._health_checkers[service.type.value]
                if asyncio.iscoroutinefunction(checker):
                    healthy = await checker(service)
                else:
                    healthy = checker(service)

                latency = (asyncio.get_event_loop().time() - start_time) * 1000

                await self.update_health(
                    service.id,
                    ServiceStatus.HEALTHY if healthy else ServiceStatus.UNHEALTHY,
                    latency_ms=latency,
                )
                return

            # Default HTTP health check
            if service.health_check_url:
                async with aiohttp.ClientSession() as session, session.get(
                    service.health_check_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency = (asyncio.get_event_loop().time() - start_time) * 1000

                    if response.status == 200:
                        await self.update_health(
                            service.id, ServiceStatus.HEALTHY, latency_ms=latency
                        )
                    else:
                        await self.update_health(
                            service.id,
                            ServiceStatus.UNHEALTHY,
                            latency_ms=latency,
                            error_message=f"HTTP {response.status}",
                        )

        except TimeoutError:
            await self.update_health(
                service.id, ServiceStatus.UNHEALTHY, error_message="Health check timeout"
            )
        except Exception as e:
            await self.update_health(service.id, ServiceStatus.UNHEALTHY, error_message=str(e))

    def on_register(self, callback: Callable):
        """Register a callback for service registration events."""
        self._on_register_callbacks.append(callback)

    def on_deregister(self, callback: Callable):
        """Register a callback for service deregistration events."""
        self._on_deregister_callbacks.append(callback)

    def on_health_change(self, callback: Callable):
        """Register a callback for health status change events."""
        self._on_health_change_callbacks.append(callback)

    async def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics."""
        services = await self._backend.get_all()

        by_type = {}
        by_status = {}

        for service in services:
            # Count by type
            type_name = service.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # Count by status
            status_name = service.health.status.value
            by_status[status_name] = by_status.get(status_name, 0) + 1

        return {
            "total_services": len(services),
            "by_type": by_type,
            "by_status": by_status,
            "healthy_count": by_status.get("healthy", 0),
            "unhealthy_count": by_status.get("unhealthy", 0),
        }


def create_service_registry(
    backend_type: str = "memory", redis_url: str | None = None, **kwargs
) -> ServiceRegistry:
    """
    Factory function to create a service registry.

    Args:
        backend_type: "memory" or "redis"
        redis_url: Redis URL for redis backend
        **kwargs: Additional registry configuration

    Returns:
        ServiceRegistry instance
    """
    if backend_type == "redis" and redis_url:
        backend = RedisRegistryBackend(redis_url=redis_url)
    else:
        backend = InMemoryRegistryBackend()

    return ServiceRegistry(backend=backend, **kwargs)
