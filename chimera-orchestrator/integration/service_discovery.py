"""
Service Discovery - Automatic service discovery mechanisms
Supports multiple discovery backends: DNS, Consul, Kubernetes, static configuration
"""

import asyncio
import contextlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from .service_registry import ServiceDefinition, ServiceEndpoint, ServiceRegistry, ServiceType

logger = logging.getLogger(__name__)


class DiscoveryMethod(Enum):
    """Methods for discovering services."""

    STATIC = "static"
    DNS = "dns"
    CONSUL = "consul"
    KUBERNETES = "kubernetes"
    ETCD = "etcd"
    ENVIRONMENT = "environment"
    FILE = "file"


@dataclass
class DiscoveryConfig:
    """Configuration for service discovery."""

    method: DiscoveryMethod = DiscoveryMethod.STATIC
    refresh_interval: int = 30
    static_services: list[dict[str, Any]] = field(default_factory=list)
    dns_domain: str = ""
    consul_host: str = "localhost"
    consul_port: int = 8500
    consul_token: str | None = None
    kubernetes_namespace: str = "default"
    kubernetes_label_selector: str = ""
    etcd_hosts: list[str] = field(default_factory=lambda: ["localhost:2379"])
    etcd_prefix: str = "/chimera/services/"
    env_prefix: str = "CHIMERA_SERVICE_"
    config_file: str = ""


class DiscoveryBackend(ABC):
    """Abstract backend for service discovery."""

    @abstractmethod
    async def discover(self) -> list[ServiceDefinition]:
        """Discover available services."""
        pass

    @abstractmethod
    async def watch(self, callback: Callable[[list[ServiceDefinition]], None]):
        """Watch for service changes."""
        pass

    async def start(self):
        """Start the discovery backend."""
        pass

    async def stop(self):
        """Stop the discovery backend."""
        pass


class StaticDiscoveryBackend(DiscoveryBackend):
    """Static configuration-based discovery."""

    def __init__(self, services: list[dict[str, Any]]):
        self._services = services

    async def discover(self) -> list[ServiceDefinition]:
        return [ServiceDefinition.from_dict(s) for s in self._services]

    async def watch(self, callback: Callable[[list[ServiceDefinition]], None]):
        pass


class EnvironmentDiscoveryBackend(DiscoveryBackend):
    """Environment variable-based discovery."""

    def __init__(self, prefix: str = "CHIMERA_SERVICE_"):
        self.prefix = prefix
        self._service_pattern = re.compile(rf"^{prefix}(\w+)_(\w+)$")

    async def discover(self) -> list[ServiceDefinition]:
        services_data: dict[str, dict[str, str]] = {}

        for key, value in os.environ.items():
            match = self._service_pattern.match(key)
            if match:
                service_name = match.group(1).lower()
                property_name = match.group(2).lower()

                if service_name not in services_data:
                    services_data[service_name] = {}
                services_data[service_name][property_name] = value

        services = []
        for name, props in services_data.items():
            service = self._create_service_from_env(name, props)
            if service:
                services.append(service)

        return services

    def _create_service_from_env(
        self, name: str, props: dict[str, str]
    ) -> ServiceDefinition | None:
        url = props.get("url", props.get("endpoint", ""))
        if not url:
            return None

        parsed = urlparse(url)
        endpoint = ServiceEndpoint(
            protocol=parsed.scheme or "http",
            host=parsed.hostname or "localhost",
            port=parsed.port or (443 if parsed.scheme == "https" else 80),
            path=parsed.path,
        )

        service_type_str = props.get("type", "microservice")
        try:
            service_type = ServiceType(service_type_str)
        except ValueError:
            service_type = ServiceType.MICROSERVICE

        return ServiceDefinition(
            name=name,
            type=service_type,
            endpoints=[endpoint],
            version=props.get("version", "1.0.0"),
            requires_auth="key" in props or "token" in props,
            auth_type="api_key" if "key" in props else None,
            metadata={
                "api_key": props.get("key", props.get("api_key", "")),
                "token": props.get("token", ""),
                "model": props.get("model", ""),
            },
        )

    async def watch(self, callback: Callable[[list[ServiceDefinition]], None]):
        pass


class FileDiscoveryBackend(DiscoveryBackend):
    """File-based service discovery."""

    def __init__(self, config_file: str):
        self.config_file = config_file
        self._last_modified: float | None = None
        self._watching = False

    async def discover(self) -> list[ServiceDefinition]:
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file not found: {self.config_file}")
            return []

        try:
            with open(self.config_file) as f:
                if self.config_file.endswith(".json"):
                    data = json.load(f)
                elif self.config_file.endswith((".yml", ".yaml")):
                    import yaml

                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            services = data.get("services", [])
            return [ServiceDefinition.from_dict(s) for s in services]

        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return []

    async def watch(self, callback: Callable[[list[ServiceDefinition]], None]):
        self._watching = True

        while self._watching:
            try:
                if os.path.exists(self.config_file):
                    mtime = os.path.getmtime(self.config_file)
                    if self._last_modified is None or mtime > self._last_modified:
                        self._last_modified = mtime
                        services = await self.discover()
                        if asyncio.iscoroutinefunction(callback):
                            await callback(services)
                        else:
                            callback(services)

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching config file: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        self._watching = False


class DNSDiscoveryBackend(DiscoveryBackend):
    """DNS-based service discovery (SRV records)."""

    def __init__(self, domain: str, service_names: list[str] | None = None):
        self.domain = domain
        self.service_names = service_names or []

    async def discover(self) -> list[ServiceDefinition]:
        services = []

        for service_name in self.service_names:
            try:
                srv_name = f"_{service_name}._tcp.{self.domain}"
                answers = await self._resolve_srv(srv_name)

                for priority, weight, port, target in answers:
                    endpoint = ServiceEndpoint(
                        protocol="http", host=str(target).rstrip("."), port=port
                    )

                    service = ServiceDefinition(
                        name=service_name,
                        type=ServiceType.MICROSERVICE,
                        endpoints=[endpoint],
                        weight=weight,
                        metadata={"priority": priority},
                    )
                    services.append(service)

            except Exception as e:
                logger.debug(f"DNS discovery failed for {service_name}: {e}")

        return services

    async def _resolve_srv(self, name: str) -> list[tuple]:
        try:
            import dns.resolver

            answers = dns.resolver.resolve(name, "SRV")
            return [(r.priority, r.weight, r.port, r.target) for r in answers]
        except ImportError:
            logger.warning("dnspython not installed, DNS discovery unavailable")
            return []
        except Exception:
            return []

    async def watch(self, callback: Callable[[list[ServiceDefinition]], None]):
        pass


class ConsulDiscoveryBackend(DiscoveryBackend):
    """HashiCorp Consul-based service discovery."""

    def __init__(self, host: str = "localhost", port: int = 8500, token: str | None = None):
        self.host = host
        self.port = port
        self.token = token
        self.base_url = f"http://{host}:{port}/v1"
        self._watching = False

    async def discover(self) -> list[ServiceDefinition]:
        import aiohttp

        services = []
        headers = {}
        if self.token:
            headers["X-Consul-Token"] = self.token

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/catalog/services", headers=headers
                ) as response:
                    if response.status != 200:
                        return []

                    service_names = await response.json()

                for service_name in service_names:
                    async with session.get(
                        f"{self.base_url}/health/service/{service_name}",
                        headers=headers,
                        params={"passing": "true"},
                    ) as response:
                        if response.status != 200:
                            continue

                        instances = await response.json()

                        for instance in instances:
                            service_data = instance.get("Service", {})

                            endpoint = ServiceEndpoint(
                                protocol="http",
                                host=service_data.get("Address", "localhost"),
                                port=service_data.get("Port", 80),
                            )

                            service = ServiceDefinition(
                                id=service_data.get("ID", ""),
                                name=service_name,
                                type=ServiceType.MICROSERVICE,
                                endpoints=[endpoint],
                                tags=set(service_data.get("Tags", [])),
                                metadata=service_data.get("Meta", {}),
                            )
                            services.append(service)

        except Exception as e:
            logger.error(f"Consul discovery error: {e}")

        return services

    async def watch(self, callback: Callable[[list[ServiceDefinition]], None]):
        self._watching = True
        last_index = 0

        import aiohttp

        headers = {}
        if self.token:
            headers["X-Consul-Token"] = self.token

        while self._watching:
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(
                        f"{self.base_url}/catalog/services",
                        headers=headers,
                        params={"index": last_index, "wait": "30s"},
                    ) as response,
                ):
                    if response.status == 200:
                        new_index = int(response.headers.get("X-Consul-Index", 0))
                        if new_index != last_index:
                            last_index = new_index
                            services = await self.discover()
                            if asyncio.iscoroutinefunction(callback):
                                await callback(services)
                            else:
                                callback(services)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consul watch error: {e}")
                await asyncio.sleep(5)

    async def stop(self):
        self._watching = False


class KubernetesDiscoveryBackend(DiscoveryBackend):
    """Kubernetes service discovery."""

    def __init__(self, namespace: str = "default", label_selector: str = ""):
        self.namespace = namespace
        self.label_selector = label_selector
        self._watching = False

    async def discover(self) -> list[ServiceDefinition]:
        try:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            v1 = client.CoreV1Api()

            services = []
            k8s_services = v1.list_namespaced_service(
                namespace=self.namespace, label_selector=self.label_selector
            )

            for svc in k8s_services.items:
                for port in svc.spec.ports or []:
                    endpoint = ServiceEndpoint(
                        protocol="http",
                        host=f"{svc.metadata.name}.{self.namespace}.svc.cluster.local",
                        port=port.port,
                    )

                    service = ServiceDefinition(
                        name=svc.metadata.name,
                        type=ServiceType.MICROSERVICE,
                        endpoints=[endpoint],
                        tags=set(svc.metadata.labels.keys()) if svc.metadata.labels else set(),
                        metadata=svc.metadata.labels or {},
                    )
                    services.append(service)

            return services

        except ImportError:
            logger.warning("kubernetes client not installed")
            return []
        except Exception as e:
            logger.error(f"Kubernetes discovery error: {e}")
            return []

    async def watch(self, callback: Callable[[list[ServiceDefinition]], None]):
        self._watching = True

        try:
            from kubernetes import client, config, watch

            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

            v1 = client.CoreV1Api()
            w = watch.Watch()

            for _event in w.stream(
                v1.list_namespaced_service,
                namespace=self.namespace,
                label_selector=self.label_selector,
            ):
                if not self._watching:
                    break

                services = await self.discover()
                if asyncio.iscoroutinefunction(callback):
                    await callback(services)
                else:
                    callback(services)

        except ImportError:
            logger.warning("kubernetes client not installed")
        except Exception as e:
            logger.error(f"Kubernetes watch error: {e}")

    async def stop(self):
        self._watching = False


class ServiceDiscovery:
    """
    Main service discovery coordinator.

    Manages multiple discovery backends and synchronizes with the service registry.
    """

    def __init__(self, registry: ServiceRegistry, config: DiscoveryConfig | None = None):
        self.registry = registry
        self.config = config or DiscoveryConfig()
        self._backends: list[DiscoveryBackend] = []
        self._running = False
        self._refresh_task: asyncio.Task | None = None
        self._watch_tasks: list[asyncio.Task] = []

    def add_backend(self, backend: DiscoveryBackend):
        """Add a discovery backend."""
        self._backends.append(backend)

    async def start(self):
        """Start service discovery."""
        self._running = True

        # Initialize backends based on config
        await self._initialize_backends()

        # Start backends
        for backend in self._backends:
            await backend.start()

        # Initial discovery
        await self.refresh()

        # Start refresh loop
        self._refresh_task = asyncio.create_task(self._refresh_loop())

        # Start watch tasks
        for backend in self._backends:
            task = asyncio.create_task(backend.watch(self._on_services_changed))
            self._watch_tasks.append(task)

        logger.info(f"Service discovery started with {len(self._backends)} backends")

    async def stop(self):
        """Stop service discovery."""
        self._running = False

        if self._refresh_task:
            self._refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresh_task

        for task in self._watch_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        for backend in self._backends:
            await backend.stop()

        logger.info("Service discovery stopped")

    async def _initialize_backends(self):
        """Initialize discovery backends based on configuration."""
        if self.config.method == DiscoveryMethod.STATIC:
            self._backends.append(StaticDiscoveryBackend(self.config.static_services))

        elif self.config.method == DiscoveryMethod.ENVIRONMENT:
            self._backends.append(EnvironmentDiscoveryBackend(self.config.env_prefix))

        elif self.config.method == DiscoveryMethod.FILE:
            self._backends.append(FileDiscoveryBackend(self.config.config_file))

        elif self.config.method == DiscoveryMethod.DNS:
            self._backends.append(DNSDiscoveryBackend(self.config.dns_domain))

        elif self.config.method == DiscoveryMethod.CONSUL:
            self._backends.append(
                ConsulDiscoveryBackend(
                    host=self.config.consul_host,
                    port=self.config.consul_port,
                    token=self.config.consul_token,
                )
            )

        elif self.config.method == DiscoveryMethod.KUBERNETES:
            self._backends.append(
                KubernetesDiscoveryBackend(
                    namespace=self.config.kubernetes_namespace,
                    label_selector=self.config.kubernetes_label_selector,
                )
            )

        # Always add environment backend as fallback
        if self.config.method != DiscoveryMethod.ENVIRONMENT:
            self._backends.append(EnvironmentDiscoveryBackend(self.config.env_prefix))

    async def refresh(self):
        """Refresh service discovery from all backends."""
        all_services: dict[str, ServiceDefinition] = {}

        for backend in self._backends:
            try:
                services = await backend.discover()
                for service in services:
                    key = f"{service.name}:{service.primary_endpoint.url if service.primary_endpoint else ''}"
                    if key not in all_services:
                        all_services[key] = service
            except Exception as e:
                logger.error(f"Discovery backend error: {e}")

        # Sync with registry
        await self._sync_with_registry(list(all_services.values()))

    async def _sync_with_registry(self, discovered_services: list[ServiceDefinition]):
        """Synchronize discovered services with the registry."""
        existing_services = await self.registry.get_all_services()
        existing_ids = {s.id for s in existing_services}
        {s.id for s in discovered_services}

        # Register new services
        for service in discovered_services:
            if service.id not in existing_ids:
                await self.registry.register_service(service)
                logger.info(f"Discovered new service: {service.name}")

        # Update existing services
        for service in discovered_services:
            if service.id in existing_ids:
                await self.registry._backend.update(service)

    async def _on_services_changed(self, services: list[ServiceDefinition]):
        """Callback when services change."""
        await self._sync_with_registry(services)

    async def _refresh_loop(self):
        """Periodic refresh loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.refresh_interval)
                await self.refresh()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Refresh loop error: {e}")

    async def discover_service(
        self, name: str, service_type: ServiceType | None = None
    ) -> ServiceDefinition | None:
        """Discover a specific service."""
        services = await self.registry.find_services(name=name, service_type=service_type)
        return services[0] if services else None


def create_service_discovery(
    registry: ServiceRegistry, method: str = "environment", **kwargs
) -> ServiceDiscovery:
    """
    Factory function to create service discovery.

    Args:
        registry: Service registry instance
        method: Discovery method (static, environment, file, dns, consul, kubernetes)
        **kwargs: Additional configuration

    Returns:
        ServiceDiscovery instance
    """
    config = DiscoveryConfig(method=DiscoveryMethod(method), **kwargs)
    return ServiceDiscovery(registry=registry, config=config)
