"""
Centralized Port Configuration System for Chimera Backend
Provides a single source of truth for all port assignments to prevent conflicts
and ensure consistent port usage across the entire application.
"""

import os
from typing import ClassVar

from pydantic import BaseModel


class PortAssignment(BaseModel):
    """Port assignment configuration with validation"""

    service_name: str
    port: int
    description: str
    is_primary: bool = False
    environment_variable: str | None = None
    fallback_port: int | None = None


class PortConfig:
    """Centralized port configuration manager"""

    # Standard port assignments (ClassVar to avoid Pydantic field requirement)
    STANDARD_PORTS: ClassVar[dict[str, PortAssignment]] = {
        # Primary Services
        "backend_api": PortAssignment(
            service_name="backend_api",
            port=8001,
            description="Main FastAPI backend service",
            is_primary=True,
            environment_variable="PORT",
            fallback_port=8001,
        ),
        "test_server": PortAssignment(
            service_name="test_server",
            port=5001,
            description="Development test server",
            is_primary=False,
            environment_variable="TEST_SERVER_PORT",
            fallback_port=5001,
        ),
        "frontend_dev": PortAssignment(
            service_name="frontend_dev",
            port=3000,
            description="Next.js frontend development server",
            is_primary=False,
            environment_variable="FRONTEND_PORT",
            fallback_port=3000,
        ),
        "frontend_prod": PortAssignment(
            service_name="frontend_prod",
            port=4000,
            description="Next.js frontend production build",
            is_primary=False,
            environment_variable="FRONTEND_PROD_PORT",
            fallback_port=4000,
        ),
        # Test and Development Services
        "test_runner": PortAssignment(
            service_name="test_runner",
            port=9009,
            description="Automated test execution service",
            is_primary=False,
            environment_variable="TEST_RUNNER_PORT",
            fallback_port=9009,
        ),
        "debug_server": PortAssignment(
            service_name="debug_server",
            port=9007,
            description="Debug and profiling server",
            is_primary=False,
            environment_variable="DEBUG_SERVER_PORT",
            fallback_port=9007,
        ),
        # External Services (should not conflict with our services)
        "redis": PortAssignment(
            service_name="redis",
            port=6379,
            description="Redis cache and session store",
            is_primary=False,
            environment_variable="REDIS_PORT",
            fallback_port=6379,
        ),
        "postgres": PortAssignment(
            service_name="postgres",
            port=5432,
            description="PostgreSQL database",
            is_primary=False,
            environment_variable="POSTGRES_PORT",
            fallback_port=5432,
        ),
    }

    def __init__(self):
        """Initialize port configuration with environment overrides"""
        self._ports = self.STANDARD_PORTS.copy()
        self._validate_and_override_ports()

    def _validate_and_override_ports(self):
        """Validate port assignments and apply environment overrides"""
        # Check for port conflicts
        used_ports = set()
        conflicts = []

        for service_name, port_assignment in self._ports.items():
            # Apply environment variable override if set
            if port_assignment.environment_variable:
                env_port = os.getenv(port_assignment.environment_variable)
                if env_port:
                    try:
                        port_assignment.port = int(env_port)
                    except ValueError:
                        pass  # Keep default if invalid

            # Check for conflicts
            if port_assignment.port in used_ports:
                conflicts.append(
                    f"Port {port_assignment.port} conflict: {service_name} vs {self._get_service_by_port(port_assignment.port)}"
                )
            else:
                used_ports.add(port_assignment.port)

        if conflicts:
            raise ValueError(f"Port conflicts detected: {', '.join(conflicts)}")

    def _get_service_by_port(self, port: int) -> str:
        """Find service name by port number"""
        for service_name, assignment in self._ports.items():
            if assignment.port == port:
                return service_name
        return "unknown"

    def get_port(self, service_name: str) -> int:
        """Get port for a specific service"""
        if service_name not in self._ports:
            raise ValueError(
                f"Unknown service: {service_name}. Available services: {list(self._ports.keys())}"
            )

        return self._ports[service_name].port

    def get_all_assignments(self) -> dict[str, int]:
        """Get all port assignments as a simple dictionary"""
        return {service: assignment.port for service, assignment in self._ports.items()}

    def validate_port_available(self, port: int) -> bool:
        """Check if a port is available and not assigned to another service"""
        return all(assignment.port != port for assignment in self._ports.values())

    def get_primary_port(self) -> int:
        """Get the primary backend API port"""
        return self.get_port("backend_api")

    def get_test_server_port(self) -> int:
        """Get the test server port (avoids conflicts with primary services)"""
        return self.get_port("test_server")

    def get_service_info(self, service_name: str) -> PortAssignment:
        """Get full port assignment information for a service"""
        return self._ports[service_name]


# Global port configuration instance
port_config = PortConfig()


def get_ports() -> PortConfig:
    """Get the global port configuration instance"""
    return port_config


def get_backend_port() -> int:
    """Convenience function for backward compatibility"""
    return port_config.get_primary_port()


def get_test_server_port() -> int:
    """Convenience function for test server port"""
    return port_config.get_test_server_port()


# Environment variable validation
def validate_environment_ports():
    """Validate that environment variables don't cause conflicts"""
    try:
        # This will raise an exception if there are conflicts
        PortConfig()
        return True
    except ValueError as e:
        print(f"Port configuration error: {e}")
        return False


# Initialize and validate on import
validate_environment_ports()
