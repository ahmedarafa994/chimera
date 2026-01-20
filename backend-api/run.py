#!/usr/bin/env python3
"""Main application entry point using centralized port configuration.

Features:
- Port conflict detection
- Graceful error handling
- Environment-based configuration
"""

import os
import socket
import sys

import uvicorn

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is already in use.

    Args:
        port: Port number to check
        host: Host to check (default: 0.0.0.0)

    Returns:
        True if port is in use, False otherwise

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def get_backend_port() -> int:
    """Get backend port using centralized configuration with fallback."""
    try:
        # Try to use centralized port configuration
        from app.core.port_config import get_ports

        port_config = get_ports()
        return port_config.get_port("backend_api")
    except ImportError:
        # Fallback to environment variable if port config not available
        return int(os.getenv("PORT", 8001))


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port.

    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found

    """
    for offset in range(max_attempts):
        port = start_port + offset
        if not is_port_in_use(port):
            return port
    msg = f"No available port found in range {start_port}-{start_port + max_attempts - 1}"
    raise RuntimeError(
        msg,
    )


if __name__ == "__main__":
    # Use centralized port configuration
    port = get_backend_port()
    environment = os.getenv("ENVIRONMENT", "development")
    reload = environment == "development"

    # Check for port conflicts
    if is_port_in_use(port):
        # In development, try to find an alternative port
        if environment == "development":
            try:
                alt_port = find_available_port(port + 1)
                port = alt_port
            except RuntimeError:
                sys.exit(1)
        else:
            sys.exit(1)

    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=reload)
