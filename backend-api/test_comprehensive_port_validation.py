#!/usr/bin/env python3
"""Comprehensive test to verify no port conflicts exist and ensure test server reliability."""

import os
import subprocess
import sys
import time

import requests

from app.core.port_config import get_ports


def test_port_conflicts() -> bool:
    """Test that no port conflicts exist in the configuration."""
    # Get port configuration
    port_config = get_ports()

    # Test all port assignments for conflicts
    all_ports = port_config.get_all_assignments()
    used_ports = set()
    conflicts = []

    for service_name, port in all_ports.items():
        if port in used_ports:
            conflicts.append(
                f"Port {port} conflict: {service_name} vs {port_config._get_service_by_port(port)}",
            )
        else:
            used_ports.add(port)

    return not conflicts


def test_test_server_reliability() -> bool | None:
    """Test that the test server starts reliably on the correct port."""
    try:
        # Start the server
        proc = subprocess.Popen(
            ["py", "tests/test_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        time.sleep(2)

        try:
            # Test if server is running on expected port (5001)
            response = requests.get("http://localhost:5001/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                expected_port = 5001
                actual_port = data.get("port", 0)

                return actual_port == expected_port
            return False
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False
        finally:
            # Clean up
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass

    except Exception:
        return False


def test_environment_variable_overrides() -> bool | None:
    """Test that environment variable overrides work correctly."""
    # Test with environment variable override
    os.environ["TEST_SERVER_PORT"] = "5002"

    try:
        from app.core.port_config import PortConfig

        test_config = PortConfig()

        # Should use the environment variable override
        test_port = test_config.get_port("test_server")

        return test_port == 5002

    except Exception:
        return False
    finally:
        # Clean up environment variable
        if "TEST_SERVER_PORT" in os.environ:
            del os.environ["TEST_SERVER_PORT"]


def test_port_validation() -> bool | None:
    """Test port validation functionality."""
    try:
        from app.core.port_config import get_ports

        port_config = get_ports()

        # Test that assigned ports are not available
        backend_available = port_config.validate_port_available(8001)  # Should be False
        test_server_available = port_config.validate_port_available(5001)  # Should be False

        # Test that unassigned ports are available
        free_port_available = port_config.validate_port_available(8080)  # Should be True

        return bool(not backend_available and not test_server_available and free_port_available)

    except Exception:
        return False


def main() -> bool:
    """Run all comprehensive port tests."""
    tests = [
        test_port_conflicts,
        test_port_validation,
        test_environment_variable_overrides,
        test_test_server_reliability,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception:
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
