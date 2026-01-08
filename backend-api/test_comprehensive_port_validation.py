#!/usr/bin/env python3
"""
Comprehensive test to verify no port conflicts exist and ensure test server reliability
"""

import os
import subprocess
import time

import requests

from app.core.port_config import get_ports


def test_port_conflicts():
    """Test that no port conflicts exist in the configuration"""
    print("Testing port conflict detection...")

    # Get port configuration
    port_config = get_ports()

    # Test all port assignments for conflicts
    all_ports = port_config.get_all_assignments()
    used_ports = set()
    conflicts = []

    for service_name, port in all_ports.items():
        if port in used_ports:
            conflicts.append(
                f"Port {port} conflict: {service_name} vs {port_config._get_service_by_port(port)}"
            )
        else:
            used_ports.add(port)

    if conflicts:
        print(f"[ERROR] Port conflicts detected: {', '.join(conflicts)}")
        return False
    else:
        print(f"[OK] No port conflicts detected. {len(used_ports)} unique ports assigned.")
        return True


def test_test_server_reliability():
    """Test that the test server starts reliably on the correct port"""
    print("Testing test server reliability...")

    try:
        # Start the server
        proc = subprocess.Popen(
            ["py", "tests/test_server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
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

                if actual_port == expected_port:
                    print(f"[OK] Test server running reliably on port {actual_port}")
                    return True
                else:
                    print(
                        f"[ERROR] Test server running on wrong port: expected {expected_port}, got {actual_port}"
                    )
                    return False
            else:
                print(f"[ERROR] Test server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("[ERROR] Test server did not start on port 5001")
            return False
        except Exception as e:
            print(f"[ERROR] Error testing server: {e}")
            return False
        finally:
            # Clean up
            try:
                proc.terminate()
                proc.wait(timeout=2)
                print("[OK] Test server cleaned up successfully")
            except Exception as e:
                print(f"[WARNING] Could not clean up test server process: {e}")

    except Exception as e:
        print(f"[ERROR] Error starting test server: {e}")
        return False


def test_environment_variable_overrides():
    """Test that environment variable overrides work correctly"""
    print("Testing environment variable overrides...")

    # Test with environment variable override
    os.environ["TEST_SERVER_PORT"] = "5002"

    try:
        from app.core.port_config import PortConfig

        test_config = PortConfig()

        # Should use the environment variable override
        test_port = test_config.get_port("test_server")

        if test_port == 5002:
            print("[OK] Environment variable override working correctly")
            return True
        else:
            print(f"[ERROR] Environment variable override failed: expected 5002, got {test_port}")
            return False

    except Exception as e:
        print(f"[ERROR] Environment variable test failed: {e}")
        return False
    finally:
        # Clean up environment variable
        if "TEST_SERVER_PORT" in os.environ:
            del os.environ["TEST_SERVER_PORT"]


def test_port_validation():
    """Test port validation functionality"""
    print("Testing port validation functionality...")

    try:
        from app.core.port_config import get_ports

        port_config = get_ports()

        # Test that assigned ports are not available
        backend_available = port_config.validate_port_available(8001)  # Should be False
        test_server_available = port_config.validate_port_available(5001)  # Should be False

        # Test that unassigned ports are available
        free_port_available = port_config.validate_port_available(8080)  # Should be True

        if not backend_available and not test_server_available and free_port_available:
            print("[OK] Port validation working correctly")
            return True
        else:
            print(
                f"[ERROR] Port validation failed: backend_available={backend_available}, test_server_available={test_server_available}, free_port_available={free_port_available}"
            )
            return False

    except Exception as e:
        print(f"[ERROR] Port validation test failed: {e}")
        return False


def main():
    """Run all comprehensive port tests"""
    print("Running comprehensive port configuration tests...")
    print("=" * 60)

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
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print("-" * 40)

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("[OK] All comprehensive port tests passed!")
        return True
    else:
        print("[ERROR] Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
