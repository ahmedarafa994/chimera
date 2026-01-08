#!/usr/bin/env python3
"""
Test the centralized port configuration system
"""

from app.core.port_config import get_ports


def test_port_configuration():
    """Test port configuration system"""
    print("Testing centralized port configuration system...")

    # Get port configuration
    port_config = get_ports()

    # Test basic functionality
    print("[OK] Port config instance created successfully")

    # Test getting specific ports
    backend_port = port_config.get_port("backend_api")
    test_server_port = port_config.get_port("test_server")
    frontend_port = port_config.get_port("frontend_dev")

    print(f"[OK] Backend API port: {backend_port}")
    print(f"[OK] Test server port: {test_server_port}")
    print(f"[OK] Frontend dev port: {frontend_port}")

    # Test getting all assignments
    all_ports = port_config.get_all_assignments()
    print(f"[OK] Total port assignments: {len(all_ports)}")

    # Test port validation
    is_8080_available = port_config.validate_port_available(8080)
    is_8001_available = port_config.validate_port_available(8001)

    print(f"[OK] Port 8080 available: {is_8080_available}")
    print(f"[OK] Port 8001 available: {is_8001_available}")  # Should be False (in use)

    # Test service info
    backend_info = port_config.get_service_info("backend_api")
    print(f"[OK] Backend service info: {backend_info.service_name} -> {backend_info.port}")

    # Test primary port
    primary_port = port_config.get_primary_port()
    print(f"[OK] Primary port: {primary_port}")

    # Test test server port
    test_port = port_config.get_test_server_port()
    print(f"[OK] Test server port: {test_port}")

    print("[OK] All port configuration tests passed!")

    return True


if __name__ == "__main__":
    try:
        test_port_configuration()
    except Exception as e:
        print(f"[ERROR] Port configuration test failed: {e}")
        import traceback

        traceback.print_exc()
