#!/usr/bin/env python3
"""Test the centralized port configuration system."""

from app.core.port_config import get_ports


def test_port_configuration() -> bool:
    """Test port configuration system."""
    # Get port configuration
    port_config = get_ports()

    # Test basic functionality

    # Test getting specific ports
    port_config.get_port("backend_api")
    port_config.get_port("test_server")
    port_config.get_port("frontend_dev")

    # Test getting all assignments
    port_config.get_all_assignments()

    # Test port validation
    port_config.validate_port_available(8080)
    port_config.validate_port_available(8001)

    # Test service info
    port_config.get_service_info("backend_api")

    # Test primary port
    port_config.get_primary_port()

    # Test test server port
    port_config.get_test_server_port()

    return True


if __name__ == "__main__":
    try:
        test_port_configuration()
    except Exception:
        import traceback

        traceback.print_exc()
