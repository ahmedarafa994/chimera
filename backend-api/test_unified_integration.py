"""Test script for Unified Provider System integration.

Tests the new REST API endpoints and WebSocket server to verify
integration with unified_registry and SelectionContext.

Usage:
    python test_unified_integration.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import httpx


async def test_rest_endpoints() -> None:
    """Test REST API endpoints."""
    base_url = "http://localhost:8001/api/v1"

    async with httpx.AsyncClient() as client:
        # Test 1: List providers
        try:
            response = await client.get(f"{base_url}/providers")
            if response.status_code == 200:
                providers = response.json()
                for _p in providers[:3]:  # Show first 3
                    pass
            else:
                pass
        except Exception:
            pass

        # Test 2: List models for a provider
        try:
            response = await client.get(f"{base_url}/providers/openai/models")
            if response.status_code == 200:
                models = response.json()
                for _m in models[:3]:  # Show first 3
                    pass
            else:
                pass
        except Exception:
            pass

        # Test 3: List all models
        try:
            response = await client.get(f"{base_url}/models")
            if response.status_code == 200:
                models = response.json()
            else:
                pass
        except Exception:
            pass

        # Test 4: Get current selection (requires SelectionMiddleware)
        try:
            response = await client.get(f"{base_url}/selection/current")
            if response.status_code == 200:
                response.json()
            else:
                pass
        except Exception:
            pass

        # Test 5: Registry stats
        try:
            response = await client.get(f"{base_url}/providers/stats")
            if response.status_code == 200:
                response.json()
            else:
                pass
        except Exception:
            pass


async def test_websocket() -> None:
    """Test WebSocket endpoint."""
    try:
        import websockets

        async with websockets.connect(
            "ws://localhost:8001/ws/model-selection?user_id=test",
            timeout=5,
        ) as websocket:
            # Should receive initial selection
            message = await asyncio.wait_for(websocket.recv(), timeout=3)
            data = json.loads(message)
            if data["type"] == "SELECTION_CHANGED":
                pass

            # Send a ping
            await websocket.send(json.dumps({"type": "PING", "data": {}}))

            # Wait for pong
            message = await asyncio.wait_for(websocket.recv(), timeout=3)
            data = json.loads(message)

    except ImportError:
        pass
    except TimeoutError:
        pass
    except Exception:
        pass


async def main() -> None:
    """Run all tests."""
    # Test REST endpoints
    await test_rest_endpoints()

    # Test WebSocket
    await test_websocket()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
