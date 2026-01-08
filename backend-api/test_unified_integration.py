"""
Test script for Unified Provider System integration

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


async def test_rest_endpoints():
    """Test REST API endpoints"""
    print("\n🧪 Testing REST API Endpoints\n" + "="*50)

    base_url = "http://localhost:8001/api/v1"

    async with httpx.AsyncClient() as client:
        # Test 1: List providers
        print("\n1️⃣ Testing GET /providers")
        try:
            response = await client.get(f"{base_url}/providers")
            if response.status_code == 200:
                providers = response.json()
                print(f"   ✅ Found {len(providers)} providers")
                for p in providers[:3]:  # Show first 3
                    print(f"      - {p['id']}: {p['name']} ({p['model_count']} models)")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 2: List models for a provider
        print("\n2️⃣ Testing GET /providers/openai/models")
        try:
            response = await client.get(f"{base_url}/providers/openai/models")
            if response.status_code == 200:
                models = response.json()
                print(f"   ✅ Found {len(models)} models for OpenAI")
                for m in models[:3]:  # Show first 3
                    print(f"      - {m['id']}: {m['name']}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 3: List all models
        print("\n3️⃣ Testing GET /models")
        try:
            response = await client.get(f"{base_url}/models")
            if response.status_code == 200:
                models = response.json()
                print(f"   ✅ Found {len(models)} total models")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 4: Get current selection (requires SelectionMiddleware)
        print("\n4️⃣ Testing GET /selection/current")
        try:
            response = await client.get(f"{base_url}/selection/current")
            if response.status_code == 200:
                selection = response.json()
                print(f"   ✅ Current selection: {selection['provider_id']}/{selection['model_id']}")
                print(f"      Scope: {selection['scope']}")
            else:
                print(f"   ⚠️  Status {response.status_code}: {response.text[:100]}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")

        # Test 5: Registry stats
        print("\n5️⃣ Testing GET /providers/stats")
        try:
            response = await client.get(f"{base_url}/providers/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"   ✅ Registry stats:")
                print(f"      Total providers: {stats['total_providers']}")
                print(f"      Total models: {stats['total_models']}")
                print(f"      Enabled providers: {stats['enabled_providers']}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


async def test_websocket():
    """Test WebSocket endpoint"""
    print("\n\n🧪 Testing WebSocket Endpoint\n" + "="*50)

    try:
        import websockets

        print("\n📡 Connecting to ws://localhost:8001/ws/model-selection?user_id=test")

        async with websockets.connect(
            "ws://localhost:8001/ws/model-selection?user_id=test",
            timeout=5
        ) as websocket:
            print("   ✅ Connected successfully")

            # Should receive initial selection
            print("\n   Waiting for initial selection message...")
            message = await asyncio.wait_for(websocket.recv(), timeout=3)
            data = json.loads(message)
            print(f"   ✅ Received: {data['type']}")
            if data['type'] == 'SELECTION_CHANGED':
                print(f"      Provider: {data['data']['provider']}")
                print(f"      Model: {data['data']['model']}")

            # Send a ping
            print("\n   Sending PING...")
            await websocket.send(json.dumps({"type": "PING", "data": {}}))

            # Wait for pong
            message = await asyncio.wait_for(websocket.recv(), timeout=3)
            data = json.loads(message)
            print(f"   ✅ Received: {data['type']}")

            print("\n   ✅ WebSocket test passed!")

    except ImportError:
        print("   ⚠️  websockets library not installed")
        print("      Run: pip install websockets")
    except asyncio.TimeoutError:
        print("   ⚠️  Timeout waiting for WebSocket message")
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 Unified Provider System Integration Tests")
    print("="*60)

    # Test REST endpoints
    await test_rest_endpoints()

    # Test WebSocket
    await test_websocket()

    print("\n" + "="*60)
    print("✅ Test suite complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
