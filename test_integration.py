
import requests

BACKEND_URL = "http://localhost:8001"

print("=" * 60)
print("CHIMERA INTEGRATION TEST")
print("=" * 60)

# Test 1: Root endpoint
print("\n[1] Testing root endpoint...")
try:
    r = requests.get(f"{BACKEND_URL}/", timeout=5)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: Health endpoint
print("\n[2] Testing health endpoint...")
try:
    r = requests.get(f"{BACKEND_URL}/health", timeout=5)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: Providers endpoint
print("\n[3] Testing providers endpoint...")
try:
    r = requests.get(f"{BACKEND_URL}/api/v1/providers", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Providers count: {data.get('count', 0)}")
        print(f"   Default provider: {data.get('default', 'N/A')}")
    else:
        print(f"   Response: {r.text[:200]}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: CORS headers
print("\n[4] Testing CORS headers...")
try:
    r = requests.options(
        f"{BACKEND_URL}/api/v1/providers",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        timeout=5,
    )
    print(f"   Status: {r.status_code}")
    cors_header = r.headers.get("access-control-allow-origin")
    print(f"   CORS header: {cors_header}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
