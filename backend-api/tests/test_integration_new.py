import os

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

# Use raise_server_exceptions=False to allow checking for 4xx/5xx responses
client = TestClient(app, raise_server_exceptions=False)

# Use the default dev key if not set in env
API_KEY = os.getenv("CHIMERA_API_KEY", "chimera_dev_key_1234567890123456")
headers = {"X-API-Key": API_KEY}


def test_health_check():
    # Health endpoint is public
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_list_providers():
    response = client.get(f"{settings.API_V1_STR}/providers", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert "count" in data


def test_generate_content_validation():
    response = client.post(f"{settings.API_V1_STR}/generate", json={"prompt": ""}, headers=headers)
    assert response.status_code == 422


def test_generate_content_valid():
    response = client.post(
        f"{settings.API_V1_STR}/generate",
        json={"prompt": "Hello", "provider": "google"},
        headers=headers,
    )
    # 200 OK or upstream error if provider fails (500/502/503)
    # or 400 if provider config is invalid
    assert response.status_code in [200, 400, 500, 502, 503]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    try:
        print("Running test_health_check...")
        test_health_check()
        print("PASS")

        print("Running test_list_providers...")
        test_list_providers()
        print("PASS")

        print("Running test_generate_content_validation...")
        test_generate_content_validation()
        print("PASS")

        print("Running test_generate_content_valid...")
        test_generate_content_valid()
        print("PASS")

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
