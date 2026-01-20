import sys

import requests


def test_login(base_url, email, password) -> bool | None:
    login_url = f"{base_url}/api/v1/auth/login"

    payload = {"username": email, "password": password}

    try:
        response = requests.post(login_url, json=payload)

        if response.status_code == 200:
            response.json()
            return True
        return False

    except Exception:
        return False


if __name__ == "__main__":
    BASE_URL = "http://127.0.0.1:8001"
    EMAIL = "admin@chimera.ai"
    PASSWORD = "ChimeraAdmin2026!"

    success = test_login(BASE_URL, EMAIL, PASSWORD)
    if not success:
        sys.exit(1)
