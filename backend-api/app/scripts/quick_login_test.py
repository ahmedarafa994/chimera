"""Quick login test script."""

import requests


def main() -> None:
    url = "http://localhost:8001/api/v1/auth/login"
    payload = {"username": "admin@chimera.ai", "password": "ChimeraAdmin2026!"}

    try:
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            response.json()
        else:
            pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
