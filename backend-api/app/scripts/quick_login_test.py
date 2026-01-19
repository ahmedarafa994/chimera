"""Quick login test script."""

import requests


def main():
    url = "http://localhost:8001/api/v1/auth/login"
    payload = {"username": "admin@chimera.ai", "password": "ChimeraAdmin2026!"}

    print(f"Testing login at: {url}")
    print(f"Payload: {payload}")

    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"\nStatus: {response.status_code}")
        print(f"Response: {response.text[:500] if response.text else 'Empty'}")

        if response.status_code == 200:
            data = response.json()
            print("\n=== LOGIN SUCCESS ===")
            print(f"User: {data.get('user', {}).get('email')}")
            print(f"Role: {data.get('user', {}).get('role')}")
            print(f"Token: {data.get('access_token', '')[:50]}...")
        else:
            print("\n=== LOGIN FAILED ===")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
