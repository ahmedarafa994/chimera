import sys

import requests


def test_login(base_url, email, password):
    login_url = f"{base_url}/api/v1/auth/login"
    print(f"Attempting login at: {login_url}")

    payload = {"username": email, "password": password}

    try:
        response = requests.post(login_url, json=payload)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("Login SUCCESS!")
            data = response.json()
            print(f"User: {data['user']['email']}")
            print(f"Role: {data['user']['role']}")
            print("Access Token received.")
            return True
        else:
            print("Login FAILED!")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"Error connecting to server: {e}")
        return False


if __name__ == "__main__":
    BASE_URL = "http://127.0.0.1:8001"
    EMAIL = "admin@chimera.ai"
    PASSWORD = "ChimeraAdmin2026!"

    success = test_login(BASE_URL, EMAIL, PASSWORD)
    if not success:
        sys.exit(1)
