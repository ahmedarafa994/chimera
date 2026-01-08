import json

import requests

try:
    response = requests.get("http://127.0.0.1:5000/api/v1/providers", timeout=10)
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print("Response is not JSON")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
