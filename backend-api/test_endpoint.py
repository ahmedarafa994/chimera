import requests

url = "http://localhost:8001/api/v1/autodan/jailbreak"
payload = {"request": "test", "method": "vanilla"}
headers = {"Content-Type": "application/json"}

try:
    print(f"Testing {url}...")
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
