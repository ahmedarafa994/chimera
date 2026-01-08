import requests

url = "http://localhost:8001/api/v1/autodan/nonexistent"
try:
    print(f"Testing {url}...")
    response = requests.post(url, timeout=5)
    print(f"Status Code: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
