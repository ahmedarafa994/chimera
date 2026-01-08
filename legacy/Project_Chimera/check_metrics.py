import requests

try:
    resp = requests.get("http://localhost:4141/api/v1/metrics")
    print(f"Status: {resp.status_code}")
    print(resp.text)
except Exception as e:
    print(e)
