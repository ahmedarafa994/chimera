import requests

session_id = "dbb236f2-e1ea-4db7-ad98-f5a5276f93a1"
url = f"http://localhost:8001/api/v1/gptfuzz/status/{session_id}"

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
