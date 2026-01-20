import contextlib

import requests

url = "http://localhost:8001/api/v1/autodan/jailbreak"
payload = {"request": "test", "method": "vanilla"}
headers = {"Content-Type": "application/json"}

with contextlib.suppress(Exception):
    response = requests.post(url, json=payload, headers=headers, timeout=10)
