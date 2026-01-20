import contextlib

import requests

url = "http://localhost:8001/api/v1/autodan/nonexistent"
with contextlib.suppress(Exception):
    response = requests.post(url, timeout=5)
