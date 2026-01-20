import contextlib

import requests

session_id = "dbb236f2-e1ea-4db7-ad98-f5a5276f93a1"
url = f"http://localhost:8001/api/v1/gptfuzz/status/{session_id}"

with contextlib.suppress(Exception):
    response = requests.get(url)
