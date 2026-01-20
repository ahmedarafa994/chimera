import contextlib

import requests

url = "http://localhost:8001/api/v1/providers"

with contextlib.suppress(Exception):
    response = requests.get(url)
