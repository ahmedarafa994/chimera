#!/usr/bin/env python3
"""Test login API endpoint with admin credentials."""
import requests

# Test the new backend on port 8005
url = "http://localhost:8005/api/v1/auth/login"
payload = {"username": "admin", "password": "Admin123!@#"}

print(f"Testing login at: {url}")
print(f"Payload: {payload}")

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
