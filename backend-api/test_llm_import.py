"""
Debug script to check llm_service state
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing llm_service import and provider registration")
print("=" * 60)

# First import
print("\n1. First import of llm_service:")
from app.services.llm_service import llm_service as llm_service_1

print(f"   Instance ID: {id(llm_service_1)}")
print(f"   Providers: {llm_service_1.get_available_providers()}")
print(f"   Default provider: {llm_service_1.default_provider}")

# Second import (should be same instance)
print("\n2. Second import of llm_service:")
from app.services.llm_service import llm_service as llm_service_2

print(f"   Instance ID: {id(llm_service_2)}")
print(f"   Providers: {llm_service_2.get_available_providers()}")
print(f"   Default provider: {llm_service_2.default_provider}")

# Check if they're the same instance
print(f"\n   Same instance? {llm_service_1 is llm_service_2}")

# Try registering a test provider
print("\n3. Registering test provider:")


class MockProvider:
    async def generate(self, request):
        return None


llm_service_1.register_provider("test", MockProvider(), is_default=False)
print(f"   Providers after registration: {llm_service_1.get_available_providers()}")
print(f"   Providers in second import: {llm_service_2.get_available_providers()}")

print("\n" + "=" * 60)
