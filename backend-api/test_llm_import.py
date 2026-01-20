"""Debug script to check llm_service state."""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# First import
from app.services.llm_service import llm_service as llm_service_1

# Second import (should be same instance)

# Check if they're the same instance

# Try registering a test provider


class MockProvider:
    async def generate(self, request) -> None:
        return None


llm_service_1.register_provider("test", MockProvider(), is_default=False)
