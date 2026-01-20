#!/usr/bin/env python3
"""Test model sync service directly."""

import asyncio

from app.services.model_sync_service import model_sync_service


async def test_model_sync_service() -> bool | None:
    """Test model sync service directly."""
    try:
        # Initialize models
        await model_sync_service.initialize_models()

        # Get available models
        await model_sync_service.get_available_models()

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_model_sync_service())
