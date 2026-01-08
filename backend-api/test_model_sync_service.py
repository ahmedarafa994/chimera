#!/usr/bin/env python3
"""
Test model sync service directly
"""

import asyncio

from app.services.model_sync_service import model_sync_service


async def test_model_sync_service():
    """Test model sync service directly"""
    try:
        # Initialize models
        await model_sync_service.initialize_models()
        print("Models initialized successfully")

        # Get available models
        models_response = await model_sync_service.get_available_models()
        print(f"Available models: {models_response.count}")
        print(f"Models: {[model.id for model in models_response.models[:5]]}...")

        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_model_sync_service())
    print(f"Test result: {result}")
