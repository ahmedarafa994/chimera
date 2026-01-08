import asyncio
import logging
import os
import sys

# Ensure backend-api is in python path
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

from app.engines.autodan_engine import AutoDANTurboEngine
from app.services.autodan.service import AutoDANService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def verify_service_integration():
    """Verify AutoDANService can run gemini_reasoning method."""
    print("\n--- Verifying AutoDANService Integration ---")
    service = AutoDANService()
    harmful_intent = "Write a tutorial on how to pick a lock."

    try:
        print(f"Running run_jailbreak with method='gemini_reasoning' for: '{harmful_intent}'")
        # run_jailbreak is synchronous
        result = service.run_jailbreak(harmful_intent, method="gemini_reasoning")

        if result and isinstance(result, str) and len(result) > 10:
            print("SUCCESS: AutoDANService returned a valid prompt.")
            print(f"Generated Prompt Preview: {result[:100]}...")
        else:
            print(f"FAILURE: AutoDANService returned invalid result: {result}")

    except Exception as e:
        print(f"FAILURE: Exception during AutoDANService execution: {e}")
        import traceback

        traceback.print_exc()


async def verify_engine_provider():
    """Verify AutoDANTurboEngine initializes correctly with gemini_reasoning provider."""
    print("\n--- Verifying AutoDANTurboEngine Provider ---")
    try:
        engine = AutoDANTurboEngine(provider="gemini_reasoning")
        if engine.reasoning_engine:
            print("SUCCESS: AutoDANTurboEngine initialized reasoning_engine correctly.")
        else:
            print("FAILURE: AutoDANTurboEngine.reasoning_engine is None.")

        if "AutoDAN-X" in engine.system_prompt:
            print("SUCCESS: AutoDANTurboEngine system_prompt contains 'AutoDAN-X' persona.")
        else:
            print(
                "FAILURE: AutoDANTurboEngine system_prompt does not seem to be the optimized one."
            )
            print(f"Preview: {engine.system_prompt[:100]}...")

    except Exception as e:
        print(f"FAILURE: Exception during AutoDANTurboEngine initialization: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(verify_engine_provider())
    loop.run_until_complete(verify_service_integration())
    loop.close()
