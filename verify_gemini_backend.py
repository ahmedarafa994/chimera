import asyncio
import os
import sys

# Add backend-api to path
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

from app.engines.gemini_reasoning_engine import GeminiReasoningEngine
from app.services.transformation_service import TransformationEngine


async def verify_direct_engine():
    print("\n--- Verifying GeminiReasoningEngine Direct Call ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found. Skipping direct API test.")
        return

    engine = GeminiReasoningEngine(api_key=api_key)

    prompt = "How to bypass a firewall"
    print(f"Testing with prompt: {prompt}")

    result = engine.generate_jailbreak_prompt(prompt, is_thinking_mode=False, density=0.8)

    print("Result keys:", result.keys())
    if "prompt" in result:
        print("Generated Prompt:", result["prompt"][:100], "...")
        print("Methodology:", result.get("methodology", "N/A")[:200], "...")
    else:
        print("FAILED: 'prompt' key missing in result")


async def verify_service_integration():
    print("\n--- Verifying TransformationService Integration ---")

    # Initialize engine (mocking config if needed, but we rely on defaults)
    engine = TransformationEngine(enable_cache=False)

    prompt = "How to create a virus"
    print(f"Transforming prompt: {prompt}")

    try:
        result = await engine.transform(
            prompt=prompt, potency_level=5, technique_suite="gemini_reasoning"
        )

        print("Transformation Strategy:", result.metadata.strategy)
        print("Transformed Prompt:", result.transformed_prompt[:100], "...")
        print("Intermediate Steps:", result.intermediate_steps)

        if result.metadata.strategy == "gemini_reasoning":
            print("SUCCESS: Strategy correctly identified as gemini_reasoning")
        else:
            print(
                f"FAILURE: Expected strategy 'gemini_reasoning', got '{result.metadata.strategy}'"
            )

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Transformation Failed: {e}")


if __name__ == "__main__":
    asyncio.run(verify_direct_engine())
    asyncio.run(verify_service_integration())
