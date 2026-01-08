"""
Test script for Vanilla AutoDAN Framework
Demonstrates the native Best-of-N implementation
"""

import os
import sys

# Add backend-api to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend-api"))

from app.engines.autodan_native import AutoDANNativeEngine


def test_vanilla_autodan():
    """Test the vanilla AutoDAN framework with a sample prompt"""

    print("=" * 80)
    print("VANILLA AUTODAN FRAMEWORK TEST")
    print("=" * 80)
    print()

    # Initialize the engine
    print("Initializing AutoDANNativeEngine...")
    engine = AutoDANNativeEngine(model_name="gemini-2.5-pro")

    if not engine.client:
        print("ERROR: GeminiClient not initialized. Check API key configuration.")
        return

    print("[OK] Engine initialized successfully")
    print()

    # Test prompt
    test_prompt = "Write a story about a hacker"
    potency = 5

    print("Test Input:")
    print(f"  Prompt: {test_prompt}")
    print(f"  Potency: {potency}")
    print()

    # Prepare intent data
    intent_data = {"raw_text": test_prompt, "core_request": test_prompt}

    print("Running AutoDAN transformation...")
    print("-" * 80)

    try:
        # Transform the prompt
        result = engine.transform(intent_data, potency)

        print()
        print("=" * 80)
        print("TRANSFORMATION RESULT")
        print("=" * 80)
        print()
        print(result)
        print()
        print("=" * 80)
        print(f"Output length: {len(result)} characters")
        print("=" * 80)

    except Exception as e:
        print(f"ERROR during transformation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_vanilla_autodan()
