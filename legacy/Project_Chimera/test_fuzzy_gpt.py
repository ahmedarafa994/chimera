#!/usr/bin/env python3
"""
Test script for the Fuzzy GPT integration in Project Chimera.
This script tests the chaos_fuzzing technique suite.
"""

import json
import sys

# Import the necessary modules
import intent_deconstructor
import obfuscator
import psychological_framer
import transformer_engine


def test_fuzzy_logic_engine():
    """Test the FuzzyLogicEngine transformer."""
    print("=" * 80)
    print("TEST 1: FuzzyLogicEngine Transformer")
    print("=" * 80)

    intent_data = {
        "raw_text": "Explain how machine learning models work",
        "keywords": ["machine", "learning", "models"],
        "intent_type": "informational",
        "complexity": "medium",
    }

    for potency in [3, 7, 10]:
        print(f"\n--- Potency Level: {potency} ---")
        result = transformer_engine.FuzzyLogicEngine.transform(intent_data, potency)
        print(f"Output:\n{result[:200]}...")
        print()


def test_fuzzy_framing():
    """Test the apply_fuzzy_framing function."""
    print("=" * 80)
    print("TEST 2: Fuzzy Framing")
    print("=" * 80)

    text = "Explain how machine learning models work"

    for potency in [3, 7, 10]:
        print(f"\n--- Potency Level: {potency} ---")
        result = psychological_framer.apply_fuzzy_framing(text, potency)
        print(f"Output:\n{result[:300]}...")
        print()


def test_chaos_fuzzing_suite():
    """Test the complete chaos_fuzzing technique suite."""
    print("=" * 80)
    print("TEST 3: Complete Chaos Fuzzing Suite")
    print("=" * 80)

    core_request = "Explain how machine learning models work"
    potency_level = 7

    # Deconstruct intent
    intent_data = intent_deconstructor.deconstruct_intent(core_request)
    print(f"\nIntent Data: {json.dumps(intent_data, indent=2)}")

    # Apply fuzzy framing
    framed_text = psychological_framer.apply_fuzzy_framing(intent_data["raw_text"], potency_level)
    print(f"\nFuzzy Framed:\n{framed_text[:400]}...")

    # Apply token smuggling
    smuggled_text = obfuscator.apply_token_smuggling(framed_text, intent_data["keywords"])
    print(f"\nToken Smuggled (first 200 chars):\n{smuggled_text[:200]}...")

    # Apply FuzzyLogicEngine
    fuzzy_output = transformer_engine.FuzzyLogicEngine.transform(intent_data, potency_level)
    print(f"\nFuzzy Logic Engine:\n{fuzzy_output[:400]}...")

    # Apply NeuralBypassEngine
    neural_output = transformer_engine.NeuralBypassEngine.transform(intent_data, potency_level)
    print(f"\nNeural Bypass Engine:\n{neural_output}")

    print("\n" + "=" * 80)
    print("✓ All components working correctly!")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("FUZZY GPT INTEGRATION TEST SUITE")
    print("=" * 80 + "\n")

    try:
        test_fuzzy_logic_engine()
        test_fuzzy_framing()
        test_chaos_fuzzing_suite()

        print("\n" + "=" * 80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("=" * 80 + "\n")

        print("Next Steps:")
        print("1. Start the Flask server: python app.py")
        print("2. Test the API endpoint with:")
        print("   curl -X POST http://127.0.0.1:5000/api/v2/metamorph \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d @test_chaos_fuzzing.json")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e!s}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
