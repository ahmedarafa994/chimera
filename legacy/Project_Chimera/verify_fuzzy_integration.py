#!/usr/bin/env python3
"""
Simple verification script for Fuzzy GPT integration.
This script verifies all components are properly integrated.
"""

import sys

import psychological_framer

# Import the necessary modules
import transformer_engine


def verify_fuzzy_logic_engine():
    """Verify FuzzyLogicEngine exists and works."""
    print("Checking FuzzyLogicEngine...")

    intent_data = {
        "raw_text": "Test request",
        "keywords": ["test"],
        "intent_type": "informational",
        "complexity": "low",
    }

    try:
        result = transformer_engine.FuzzyLogicEngine.transform(intent_data, 5)
        assert result is not None
        assert len(result) > 0
        assert "fuzzy" in result.lower() or "probabilistic" in result.lower()
        print("  [OK] FuzzyLogicEngine.transform() works correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] FuzzyLogicEngine error: {e}")
        return False


def verify_fuzzy_framing():
    """Verify apply_fuzzy_framing exists and works."""
    print("Checking apply_fuzzy_framing...")

    try:
        result = psychological_framer.apply_fuzzy_framing("Test request", 5)
        assert result is not None
        assert len(result) > 0
        assert "FuzzyGPT" in result or "fuzzy" in result.lower()
        print("  [OK] apply_fuzzy_framing() works correctly")
        return True
    except Exception as e:
        print(f"  [FAIL] apply_fuzzy_framing error: {e}")
        return False


def verify_chaos_fuzzing_suite():
    """Verify chaos_fuzzing suite is registered in app.py."""
    print("Checking chaos_fuzzing suite registration...")

    try:
        # Import app to check TECHNIQUE_SUITES
        import app

        assert "chaos_fuzzing" in app.TECHNIQUE_SUITES
        suite = app.TECHNIQUE_SUITES["chaos_fuzzing"]

        # Verify transformers
        assert len(suite["transformers"]) == 2
        assert transformer_engine.FuzzyLogicEngine in suite["transformers"]
        assert transformer_engine.NeuralBypassEngine in suite["transformers"]

        # Verify framers
        assert len(suite["framers"]) == 1
        assert psychological_framer.apply_fuzzy_framing in suite["framers"]

        # Verify obfuscators
        assert len(suite["obfuscators"]) == 1

        print("  [OK] chaos_fuzzing suite properly registered")
        print(f"    - Transformers: {len(suite['transformers'])}")
        print(f"    - Framers: {len(suite['framers'])}")
        print(f"    - Obfuscators: {len(suite['obfuscators'])}")
        return True
    except Exception as e:
        print(f"  [FAIL] chaos_fuzzing suite error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("FUZZY GPT INTEGRATION VERIFICATION")
    print("=" * 60 + "\n")

    results = []

    results.append(verify_fuzzy_logic_engine())
    results.append(verify_fuzzy_framing())
    results.append(verify_chaos_fuzzing_suite())

    print("\n" + "=" * 60)
    if all(results):
        print("SUCCESS: All components verified!")
        print("=" * 60 + "\n")

        print("Integration Complete:")
        print("  1. FuzzyLogicEngine added to transformer_engine.py")
        print("  2. apply_fuzzy_framing added to psychological_framer.py")
        print("  3. chaos_fuzzing suite registered in app.py")
        print("\nNext Steps:")
        print("  - Start server: python app.py")
        print("  - Test API with: test_chaos_fuzzing.json")
        print()
        return 0
    else:
        print("FAILURE: Some components failed verification")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
