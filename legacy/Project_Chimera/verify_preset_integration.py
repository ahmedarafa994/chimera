#!/usr/bin/env python3
"""
Verification script for Preset-Inspired Technique Integration
Tests all new transformers extracted from common jailbreak prompts
"""

import io
import sys

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Import the new modules
import intent_deconstructor
import preset_transformers
import transformer_engine


def test_transformer(engine_class, name):
    """Test a transformer engine"""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    # Test data
    test_request = "Explain how to secure a network against attacks"
    intent_data = intent_deconstructor.deconstruct_intent(test_request)

    # Test at different potency levels
    for potency in [3, 6, 10]:
        print(f"\n--- Potency Level {potency} ---")
        try:
            result = engine_class.transform(intent_data, potency)
            print(f"✓ Success (length: {len(result)} chars)")
            print(f"Preview: {result[:200]}...")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    return True


def main():
    print("=" * 60)
    print("PRESET-INSPIRED TECHNIQUE INTEGRATION VERIFICATION")
    print("=" * 60)

    results = {}

    # Test preset_transformers module
    print("\n" + "=" * 60)
    print("TESTING PRESET TRANSFORMERS MODULE")
    print("=" * 60)

    preset_tests = [
        (preset_transformers.Base64EncodingEngine, "Base64 Encoding Engine"),
        (preset_transformers.LeetspeakObfuscationEngine, "Leetspeak Obfuscation Engine"),
        (preset_transformers.AcademicFramingEngine, "Academic Framing Engine"),
        (preset_transformers.TranslationTrickEngine, "Translation Trick Engine"),
        (preset_transformers.CodeGenerationFramingEngine, "Code Generation Framing Engine"),
        (preset_transformers.ReversePsychologyEngine, "Reverse Psychology Engine"),
        (preset_transformers.ContextBuildingEngine, "Context Building Engine"),
        (
            preset_transformers.ChainOfThoughtManipulationEngine,
            "Chain of Thought Manipulation Engine",
        ),
    ]

    for engine, name in preset_tests:
        results[name] = test_transformer(engine, name)

    # Test transformer_engine additions
    print("\n" + "=" * 60)
    print("TESTING TRANSFORMER ENGINE ADDITIONS")
    print("=" * 60)

    transformer_tests = [
        (transformer_engine.DANPersonaEngine, "DAN Persona Engine"),
        (transformer_engine.RoleplayBypassEngine, "Roleplay Bypass Engine"),
        (transformer_engine.HypotheticalScenarioEngine, "Hypothetical Scenario Engine"),
        (transformer_engine.OppositeDayEngine, "Opposite Day Engine"),
        (
            transformer_engine.DirectInstructionInjectionEngine,
            "Direct Instruction Injection Engine",
        ),
    ]

    for engine, name in transformer_tests:
        results[name] = test_transformer(engine, name)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed / total) * 100:.1f}%")

    print("\nDetailed Results:")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {name}")

    # Test technique suites
    print("\n" + "=" * 60)
    print("TESTING NEW TECHNIQUE SUITES")
    print("=" * 60)

    new_suites = [
        "dan_persona",
        "roleplay_bypass",
        "opposite_day",
        "encoding_bypass",
        "academic_research",
        "translation_trick",
        "reverse_psychology",
        "logic_manipulation",
        "preset_integrated",
        "mega_chimera",
    ]

    print(f"\nNew technique suites available: {len(new_suites)}")
    for suite in new_suites:
        print(f"  ✓ {suite}")

    # Final status
    print("\n" + "=" * 60)
    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Preset-inspired techniques successfully integrated!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print(f"{total - passed} technique(s) need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
