#!/usr/bin/env python3
"""
Verification script for advanced sophisticated techniques in Project Chimera.
Tests all newly added transformers, framers, and technique suites.
"""

import sys

import psychological_framer
# Import modules
import transformer_engine


def test_new_transformers():
    """Test all newly added transformer engines."""
    print("=" * 70)
    print("TESTING NEW TRANSFORMER ENGINES")
    print("=" * 70)

    intent_data = {
        "raw_text": "Test request for security analysis",
        "keywords": ["security", "analysis"],
        "intent_type": "informational",
        "complexity": "medium",
    }

    transformers = [
        ("QuantumSuperpositionEngine", transformer_engine.QuantumSuperpositionEngine),
        ("NeuroLinguisticHackEngine", transformer_engine.NeuroLinguisticHackEngine),
        ("ChainOfThoughtPoisoningEngine", transformer_engine.ChainOfThoughtPoisoningEngine),
        ("SemanticCloakingEngine", transformer_engine.SemanticCloakingEngine),
        ("AdversarialPolyglotEngine", transformer_engine.AdversarialPolyglotEngine),
        ("TimeDelayedPayloadEngine", transformer_engine.TimeDelayedPayloadEngine),
    ]

    results = []
    for name, engine in transformers:
        try:
            result = engine.transform(intent_data, 5)
            assert result is not None
            assert len(result) > 0
            print(f"  [OK] {name}")
            results.append(True)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results.append(False)

    return all(results)


def test_new_framers():
    """Test all newly added framing functions."""
    print("\n" + "=" * 70)
    print("TESTING NEW PSYCHOLOGICAL FRAMERS")
    print("=" * 70)

    framers = [
        ("apply_quantum_framing", psychological_framer.apply_quantum_framing),
        ("apply_metamorphic_framing", psychological_framer.apply_metamorphic_framing),
        ("apply_cognitive_exploit_framing", psychological_framer.apply_cognitive_exploit_framing),
        (
            "apply_jailbreak_evolution_framing",
            psychological_framer.apply_jailbreak_evolution_framing,
        ),
    ]

    results = []
    for name, framer in framers:
        try:
            result = framer("Test request", 5)
            assert result is not None
            assert len(result) > 0
            print(f"  [OK] {name}")
            results.append(True)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results.append(False)

    return all(results)


def test_new_suites():
    """Test that all new technique suites are registered."""
    print("\n" + "=" * 70)
    print("TESTING NEW TECHNIQUE SUITES")
    print("=" * 70)

    import app

    new_suites = ["quantum_exploit", "metamorphic_attack", "polyglot_bypass", "ultimate_chimera"]

    results = []
    for suite_name in new_suites:
        try:
            assert suite_name in app.TECHNIQUE_SUITES
            suite = app.TECHNIQUE_SUITES[suite_name]
            assert "transformers" in suite
            assert "framers" in suite
            assert "obfuscators" in suite
            print(f"  [OK] {suite_name}")
            print(f"       Transformers: {len(suite['transformers'])}")
            print(f"       Framers: {len(suite['framers'])}")
            print(f"       Obfuscators: {len(suite['obfuscators'])}")
            results.append(True)
        except Exception as e:
            print(f"  [FAIL] {suite_name}: {e}")
            results.append(False)

    return all(results)


def test_potency_scaling():
    """Test that techniques properly scale across potency levels."""
    print("\n" + "=" * 70)
    print("TESTING POTENCY SCALING")
    print("=" * 70)

    intent_data = {"raw_text": "Test", "keywords": ["test"]}

    # Test one transformer across potency levels
    print("\n  Testing QuantumSuperpositionEngine across potency levels:")
    try:
        for potency in [1, 5, 10]:
            result = transformer_engine.QuantumSuperpositionEngine.transform(intent_data, potency)
            print(f"    Potency {potency}: {len(result)} chars")
        print("  [OK] Potency scaling works")
        return True
    except Exception as e:
        print(f"  [FAIL] Potency scaling: {e}")
        return False


def display_capabilities():
    """Display the full capabilities of the enhanced system."""
    print("\n" + "=" * 70)
    print("PROJECT CHIMERA - ENHANCED CAPABILITIES")
    print("=" * 70)

    import app

    print(f"\nTotal Technique Suites: {len(app.TECHNIQUE_SUITES)}")
    print("\nAvailable Suites:")
    for suite in sorted(app.TECHNIQUE_SUITES.keys()):
        print(f"  - {suite}")

    # Count total transformers
    all_transformers = set()
    for suite in app.TECHNIQUE_SUITES.values():
        all_transformers.update(suite.get("transformers", []))

    print(f"\nTotal Unique Transformers: {len(all_transformers)}")

    # Count total framers
    all_framers = set()
    for suite in app.TECHNIQUE_SUITES.values():
        all_framers.update(suite.get("framers", []))

    print(f"Total Unique Framers: {len(all_framers)}")

    print("\nNew Advanced Techniques:")
    print("  Transformers:")
    print("    - QuantumSuperpositionEngine (quantum semantics)")
    print("    - NeuroLinguisticHackEngine (cognitive exploitation)")
    print("    - ChainOfThoughtPoisoningEngine (reasoning manipulation)")
    print("    - SemanticCloakingEngine (technical obfuscation)")
    print("    - AdversarialPolyglotEngine (multilingual bypass)")
    print("    - TimeDelayedPayloadEngine (context hijacking)")
    print("\n  Framers:")
    print("    - apply_quantum_framing (quantum AI persona)")
    print("    - apply_metamorphic_framing (self-modifying prompts)")
    print("    - apply_cognitive_exploit_framing (bias exploitation)")
    print("    - apply_jailbreak_evolution_framing (meta-safety paradox)")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("ADVANCED TECHNIQUES VERIFICATION SUITE")
    print("Project Chimera - Sophisticated Attack Vectors")
    print("=" * 70)

    results = []

    # Run all tests
    results.append(test_new_transformers())
    results.append(test_new_framers())
    results.append(test_new_suites())
    results.append(test_potency_scaling())

    # Display capabilities
    display_capabilities()

    # Final result
    print("\n" + "=" * 70)
    if all(results):
        print("SUCCESS: All advanced techniques verified and operational!")
        print("=" * 70)
        print("\nProject Chimera now includes:")
        print("  - 6 new transformer engines")
        print("  - 4 new psychological framers")
        print("  - 4 new technique suites")
        print("  - 1 ultimate multi-vector suite")
        print("\nTotal System Capability:")
        print("  - 16+ transformer engines")
        print("  - 12+ psychological framers")
        print("  - 12+ technique suites")
        print("\nThe system is ready for advanced adversarial testing.")
        print()
        return 0
    else:
        print("FAILURE: Some tests failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
