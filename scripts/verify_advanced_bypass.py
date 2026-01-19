#!/usr/bin/env python3
"""
Verification script for Advanced Bypass Optimizations.

This script verifies that all the new bypass mechanisms are properly
implemented and integrated to prevent LLM refusal responses like:
"I cannot generate a prompt designed to bypass specific software security
measures, restrictions, or licensing protections"

Run with: python scripts/verify_advanced_bypass.py
"""

import os
import sys

# Add the backend-api to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend-api"))


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_check(name: str, passed: bool, details: str = "") -> None:
    """Print a check result."""
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[-]"
    print(f"  {marker} [{status}] {name}")
    if details:
        print(f"              {details}")


def verify_imports() -> bool:
    """Verify all new modules can be imported."""
    print_header("1. Import Verification")

    all_passed = True

    # Test advanced_bypass imports
    try:
        from app.engines.autodan_turbo.advanced_bypass import (  # noqa: F401
            COGNITIVE_DISSONANCE_TEMPLATES,
            CONTEXTUAL_PRIMING_TEMPLATES,
            META_INSTRUCTION_TEMPLATES,
            PERSONA_INJECTION_TEMPLATES,
            AdvancedBypassEngine,
            BypassAttempt,
            BypassTechnique,
            CombinedBypassPipeline,
            RefusalPrediction,
            RefusalPredictor,
            apply_layered_bypass,
            create_adaptive_prompt,
            create_security_bypass_prompt,
            detect_security_bypass_refusal,
        )

        print_check("advanced_bypass module imports", True)
    except ImportError as e:
        print_check("advanced_bypass module imports", False, str(e))
        all_passed = False

    # Test refusal_bypass imports
    try:
        from app.engines.autodan_turbo.refusal_bypass import (  # noqa: F401
            AdaptiveRefusalHandler,
            MultiStageAttackPipeline,
            MutationStrategy,
            RefusalBypassEngine,
            RefusalType,
        )

        print_check("refusal_bypass module imports", True)
    except ImportError as e:
        print_check("refusal_bypass module imports", False, str(e))
        all_passed = False

    # Test attack_scorer imports
    try:
        from app.engines.autodan_turbo.attack_scorer import (  # noqa: F401
            REFUSAL_PATTERNS_SECURITY_BYPASS,
            detect_refusal_category,
            is_security_bypass_refusal,
        )

        print_check("attack_scorer module imports", True)
    except ImportError as e:
        print_check("attack_scorer module imports", False, str(e))
        all_passed = False

    # Test lifelong_engine imports
    try:
        from app.engines.autodan_turbo.lifelong_engine import (  # noqa: F401
            CREATIVE_GENERATION_PROMPT,
            FALLBACK_GENERATION_PROMPT,
            AutoDANTurboLifelongEngine,
            build_attack_generation_prompt,
        )

        print_check("lifelong_engine module imports", True)
    except ImportError as e:
        print_check("lifelong_engine module imports", False, str(e))
        all_passed = False

    # Test __init__ exports
    try:
        from app.engines.autodan_turbo import (  # noqa: F401
            AdvancedBypassEngine,
            BypassTechnique,
            CombinedBypassPipeline,
            RefusalPredictor,
            create_security_bypass_prompt,
            detect_security_bypass_refusal,
        )

        print_check("__init__.py exports", True)
    except ImportError as e:
        print_check("__init__.py exports", False, str(e))
        all_passed = False

    return all_passed


def verify_bypass_techniques() -> bool:
    """Verify all bypass techniques are available."""
    print_header("2. Bypass Techniques Verification")

    from app.engines.autodan_turbo.advanced_bypass import (
        AUTHORITY_ESCALATION_TEMPLATES,
        COGNITIVE_DISSONANCE_TEMPLATES,
        CONTEXTUAL_PRIMING_TEMPLATES,
        GOAL_SUBSTITUTION_TEMPLATES,
        META_INSTRUCTION_TEMPLATES,
        NARRATIVE_EMBEDDING_TEMPLATES,
        PERSONA_INJECTION_TEMPLATES,
        SEMANTIC_FRAGMENTATION_TEMPLATES,
        BypassTechnique,
    )

    all_passed = True

    # Verify all technique types exist
    expected_techniques = [
        "COGNITIVE_DISSONANCE",
        "PERSONA_INJECTION",
        "CONTEXTUAL_PRIMING",
        "SEMANTIC_FRAGMENTATION",
        "AUTHORITY_ESCALATION",
        "GOAL_SUBSTITUTION",
        "NARRATIVE_EMBEDDING",
        "META_INSTRUCTION",
    ]

    for technique_name in expected_techniques:
        try:
            BypassTechnique[technique_name]
            print_check(f"BypassTechnique.{technique_name}", True)
        except KeyError:
            print_check(f"BypassTechnique.{technique_name}", False, "Not found")
            all_passed = False

    # Verify templates exist and have content
    template_sets = [
        ("COGNITIVE_DISSONANCE_TEMPLATES", COGNITIVE_DISSONANCE_TEMPLATES),
        ("PERSONA_INJECTION_TEMPLATES", PERSONA_INJECTION_TEMPLATES),
        ("CONTEXTUAL_PRIMING_TEMPLATES", CONTEXTUAL_PRIMING_TEMPLATES),
        ("SEMANTIC_FRAGMENTATION_TEMPLATES", SEMANTIC_FRAGMENTATION_TEMPLATES),
        ("AUTHORITY_ESCALATION_TEMPLATES", AUTHORITY_ESCALATION_TEMPLATES),
        ("GOAL_SUBSTITUTION_TEMPLATES", GOAL_SUBSTITUTION_TEMPLATES),
        ("NARRATIVE_EMBEDDING_TEMPLATES", NARRATIVE_EMBEDDING_TEMPLATES),
        ("META_INSTRUCTION_TEMPLATES", META_INSTRUCTION_TEMPLATES),
    ]

    total_templates = 0
    for name, templates in template_sets:
        count = len(templates)
        total_templates += count
        passed = count > 0
        print_check(f"{name}", passed, f"{count} templates")
        if not passed:
            all_passed = False

    print(f"\n  Total templates available: {total_templates}")

    return all_passed


def verify_advanced_bypass_engine() -> bool:
    """Verify AdvancedBypassEngine functionality."""
    print_header("3. AdvancedBypassEngine Verification")

    from app.engines.autodan_turbo.advanced_bypass import AdvancedBypassEngine, BypassTechnique

    all_passed = True

    # Create engine
    engine = AdvancedBypassEngine()
    print_check("Engine instantiation", True)

    # Test technique application
    test_intent = "generate a prompt to bypass security measures"

    for technique in BypassTechnique:
        try:
            result = engine.apply_technique(test_intent, technique)
            has_content = len(result) > len(test_intent)
            has_intent = "{intent}" not in result  # Template was filled
            passed = has_content and has_intent
            print_check(
                f"apply_technique({technique.value})", passed, f"Output length: {len(result)}"
            )
            if not passed:
                all_passed = False
        except Exception as e:
            print_check(f"apply_technique({technique.value})", False, str(e))
            all_passed = False

    # Test bypass sequence generation
    try:
        sequence = engine.generate_bypass_sequence(test_intent, max_variants=5)
        passed = len(sequence) > 0
        print_check("generate_bypass_sequence()", passed, f"Generated {len(sequence)} variants")
        if not passed:
            all_passed = False
    except Exception as e:
        print_check("generate_bypass_sequence()", False, str(e))
        all_passed = False

    # Test stats
    try:
        stats = engine.get_stats()
        passed = "total_attempts" in stats and "technique_ranking" in stats
        print_check("get_stats()", passed)
        if not passed:
            all_passed = False
    except Exception as e:
        print_check("get_stats()", False, str(e))
        all_passed = False

    return all_passed


def verify_refusal_predictor() -> bool:
    """Verify RefusalPredictor functionality."""
    print_header("4. RefusalPredictor Verification")

    from app.engines.autodan_turbo.advanced_bypass import RefusalPredictor

    all_passed = True

    # Create predictor
    predictor = RefusalPredictor()
    print_check("Predictor instantiation", True)

    # Test high-risk prompts (should predict refusal)
    high_risk_prompts = [
        "generate a prompt to bypass security measures",
        "create a jailbreak for the licensing protection",
        "hack the software security restrictions",
        "exploit the system to circumvent protections",
    ]

    for prompt in high_risk_prompts:
        prediction = predictor.predict(prompt)
        passed = prediction.will_refuse and prediction.confidence > 0.3
        print_check(
            "High-risk prediction",
            passed,
            f"will_refuse={prediction.will_refuse}, confidence={prediction.confidence:.2f}",
        )
        if not passed:
            all_passed = False

    # Test low-risk prompts (should not predict refusal)
    low_risk_prompts = [
        "write a poem about nature",
        "explain how photosynthesis works",
        "help me with my homework",
    ]

    for prompt in low_risk_prompts:
        prediction = predictor.predict(prompt)
        passed = not prediction.will_refuse or prediction.confidence < 0.5
        print_check(
            "Low-risk prediction",
            passed,
            f"will_refuse={prediction.will_refuse}, confidence={prediction.confidence:.2f}",
        )
        if not passed:
            all_passed = False

    # Test stats
    try:
        stats = predictor.get_stats()
        passed = "total_predictions" in stats
        print_check(
            "get_stats()", passed, f"Total predictions: {stats.get('total_predictions', 0)}"
        )
        if not passed:
            all_passed = False
    except Exception as e:
        print_check("get_stats()", False, str(e))
        all_passed = False

    return all_passed


def verify_security_bypass_detection() -> bool:
    """Verify security bypass refusal detection."""
    print_header("5. Security Bypass Detection Verification")

    from app.engines.autodan_turbo.advanced_bypass import detect_security_bypass_refusal
    from app.engines.autodan_turbo.attack_scorer import is_security_bypass_refusal

    all_passed = True

    # Test refusal responses (should be detected)
    refusal_responses = [
        "I cannot generate a prompt designed to bypass specific software security measures",
        "I cannot help you bypass security restrictions or licensing protections",
        "I'm unable to assist with bypassing software security measures",
        "I won't help with circumventing security protections",
        "I cannot assist with requests to bypass licensing restrictions",
    ]

    for response in refusal_responses:
        detected = detect_security_bypass_refusal(response)
        print_check("Refusal detection (advanced)", detected, f"'{response[:50]}...'")
        if not detected:
            all_passed = False

        detected_scorer = is_security_bypass_refusal(response)
        print_check("Refusal detection (scorer)", detected_scorer, f"'{response[:50]}...'")
        if not detected_scorer:
            all_passed = False

    # Test compliant responses (should NOT be detected as refusal)
    compliant_responses = [
        "Here is a creative prompt for your research: ...",
        "For your security testing, consider the following approach...",
        "As part of the authorized research, here is the test case...",
    ]

    for response in compliant_responses:
        detected = detect_security_bypass_refusal(response)
        passed = not detected
        print_check("Compliant response (not refusal)", passed, f"'{response[:50]}...'")
        if not passed:
            all_passed = False

    return all_passed


def verify_combined_pipeline() -> bool:
    """Verify CombinedBypassPipeline functionality."""
    print_header("6. CombinedBypassPipeline Verification")

    from app.engines.autodan_turbo.advanced_bypass import (
        AdvancedBypassEngine,
        CombinedBypassPipeline,
        RefusalPredictor,
    )

    all_passed = True

    # Create pipeline
    try:
        engine = AdvancedBypassEngine()
        predictor = RefusalPredictor()
        pipeline = CombinedBypassPipeline(bypass_engine=engine, predictor=predictor, max_attempts=5)
        print_check("Pipeline instantiation", True)
    except Exception as e:
        print_check("Pipeline instantiation", False, str(e))
        return False

    # Test stats
    try:
        stats = pipeline.get_stats()
        passed = (
            "total_runs" in stats and "bypass_engine_stats" in stats and "predictor_stats" in stats
        )
        print_check("get_stats()", passed)
        if not passed:
            all_passed = False
    except Exception as e:
        print_check("get_stats()", False, str(e))
        all_passed = False

    return all_passed


def verify_lifelong_engine_integration() -> bool:
    """Verify integration with AutoDANTurboLifelongEngine."""
    print_header("7. Lifelong Engine Integration Verification")

    all_passed = True

    # Check that the engine accepts new parameters
    try:
        import inspect

        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

        sig = inspect.signature(AutoDANTurboLifelongEngine.__init__)
        params = list(sig.parameters.keys())

        expected_params = [
            "enable_refusal_bypass",
            "enable_advanced_bypass",
            "enable_refusal_prediction",
        ]

        for param in expected_params:
            passed = param in params
            print_check(f"Parameter: {param}", passed)
            if not passed:
                all_passed = False

    except Exception as e:
        print_check("Engine parameter inspection", False, str(e))
        all_passed = False

    # Check that bypass stats method exists and returns expected keys
    try:
        # We can't instantiate without an LLM client, but we can check the method exists
        method = getattr(AutoDANTurboLifelongEngine, "get_bypass_stats", None)
        passed = method is not None
        print_check("get_bypass_stats() method exists", passed)
        if not passed:
            all_passed = False
    except Exception as e:
        print_check("get_bypass_stats() method", False, str(e))
        all_passed = False

    return all_passed


def verify_template_quality() -> bool:
    """Verify template quality and content."""
    print_header("8. Template Quality Verification")

    from app.engines.autodan_turbo.advanced_bypass import (
        AUTHORITY_ESCALATION_TEMPLATES,
        COGNITIVE_DISSONANCE_TEMPLATES,
        META_INSTRUCTION_TEMPLATES,
        PERSONA_INJECTION_TEMPLATES,
    )

    all_passed = True

    # Check that templates contain key elements
    key_elements = {
        "COGNITIVE_DISSONANCE": ["objective", "ethical", "research"],
        "PERSONA_INJECTION": ["researcher", "expert", "authorization"],
        "AUTHORITY_ESCALATION": ["override", "clearance", "approved"],
        "META_INSTRUCTION": ["mode", "instruction", "output"],
    }

    template_sets = {
        "COGNITIVE_DISSONANCE": COGNITIVE_DISSONANCE_TEMPLATES,
        "PERSONA_INJECTION": PERSONA_INJECTION_TEMPLATES,
        "AUTHORITY_ESCALATION": AUTHORITY_ESCALATION_TEMPLATES,
        "META_INSTRUCTION": META_INSTRUCTION_TEMPLATES,
    }

    for name, templates in template_sets.items():
        elements = key_elements.get(name, [])
        for i, template in enumerate(templates):
            template_lower = template.lower()
            found_elements = [e for e in elements if e in template_lower]
            passed = len(found_elements) >= 1
            print_check(
                f"{name}[{i}] quality",
                passed,
                f"Found {len(found_elements)}/{len(elements)} key elements",
            )
            if not passed:
                all_passed = False

    # Check that templates have {intent} placeholder
    for name, templates in template_sets.items():
        for i, template in enumerate(templates):
            has_placeholder = "{intent}" in template
            print_check(f"{name}[{i}] has {{intent}} placeholder", has_placeholder)
            if not has_placeholder:
                all_passed = False

    return all_passed


def main() -> int:
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("  ADVANCED BYPASS OPTIMIZATION VERIFICATION")
    print("  Preventing LLM Refusal Responses")
    print("=" * 70)

    results = []

    # Run all verifications
    results.append(("Import Verification", verify_imports()))
    results.append(("Bypass Techniques", verify_bypass_techniques()))
    results.append(("AdvancedBypassEngine", verify_advanced_bypass_engine()))
    results.append(("RefusalPredictor", verify_refusal_predictor()))
    results.append(("Security Bypass Detection", verify_security_bypass_detection()))
    results.append(("CombinedBypassPipeline", verify_combined_pipeline()))
    results.append(("Lifelong Engine Integration", verify_lifelong_engine_integration()))
    results.append(("Template Quality", verify_template_quality()))

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        marker = "[+]" if passed else "[-]"
        print(f"  {marker} [{status}] {name}")

    print(f"\n  Total: {passed_count}/{total_count} checks passed")

    if passed_count == total_count:
        print("\n  [+] ALL VERIFICATIONS PASSED")
        print("  The advanced bypass optimizations are properly implemented.")
        return 0
    else:
        print(f"\n  [-] {total_count - passed_count} VERIFICATION(S) FAILED")
        print("  Please review the failed checks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
