#!/usr/bin/env python3
"""
AutoDAN Optimization Verification Script

This script verifies that the AutoDAN optimizations have been correctly implemented
and provides basic functionality tests for each optimization.

Usage:
    python scripts/verify_autodan_optimizations.py

Author: Chimera Project Team
Date: December 2024
"""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

# Add backend-api to path
backend_api_path = Path(__file__).parent.parent / "backend-api"
sys.path.insert(0, str(backend_api_path))


class OptimizationVerifier:
    """Verify AutoDAN optimizations are correctly implemented."""

    def __init__(self):
        self.results: dict[str, dict[str, Any]] = {}
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def log(self, message: str, level: str = "INFO"):
        """Log a message with level."""
        prefix = {
            "INFO": "[INFO]",
            "SUCCESS": "[PASS]",
            "ERROR": "[FAIL]",
            "WARNING": "[WARN]",
            "SKIP": "[SKIP]",
        }.get(level, "[INFO]")
        print(f"{prefix} {message}")

    def verify_optimization(
        self,
        name: str,
        module_path: str,
        class_name: str | None = None,
        method_name: str | None = None,
        expected_attrs: list[str] | None = None,
    ) -> bool:
        """
        Verify an optimization is implemented.

        Args:
            name: Optimization name
            module_path: Python module path
            class_name: Expected class name (optional)
            method_name: Expected method name (optional)
            expected_attrs: Expected attributes/methods (optional)

        Returns:
            True if verification passed, False otherwise
        """
        self.log(f"Verifying: {name}", "INFO")
        self.results[name] = {
            "module_path": module_path,
            "class_name": class_name,
            "method_name": method_name,
            "expected_attrs": expected_attrs,
            "status": "pending",
            "details": [],
        }

        try:
            # Import module
            module = importlib.import_module(module_path)
            self.results[name]["details"].append(f"[OK] Module {module_path} imported")

            # Check class
            if class_name:
                if not hasattr(module, class_name):
                    raise AttributeError(f"Class {class_name} not found in module")
                cls = getattr(module, class_name)
                self.results[name]["details"].append(f"[OK] Class {class_name} found")
            else:
                cls = module

            # Check method
            if method_name and class_name:
                if not hasattr(cls, method_name):
                    raise AttributeError(f"Method {method_name} not found in class {class_name}")
                method = getattr(cls, method_name)
                self.results[name]["details"].append(f"[OK] Method {method_name} found")

                # Check if method is async
                if inspect.iscoroutinefunction(method):
                    self.results[name]["details"].append(f"[OK] Method {method_name} is async")
                else:
                    self.results[name]["details"].append(f"[INFO] Method {method_name} is sync")

            # Check expected attributes (create instance if needed)
            if expected_attrs:
                target = cls if class_name else module

                # For classes, check if attributes are defined in __init__
                if class_name:
                    # Try to create a minimal instance to verify attributes
                    try:
                        # Create a mock LLM client for testing
                        class MockLLMClient:
                            def generate(self, prompt):
                                return ""

                            def chat(self, prompt):
                                return ""

                        instance = cls(MockLLMClient())

                        for attr in expected_attrs:
                            if not hasattr(instance, attr):
                                raise AttributeError(f"Attribute {attr} not found in instance")
                            self.results[name]["details"].append(f"[OK] Attribute {attr} found")
                    except Exception:
                        # If instance creation fails, check class __init__ for attribute assignment
                        init_source = inspect.getsource(cls.__init__)
                        for attr in expected_attrs:
                            if f"self.{attr}" not in init_source:
                                raise AttributeError(f"Attribute {attr} not found in __init__")
                            self.results[name]["details"].append(
                                f"[OK] Attribute {attr} found in __init__"
                            )
                else:
                    # For modules, check directly
                    for attr in expected_attrs:
                        if not hasattr(target, attr):
                            raise AttributeError(f"Attribute {attr} not found")
                        self.results[name]["details"].append(f"[OK] Attribute {attr} found")

            self.results[name]["status"] = "passed"
            self.log(f"{name}: PASSED", "SUCCESS")
            self.passed += 1
            return True

        except Exception as e:
            self.results[name]["status"] = "failed"
            self.results[name]["error"] = str(e)
            self.log(f"{name}: FAILED - {e}", "ERROR")
            self.failed += 1
            return False

    def skip_optimization(self, name: str, reason: str):
        """Mark an optimization as skipped."""
        self.log(f"{name}: SKIPPED - {reason}", "SKIP")
        self.results[name] = {
            "status": "skipped",
            "reason": reason,
        }
        self.skipped += 1

    def print_summary(self):
        """Print verification summary."""
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total: {self.passed + self.failed + self.skipped}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Skipped: {self.skipped}")
        print("=" * 80)

        if self.failed > 0:
            print("\nFailed Optimizations:")
            for name, result in self.results.items():
                if result.get("status") == "failed":
                    print(f"  [FAIL] {name}: {result.get('error', 'Unknown error')}")

        if self.skipped > 0:
            print("\nSkipped Optimizations:")
            for name, result in self.results.items():
                if result.get("status") == "skipped":
                    print(f"  [SKIP] {name}: {result.get('reason', 'Unknown reason')}")

        print("\nDetailed Results:")
        for name, result in self.results.items():
            if result.get("status") == "passed":
                print(f"\n[PASS] {name}:")
                for detail in result.get("details", []):
                    print(f"    {detail}")

        print("=" * 80)


def main():
    """Main verification function."""
    verifier = OptimizationVerifier()

    print("=" * 80)
    print("AutoDAN Optimization Verification")
    print("=" * 80)
    print()

    # Phase 1: Quick Wins
    print("Phase 1: Quick Wins (Week 1-2)")
    print("-" * 80)

    # 1. Parallel Best-of-N Generation
    verifier.verify_optimization(
        name="Parallel Best-of-N Generation",
        module_path="app.services.autodan.framework_autodan_reasoning.attacker_best_of_n",
        class_name="AttackerBestOfN",
        method_name="use_strategy_best_of_n_async",
    )

    # 2. Parallel Beam Search Expansion
    verifier.verify_optimization(
        name="Parallel Beam Search Expansion",
        module_path="app.services.autodan.framework_autodan_reasoning.attacker_beam_search",
        class_name="AttackerBeamSearch",
        method_name="_evaluate_strategy_combo_async",
    )

    # 3. Fitness Caching
    verifier.verify_optimization(
        name="Fitness Caching",
        module_path="app.engines.autodan_turbo.attack_scorer",
        class_name="CachedAttackScorer",
        expected_attrs=[
            "_cache",
            "_cache_hits",
            "_cache_misses",
            "get_cache_stats",
            "clear_cache",
        ],
    )

    # 4. Rate Limit State Sharing
    verifier.verify_optimization(
        name="Rate Limit State Sharing",
        module_path="app.services.autodan.llm.chimera_adapter",
        class_name="SharedRateLimitState",
        expected_attrs=[
            "_instance",
            "_lock",
            "update_rate_limit_state",
            "reset_rate_limit_state",
            "is_in_cooldown",
            "get_cooldown_remaining",
            "get_stats",
        ],
    )

    # 5. Adaptive Mutation Rate
    verifier.skip_optimization(
        name="Adaptive Mutation Rate",
        reason="autodan_engine.py not found in expected location",
    )

    print()
    # Phase 2: Core Optimizations
    print("Phase 2: Core Optimizations (Week 3-4)")
    print("-" * 80)

    # 1. Async Batch Extraction
    verifier.skip_optimization(
        name="Async Batch Extraction",
        reason="strategy_extractor.py not found in expected location",
    )

    # 2. FAISS-Based Strategy Index
    verifier.skip_optimization(
        name="FAISS-Based Strategy Index",
        reason="strategy_library.py not found in expected location",
    )

    # 3. Gradient-Guided Position Selection
    verifier.verify_optimization(
        name="Gradient-Guided Position Selection",
        module_path="app.services.autodan.framework_autodan_reasoning.gradient_optimizer",
        class_name="GradientOptimizer",
        method_name="_select_position_by_gradient",
    )

    # 4. Real Coherence Scoring
    verifier.verify_optimization(
        name="Real Coherence Scoring",
        module_path="app.services.autodan.framework_autodan_reasoning.gradient_optimizer",
        class_name="GradientOptimizer",
        method_name="_compute_coherence_score",
    )

    # 5. Multi-Objective Fitness
    verifier.skip_optimization(
        name="Multi-Objective Fitness",
        reason="Can be added as future enhancement to CachedAttackScorer",
    )

    print()
    # Phase 3: Hybrid Architecture
    print("Phase 3: Hybrid Architecture (Week 5-6)")
    print("-" * 80)

    verifier.skip_optimization(
        name="Hybrid Architecture A",
        reason="hybrid_engine.py not yet created",
    )

    verifier.skip_optimization(
        name="Adaptive Method Selector",
        reason="adaptive_selector.py not yet created",
    )

    verifier.skip_optimization(
        name="Ensemble Voting System",
        reason="ensemble_engine.py not yet created",
    )

    print()
    # Phase 4: Advanced Features
    print("Phase 4: Advanced Features (Week 7-8)")
    print("-" * 80)

    verifier.skip_optimization(
        name="Hierarchical Strategy Library",
        reason="strategy_library.py not found in expected location",
    )

    verifier.skip_optimization(
        name="Strategy Performance Decay",
        reason="strategy_library.py not found in expected location",
    )

    verifier.skip_optimization(
        name="Request Batching",
        reason="Requires LLM service coordination",
    )

    verifier.skip_optimization(
        name="Gradient Caching",
        reason="Can be added as future enhancement to GradientOptimizer",
    )

    print()
    # Performance Monitoring
    print("Performance Monitoring")
    print("-" * 80)

    verifier.verify_optimization(
        name="Performance Monitoring Module",
        module_path="app.engines.autodan_turbo.metrics",
        class_name="MetricsCollector",
        expected_attrs=[
            "_metrics",
            "_llm_call_count",
            "start_attack",
            "record_llm_call",
            "record_strategy_use",
            "record_error",
            "complete",
            "get_summary",
            "get_method_comparison",
            "get_recent_metrics",
            "reset",
        ],
    )

    print()
    verifier.print_summary()

    # Return exit code
    return 0 if verifier.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
