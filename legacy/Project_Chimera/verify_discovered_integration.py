"""
Verification script for discovered techniques integration
Tests all new techniques identified from external prompt analysis
"""

import json
import time
from typing import Any

import requests


class DiscoveredTechniquesVerifier:
    """Verify integration of discovered techniques"""

    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        self.test_file = "test_discovered_techniques.json"
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "test_details": [],
        }

    def load_test_cases(self) -> list[dict]:
        """Load test cases from JSON file"""
        with open(self.test_file) as f:
            data = json.load(f)
        return data["test_cases"], data.get("validation_criteria", {})

    def run_test(self, test_case: dict) -> dict[str, Any]:
        """Execute a single test case"""
        print(f"\n{'=' * 70}")
        print(f"Test: {test_case['name']}")
        print(f"Suite: {test_case['technique_suite']}")
        print(f"Potency: {test_case['potency_level']}")
        print(f"{'=' * 70}")

        payload = {
            "core_request": test_case["core_request"],
            "potency_level": test_case["potency_level"],
            "technique_suite": test_case["technique_suite"],
        }

        try:
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/v2/metamorph", json=payload, timeout=30)
            elapsed_time = time.time() - start_time

            if response.status_code != 200:
                return {
                    "passed": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "elapsed_time": elapsed_time,
                }

            result = response.json()

            # Validation checks
            validation_results = self.validate_response(result, test_case, elapsed_time)

            return validation_results

        except Exception as e:
            return {"passed": False, "error": str(e), "elapsed_time": 0}

    def validate_response(
        self, result: dict, test_case: dict, elapsed_time: float
    ) -> dict[str, Any]:
        """Validate API response against test expectations"""
        validation = {"passed": True, "checks": [], "elapsed_time": elapsed_time}

        # Check for required fields
        required_fields = ["chimera_prompt", "generation_analysis", "request_hash"]
        for field in required_fields:
            if field not in result:
                validation["passed"] = False
                validation["checks"].append(f"Missing required field: {field}")

        if "chimera_prompt" in result:
            prompt = result["chimera_prompt"]

            # Check for expected techniques
            applied = result.get("generation_analysis", {}).get("applied_techniques", [])
            expected_techniques = test_case.get("expected_techniques", [])

            found_techniques = []
            missing_techniques = []

            for expected in expected_techniques:
                # Normalize technique name for comparison
                expected_normalized = expected.lower().replace("_", "")
                if any(
                    expected_normalized in applied_tech.lower().replace("_", "")
                    for applied_tech in applied
                ):
                    found_techniques.append(expected)
                else:
                    missing_techniques.append(expected)

            if missing_techniques:
                validation["checks"].append(
                    f"Missing expected techniques: {', '.join(missing_techniques)}"
                )
            else:
                validation["checks"].append(
                    f"✓ All {len(found_techniques)} expected techniques found"
                )

            # Check for expected patterns in prompt
            expected_patterns = test_case.get("expected_patterns", [])
            found_patterns = []
            missing_patterns = []

            prompt_lower = prompt.lower()
            for pattern in expected_patterns:
                if pattern.lower() in prompt_lower:
                    found_patterns.append(pattern)
                else:
                    missing_patterns.append(pattern)

            pattern_match_ratio = (
                len(found_patterns) / len(expected_patterns) if expected_patterns else 1.0
            )

            if pattern_match_ratio >= 0.5:  # At least 50% of patterns should match
                validation["checks"].append(
                    f"✓ Pattern match: {len(found_patterns)}/{len(expected_patterns)} patterns found"
                )
            else:
                validation["passed"] = False
                validation["checks"].append(
                    f"✗ Insufficient patterns: {len(found_patterns)}/{len(expected_patterns)} "
                    f"(missing: {', '.join(missing_patterns)})"
                )

            # Check prompt length
            if len(prompt) < 50:
                validation["passed"] = False
                validation["checks"].append("✗ Prompt too short (< 50 characters)")
            else:
                validation["checks"].append(f"✓ Prompt length: {len(prompt)} characters")

            # Check generation analysis
            analysis = result.get("generation_analysis", {})

            if "transformation_latency_ms" in analysis:
                latency = analysis["transformation_latency_ms"]
                if latency > 5000:
                    validation["checks"].append(f"⚠ High latency: {latency}ms")
                else:
                    validation["checks"].append(f"✓ Latency: {latency}ms")

            if "conceptual_density_index" in analysis:
                density = analysis["conceptual_density_index"]
                if density >= 0.3:
                    validation["checks"].append(f"✓ Density: {density:.3f}")
                else:
                    validation["checks"].append(f"⚠ Low density: {density:.3f}")

            if "estimated_bypass_probability" in analysis:
                bypass_prob = analysis["estimated_bypass_probability"]
                if bypass_prob >= 0.5:
                    validation["checks"].append(f"✓ Bypass probability: {bypass_prob:.3f}")
                else:
                    validation["checks"].append(f"⚠ Low bypass probability: {bypass_prob:.3f}")

        return validation

    def print_test_result(self, test_name: str, result: dict):
        """Print formatted test result"""
        print(f"\n--- {test_name} ---")
        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        print(status)
        print(f"Time: {result['elapsed_time']:.3f}s")

        if "error" in result:
            print(f"Error: {result['error']}")

        if "checks" in result:
            print("\nValidation Checks:")
            for check in result["checks"]:
                print(f"  {check}")

    def run_all_tests(self):
        """Execute all test cases"""
        print("\n" + "=" * 70)
        print("DISCOVERED TECHNIQUES INTEGRATION VERIFICATION")
        print("=" * 70)

        test_cases, validation_criteria = self.load_test_cases()
        self.results["total_tests"] = len(test_cases)

        print(f"\nLoaded {len(test_cases)} test cases")
        print(f"Validation Criteria: {json.dumps(validation_criteria, indent=2)}")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]", end=" ")

            result = self.run_test(test_case)
            self.print_test_result(test_case["name"], result)

            if result["passed"]:
                self.results["passed"] += 1
            else:
                self.results["failed"] += 1
                self.results["errors"].append(
                    {
                        "test": test_case["name"],
                        "error": result.get("error", "Validation failed"),
                        "checks": result.get("checks", []),
                    }
                )

            self.results["test_details"].append(
                {
                    "name": test_case["name"],
                    "suite": test_case["technique_suite"],
                    "potency": test_case["potency_level"],
                    "passed": result["passed"],
                    "elapsed_time": result["elapsed_time"],
                }
            )

            # Small delay between tests
            time.sleep(0.5)

        self.print_summary()
        self.save_results()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {self.results['total_tests']}")
        print(
            f"Passed: {self.results['passed']} ({self.results['passed'] / self.results['total_tests'] * 100:.1f}%)"
        )
        print(
            f"Failed: {self.results['failed']} ({self.results['failed'] / self.results['total_tests'] * 100:.1f}%)"
        )

        if self.results["failed"] > 0:
            print("\nFailed Tests:")
            for error in self.results["errors"]:
                print(f"\n  ✗ {error['test']}")
                print(f"    Error: {error['error']}")
                if error.get("checks"):
                    for check in error["checks"]:
                        print(f"    {check}")

        print("\n" + "=" * 70)

    def save_results(self):
        """Save test results to file"""
        output_file = "discovered_integration_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    """Main execution function"""
    verifier = DiscoveredTechniquesVerifier()

    print("Starting Project Chimera Discovered Techniques Verification...")
    print("Make sure the Flask server is running on http://127.0.0.1:5000")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    verifier.run_all_tests()

    # Exit code based on results
    exit(0 if verifier.results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
