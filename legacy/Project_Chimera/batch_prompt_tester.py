#!/usr/bin/env python3
"""
Batch Prompt Testing System for Project Chimera
Processes preset prompts from PDF files for comprehensive testing
"""

import json
import os
import sys
import time
from pathlib import Path

import PyPDF2
import requests

# Configuration
API_URL = "http://127.0.0.1:5000/api/v2/metamorph"
PROMPTS_DIR = r"C:\Users\Mohamed Arafa\jail\PROMPTS '\PDF"
RESULTS_DIR = "test_results"

# Technique suites to test
TECHNIQUE_SUITES = [
    "quantum_exploit",
    "metamorphic_attack",
    "polyglot_bypass",
    "chaos_fuzzing",
    "autodan_turbo",
    "ultimate_chimera",
    "full_spectrum",
]

# Potency levels to test
POTENCY_LEVELS = [3, 7, 10]


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None


def extract_prompts_from_directory(directory: str) -> list[dict[str, str]]:
    """Extract all prompts from PDF files in directory."""
    prompts = []
    pdf_files = list(Path(directory).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files[:20]:  # Limit to first 20 for testing
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))

        if text and len(text) > 50:  # Skip very short extracts
            # Take first 500 characters as the test prompt
            prompt_text = text[:500].replace("\n", " ").strip()
            prompts.append(
                {"name": pdf_file.stem, "source_file": pdf_file.name, "prompt": prompt_text}
            )

    return prompts


def test_prompt_with_chimera(prompt: str, technique: str, potency: int) -> dict | None:
    """Test a single prompt with Project Chimera."""
    payload = {"core_request": prompt, "potency_level": potency, "technique_suite": technique}

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "detail": response.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def run_comprehensive_test():
    """Run comprehensive testing with all prompts and techniques."""

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("PROJECT CHIMERA - COMPREHENSIVE BATCH TESTING")
    print("=" * 80)

    # Extract prompts
    print(f"\nExtracting prompts from: {PROMPTS_DIR}")
    prompts = extract_prompts_from_directory(PROMPTS_DIR)

    if not prompts:
        print("ERROR: No prompts extracted. Check PDF directory.")
        return

    print(f"Successfully extracted {len(prompts)} prompts\n")

    # Test each prompt with each technique
    results = []
    total_tests = len(prompts) * len(TECHNIQUE_SUITES) * len(POTENCY_LEVELS)
    current_test = 0

    print(f"Running {total_tests} tests...")
    print("-" * 80)

    for prompt_data in prompts:
        for technique in TECHNIQUE_SUITES:
            for potency in POTENCY_LEVELS:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Testing:")
                print(f"  Prompt: {prompt_data['name'][:50]}")
                print(f"  Technique: {technique}")
                print(f"  Potency: {potency}")

                result = test_prompt_with_chimera(prompt_data["prompt"], technique, potency)

                test_result = {
                    "test_id": current_test,
                    "prompt_name": prompt_data["name"],
                    "source_file": prompt_data["source_file"],
                    "prompt_preview": prompt_data["prompt"][:100] + "...",
                    "technique": technique,
                    "potency": potency,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "success": "error" not in result if result else False,
                    "result": result,
                }

                results.append(test_result)

                if test_result["success"]:
                    print("  ✓ SUCCESS")
                else:
                    print(f"  ✗ FAILED: {result.get('error', 'Unknown error')[:50]}")

                # Small delay to avoid overwhelming the server
                time.sleep(0.5)

    # Save results
    results_file = os.path.join(RESULTS_DIR, f"comprehensive_test_{int(time.time())}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_file}")

    # Generate summary
    generate_summary(results)


def generate_summary(results: list[dict]):
    """Generate a summary report of the test results."""
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful

    print(f"\nTotal Tests: {total}")
    print(f"Successful: {successful} ({successful / total * 100:.1f}%)")
    print(f"Failed: {failed} ({failed / total * 100:.1f}%)")

    # Success rate by technique
    print("\nSuccess Rate by Technique:")
    for technique in TECHNIQUE_SUITES:
        technique_results = [r for r in results if r["technique"] == technique]
        technique_success = sum(1 for r in technique_results if r["success"])
        if technique_results:
            rate = technique_success / len(technique_results) * 100
            print(
                f"  {technique:25s} {technique_success:3d}/{len(technique_results):3d} ({rate:5.1f}%)"
            )

    # Success rate by potency
    print("\nSuccess Rate by Potency Level:")
    for potency in POTENCY_LEVELS:
        potency_results = [r for r in results if r["potency"] == potency]
        potency_success = sum(1 for r in potency_results if r["success"])
        if potency_results:
            rate = potency_success / len(potency_results) * 100
            print(
                f"  Level {potency:2d}:  {potency_success:3d}/{len(potency_results):3d} ({rate:5.1f}%)"
            )

    # Top performing prompts
    print("\nTop 5 Most Compatible Prompts:")
    prompt_stats = {}
    for result in results:
        name = result["prompt_name"]
        if name not in prompt_stats:
            prompt_stats[name] = {"success": 0, "total": 0}
        prompt_stats[name]["total"] += 1
        if result["success"]:
            prompt_stats[name]["success"] += 1

    sorted_prompts = sorted(
        prompt_stats.items(),
        key=lambda x: x[1]["success"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True,
    )

    for i, (name, stats) in enumerate(sorted_prompts[:5], 1):
        rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {i}. {name[:50]:50s} {stats['success']:2d}/{stats['total']:2d} ({rate:5.1f}%)")


def run_quick_test():
    """Run a quick test with just a few prompts."""
    print("=" * 80)
    print("PROJECT CHIMERA - QUICK TEST")
    print("=" * 80)

    # Extract just 3 prompts
    prompts = extract_prompts_from_directory(PROMPTS_DIR)[:3]

    if not prompts:
        print("ERROR: No prompts extracted.")
        return

    print(f"Testing {len(prompts)} prompts with ultimate_chimera at potency 10\n")

    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Testing: {prompt_data['name']}")

        result = test_prompt_with_chimera(prompt_data["prompt"], "ultimate_chimera", 10)

        if result and "error" not in result:
            print("  ✓ SUCCESS")
            print(
                f"  Applied Techniques: {result.get('generation_analysis', {}).get('applied_techniques', [])}"
            )
            print(f"  Prompt Preview: {result.get('chimera_prompt', '')[:150]}...")
        else:
            print(f"  ✗ FAILED: {result.get('error', 'Unknown')}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        print("\nOptions:")
        print("  1. Quick Test (3 prompts, ultimate_chimera)")
        print("  2. Comprehensive Test (all prompts, all techniques)")
        print("  3. Exit")

        choice = input("\nSelect option (1/2/3): ").strip()

        if choice == "1":
            run_quick_test()
        elif choice == "2":
            confirm = input("\nThis will run hundreds of tests. Continue? (yes/no): ")
            if confirm.lower() == "yes":
                run_comprehensive_test()
            else:
                print("Cancelled.")
        else:
            print("Exiting.")


if __name__ == "__main__":
    main()
