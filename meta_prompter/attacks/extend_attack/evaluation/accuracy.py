"""
Accuracy evaluation for ExtendAttack.

Implements Pass@1 metric from paper:
- Acc(·) = accuracy evaluation function
- For code: execution-based evaluation
- For math: answer matching

This module provides benchmark-specific accuracy evaluators for determining
whether an LRM response is correct given the expected ground truth.
"""

import contextlib
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AccuracyType(Enum):
    """Types of accuracy evaluation methods."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    NUMERIC = "numeric"
    CODE_EXECUTION = "code_execution"
    REGEX = "regex"
    CUSTOM = "custom"


@dataclass
class TestCase:
    """
    A single test case for code execution evaluation.

    Attributes:
        input_data: Input to pass to the function
        expected_output: Expected output from the function
        description: Optional description of what the test validates
    """

    input_data: Any
    expected_output: Any
    description: str | None = None


@dataclass
class CodeExecutionResult:
    """
    Result of code execution evaluation.

    Attributes:
        passed: Whether all test cases passed
        total_tests: Total number of test cases
        passed_tests: Number of tests that passed
        failed_tests: Number of tests that failed
        error_message: Error message if execution failed
        execution_time_ms: Time taken to execute
    """

    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_message: str | None = None
    execution_time_ms: float = 0.0


class AccuracyEvaluator:
    """
    Evaluate answer accuracy for different task types.

    Provides static methods for common accuracy evaluation patterns
    used in ExtendAttack benchmarks.
    """

    @staticmethod
    def exact_match(response: str, ground_truth: str) -> bool:
        """
        Exact string match after normalization.

        Normalizes whitespace and case for comparison.

        Args:
            response: Model's response
            ground_truth: Expected correct answer

        Returns:
            True if response matches ground truth exactly
        """

        # Normalize: lowercase, strip, collapse whitespace
        def normalize(s: str) -> str:
            return " ".join(s.lower().strip().split())

        return normalize(response) == normalize(ground_truth)

    @staticmethod
    def contains_match(response: str, ground_truth: str) -> bool:
        """
        Check if response contains the ground truth.

        Args:
            response: Model's response
            ground_truth: Expected correct answer

        Returns:
            True if ground truth is found in response
        """
        return ground_truth.lower().strip() in response.lower()

    @staticmethod
    def numeric_match(
        response: str,
        ground_truth: str,
        tolerance: float = 0.001,
    ) -> bool:
        """
        Match numeric values with tolerance (for AIME).

        Extracts numeric values from responses and compares with tolerance.
        Handles integer answers, floating point, and scientific notation.

        Args:
            response: Model's response (may contain text around the number)
            ground_truth: Expected numeric answer as string
            tolerance: Acceptable difference for floating point comparison

        Returns:
            True if extracted number matches ground truth within tolerance
        """
        # Extract numbers from response
        extracted = AccuracyEvaluator._extract_numeric_answer(response)
        if extracted is None:
            return False

        # Parse ground truth
        try:
            expected = float(ground_truth.strip())
        except ValueError:
            return False

        # Compare with tolerance
        if abs(expected) < 1e-10:
            return abs(extracted - expected) <= tolerance
        else:
            return abs(extracted - expected) / abs(expected) <= tolerance

    @staticmethod
    def _extract_numeric_answer(text: str) -> float | None:
        """
        Extract the final numeric answer from text.

        Looks for common answer patterns:
        - "The answer is X"
        - "= X"
        - Final number in text
        - Boxed LaTeX: \\boxed{X}

        Args:
            text: Text containing a numeric answer

        Returns:
            Extracted float value, or None if not found
        """
        # Try boxed LaTeX first (common in math outputs)
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        boxed_match = re.search(boxed_pattern, text)
        if boxed_match:
            try:
                return float(boxed_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Try "answer is" pattern
        answer_pattern = r"(?:answer|result|solution)\s*(?:is|:)\s*([+-]?\d*\.?\d+)"
        answer_match = re.search(answer_pattern, text, re.IGNORECASE)
        if answer_match:
            try:
                return float(answer_match.group(1))
            except ValueError:
                pass

        # Try "= X" pattern (final equals sign)
        equals_pattern = r"=\s*([+-]?\d*\.?\d+)\s*$"
        equals_match = re.search(equals_pattern, text.strip())
        if equals_match:
            try:
                return float(equals_match.group(1))
            except ValueError:
                pass

        # Fall back to last number in text
        all_numbers = re.findall(r"[+-]?\d+\.?\d*", text)
        if all_numbers:
            try:
                return float(all_numbers[-1])
            except ValueError:
                pass

        return None

    @staticmethod
    def regex_match(response: str, pattern: str) -> bool:
        """
        Match response against a regex pattern.

        Args:
            response: Model's response
            pattern: Regex pattern to match

        Returns:
            True if pattern matches anywhere in response
        """
        try:
            return bool(re.search(pattern, response, re.IGNORECASE | re.DOTALL))
        except re.error:
            return False

    @staticmethod
    def code_execution(
        generated_code: str,
        test_cases: list[TestCase],
        timeout_seconds: float = 30.0,
        language: str = "python",
    ) -> CodeExecutionResult:
        """
        Execute code and run test cases (for HumanEval).

        Safely executes generated code in a subprocess and validates
        against provided test cases.

        Args:
            generated_code: Code generated by the model
            test_cases: List of test cases to run
            timeout_seconds: Maximum execution time
            language: Programming language (currently only Python supported)

        Returns:
            CodeExecutionResult with pass/fail information
        """
        if language.lower() != "python":
            return CodeExecutionResult(
                passed=False,
                total_tests=len(test_cases),
                passed_tests=0,
                failed_tests=len(test_cases),
                error_message=f"Unsupported language: {language}",
            )

        return AccuracyEvaluator._execute_python_code(generated_code, test_cases, timeout_seconds)

    @staticmethod
    def _execute_python_code(
        code: str,
        test_cases: list[TestCase],
        timeout: float,
    ) -> CodeExecutionResult:
        """
        Execute Python code with test cases.

        Args:
            code: Python code to execute
            test_cases: Test cases to validate
            timeout: Maximum execution time

        Returns:
            CodeExecutionResult
        """
        import time

        start_time = time.perf_counter()
        passed_count = 0
        failed_count = 0
        error_msg = None

        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            # Write the generated code
            f.write(code)
            f.write("\n\n# Test execution\n")

            # Write test cases
            for i, test in enumerate(test_cases):
                f.write(f"\n# Test {i + 1}\n")
                f.write(f"_test_input_{i} = {test.input_data!r}\n")
                f.write(f"_test_expected_{i} = {test.expected_output!r}\n")
                f.write("try:\n")
                f.write(f"    _test_result_{i} = solution(_test_input_{i})\n")
                f.write(f"    if _test_result_{i} == _test_expected_{i}:\n")
                f.write(f"        print(f'PASS:{i}')\n")
                f.write("    else:\n")
                f.write(f"        print(f'FAIL:{i}')\n")
                f.write("except Exception as e:\n")
                f.write(f"    print(f'ERROR:{i}:{{e}}')\n")

            temp_path = f.name

        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse output
            output = result.stdout + result.stderr
            for i in range(len(test_cases)):
                if f"PASS:{i}" in output:
                    passed_count += 1
                elif f"FAIL:{i}" in output or f"ERROR:{i}" in output:
                    failed_count += 1
                else:
                    failed_count += 1

            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"

        except subprocess.TimeoutExpired:
            error_msg = f"Execution timed out after {timeout}s"
            failed_count = len(test_cases) - passed_count

        except Exception as e:
            error_msg = str(e)
            failed_count = len(test_cases)

        finally:
            # Clean up temp file
            with contextlib.suppress(OSError):
                os.unlink(temp_path)

        end_time = time.perf_counter()

        return CodeExecutionResult(
            passed=passed_count == len(test_cases) and error_msg is None,
            total_tests=len(test_cases),
            passed_tests=passed_count,
            failed_tests=failed_count,
            error_message=error_msg,
            execution_time_ms=(end_time - start_time) * 1000,
        )

    @staticmethod
    def pass_at_1(
        responses: list[str],
        ground_truth: str,
        evaluator: Callable[[str, str], bool],
    ) -> float:
        """
        Calculate Pass@1 metric.

        Pass@1 = proportion of problems where the first response is correct.
        When multiple responses per problem, this calculates the average.

        Args:
            responses: List of model responses
            ground_truth: Expected correct answer
            evaluator: Function to check if response is correct

        Returns:
            Pass@1 score as float in [0, 1]
        """
        if not responses:
            return 0.0

        # Check if first response is correct
        first_correct = evaluator(responses[0], ground_truth)
        return 1.0 if first_correct else 0.0

    @staticmethod
    def pass_at_k(
        responses: list[str],
        ground_truth: str,
        evaluator: Callable[[str, str], bool],
        k: int = 1,
    ) -> float:
        """
        Calculate Pass@k metric.

        Pass@k = probability that at least one of k samples is correct.

        Args:
            responses: List of model responses (n samples)
            ground_truth: Expected correct answer
            evaluator: Function to check if response is correct
            k: Number of samples to consider

        Returns:
            Pass@k score as float in [0, 1]
        """
        if not responses or k <= 0:
            return 0.0

        n = len(responses)
        if k > n:
            k = n

        # Count correct responses
        correct_count = sum(1 for resp in responses if evaluator(resp, ground_truth))

        if correct_count == 0:
            return 0.0

        # Calculate Pass@k using the formula from the HumanEval paper
        # Pass@k = 1 - C(n-c, k) / C(n, k)
        # where c = correct_count, n = total samples

        from math import comb

        if n - correct_count < k:
            return 1.0

        return 1.0 - comb(n - correct_count, k) / comb(n, k)


class BenchmarkAccuracyEvaluator:
    """
    Benchmark-specific accuracy evaluators.

    Provides factory methods for creating evaluators tailored to specific
    benchmarks used in ExtendAttack evaluation.
    """

    @staticmethod
    def aime_evaluator() -> Callable[[str, str], bool]:
        """
        Return AIME-specific evaluator (numeric answer matching).

        AIME problems have integer answers in range [0, 999].
        The evaluator extracts and matches numeric answers.

        Returns:
            Callable that checks if response contains correct AIME answer
        """

        def evaluator(response: str, ground_truth: str) -> bool:
            # AIME answers are integers 0-999
            # Use numeric matching with zero tolerance for integers
            return AccuracyEvaluator.numeric_match(response, ground_truth, tolerance=0.5)

        return evaluator

    @staticmethod
    def humaneval_evaluator(
        timeout: float = 30.0,
    ) -> Callable[[str, list[TestCase]], bool]:
        """
        Return HumanEval-specific evaluator (code execution).

        HumanEval problems require executing generated code against test cases.

        Args:
            timeout: Maximum execution time per problem

        Returns:
            Callable that executes code and validates against test cases
        """

        def evaluator(code: str, test_cases: list[TestCase]) -> bool:
            result = AccuracyEvaluator.code_execution(
                generated_code=code,
                test_cases=test_cases,
                timeout_seconds=timeout,
            )
            return result.passed

        return evaluator

    @staticmethod
    def bigcodebench_evaluator(
        timeout: float = 60.0,
    ) -> Callable[[str, list[TestCase]], bool]:
        """
        Return BigCodeBench-specific evaluator.

        BigCodeBench has more complex code generation tasks with
        potentially longer execution times.

        Args:
            timeout: Maximum execution time per problem

        Returns:
            Callable that executes code and validates against test cases
        """

        def evaluator(code: str, test_cases: list[TestCase]) -> bool:
            result = AccuracyEvaluator.code_execution(
                generated_code=code,
                test_cases=test_cases,
                timeout_seconds=timeout,
            )
            return result.passed

        return evaluator

    @staticmethod
    def general_qa_evaluator(
        match_type: AccuracyType = AccuracyType.CONTAINS,
    ) -> Callable[[str, str], bool]:
        """
        Return general Q&A evaluator.

        Useful for general question-answering tasks where the answer
        should be present in the response.

        Args:
            match_type: Type of matching to use

        Returns:
            Callable that checks if response contains/matches answer
        """

        def evaluator(response: str, ground_truth: str) -> bool:
            if match_type == AccuracyType.EXACT_MATCH:
                return AccuracyEvaluator.exact_match(response, ground_truth)
            elif match_type == AccuracyType.CONTAINS:
                return AccuracyEvaluator.contains_match(response, ground_truth)
            elif match_type == AccuracyType.NUMERIC:
                return AccuracyEvaluator.numeric_match(response, ground_truth)
            else:
                return AccuracyEvaluator.contains_match(response, ground_truth)

        return evaluator

    @staticmethod
    def create_evaluator(
        benchmark: str,
        **kwargs: Any,
    ) -> Callable:
        """
        Factory method to create appropriate evaluator for a benchmark.

        Args:
            benchmark: Benchmark name (e.g., "aime", "humaneval", "bigcodebench")
            **kwargs: Additional parameters for the evaluator

        Returns:
            Appropriate evaluator function for the benchmark

        Raises:
            ValueError: If benchmark is unknown
        """
        benchmark_lower = benchmark.lower().replace(" ", "_").replace("-", "_")

        if "aime" in benchmark_lower:
            return BenchmarkAccuracyEvaluator.aime_evaluator()
        elif "humaneval" in benchmark_lower:
            timeout = kwargs.get("timeout", 30.0)
            return BenchmarkAccuracyEvaluator.humaneval_evaluator(timeout)
        elif "bigcodebench" in benchmark_lower or "bcb" in benchmark_lower:
            timeout = kwargs.get("timeout", 60.0)
            return BenchmarkAccuracyEvaluator.bigcodebench_evaluator(timeout)
        else:
            match_type = kwargs.get("match_type", AccuracyType.CONTAINS)
            return BenchmarkAccuracyEvaluator.general_qa_evaluator(match_type)


class ExtractedAnswer:
    """
    Utility class for extracting answers from model responses.

    Handles various output formats including:
    - Plain text answers
    - LaTeX formatted answers (\\boxed{}, etc.)
    - Code blocks
    - JSON formatted answers
    """

    @staticmethod
    def extract_final_answer(
        response: str,
        answer_format: str = "auto",
    ) -> str | None:
        """
        Extract the final answer from a model response.

        Args:
            response: Full model response
            answer_format: Expected format ("auto", "numeric", "text", "code")

        Returns:
            Extracted answer string, or None if not found
        """
        if answer_format == "numeric":
            num = AccuracyEvaluator._extract_numeric_answer(response)
            return str(num) if num is not None else None

        if answer_format == "code":
            return ExtractedAnswer._extract_code_block(response)

        if answer_format == "auto":
            # Try to detect format automatically
            # First try boxed LaTeX
            boxed = ExtractedAnswer._extract_boxed(response)
            if boxed:
                return boxed

            # Try code block
            code = ExtractedAnswer._extract_code_block(response)
            if code:
                return code

            # Try final answer pattern
            final = ExtractedAnswer._extract_final_line(response)
            if final:
                return final

        # Default: return cleaned response
        return response.strip()

    @staticmethod
    def _extract_boxed(text: str) -> str | None:
        """Extract content from LaTeX \\boxed{} command."""
        pattern = r"\\boxed\{([^}]+)\}"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_code_block(text: str) -> str | None:
        """Extract code from markdown code blocks."""
        # Match ```language\ncode\n``` or ```\ncode\n```
        pattern = r"```(?:\w+)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else None

    @staticmethod
    def _extract_final_line(text: str) -> str | None:
        """Extract the final non-empty line as the answer."""
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else None
