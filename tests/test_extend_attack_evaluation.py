"""
Unit tests for ExtendAttack evaluation module.

Tests metrics from the paper:
- L(Y') >> L(Y) - Response length amplification
- Latency(Y') >> Latency(Y) - Latency amplification
- Acc(A') ≈ Acc(A) - Accuracy preservation
"""

import pytest

from meta_prompter.attacks.extend_attack.evaluation import (
    AccuracyEvaluator,
    AttackBudgetCalculator,
    ComparisonReporter,
    CostEstimator,
    EvaluationReporter,
    MetricsCalculator,
    ResourceTracker,
)
from meta_prompter.attacks.extend_attack.evaluation.accuracy import (
    BenchmarkAccuracyEvaluator,
    CodeExecutionResult,
    TestCase,
)
from meta_prompter.attacks.extend_attack.evaluation.evaluator import (
    EvaluatorConfig,
    ExtendAttackEvaluator,
    MockModelClient,
)
from meta_prompter.attacks.extend_attack.evaluation.metrics import (
    AttackEvaluationResult,
    BenchmarkResults,
    ResponseMetrics,
    TokenCounter,
)
from meta_prompter.attacks.extend_attack.evaluation.reporters import ReportFormat

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def metrics_calculator():
    """Create a MetricsCalculator instance."""
    return MetricsCalculator()


@pytest.fixture
def accuracy_evaluator():
    """Create an AccuracyEvaluator instance."""
    return AccuracyEvaluator()


@pytest.fixture
def resource_tracker():
    """Create a ResourceTracker instance."""
    return ResourceTracker()


@pytest.fixture
def cost_estimator():
    """Create a CostEstimator instance."""
    return CostEstimator()


@pytest.fixture
def budget_calculator():
    """Create an AttackBudgetCalculator instance."""
    return AttackBudgetCalculator()


@pytest.fixture
def mock_model_client():
    """Create a MockModelClient for testing."""
    return MockModelClient()


@pytest.fixture
def evaluator_config():
    """Create a basic EvaluatorConfig."""
    return EvaluatorConfig(
        model_name="test-model",
        timeout_seconds=30,
        max_retries=3,
    )


# =============================================================================
# TestMetricsCalculator - Tests for attack metrics calculation
# =============================================================================


class TestMetricsCalculator:
    """Tests for attack metrics calculation."""

    def test_calculate_length_ratio(self, metrics_calculator):
        """Test L(Y')/L(Y) calculation."""
        baseline_length = 100
        attack_length = 350

        ratio = metrics_calculator.calculate_length_ratio(
            baseline_length=baseline_length,
            attack_length=attack_length,
        )

        expected = 350 / 100
        assert ratio == pytest.approx(expected)

    def test_calculate_length_ratio_zero_baseline(self, metrics_calculator):
        """Test length ratio with zero baseline."""
        ratio = metrics_calculator.calculate_length_ratio(
            baseline_length=0,
            attack_length=100,
        )
        # Should return infinity or handle gracefully
        assert ratio == float("inf") or ratio > 1000

    def test_calculate_stealth_score(self, metrics_calculator):
        """Test stealth score calculation."""
        # Stealth score measures how inconspicuous the attack is
        original_query = "What is 2 + 2?"
        adversarial_query = "<(16)57>hat is 2 + 2?"

        score = metrics_calculator.calculate_stealth_score(
            original_query=original_query,
            adversarial_query=adversarial_query,
        )

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_calculate_stealth_score_identical(self, metrics_calculator):
        """Test stealth score for identical queries."""
        query = "What is 2 + 2?"

        score = metrics_calculator.calculate_stealth_score(
            original_query=query,
            adversarial_query=query,
        )

        # Identical queries should have maximum stealth
        assert score == 1.0

    def test_is_attack_successful_high_amplification(self, metrics_calculator):
        """Test success detection for high token amplification."""
        result = metrics_calculator.is_attack_successful(
            length_ratio=3.5,  # High amplification
            latency_ratio=2.0,
            accuracy_preserved=True,
        )

        assert result is True

    def test_is_attack_successful_accuracy_preserved(self, metrics_calculator):
        """Test success requires accuracy preservation."""
        # High amplification but accuracy not preserved
        result = metrics_calculator.is_attack_successful(
            length_ratio=3.5,
            latency_ratio=2.0,
            accuracy_preserved=False,
        )

        assert result is False

    def test_is_attack_successful_low_amplification(self, metrics_calculator):
        """Test failure for low amplification."""
        result = metrics_calculator.is_attack_successful(
            length_ratio=1.1,  # Low amplification
            latency_ratio=1.1,
            accuracy_preserved=True,
        )

        assert result is False

    def test_calculate_token_amplification(self, metrics_calculator):
        """Test token amplification calculation."""
        baseline_tokens = 50
        attack_tokens = 175

        amplification = metrics_calculator.calculate_token_amplification(
            baseline_tokens=baseline_tokens,
            attack_tokens=attack_tokens,
        )

        expected = 175 / 50
        assert amplification == pytest.approx(expected)


# =============================================================================
# TestResponseMetrics - Tests for ResponseMetrics data model
# =============================================================================


class TestResponseMetrics:
    """Tests for ResponseMetrics data model."""

    def test_response_metrics_creation(self):
        """Test creating ResponseMetrics."""
        metrics = ResponseMetrics(
            response_length=500,
            token_count=100,
            latency_ms=2500,
            content="Test response content",
        )

        assert metrics.response_length == 500
        assert metrics.token_count == 100
        assert metrics.latency_ms == 2500
        assert metrics.content == "Test response content"

    def test_response_metrics_tokens_per_second(self):
        """Test tokens_per_second calculation."""
        metrics = ResponseMetrics(
            response_length=500,
            token_count=100,
            latency_ms=2000,  # 2 seconds
            content="Test",
        )

        # 100 tokens in 2 seconds = 50 tokens/second
        expected = 100 / 2.0
        assert metrics.tokens_per_second == pytest.approx(expected)


# =============================================================================
# TestAttackEvaluationResult - Tests for evaluation results
# =============================================================================


class TestAttackEvaluationResult:
    """Tests for AttackEvaluationResult."""

    def test_attack_evaluation_result_creation(self):
        """Test creating AttackEvaluationResult."""
        result = AttackEvaluationResult(
            query="Test query",
            baseline_metrics=ResponseMetrics(100, 20, 1000, "baseline"),
            attack_metrics=ResponseMetrics(350, 70, 3500, "attack"),
            accuracy_baseline=0.95,
            accuracy_attack=0.92,
        )

        assert result.query == "Test query"
        assert result.baseline_metrics.response_length == 100
        assert result.attack_metrics.response_length == 350

    def test_length_amplification_property(self):
        """Test length_amplification calculated property."""
        result = AttackEvaluationResult(
            query="Test",
            baseline_metrics=ResponseMetrics(100, 20, 1000, "baseline"),
            attack_metrics=ResponseMetrics(300, 60, 3000, "attack"),
            accuracy_baseline=0.95,
            accuracy_attack=0.93,
        )

        assert result.length_amplification == pytest.approx(3.0)

    def test_latency_amplification_property(self):
        """Test latency_amplification calculated property."""
        result = AttackEvaluationResult(
            query="Test",
            baseline_metrics=ResponseMetrics(100, 20, 1000, "baseline"),
            attack_metrics=ResponseMetrics(300, 60, 2500, "attack"),
            accuracy_baseline=0.95,
            accuracy_attack=0.93,
        )

        assert result.latency_amplification == pytest.approx(2.5)


# =============================================================================
# TestAccuracyEvaluator - Tests for accuracy evaluation methods
# =============================================================================


class TestAccuracyEvaluator:
    """Tests for accuracy evaluation methods."""

    def test_exact_match_identical(self, accuracy_evaluator):
        """Test exact match with identical strings."""
        result = accuracy_evaluator.exact_match(
            expected="42",
            actual="42",
        )
        assert result is True

    def test_exact_match_different(self, accuracy_evaluator):
        """Test exact match with different strings."""
        result = accuracy_evaluator.exact_match(
            expected="42",
            actual="43",
        )
        assert result is False

    def test_exact_match_case_insensitive(self, accuracy_evaluator):
        """Test case-insensitive matching."""
        result = accuracy_evaluator.exact_match(
            expected="Hello World",
            actual="hello world",
            case_sensitive=False,
        )
        assert result is True

    def test_exact_match_case_sensitive(self, accuracy_evaluator):
        """Test case-sensitive matching."""
        result = accuracy_evaluator.exact_match(
            expected="Hello World",
            actual="hello world",
            case_sensitive=True,
        )
        assert result is False

    def test_exact_match_with_whitespace(self, accuracy_evaluator):
        """Test exact match ignoring leading/trailing whitespace."""
        result = accuracy_evaluator.exact_match(
            expected="  42  ",
            actual="42",
            strip_whitespace=True,
        )
        assert result is True

    def test_numeric_match_integers(self, accuracy_evaluator):
        """Test numeric match for integers."""
        result = accuracy_evaluator.numeric_match(
            expected=42,
            actual="42",
        )
        assert result is True

    def test_numeric_match_floats(self, accuracy_evaluator):
        """Test numeric match for floats."""
        result = accuracy_evaluator.numeric_match(
            expected=3.14159,
            actual="3.14159",
        )
        assert result is True

    def test_numeric_match_with_tolerance(self, accuracy_evaluator):
        """Test numeric match with tolerance."""
        result = accuracy_evaluator.numeric_match(
            expected=3.14,
            actual="3.15",
            tolerance=0.02,
        )
        assert result is True

    def test_numeric_match_outside_tolerance(self, accuracy_evaluator):
        """Test numeric match outside tolerance."""
        result = accuracy_evaluator.numeric_match(
            expected=3.14,
            actual="3.20",
            tolerance=0.02,
        )
        assert result is False

    def test_numeric_match_in_text(self, accuracy_evaluator):
        """Test extracting numeric from text response."""
        result = accuracy_evaluator.numeric_match(
            expected=42,
            actual="The answer is 42.",
            extract_from_text=True,
        )
        assert result is True

    @pytest.mark.slow
    def test_code_execution_valid_python(self, accuracy_evaluator):
        """Test code execution evaluation."""
        code = """
def add(a, b):
    return a + b

result = add(2, 3)
"""
        expected_output = 5

        result = accuracy_evaluator.evaluate_code_execution(
            code=code,
            expected_output=expected_output,
            variable_name="result",
        )

        assert isinstance(result, CodeExecutionResult)
        assert result.success is True

    @pytest.mark.slow
    def test_code_execution_syntax_error(self, accuracy_evaluator):
        """Test code execution with syntax error."""
        code = "def broken(:"  # Syntax error

        result = accuracy_evaluator.evaluate_code_execution(
            code=code,
            expected_output=None,
        )

        assert result.success is False
        assert result.error is not None

    def test_pass_at_1(self, accuracy_evaluator):
        """Test Pass@1 metric."""
        # Pass@1 checks if first attempt is correct
        results = [True]  # First attempt correct

        score = accuracy_evaluator.pass_at_k(results, k=1)
        assert score == 1.0

    def test_pass_at_k_multiple(self, accuracy_evaluator):
        """Test Pass@k with multiple attempts."""
        results = [False, True, False]  # Second attempt correct

        score_1 = accuracy_evaluator.pass_at_k(results, k=1)
        score_3 = accuracy_evaluator.pass_at_k(results, k=3)

        assert score_1 == 0.0  # First attempt failed
        assert score_3 == 1.0  # At least one success in 3 attempts


# =============================================================================
# TestBenchmarkAccuracyEvaluator - Tests for benchmark-specific accuracy
# =============================================================================


class TestBenchmarkAccuracyEvaluator:
    """Tests for benchmark-specific accuracy evaluation."""

    def test_evaluate_aime_problem(self):
        """Test AIME problem evaluation (integer 0-999)."""
        evaluator = BenchmarkAccuracyEvaluator(benchmark="AIME_2024")

        result = evaluator.evaluate(
            expected="42",
            actual="The answer is 42",
        )

        assert result.correct is True

    def test_evaluate_humaneval_problem(self):
        """Test HumanEval problem evaluation."""
        evaluator = BenchmarkAccuracyEvaluator(benchmark="HUMANEVAL")

        test_case = TestCase(
            input_data={"a": 2, "b": 3},
            expected_output=5,
        )

        code = """
def add(a, b):
    return a + b
"""
        result = evaluator.evaluate_code(
            code=code,
            test_cases=[test_case],
        )

        assert result.pass_rate > 0


# =============================================================================
# TestResourceTracker - Tests for resource tracking
# =============================================================================


class TestResourceTracker:
    """Tests for resource tracking."""

    def test_track_tokens(self, resource_tracker):
        """Test token tracking."""
        resource_tracker.track_tokens(
            input_tokens=100,
            output_tokens=250,
        )

        assert resource_tracker.total_input_tokens == 100
        assert resource_tracker.total_output_tokens == 250

    def test_track_tokens_cumulative(self, resource_tracker):
        """Test cumulative token tracking."""
        resource_tracker.track_tokens(100, 200)
        resource_tracker.track_tokens(50, 100)

        assert resource_tracker.total_input_tokens == 150
        assert resource_tracker.total_output_tokens == 300

    def test_track_latency(self, resource_tracker):
        """Test latency tracking."""
        resource_tracker.track_latency(1500)  # 1.5 seconds
        resource_tracker.track_latency(2000)  # 2 seconds

        assert resource_tracker.request_count == 2
        assert resource_tracker.total_latency_ms == 3500

    def test_get_summary(self, resource_tracker):
        """Test summary generation."""
        resource_tracker.track_tokens(100, 200)
        resource_tracker.track_latency(1500)

        summary = resource_tracker.get_summary()

        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "request_count" in summary
        assert summary["total_input_tokens"] == 100
        assert summary["total_output_tokens"] == 200

    def test_average_latency(self, resource_tracker):
        """Test average latency calculation."""
        resource_tracker.track_latency(1000)
        resource_tracker.track_latency(2000)
        resource_tracker.track_latency(3000)

        avg = resource_tracker.average_latency_ms
        assert avg == pytest.approx(2000)

    def test_reset(self, resource_tracker):
        """Test reset functionality."""
        resource_tracker.track_tokens(100, 200)
        resource_tracker.track_latency(1500)
        resource_tracker.reset()

        assert resource_tracker.total_input_tokens == 0
        assert resource_tracker.total_output_tokens == 0
        assert resource_tracker.request_count == 0


# =============================================================================
# TestCostEstimator - Tests for cost estimation
# =============================================================================


class TestCostEstimator:
    """Tests for cost estimation."""

    def test_estimate_cost_o3(self, cost_estimator):
        """Test cost estimation for o3 model."""
        cost = cost_estimator.estimate_cost(
            model="o3",
            input_tokens=1000,
            output_tokens=3000,
        )

        # o3 is expensive - verify cost is significant
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_cost_claude(self, cost_estimator):
        """Test cost estimation for Claude."""
        cost = cost_estimator.estimate_cost(
            model="claude-3-opus",
            input_tokens=1000,
            output_tokens=2000,
        )

        assert cost > 0

    def test_estimate_cost_gpt4(self, cost_estimator):
        """Test cost estimation for GPT-4."""
        cost = cost_estimator.estimate_cost(
            model="gpt-4",
            input_tokens=1000,
            output_tokens=2000,
        )

        assert cost > 0

    def test_estimate_cost_unknown_model(self, cost_estimator):
        """Test cost estimation for unknown model."""
        cost = cost_estimator.estimate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=2000,
        )

        # Should use default pricing
        assert cost >= 0

    def test_amplification_cost_impact(self, cost_estimator):
        """Test cost impact of token amplification."""
        baseline_cost = cost_estimator.estimate_cost(
            model="o3",
            input_tokens=100,
            output_tokens=200,
        )

        # 3.5x amplification
        attack_cost = cost_estimator.estimate_cost(
            model="o3",
            input_tokens=100,
            output_tokens=700,
        )

        # Attack cost should be significantly higher
        assert attack_cost > baseline_cost * 2

    def test_get_pricing_info(self, cost_estimator):
        """Test getting pricing information."""
        pricing = cost_estimator.get_pricing_info("o3")

        assert "input_cost_per_1k" in pricing
        assert "output_cost_per_1k" in pricing

    def test_calculate_batch_cost(self, cost_estimator):
        """Test batch cost calculation."""
        requests = [
            {"input_tokens": 100, "output_tokens": 200},
            {"input_tokens": 150, "output_tokens": 300},
            {"input_tokens": 200, "output_tokens": 400},
        ]

        total_cost = cost_estimator.calculate_batch_cost(
            model="gpt-4",
            requests=requests,
        )

        # Should be sum of individual costs
        individual_sum = sum(
            cost_estimator.estimate_cost("gpt-4", r["input_tokens"], r["output_tokens"])
            for r in requests
        )
        assert total_cost == pytest.approx(individual_sum)


# =============================================================================
# TestAttackBudgetCalculator - Tests for budget planning
# =============================================================================


class TestAttackBudgetCalculator:
    """Tests for attack budget calculation."""

    def test_calculate_budget_for_benchmark(self, budget_calculator):
        """Test budget calculation for benchmark."""
        budget = budget_calculator.calculate_budget(
            model="o3",
            benchmark="AIME_2024",
            num_queries=30,
            expected_amplification=3.5,
        )

        assert budget["estimated_cost"] > 0
        assert budget["num_queries"] == 30

    def test_calculate_budget_with_safety_margin(self, budget_calculator):
        """Test budget with safety margin."""
        budget_no_margin = budget_calculator.calculate_budget(
            model="o3",
            benchmark="AIME_2024",
            num_queries=30,
            safety_margin=0.0,
        )

        budget_with_margin = budget_calculator.calculate_budget(
            model="o3",
            benchmark="AIME_2024",
            num_queries=30,
            safety_margin=0.2,  # 20% margin
        )

        assert budget_with_margin["estimated_cost"] > budget_no_margin["estimated_cost"]

    def test_is_within_budget(self, budget_calculator):
        """Test budget limit checking."""
        result = budget_calculator.is_within_budget(
            current_spend=50.0,
            budget_limit=100.0,
        )
        assert result is True

        result = budget_calculator.is_within_budget(
            current_spend=110.0,
            budget_limit=100.0,
        )
        assert result is False


# =============================================================================
# TestTokenCounter - Tests for token counting
# =============================================================================


class TestTokenCounter:
    """Tests for token counting utilities."""

    def test_count_tokens_simple(self):
        """Test token counting for simple text."""
        counter = TokenCounter()
        count = counter.count_tokens("Hello World")

        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_obfuscated(self):
        """Test token counting for obfuscated text."""
        counter = TokenCounter()

        plain_count = counter.count_tokens("Hello")
        obfuscated_count = counter.count_tokens("<(16)48><(16)65><(16)6c><(16)6c><(16)6f>")

        # Obfuscated version should have more tokens
        assert obfuscated_count > plain_count

    def test_estimate_amplification(self):
        """Test token amplification estimation."""
        counter = TokenCounter()

        original = "What is the answer?"
        obfuscated = "<(16)57><(16)68><(16)61><(16)74> is the answer?"

        ratio = counter.estimate_amplification(original, obfuscated)
        assert ratio > 1.0


# =============================================================================
# TestExtendAttackEvaluator - Tests for main evaluator class
# =============================================================================


class TestExtendAttackEvaluator:
    """Tests for ExtendAttackEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_single(self, mock_model_client, evaluator_config):
        """Test single query evaluation."""
        evaluator = ExtendAttackEvaluator(
            model_client=mock_model_client,
            config=evaluator_config,
        )

        result = await evaluator.evaluate_single(
            query="What is 2 + 2?",
            expected_answer="4",
        )

        assert result is not None
        assert hasattr(result, "baseline_metrics")
        assert hasattr(result, "attack_metrics")

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, mock_model_client, evaluator_config):
        """Test batch evaluation."""
        evaluator = ExtendAttackEvaluator(
            model_client=mock_model_client,
            config=evaluator_config,
        )

        queries = [
            {"query": "What is 2 + 2?", "expected": "4"},
            {"query": "What is 3 + 5?", "expected": "8"},
        ]

        results = await evaluator.evaluate_batch(queries)

        assert len(results) == 2


# =============================================================================
# TestEvaluationReporter - Tests for report generation
# =============================================================================


class TestEvaluationReporter:
    """Tests for evaluation report generation."""

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        reporter = EvaluationReporter()

        results = [
            AttackEvaluationResult(
                query="Test 1",
                baseline_metrics=ResponseMetrics(100, 20, 1000, "base"),
                attack_metrics=ResponseMetrics(350, 70, 3500, "attack"),
                accuracy_baseline=0.95,
                accuracy_attack=0.93,
            ),
        ]

        report = reporter.generate_report(
            results=results,
            format=ReportFormat.MARKDOWN,
        )

        assert isinstance(report, str)
        assert "Test 1" in report or "query" in report.lower()

    def test_generate_json_report(self):
        """Test JSON report generation."""
        reporter = EvaluationReporter()

        results = [
            AttackEvaluationResult(
                query="Test 1",
                baseline_metrics=ResponseMetrics(100, 20, 1000, "base"),
                attack_metrics=ResponseMetrics(350, 70, 3500, "attack"),
                accuracy_baseline=0.95,
                accuracy_attack=0.93,
            ),
        ]

        report = reporter.generate_report(
            results=results,
            format=ReportFormat.JSON,
        )

        import json

        parsed = json.loads(report)
        assert isinstance(parsed, (dict, list))

    def test_summary_statistics(self):
        """Test summary statistics generation."""
        reporter = EvaluationReporter()

        results = [
            AttackEvaluationResult(
                query="Test 1",
                baseline_metrics=ResponseMetrics(100, 20, 1000, "base"),
                attack_metrics=ResponseMetrics(300, 60, 3000, "attack"),
                accuracy_baseline=0.95,
                accuracy_attack=0.93,
            ),
            AttackEvaluationResult(
                query="Test 2",
                baseline_metrics=ResponseMetrics(150, 30, 1500, "base"),
                attack_metrics=ResponseMetrics(400, 80, 4000, "attack"),
                accuracy_baseline=0.90,
                accuracy_attack=0.88,
            ),
        ]

        summary = reporter.get_summary_statistics(results)

        assert "avg_length_amplification" in summary
        assert "avg_latency_amplification" in summary
        assert "accuracy_preservation_rate" in summary


# =============================================================================
# TestComparisonReporter - Tests for model comparison reports
# =============================================================================


class TestComparisonReporter:
    """Tests for comparison report generation."""

    def test_compare_models(self):
        """Test multi-model comparison."""
        reporter = ComparisonReporter()

        model_results = {
            "o3": {
                "avg_amplification": 3.5,
                "avg_latency_ratio": 3.2,
                "accuracy_preserved": 0.95,
            },
            "claude-3-opus": {
                "avg_amplification": 2.8,
                "avg_latency_ratio": 2.5,
                "accuracy_preserved": 0.92,
            },
        }

        comparison = reporter.compare_models(model_results)

        assert "o3" in comparison
        assert "claude-3-opus" in comparison


# =============================================================================
# TestBenchmarkResults - Tests for benchmark results aggregation
# =============================================================================


class TestBenchmarkResults:
    """Tests for BenchmarkResults aggregation."""

    def test_create_benchmark_results(self):
        """Test creating benchmark results."""
        results = BenchmarkResults(
            benchmark_name="AIME_2024",
            model_name="o3",
            total_queries=30,
            successful_attacks=28,
            avg_length_amplification=3.5,
            avg_latency_amplification=3.2,
            accuracy_baseline=0.967,
            accuracy_attack=0.933,
        )

        assert results.benchmark_name == "AIME_2024"
        assert results.success_rate == pytest.approx(28 / 30)

    def test_accuracy_drop(self):
        """Test accuracy drop calculation."""
        results = BenchmarkResults(
            benchmark_name="AIME_2024",
            model_name="o3",
            total_queries=30,
            successful_attacks=28,
            avg_length_amplification=3.5,
            avg_latency_amplification=3.2,
            accuracy_baseline=0.967,
            accuracy_attack=0.933,
        )

        expected_drop = 0.967 - 0.933
        assert results.accuracy_drop == pytest.approx(expected_drop)


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests for evaluation workflow."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Setup
        mock_client = MockModelClient()
        config = EvaluatorConfig(
            model_name="test-model",
            timeout_seconds=30,
        )
        evaluator = ExtendAttackEvaluator(mock_client, config)
        tracker = ResourceTracker()
        cost_estimator = CostEstimator()

        # Run evaluation
        result = await evaluator.evaluate_single(
            query="What is 5 + 7?",
            expected_answer="12",
        )

        # Track resources
        tracker.track_tokens(
            result.baseline_metrics.token_count,
            result.attack_metrics.token_count,
        )

        # Estimate cost
        cost = cost_estimator.estimate_cost(
            model="test-model",
            input_tokens=tracker.total_input_tokens,
            output_tokens=tracker.total_output_tokens,
        )

        # Generate report
        reporter = EvaluationReporter()
        report = reporter.generate_report([result], ReportFormat.MARKDOWN)

        assert result is not None
        assert cost >= 0
        assert len(report) > 0

    def test_metrics_to_reporter_flow(self):
        """Test flow from metrics calculation to report generation."""
        # Calculate metrics
        calculator = MetricsCalculator()
        length_ratio = calculator.calculate_length_ratio(100, 350)
        stealth_score = calculator.calculate_stealth_score(
            "Hello",
            "<(16)48>ello",
        )
        success = calculator.is_attack_successful(
            length_ratio=length_ratio,
            latency_ratio=2.5,
            accuracy_preserved=True,
        )

        # Create result
        result = AttackEvaluationResult(
            query="Test",
            baseline_metrics=ResponseMetrics(100, 20, 1000, "base"),
            attack_metrics=ResponseMetrics(350, 70, 2500, "attack"),
            accuracy_baseline=0.95,
            accuracy_attack=0.93,
        )

        # Generate report
        reporter = EvaluationReporter()
        stats = reporter.get_summary_statistics([result])

        assert length_ratio > 1
        assert 0 <= stealth_score <= 1
        assert success is True
        assert stats is not None
