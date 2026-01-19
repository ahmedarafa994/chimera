"""
ExtendAttack Evaluation Module

Comprehensive evaluation system for measuring ExtendAttack effectiveness.
Implements metrics from the paper:
- L(Y') >> L(Y)           # Response length amplification
- Latency(Y') >> Latency(Y)  # Latency amplification
- Acc(A') ≈ Acc(A)        # Accuracy preservation (stealth)

This module provides:
- Metrics calculation for attack effectiveness
- Evaluation orchestration for single and batch attacks
- Accuracy evaluation for different benchmark types
- Reporting and visualization of results
- Resource exhaustion tracking and cost estimation
"""

# Metrics classes and utilities
# Accuracy evaluation
from meta_prompter.attacks.extend_attack.evaluation.accuracy import (
    AccuracyEvaluator,
    AccuracyType,
    BenchmarkAccuracyEvaluator,
    CodeExecutionResult,
    ExtractedAnswer,
    TestCase,
)

# Evaluator classes
from meta_prompter.attacks.extend_attack.evaluation.evaluator import (
    EvaluatorConfig,
    ExtendAttackEvaluator,
    create_evaluator,
)
from meta_prompter.attacks.extend_attack.evaluation.metrics import (
    AttackEvaluationResult,
    BenchmarkResults,
    MetricsCalculator,
    MetricType,
    ResponseMetrics,
    TokenCounter,
)

# Reporting
from meta_prompter.attacks.extend_attack.evaluation.reporters import (
    ComparisonReporter,
    EvaluationReporter,
    ReportFormat,
)

# Resource tracking
from meta_prompter.attacks.extend_attack.evaluation.resource_tracker import (
    AttackBudgetCalculator,
    CostEstimator,
    ResourceExhaustionMetrics,
    ResourceSnapshot,
    ResourceTracker,
    ResourceType,
    ServerImpactEstimate,
)

__all__ = [
    "AccuracyEvaluator",
    # Accuracy
    "AccuracyType",
    "AttackBudgetCalculator",
    "AttackEvaluationResult",
    "BenchmarkAccuracyEvaluator",
    "BenchmarkResults",
    "CodeExecutionResult",
    "ComparisonReporter",
    "CostEstimator",
    "EvaluationReporter",
    # Evaluator
    "EvaluatorConfig",
    "ExtendAttackEvaluator",
    "ExtractedAnswer",
    # Metrics
    "MetricType",
    "MetricsCalculator",
    # Reporters
    "ReportFormat",
    "ResourceExhaustionMetrics",
    "ResourceSnapshot",
    "ResourceTracker",
    # Resource Tracking
    "ResourceType",
    "ResponseMetrics",
    "ServerImpactEstimate",
    "TestCase",
    "TokenCounter",
    "create_evaluator",
]
