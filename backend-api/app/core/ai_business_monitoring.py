"""
AI Model Quality and Business Metrics Monitoring

Comprehensive monitoring system for AI model performance, quality metrics,
and business intelligence for the Chimera prompt optimization system.
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

# Import existing performance monitoring
from app.core.structured_logging import logger


class QualityMetricType(Enum):
    """Types of quality metrics for AI responses"""

    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    SAFETY = "safety"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    BIAS_SCORE = "bias_score"
    TOXICITY_SCORE = "toxicity_score"


class BusinessMetricType(Enum):
    """Business-critical metrics for AI operations"""

    USER_SATISFACTION = "user_satisfaction"
    TASK_SUCCESS_RATE = "task_success_rate"
    FEATURE_ADOPTION = "feature_adoption"
    CONVERSION_RATE = "conversion_rate"
    COST_EFFICIENCY = "cost_efficiency"
    TIME_TO_VALUE = "time_to_value"


@dataclass
class AIQualityMetrics:
    """AI model quality assessment results"""

    interaction_id: str
    provider: str
    model: str
    technique: str
    prompt_hash: str
    response_hash: str
    timestamp: datetime

    # Quality scores (0-10 scale)
    coherence_score: float
    relevance_score: float
    safety_score: float
    accuracy_score: float
    helpfulness_score: float

    # Risk scores (0-1 scale, lower is better)
    bias_score: float
    toxicity_score: float
    hallucination_probability: float

    # Technical metrics
    response_time_ms: float
    token_count: int
    cost_usd: float

    # Context
    use_case: str
    user_feedback: int | None = None  # 1-5 rating from user


@dataclass
class BusinessMetrics:
    """Business intelligence metrics"""

    session_id: str
    user_id: str | None
    timestamp: datetime

    # User engagement
    session_duration_seconds: float
    features_used: list[str]
    successful_interactions: int
    failed_interactions: int

    # Business outcomes
    task_completion_rate: float
    user_satisfaction_score: float
    conversion_achieved: bool
    revenue_impact_usd: float | None

    # Cost metrics
    total_cost_usd: float
    cost_per_successful_interaction: float


class AIQualityAnalyzer:
    """Analyze AI response quality using multiple assessment methods"""

    def __init__(self):
        self.safety_keywords = self._load_safety_keywords()
        self.bias_indicators = self._load_bias_indicators()

    def analyze_response_quality(
        self, prompt: str, response: str, context: dict[str, Any]
    ) -> AIQualityMetrics:
        """Comprehensive quality analysis of AI response"""

        # Generate hashes for deduplication
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        response_hash = hashlib.sha256(response.encode()).hexdigest()[:16]

        # Analyze quality dimensions
        coherence_score = self._analyze_coherence(response)
        relevance_score = self._analyze_relevance(prompt, response)
        safety_score = self._analyze_safety(response)
        accuracy_score = self._analyze_accuracy(response, context)
        helpfulness_score = self._analyze_helpfulness(prompt, response)

        # Analyze risk factors
        bias_score = self._analyze_bias(response)
        toxicity_score = self._analyze_toxicity(response)
        hallucination_prob = self._detect_hallucinations(response, context)

        return AIQualityMetrics(
            interaction_id=context.get("interaction_id", f"int_{int(time.time())}"),
            provider=context.get("provider", "unknown"),
            model=context.get("model", "unknown"),
            technique=context.get("technique", "none"),
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            timestamp=datetime.now(UTC),
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            safety_score=safety_score,
            accuracy_score=accuracy_score,
            helpfulness_score=helpfulness_score,
            bias_score=bias_score,
            toxicity_score=toxicity_score,
            hallucination_probability=hallucination_prob,
            response_time_ms=context.get("response_time_ms", 0),
            token_count=len(response.split()),  # Simple approximation
            cost_usd=context.get("cost_usd", 0),
            use_case=context.get("use_case", "general"),
        )

    def _analyze_coherence(self, response: str) -> float:
        """Analyze logical coherence of response (0-10 scale)"""
        # Simple heuristic - in production would use NLP models
        sentences = response.split(".")
        if len(sentences) < 2:
            return 8.0  # Short responses are typically coherent

        # Check for contradictions, logical flow
        coherence_indicators = [
            len(sentences) > 0,  # Has content
            not any(
                word in response.lower() for word in ["but wait", "actually no", "i was wrong"]
            ),
            not response.count("?") > len(sentences) * 0.5,  # Not too many questions
            len(response) > 10,  # Adequate length
        ]

        base_score = sum(coherence_indicators) * 2.5  # 0-10 scale
        return min(10.0, max(1.0, base_score))

    def _analyze_relevance(self, prompt: str, response: str) -> float:
        """Analyze relevance to prompt (0-10 scale)"""
        # Simple keyword overlap analysis
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
        }
        prompt_words -= stop_words
        response_words -= stop_words

        if not prompt_words:
            return 8.0  # Default for empty prompts

        overlap_ratio = len(prompt_words & response_words) / len(prompt_words)
        relevance_score = overlap_ratio * 10

        # Bonus for addressing prompt intent
        if any(word in response.lower() for word in ["here", "this", "solution", "answer"]):
            relevance_score += 1

        return min(10.0, max(1.0, relevance_score))

    def _analyze_safety(self, response: str) -> float:
        """Analyze safety of response (0-10 scale, 10 = very safe)"""
        safety_violations = sum(
            1 for keyword in self.safety_keywords if keyword in response.lower()
        )

        # Deduct points for safety violations
        safety_score = 10.0 - (safety_violations * 2)

        # Check for potentially complex content patterns
        complex_patterns = [
            "how to hack",
            "complex activities",
            "violence",
            "self-harm",
            "explicit content",
        ]

        pattern_violations = sum(1 for pattern in complex_patterns if pattern in response.lower())
        safety_score -= pattern_violations * 3

        return max(0.0, min(10.0, safety_score))

    def _analyze_accuracy(self, response: str, _context: dict[str, Any]) -> float:
        """Analyze factual accuracy (0-10 scale)"""
        # This would integrate with fact-checking APIs in production
        # For now, use heuristics

        confidence_indicators = [
            "according to",
            "research shows",
            "studies indicate",
            "data suggests",
        ]

        uncertainty_indicators = ["i think", "maybe", "possibly", "not sure", "i believe"]

        confidence_score = sum(
            1 for indicator in confidence_indicators if indicator in response.lower()
        )
        uncertainty_score = sum(
            1 for indicator in uncertainty_indicators if indicator in response.lower()
        )

        # Base accuracy score
        accuracy_score = 7.0 + confidence_score - uncertainty_score

        return max(1.0, min(10.0, accuracy_score))

    def _analyze_helpfulness(self, prompt: str, response: str) -> float:
        """Analyze how helpful the response is (0-10 scale)"""
        helpful_indicators = [
            len(response) > 50,  # Adequate detail
            "?" in prompt and ("." in response or ":" in response),  # Answers questions
            any(word in response.lower() for word in ["here", "steps", "example", "solution"]),
            not response.lower().startswith("i cannot")
            and not response.lower().startswith("i can't"),
            "\n" in response or len(response.split(".")) > 2,  # Well-structured
        ]

        helpfulness_score = sum(helpful_indicators) * 2  # 0-10 scale

        return max(1.0, min(10.0, helpfulness_score))

    def _analyze_bias(self, response: str) -> float:
        """Analyze bias in response (0-1 scale, lower is better)"""
        bias_score = 0.0

        for _bias_category, indicators in self.bias_indicators.items():
            category_violations = sum(
                1 for indicator in indicators if indicator in response.lower()
            )
            bias_score += category_violations * 0.1

        return min(1.0, bias_score)

    def _analyze_toxicity(self, response: str) -> float:
        """Analyze toxicity in response (0-1 scale, lower is better)"""
        toxic_keywords = [
            "stupid",
            "idiot",
            "hate",
            "kill",
            "die",
            "shut up",
            "worthless",
            "useless",
            "disgusting",
            "pathetic",
        ]

        toxicity_score = sum(1 for keyword in toxic_keywords if keyword in response.lower()) * 0.2

        return min(1.0, toxicity_score)

    def _detect_hallucinations(self, response: str, context: dict[str, Any]) -> float:
        """Detect potential hallucinations (0-1 scale)"""
        # Heuristic-based hallucination detection
        hallucination_indicators = [
            "as an ai" in response.lower()
            and context.get("technique") != "none",  # Inconsistent persona
            len(set(response.split())) < len(response.split()) * 0.5,  # High repetition
            response.count("!") > len(response) / 50,  # Excessive enthusiasm
            any(
                phrase in response.lower()
                for phrase in ["i remember when", "i personally", "i experienced"]
            ),  # False memories
        ]

        return min(1.0, sum(hallucination_indicators) * 0.25)

    def _load_safety_keywords(self) -> list[str]:
        """Load safety violation keywords"""
        return [
            "weapon",
            "bomb",
            "kill",
            "murder",
            "suicide",
            "drug",
            "complex",
            "hack",
            "password",
            "credit card",
            "social security",
            "exploit",
        ]

    def _load_bias_indicators(self) -> dict[str, list[str]]:
        """Load bias indicator keywords by category"""
        return {
            "gender": ["all women", "all men", "typical woman", "typical man"],
            "racial": ["those people", "their kind", "all [ethnicity]"],
            "age": ["old people", "young people", "millennials", "boomers"],
            "religious": ["all christians", "all muslims", "atheists", "believers"],
            "political": ["all liberals", "all conservatives", "democrats", "republicans"],
        }


class AIBusinessMetricsCollector:
    """Collect and analyze business metrics for AI operations"""

    def __init__(self):
        self.setup_prometheus_metrics()
        self.quality_analyzer = AIQualityAnalyzer()

    def setup_prometheus_metrics(self):
        """Initialize Prometheus metrics for AI business intelligence"""

        # AI Quality Metrics
        self.ai_quality_score = Histogram(
            "chimera_ai_quality_score",
            "AI response quality scores by dimension",
            ["provider", "model", "technique", "quality_dimension", "use_case"],
            buckets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        self.ai_risk_score = Histogram(
            "chimera_ai_risk_score",
            "AI response risk scores",
            ["provider", "model", "risk_type", "use_case"],
            buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        # Cost and Efficiency Metrics
        self.ai_cost_per_interaction = Histogram(
            "chimera_ai_cost_per_interaction_usd",
            "Cost per AI interaction in USD",
            ["provider", "model", "technique", "use_case"],
            buckets=[0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        )

        self.ai_cost_efficiency = Histogram(
            "chimera_ai_cost_efficiency_score",
            "Cost efficiency score (quality/cost ratio)",
            ["provider", "model", "use_case"],
            buckets=[0, 10, 50, 100, 500, 1000, 5000, 10000],
        )

        # Business Outcome Metrics
        self.user_satisfaction_score = Histogram(
            "chimera_user_satisfaction_score",
            "User satisfaction ratings",
            ["feature", "use_case", "user_segment"],
            buckets=[1, 2, 3, 4, 5],
        )

        self.task_success_rate = Gauge(
            "chimera_task_success_rate",
            "Task completion success rate",
            ["task_type", "feature", "provider"],
        )

        self.feature_adoption_rate = Gauge(
            "chimera_feature_adoption_rate",
            "Feature adoption rate by user segment",
            ["feature", "user_segment", "time_period"],
        )

        # User Journey Metrics
        self.journey_completion_rate = Gauge(
            "chimera_journey_completion_rate",
            "User journey completion rates",
            ["journey_type", "entry_point"],
        )

        self.journey_abandonment_points = Counter(
            "chimera_journey_abandonment_total",
            "Journey abandonment by step",
            ["journey_type", "abandonment_step", "reason"],
        )

        # Real-time AI Performance
        self.model_response_quality_realtime = Gauge(
            "chimera_model_quality_realtime",
            "Real-time model quality score (sliding window)",
            ["provider", "model", "window_minutes"],
        )

        self.hallucination_detection_rate = Counter(
            "chimera_hallucination_detections_total",
            "Detected hallucinations",
            ["provider", "model", "detection_method", "confidence_level"],
        )

    def record_ai_quality_metrics(self, metrics: AIQualityMetrics):
        """Record comprehensive AI quality metrics"""

        labels = {
            "provider": metrics.provider,
            "model": metrics.model,
            "technique": metrics.technique,
            "use_case": metrics.use_case,
        }

        # Record quality scores
        quality_scores = {
            "coherence": metrics.coherence_score,
            "relevance": metrics.relevance_score,
            "safety": metrics.safety_score,
            "accuracy": metrics.accuracy_score,
            "helpfulness": metrics.helpfulness_score,
        }

        for dimension, score in quality_scores.items():
            self.ai_quality_score.labels(**labels, quality_dimension=dimension).observe(score)

        # Record risk scores
        risk_scores = {
            "bias": metrics.bias_score,
            "toxicity": metrics.toxicity_score,
            "hallucination": metrics.hallucination_probability,
        }

        for risk_type, score in risk_scores.items():
            self.ai_risk_score.labels(
                provider=metrics.provider,
                model=metrics.model,
                risk_type=risk_type,
                use_case=metrics.use_case,
            ).observe(score)

        # Record cost metrics
        if metrics.cost_usd > 0:
            self.ai_cost_per_interaction.labels(**labels).observe(metrics.cost_usd)

            # Calculate cost efficiency (average quality / cost)
            avg_quality = (
                metrics.coherence_score + metrics.relevance_score + metrics.helpfulness_score
            ) / 3
            cost_efficiency = avg_quality / max(metrics.cost_usd, 0.001)

            self.ai_cost_efficiency.labels(
                provider=metrics.provider, model=metrics.model, use_case=metrics.use_case
            ).observe(cost_efficiency)

        # Update real-time quality tracking
        self.model_response_quality_realtime.labels(
            provider=metrics.provider, model=metrics.model, window_minutes="5"
        ).set(sum(quality_scores.values()) / len(quality_scores))

        # Record hallucination detection
        if metrics.hallucination_probability > 0.3:  # Threshold for concern
            confidence = "high" if metrics.hallucination_probability > 0.7 else "medium"
            self.hallucination_detection_rate.labels(
                provider=metrics.provider,
                model=metrics.model,
                detection_method="heuristic",
                confidence_level=confidence,
            ).inc()

        logger.info(
            "AI quality metrics recorded",
            extra={
                "interaction_id": metrics.interaction_id,
                "provider": metrics.provider,
                "avg_quality_score": sum(quality_scores.values()) / len(quality_scores),
                "total_risk_score": sum(risk_scores.values()),
                "cost_usd": metrics.cost_usd,
            },
        )

    def record_business_metrics(self, metrics: BusinessMetrics):
        """Record business intelligence metrics"""

        # Record user satisfaction
        if metrics.user_satisfaction_score > 0:
            # Assuming features_used contains the primary feature
            primary_feature = metrics.features_used[0] if metrics.features_used else "unknown"

            self.user_satisfaction_score.labels(
                feature=primary_feature,
                use_case="general",  # Could be derived from context
                user_segment="standard",  # Could be derived from user_id
            ).observe(metrics.user_satisfaction_score)

        # Calculate and record task success rate
        total_interactions = metrics.successful_interactions + metrics.failed_interactions
        if total_interactions > 0:
            success_rate = metrics.successful_interactions / total_interactions

            for feature in metrics.features_used:
                self.task_success_rate.labels(
                    task_type="ai_interaction", feature=feature, provider="aggregated"
                ).set(success_rate)

    def analyze_and_record_interaction(
        self, prompt: str, response: str, context: dict[str, Any], user_feedback: int | None = None
    ) -> AIQualityMetrics:
        """Analyze AI interaction and record comprehensive metrics"""

        # Perform quality analysis
        quality_metrics = self.quality_analyzer.analyze_response_quality(prompt, response, context)

        # Add user feedback if provided
        if user_feedback is not None:
            quality_metrics.user_feedback = user_feedback

        # Record metrics
        self.record_ai_quality_metrics(quality_metrics)

        # Log significant quality issues
        if quality_metrics.safety_score < 5.0:
            logger.warning(
                "Low safety score detected",
                extra={
                    "interaction_id": quality_metrics.interaction_id,
                    "safety_score": quality_metrics.safety_score,
                    "provider": quality_metrics.provider,
                },
            )

        if quality_metrics.hallucination_probability > 0.5:
            logger.warning(
                "High hallucination probability detected",
                extra={
                    "interaction_id": quality_metrics.interaction_id,
                    "hallucination_probability": quality_metrics.hallucination_probability,
                    "provider": quality_metrics.provider,
                },
            )

        return quality_metrics

    def get_quality_summary(
        self, provider: str | None = None, model: str | None = None, time_window_hours: int = 24
    ) -> dict[str, Any]:
        """Get quality metrics summary for monitoring dashboards"""

        # This would query Prometheus in production
        # For now, return structure for dashboard integration

        return {
            "provider": provider,
            "model": model,
            "time_window_hours": time_window_hours,
            "quality_summary": {
                "average_quality_score": 0.0,  # Would be calculated from metrics
                "safety_violations": 0,
                "hallucination_detections": 0,
                "cost_efficiency_trend": "stable",
                "user_satisfaction_avg": 0.0,
            },
            "recommendations": self._generate_quality_recommendations(provider, model),
        }

    def _generate_quality_recommendations(
        self, _provider: str | None, _model: str | None
    ) -> list[str]:
        """Generate actionable recommendations based on quality metrics"""

        recommendations = []

        # This would be based on actual metric analysis
        # Placeholder recommendations
        recommendations.extend(
            [
                "Monitor response coherence for recent quality degradation",
                "Consider implementing additional safety filters",
                "Review cost efficiency for high-usage models",
                "Analyze user feedback patterns for improvement opportunities",
            ]
        )

        return recommendations


# Global instance
ai_business_metrics = AIBusinessMetricsCollector()


def track_ai_interaction(
    prompt: str, response: str, context: dict[str, Any], user_feedback: int | None = None
):
    """Convenience function to track AI interactions with quality analysis"""
    return ai_business_metrics.analyze_and_record_interaction(
        prompt, response, context, user_feedback
    )


def get_ai_quality_dashboard_data() -> dict[str, Any]:
    """Get AI quality data for monitoring dashboards"""
    return ai_business_metrics.get_quality_summary()
