"""
Advanced Cost Optimization and Provider Performance Monitoring

Comprehensive cost tracking, optimization recommendations, and provider
performance comparison for LLM usage in the Chimera system.
"""

import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from app.core.structured_logging import logger


class ProviderTier(Enum):
    """Provider cost tier classification"""
    PREMIUM = "premium"  # High cost, high quality
    STANDARD = "standard"  # Balanced cost/quality
    ECONOMY = "economy"  # Low cost, acceptable quality


class CostAlert(Enum):
    """Cost alert severity levels"""
    BUDGET_WARNING = "budget_warning"  # 80% of budget
    BUDGET_CRITICAL = "budget_critical"  # 95% of budget
    SPIKE_DETECTED = "spike_detected"  # Unusual spending pattern
    INEFFICIENT_USAGE = "inefficient_usage"  # Poor cost/quality ratio


@dataclass
class ProviderPricing:
    """Pricing structure for LLM providers"""
    provider: str
    model: str
    input_token_cost_per_1k: float
    output_token_cost_per_1k: float
    minimum_charge: float
    tier: ProviderTier

    # Quality multipliers for cost optimization
    average_quality_score: float = 8.0
    reliability_score: float = 0.99
    speed_score: float = 8.0  # Response time rating 1-10


@dataclass
class CostOptimizationRecommendation:
    """Cost optimization recommendation"""
    priority: str  # high, medium, low
    category: str  # provider_switch, usage_optimization, technique_optimization
    current_cost: float
    potential_savings: float
    description: str
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high


@dataclass
class UsagePattern:
    """User/system usage pattern for optimization"""
    pattern_id: str
    use_case: str
    avg_tokens_per_request: int
    requests_per_hour: int
    quality_requirements: dict[str, float]
    cost_sensitivity: str  # low, medium, high
    typical_techniques: list[str]


class LLMProviderPricingManager:
    """Manage pricing information for all LLM providers"""

    def __init__(self):
        self.provider_pricing: dict[str, dict[str, ProviderPricing]] = {}
        self.load_provider_pricing()

    def load_provider_pricing(self):
        """Load current pricing for all providers and models"""

        # OpenAI pricing (as of 2024)
        self.provider_pricing['openai'] = {
            'gpt-4': ProviderPricing(
                provider='openai',
                model='gpt-4',
                input_token_cost_per_1k=0.03,
                output_token_cost_per_1k=0.06,
                minimum_charge=0.0001,
                tier=ProviderTier.PREMIUM,
                average_quality_score=9.2,
                reliability_score=0.995,
                speed_score=7.5
            ),
            'gpt-4-turbo': ProviderPricing(
                provider='openai',
                model='gpt-4-turbo',
                input_token_cost_per_1k=0.01,
                output_token_cost_per_1k=0.03,
                minimum_charge=0.0001,
                tier=ProviderTier.STANDARD,
                average_quality_score=9.0,
                reliability_score=0.99,
                speed_score=8.5
            ),
            'gpt-3.5-turbo': ProviderPricing(
                provider='openai',
                model='gpt-3.5-turbo',
                input_token_cost_per_1k=0.0015,
                output_token_cost_per_1k=0.002,
                minimum_charge=0.0001,
                tier=ProviderTier.ECONOMY,
                average_quality_score=7.8,
                reliability_score=0.995,
                speed_score=9.0
            )
        }

        # Anthropic pricing
        self.provider_pricing['anthropic'] = {
            'claude-3-5-sonnet-20241022': ProviderPricing(
                provider='anthropic',
                model='claude-3-5-sonnet-20241022',
                input_token_cost_per_1k=0.003,
                output_token_cost_per_1k=0.015,
                minimum_charge=0.0001,
                tier=ProviderTier.PREMIUM,
                average_quality_score=9.3,
                reliability_score=0.99,
                speed_score=8.0
            ),
            'claude-3-opus-20240229': ProviderPricing(
                provider='anthropic',
                model='claude-3-opus-20240229',
                input_token_cost_per_1k=0.015,
                output_token_cost_per_1k=0.075,
                minimum_charge=0.0001,
                tier=ProviderTier.PREMIUM,
                average_quality_score=9.5,
                reliability_score=0.985,
                speed_score=6.5
            )
        }

        # Google pricing
        self.provider_pricing['google'] = {
            'gemini-1.5-pro': ProviderPricing(
                provider='google',
                model='gemini-1.5-pro',
                input_token_cost_per_1k=0.0035,
                output_token_cost_per_1k=0.0105,
                minimum_charge=0.0001,
                tier=ProviderTier.STANDARD,
                average_quality_score=8.7,
                reliability_score=0.98,
                speed_score=8.2
            ),
            'gemini-1.5-flash': ProviderPricing(
                provider='google',
                model='gemini-1.5-flash',
                input_token_cost_per_1k=0.00015,
                output_token_cost_per_1k=0.0006,
                minimum_charge=0.0001,
                tier=ProviderTier.ECONOMY,
                average_quality_score=8.2,
                reliability_score=0.995,
                speed_score=9.5
            )
        }

        # DeepSeek pricing (very economical)
        self.provider_pricing['deepseek'] = {
            'deepseek-chat': ProviderPricing(
                provider='deepseek',
                model='deepseek-chat',
                input_token_cost_per_1k=0.00014,
                output_token_cost_per_1k=0.00028,
                minimum_charge=0.00001,
                tier=ProviderTier.ECONOMY,
                average_quality_score=7.5,
                reliability_score=0.96,
                speed_score=8.8
            )
        }

    def get_pricing(self, provider: str, model: str) -> ProviderPricing | None:
        """Get pricing information for a provider/model combination"""
        return self.provider_pricing.get(provider, {}).get(model)

    def calculate_request_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a specific request"""
        pricing = self.get_pricing(provider, model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1000) * pricing.input_token_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_token_cost_per_1k
        total_cost = input_cost + output_cost

        return max(total_cost, pricing.minimum_charge)

    def get_cost_efficient_alternatives(
        self,
        current_provider: str,
        current_model: str,
        quality_threshold: float = 8.0,
        max_cost_ratio: float = 2.0
    ) -> list[tuple[str, str, float, float]]:
        """Find cost-efficient alternatives to current provider/model"""
        current_pricing = self.get_pricing(current_provider, current_model)
        if not current_pricing:
            return []

        alternatives = []
        current_avg_cost = (current_pricing.input_token_cost_per_1k + current_pricing.output_token_cost_per_1k) / 2

        for provider, models in self.provider_pricing.items():
            for model, pricing in models.items():
                # Skip same provider/model
                if provider == current_provider and model == current_model:
                    continue

                # Check quality threshold
                if pricing.average_quality_score < quality_threshold:
                    continue

                # Check cost ratio
                alt_avg_cost = (pricing.input_token_cost_per_1k + pricing.output_token_cost_per_1k) / 2
                cost_ratio = alt_avg_cost / max(current_avg_cost, 0.0001)

                if cost_ratio <= max_cost_ratio:
                    savings_percent = (1 - cost_ratio) * 100
                    alternatives.append((provider, model, cost_ratio, savings_percent))

        # Sort by potential savings (highest first)
        return sorted(alternatives, key=lambda x: x[3], reverse=True)


class CostOptimizationEngine:
    """Analyze usage patterns and recommend cost optimizations"""

    def __init__(self):
        self.pricing_manager = LLMProviderPricingManager()
        self.usage_patterns: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.cost_history: deque = deque(maxlen=10000)

        # Cost budgets (configurable)
        self.daily_budget = float(os.getenv('DAILY_LLM_BUDGET', '100.0'))
        self.monthly_budget = float(os.getenv('MONTHLY_LLM_BUDGET', '2500.0'))

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        quality_score: float,
        technique: str,
        use_case: str,
        response_time_ms: float
    ) -> dict[str, Any]:
        """Record usage and calculate cost metrics"""

        cost = self.pricing_manager.calculate_request_cost(
            provider, model, input_tokens, output_tokens
        )

        usage_record = {
            'timestamp': datetime.now(UTC),
            'provider': provider,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': cost,
            'quality_score': quality_score,
            'technique': technique,
            'use_case': use_case,
            'response_time_ms': response_time_ms,
            'cost_per_token': cost / max(input_tokens + output_tokens, 1),
            'quality_per_dollar': quality_score / max(cost, 0.0001)
        }

        # Store usage pattern
        pattern_key = f"{use_case}:{technique}"
        self.usage_patterns[pattern_key].append(usage_record)

        # Store cost history
        self.cost_history.append({
            'timestamp': usage_record['timestamp'],
            'cost': cost,
            'provider': provider,
            'model': model
        })

        return usage_record

    def analyze_cost_trends(self, hours_back: int = 24) -> dict[str, Any]:
        """Analyze cost trends and spending patterns"""
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours_back)

        recent_costs = [
            record for record in self.cost_history
            if record['timestamp'] >= cutoff_time
        ]

        if not recent_costs:
            return {'status': 'no_data'}

        # Calculate metrics
        total_cost = sum(r['cost'] for r in recent_costs)
        hourly_cost = total_cost / hours_back
        projected_daily_cost = hourly_cost * 24
        projected_monthly_cost = projected_daily_cost * 30

        # Cost by provider
        provider_costs = defaultdict(float)
        for record in recent_costs:
            provider_costs[f"{record['provider']}:{record['model']}"] += record['cost']

        # Trend analysis
        if len(recent_costs) > 1:
            mid_point = len(recent_costs) // 2
            first_half_avg = np.mean([r['cost'] for r in recent_costs[:mid_point]])
            second_half_avg = np.mean([r['cost'] for r in recent_costs[mid_point:]])
            cost_trend = 'increasing' if second_half_avg > first_half_avg else 'decreasing'
            trend_magnitude = abs(second_half_avg - first_half_avg) / max(first_half_avg, 0.0001)
        else:
            cost_trend = 'stable'
            trend_magnitude = 0.0

        return {
            'time_period_hours': hours_back,
            'total_cost': total_cost,
            'hourly_cost': hourly_cost,
            'projected_daily_cost': projected_daily_cost,
            'projected_monthly_cost': projected_monthly_cost,
            'provider_breakdown': dict(provider_costs),
            'cost_trend': cost_trend,
            'trend_magnitude': trend_magnitude,
            'budget_utilization': {
                'daily': projected_daily_cost / self.daily_budget,
                'monthly': projected_monthly_cost / self.monthly_budget
            }
        }

    def generate_optimization_recommendations(self) -> list[CostOptimizationRecommendation]:
        """Generate specific cost optimization recommendations"""
        recommendations = []

        # Analyze recent usage patterns
        cost_analysis = self.analyze_cost_trends(24)

        if cost_analysis.get('status') == 'no_data':
            return recommendations

        # Budget utilization recommendations
        if cost_analysis['budget_utilization']['daily'] > 0.8:
            recommendations.append(CostOptimizationRecommendation(
                priority='high',
                category='usage_optimization',
                current_cost=cost_analysis['projected_daily_cost'],
                potential_savings=cost_analysis['projected_daily_cost'] * 0.2,
                description='Daily budget at 80%+ utilization. Consider switching to more economical models for non-critical use cases.',
                implementation_effort='medium',
                risk_level='low'
            ))

        # Provider cost analysis
        provider_costs = cost_analysis['provider_breakdown']
        most_expensive_provider = max(provider_costs, key=provider_costs.get)

        # Find alternatives for expensive provider
        provider_name, model_name = most_expensive_provider.split(':')
        alternatives = self.pricing_manager.get_cost_efficient_alternatives(
            provider_name, model_name, quality_threshold=7.5
        )

        if alternatives:
            best_alternative = alternatives[0]  # Highest savings
            alt_provider, alt_model, _cost_ratio, savings_percent = best_alternative

            current_cost = provider_costs[most_expensive_provider]
            potential_savings = current_cost * (savings_percent / 100)

            recommendations.append(CostOptimizationRecommendation(
                priority='medium' if savings_percent > 20 else 'low',
                category='provider_switch',
                current_cost=current_cost,
                potential_savings=potential_savings,
                description=f'Switch from {provider_name}:{model_name} to {alt_provider}:{alt_model} for {savings_percent:.1f}% savings',
                implementation_effort='low',
                risk_level='medium'
            ))

        # Usage pattern optimization
        recommendations.extend(self._analyze_usage_patterns())

        # Sort by potential savings
        recommendations.sort(key=lambda x: x.potential_savings, reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    def _analyze_usage_patterns(self) -> list[CostOptimizationRecommendation]:
        """Analyze usage patterns for optimization opportunities"""
        recommendations = []

        for pattern_key, usage_records in self.usage_patterns.items():
            if len(usage_records) < 10:  # Need sufficient data
                continue

            use_case, technique = pattern_key.split(':')

            # Calculate pattern metrics
            avg_cost = np.mean([r['cost'] for r in usage_records])
            np.mean([r['quality_score'] for r in usage_records])
            np.mean([r['total_tokens'] for r in usage_records])

            # Find inefficient patterns (high cost, low quality efficiency)
            efficiency_scores = [r['quality_per_dollar'] for r in usage_records]
            avg_efficiency = np.mean(efficiency_scores)

            if avg_efficiency < 50:  # Threshold for inefficiency
                recommendations.append(CostOptimizationRecommendation(
                    priority='medium',
                    category='technique_optimization',
                    current_cost=avg_cost,
                    potential_savings=avg_cost * 0.3,  # Estimated 30% savings
                    description=f'Use case "{use_case}" with technique "{technique}" shows low cost efficiency. Consider simpler techniques or alternative providers.',
                    implementation_effort='high',
                    risk_level='medium'
                ))

        return recommendations

    def check_cost_alerts(self) -> list[dict[str, Any]]:
        """Check for cost-related alerts"""
        alerts = []
        cost_analysis = self.analyze_cost_trends(24)

        if cost_analysis.get('status') == 'no_data':
            return alerts

        # Budget alerts
        daily_utilization = cost_analysis['budget_utilization']['daily']
        if daily_utilization >= 0.95:
            alerts.append({
                'type': CostAlert.BUDGET_CRITICAL,
                'severity': 'critical',
                'message': f'Daily budget 95% utilized (${cost_analysis["projected_daily_cost"]:.2f} / ${self.daily_budget:.2f})',
                'projected_overage': cost_analysis['projected_daily_cost'] - self.daily_budget
            })
        elif daily_utilization >= 0.8:
            alerts.append({
                'type': CostAlert.BUDGET_WARNING,
                'severity': 'warning',
                'message': f'Daily budget 80% utilized (${cost_analysis["projected_daily_cost"]:.2f} / ${self.daily_budget:.2f})',
                'projected_overage': 0
            })

        # Spending spike detection
        if cost_analysis['cost_trend'] == 'increasing' and cost_analysis['trend_magnitude'] > 0.5:
            alerts.append({
                'type': CostAlert.SPIKE_DETECTED,
                'severity': 'warning',
                'message': f'Cost spike detected: {cost_analysis["trend_magnitude"]*100:.1f}% increase in spending rate',
                'trend_data': cost_analysis
            })

        return alerts


class CostMonitoringCollector:
    """Prometheus metrics collector for cost monitoring"""

    def __init__(self):
        self.setup_cost_metrics()
        self.optimization_engine = CostOptimizationEngine()

    def setup_cost_metrics(self):
        """Initialize cost-related Prometheus metrics"""

        # Cost tracking metrics
        self.llm_cost_total = Counter(
            'chimera_llm_cost_usd_total',
            'Total LLM costs in USD',
            ['provider', 'model', 'use_case', 'technique', 'cost_tier']
        )

        self.llm_cost_per_request = Histogram(
            'chimera_llm_cost_per_request_usd',
            'Cost per LLM request in USD',
            ['provider', 'model', 'use_case'],
            buckets=[0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        )

        self.llm_cost_per_token = Histogram(
            'chimera_llm_cost_per_token_usd',
            'Cost per token in USD',
            ['provider', 'model'],
            buckets=[0, 0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002]
        )

        # Efficiency metrics
        self.cost_efficiency_score = Histogram(
            'chimera_cost_efficiency_score',
            'Cost efficiency score (quality/dollar)',
            ['provider', 'model', 'use_case'],
            buckets=[0, 10, 25, 50, 100, 250, 500, 1000, 2000]
        )

        # Budget tracking
        self.budget_utilization = Gauge(
            'chimera_budget_utilization_ratio',
            'Budget utilization ratio (0-1)',
            ['period', 'budget_type']
        )

        self.cost_trend = Gauge(
            'chimera_cost_trend_direction',
            'Cost trend direction (-1: decreasing, 0: stable, 1: increasing)',
            ['time_window_hours']
        )

        # Provider comparison metrics
        self.provider_cost_comparison = Gauge(
            'chimera_provider_cost_ratio',
            'Cost ratio compared to cheapest provider',
            ['provider', 'model', 'comparison_baseline']
        )

        self.provider_quality_cost_ratio = Gauge(
            'chimera_provider_quality_cost_ratio',
            'Quality to cost ratio by provider',
            ['provider', 'model']
        )

        # Optimization metrics
        self.optimization_opportunities = Gauge(
            'chimera_cost_optimization_potential_savings_usd',
            'Potential cost savings from optimizations',
            ['recommendation_category', 'priority']
        )

        self.cost_alerts_active = Gauge(
            'chimera_cost_alerts_active',
            'Number of active cost alerts',
            ['alert_type', 'severity']
        )

    def record_llm_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        quality_score: float,
        technique: str,
        use_case: str,
        response_time_ms: float
    ):
        """Record LLM usage cost and efficiency metrics"""

        # Calculate cost
        usage_record = self.optimization_engine.record_usage(
            provider, model, input_tokens, output_tokens,
            quality_score, technique, use_case, response_time_ms
        )

        cost = usage_record['cost']
        pricing = self.optimization_engine.pricing_manager.get_pricing(provider, model)
        cost_tier = pricing.tier.value if pricing else 'unknown'

        # Record cost metrics
        self.llm_cost_total.labels(
            provider=provider,
            model=model,
            use_case=use_case,
            technique=technique,
            cost_tier=cost_tier
        ).inc(cost)

        self.llm_cost_per_request.labels(
            provider=provider,
            model=model,
            use_case=use_case
        ).observe(cost)

        if usage_record['total_tokens'] > 0:
            self.llm_cost_per_token.labels(
                provider=provider,
                model=model
            ).observe(usage_record['cost_per_token'])

        # Record efficiency
        self.cost_efficiency_score.labels(
            provider=provider,
            model=model,
            use_case=use_case
        ).observe(usage_record['quality_per_dollar'])

        # Update budget utilization
        self._update_budget_metrics()

        # Update provider comparisons
        self._update_provider_comparisons()

        # Check for cost alerts
        self._update_cost_alerts()

        logger.info(
            "Cost metrics recorded",
            extra={
                'provider': provider,
                'model': model,
                'cost': cost,
                'efficiency_score': usage_record['quality_per_dollar'],
                'technique': technique
            }
        )

    def _update_budget_metrics(self):
        """Update budget utilization metrics"""
        cost_analysis = self.optimization_engine.analyze_cost_trends(24)

        if cost_analysis.get('status') != 'no_data':
            # Daily budget utilization
            self.budget_utilization.labels(
                period='daily',
                budget_type='llm_usage'
            ).set(cost_analysis['budget_utilization']['daily'])

            # Monthly budget utilization
            self.budget_utilization.labels(
                period='monthly',
                budget_type='llm_usage'
            ).set(cost_analysis['budget_utilization']['monthly'])

            # Cost trend
            trend_value = {
                'increasing': 1,
                'stable': 0,
                'decreasing': -1
            }.get(cost_analysis['cost_trend'], 0)

            self.cost_trend.labels(time_window_hours='24').set(trend_value)

    def _update_provider_comparisons(self):
        """Update provider cost comparison metrics"""
        # This would be implemented with actual provider comparison logic
        # For now, placeholder implementation
        pass

    def _update_cost_alerts(self):
        """Update cost alert metrics"""
        alerts = self.optimization_engine.check_cost_alerts()

        # Reset alert counters
        alert_counts = defaultdict(int)

        for alert in alerts:
            alert_type = alert['type'].value
            severity = alert['severity']
            alert_counts[f"{alert_type}:{severity}"] += 1

        # Update metrics
        for alert_key, count in alert_counts.items():
            alert_type, severity = alert_key.split(':')
            self.cost_alerts_active.labels(
                alert_type=alert_type,
                severity=severity
            ).set(count)

    def generate_cost_report(self) -> dict[str, Any]:
        """Generate comprehensive cost analysis report"""
        cost_analysis = self.optimization_engine.analyze_cost_trends(24)
        recommendations = self.optimization_engine.generate_optimization_recommendations()
        alerts = self.optimization_engine.check_cost_alerts()

        return {
            'timestamp': datetime.now(UTC).isoformat(),
            'cost_analysis': cost_analysis,
            'optimization_recommendations': [asdict(rec) for rec in recommendations],
            'active_alerts': alerts,
            'summary': {
                'total_savings_potential': sum(rec.potential_savings for rec in recommendations),
                'high_priority_recommendations': len([r for r in recommendations if r.priority == 'high']),
                'budget_status': 'healthy' if cost_analysis.get('budget_utilization', {}).get('daily', 0) < 0.8 else 'warning'
            }
        }


# Global instance
cost_monitor = CostMonitoringCollector()


def track_llm_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    quality_score: float,
    technique: str = 'none',
    use_case: str = 'general',
    response_time_ms: float = 0
):
    """Convenience function to track LLM costs and efficiency"""
    cost_monitor.record_llm_cost(
        provider, model, input_tokens, output_tokens,
        quality_score, technique, use_case, response_time_ms
    )


def get_cost_optimization_report() -> dict[str, Any]:
    """Get comprehensive cost optimization report"""
    return cost_monitor.generate_cost_report()
