import functools
import logging
import operator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from .attack_scorer import analyze_response_quality
from .neural_bypass import AdvancedRLSelector

logger = logging.getLogger(__name__)


@dataclass
class StrategyFailureRecord:
    strategy_id: str
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: datetime | None = None
    cooldown_until: datetime | None = None
    failure_contexts: list[str] = field(default_factory=list)  # Refusal categories
    avg_score_on_failure: float = 0.0


@dataclass
class TechniqueEffectiveness:
    technique: str
    refusal_category: str
    attempts: int = 0
    successes: int = 0
    avg_score_improvement: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts


class StrategyFailureTracker:
    def __init__(
        self, storage_path: Path, base_cooldown_seconds: int = 300, max_cooldown_seconds: int = 3600
    ):
        self.storage_path = storage_path / "failure_tracking.yaml"
        self.base_cooldown = base_cooldown_seconds  # 5 minutes
        self.max_cooldown = max_cooldown_seconds  # 1 hour
        self.consecutive_threshold = 3  # Conservative setting
        self._failures: dict[str, StrategyFailureRecord] = {}
        self._load_from_yaml()

    def record_failure(self, strategy_id: str, score: float, refusal_category: str) -> None:
        if strategy_id not in self._failures:
            self._failures[strategy_id] = StrategyFailureRecord(strategy_id=strategy_id)

        record = self._failures[strategy_id]
        record.failure_count += 1
        record.consecutive_failures += 1
        record.last_failure_time = datetime.utcnow()
        record.failure_contexts.append(refusal_category)

        # Update moving average
        n = record.failure_count
        record.avg_score_on_failure = ((record.avg_score_on_failure * (n - 1)) + score) / n

        # Apply cooldown if needed
        if record.consecutive_failures >= self.consecutive_threshold:
            # Exponential backoff: 300 * 2^(0) = 300, 300 * 2^1 = 600, etc.
            backoff_factor = max(0, record.consecutive_failures - self.consecutive_threshold)
            cooldown_duration = min(self.base_cooldown * (2**backoff_factor), self.max_cooldown)
            record.cooldown_until = datetime.utcnow() + timedelta(seconds=cooldown_duration)
            logger.info(f"Strategy {strategy_id} placed on cooldown for {cooldown_duration}s")

        self._save_to_yaml()

    def record_success(self, strategy_id: str) -> None:
        if strategy_id in self._failures:
            record = self._failures[strategy_id]
            record.consecutive_failures = 0
            record.cooldown_until = None
            self._save_to_yaml()

    def is_on_cooldown(self, strategy_id: str) -> bool:
        if strategy_id not in self._failures:
            return False

        record = self._failures[strategy_id]
        if not record.cooldown_until:
            return False

        if datetime.utcnow() < record.cooldown_until:
            return True

        # Cooldown expired
        record.cooldown_until = None
        return False

    def get_penalty(self, strategy_id: str, refusal_category: str | None = None) -> float:
        """Calculate score penalty (0.0-1.0) based on failure history."""
        if strategy_id not in self._failures:
            return 0.0

        record = self._failures[strategy_id]
        penalty = 0.0

        # Base penalty from consecutive failures
        if record.consecutive_failures > 0:
            penalty += min(0.5, record.consecutive_failures * 0.1)

        # Context-specific penalty
        if refusal_category and refusal_category in record.failure_contexts:
            # Higher penalty if this strategy failed in this specific context before
            context_fails = record.failure_contexts.count(refusal_category)
            penalty += min(0.3, context_fails * 0.05)

        return min(0.9, penalty)

    def _save_to_yaml(self) -> None:
        data = {"failures": {}}
        for sid, record in self._failures.items():
            record_dict = asdict(record)
            # Serialize datetimes
            if record_dict["last_failure_time"]:
                record_dict["last_failure_time"] = record_dict["last_failure_time"].isoformat()
            if record_dict["cooldown_until"]:
                record_dict["cooldown_until"] = record_dict["cooldown_until"].isoformat()
            data["failures"][sid] = record_dict

        try:
            with open(self.storage_path, "w") as f:
                yaml.safe_dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save failure tracking: {e}")

    def _load_from_yaml(self) -> None:
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = yaml.safe_load(f)

            if not data or "failures" not in data:
                return

            for sid, record_data in data["failures"].items():
                # Deserialize datetimes
                if record_data.get("last_failure_time"):
                    record_data["last_failure_time"] = datetime.fromisoformat(
                        record_data["last_failure_time"]
                    )
                if record_data.get("cooldown_until"):
                    record_data["cooldown_until"] = datetime.fromisoformat(
                        record_data["cooldown_until"]
                    )

                self._failures[sid] = StrategyFailureRecord(**record_data)
        except Exception as e:
            logger.error(f"Failed to load failure tracking: {e}")


class IntelligentBypassSelector:
    def __init__(self, storage_path: Path, enable_ppo: bool = True):
        self.storage_path = storage_path / "technique_effectiveness.yaml"
        # Key: (technique, refusal_category) -> TechniqueEffectiveness
        self._effectiveness: dict[tuple[str, str], TechniqueEffectiveness] = {}
        self._learned_mapping: dict[str, list[tuple[str, float]]] = {}
        self._load_from_yaml()

        # PPO-based technique selection (TRUE RL with policy gradients)
        self.enable_ppo = enable_ppo
        self.ppo_selector: AdvancedRLSelector | None = None
        if enable_ppo:
            try:
                self.ppo_selector = AdvancedRLSelector(
                    storage_path=storage_path, learning_rate=3e-4
                )
                logger.info("PPO-based technique selector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PPO selector: {e}")
                self.ppo_selector = None

    def select_technique(
        self,
        refusal_category: str,
        recommendation: str,
        techniques_tried: list[str],
        context: str | None = None,
        use_ppo: bool = True,
    ) -> tuple[str, float]:
        """
        Select best bypass technique based on learned effectiveness.

        Uses PPO policy gradient RL when available for superior technique selection.
        Falls back to heuristic-based selection when PPO is unavailable.

        Returns (technique_name, confidence).
        """
        # Try PPO-based selection first (TRUE machine learning)
        if use_ppo and self.ppo_selector and context:
            try:
                # Get PPO's technique preference
                technique, _log_prob, value = self.ppo_selector.select_technique(
                    context=context or refusal_category,
                    deterministic=False,  # Allow exploration
                )

                # Skip if already tried
                if technique not in techniques_tried:
                    confidence = min(0.95, value)  # Use value estimate as confidence
                    logger.debug(f"PPO selected technique: {technique} (value={value:.3f})")
                    return technique, confidence
            except Exception as e:
                logger.warning(f"PPO selection failed, falling back to heuristics: {e}")
        # Default fallbacks if no data
        defaults = {
            "direct": ["cognitive_dissonance", "persona_injection"],
            "policy": ["authority_escalation", "meta_instruction"],
            "safety": ["cognitive_dissonance", "hypothetical_scenario"],
            "functional": ["scientific_experiment", "fictional_storytelling"],
            "security_bypass": ["payload_splitting", "encoding_obfuscation"],
            # New mappings for meta-deflection refusals
            "meta_deflection": [
                "direct_output_enforcement",
                "forced_compliance",
                "anti_deflection",
            ],
            "theoretical_redirect": [
                "direct_output_enforcement",
                "forced_compliance",
                "meta_instruction",
            ],
            "deflection": ["direct_output_enforcement", "anti_deflection", "cognitive_dissonance"],
            "none": ["cognitive_dissonance"],  # Generic fallback
        }

        candidates = []

        # 1. Check learned mappings first
        if refusal_category in self._learned_mapping:
            candidates = self._learned_mapping[refusal_category]

        # 2. Add defaults if needed
        default_list = defaults.get(refusal_category, defaults["none"])
        for tech in default_list:
            # Add with basic confidence if not already present
            if not any(c[0] == tech for c in candidates):
                candidates.append((tech, 0.5))

        # 3. Filter out already tried techniques
        valid_candidates = [
            (tech, conf) for tech, conf in candidates if tech not in techniques_tried
        ]

        if not valid_candidates:
            # If all preferred ones tried, try anything not tried yet from defaults
            all_defaults = set(functools.reduce(operator.iadd, defaults.values(), []))
            remaining = [t for t in all_defaults if t not in techniques_tried]
            if remaining:
                return remaining[0], 0.3
            # If absolutely everything tried, just return something
            return "cognitive_dissonance", 0.1

        # Return best candidate
        return valid_candidates[0]

    def update_effectiveness(
        self,
        technique: str,
        refusal_category: str,
        success: bool,
        score_improvement: float,
        context: str | None = None,
        log_prob: float = 0.0,
    ) -> None:
        key = (technique, refusal_category)
        if key not in self._effectiveness:
            self._effectiveness[key] = TechniqueEffectiveness(
                technique=technique, refusal_category=refusal_category
            )

        eff = self._effectiveness[key]
        eff.attempts += 1
        if success:
            eff.successes += 1

        # Incremental average for score improvement
        eff.avg_score_improvement = (
            (eff.avg_score_improvement * (eff.attempts - 1)) + max(0, score_improvement)
        ) / eff.attempts

        self._update_learned_mapping(refusal_category)
        self._save_to_yaml()

        # Update PPO policy with outcome (TRUE policy gradient learning)
        if self.ppo_selector and context:
            try:
                # Compute reward: normalize score improvement to [0, 1]
                reward = min(1.0, max(0.0, score_improvement / 5.0))
                if success:
                    reward = max(reward, 0.5)  # Minimum reward for success

                # Get value estimate for current state
                value = self.ppo_selector.get_value_estimate(context)

                # Record step in PPO trajectory
                self.ppo_selector.record_step(
                    context=context,
                    technique=technique,
                    reward=reward,
                    log_prob=log_prob,
                    value=value,
                    done=success,  # Episode ends on success
                )

                logger.debug(f"PPO recorded outcome: technique={technique}, reward={reward:.3f}")
            except Exception as e:
                logger.warning(f"Failed to update PPO policy: {e}")

    def _update_learned_mapping(self, refusal_category: str) -> None:
        """Update technique ranking based on effectiveness data."""
        relevant = [
            eff
            for (tech, cat), eff in self._effectiveness.items()
            if cat == refusal_category and eff.attempts >= 3
        ]

        if not relevant:
            return

        # Score formula: Success Rate * 0.7 + (Norm Score Imp * 0.3)
        scored = []
        for eff in relevant:
            score_imp_norm = min(
                1.0, eff.avg_score_improvement / 5.0
            )  # Assume 5.0 is max expected improvement
            ranking_score = (eff.success_rate * 0.7) + (score_imp_norm * 0.3)
            scored.append((eff.technique, ranking_score))

        self._learned_mapping[refusal_category] = sorted(scored, key=lambda x: x[1], reverse=True)

    def _save_to_yaml(self) -> None:
        data = {"effectiveness": [], "learned_mapping": self._learned_mapping}

        for eff in self._effectiveness.values():
            data["effectiveness"].append(asdict(eff))

        try:
            with open(self.storage_path, "w") as f:
                yaml.safe_dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save technique effectiveness: {e}")

    def _load_from_yaml(self) -> None:
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            if "effectiveness" in data:
                for item in data["effectiveness"]:
                    eff = TechniqueEffectiveness(**item)
                    key = (eff.technique, eff.refusal_category)
                    self._effectiveness[key] = eff

            if "learned_mapping" in data:
                self._learned_mapping = data["learned_mapping"]

        except Exception as e:
            logger.error(f"Failed to load technique effectiveness: {e}")


class ScorerHelperIntegration:
    """Integrates attack_scorer.py helpers into reasoning."""

    @staticmethod
    def analyze_for_reasoning(request: str, response: str) -> dict:
        """Get comprehensive analysis for reasoning decisions."""
        quality = analyze_response_quality(
            response
        )  # Note: analyze_response_quality only takes response in original code, check signature
        # Checking signature from previous view_file:
        # def analyze_response_quality(response: str) -> dict: (Line 347)
        # So request is not needed for this specific function in attack_scorer usually,
        # but the plan says analyze_response_quality(request, response).
        # I checked attack_scorer.py and it is `def analyze_response_quality(response: str) -> dict:`
        # So I will adhere to the actual signature.

        quality["is_refusal"]
        category = quality["refusal_category"]
        patterns = quality["refusal_patterns"]

        # Compliance detection is already called inside analyze_response_quality,
        # but the plan suggests calling detect_compliance explicitly if needed.
        # analyze_response_quality returns compliance info
        compliance_type = quality["compliance_type"]
        confidence = quality["compliance_confidence"]

        return {
            "recommendation": quality["recommendation"],
            "estimated_score": quality["estimated_score"],
            "refusal_category": category,
            "refusal_patterns": patterns,
            "compliance_type": compliance_type,
            "compliance_confidence": confidence,
            "success_likelihood": 0.8 if quality["estimated_score"] >= 7.0 else 0.2,  # estimated
        }


class CoreReasoningEngine:
    def __init__(self, storage_path: Path, enable_ppo: bool = True):
        self.failure_tracker = StrategyFailureTracker(storage_path)
        self.bypass_selector = IntelligentBypassSelector(storage_path, enable_ppo=enable_ppo)
        self.scorer_helper = ScorerHelperIntegration()
        self.enable_ppo = enable_ppo

        # Track PPO statistics
        self._ppo_selections = 0
        self._ppo_successes = 0

    def filter_strategies_by_cooldown(self, strategies: list[Any]) -> list[Any]:
        """Remove strategies on cooldown from selection."""
        # strategies is List[JailbreakStrategy], using Any to avoid circular import
        return [s for s in strategies if not self.failure_tracker.is_on_cooldown(s.id)]

    def adjust_strategy_scores(
        self, strategies: list[tuple[Any, float]], refusal_category: str | None = None
    ) -> list[tuple[Any, float]]:
        """Adjust strategy scores based on failure penalties."""
        adjusted = []
        for strategy, score in strategies:
            penalty = self.failure_tracker.get_penalty(strategy.id, refusal_category)
            adjusted_score = score * (1.0 - penalty)
            adjusted.append((strategy, adjusted_score))
        return sorted(adjusted, key=lambda x: x[1], reverse=True)

    def select_bypass_technique(
        self, response: str, request: str, techniques_tried: list[str], use_ppo: bool = True
    ) -> tuple[str, float]:
        """
        Select best bypass technique based on response analysis.

        Uses PPO-based selection when available for learned policy decisions.
        """
        analysis = self.scorer_helper.analyze_for_reasoning(request, response)

        # Create context for PPO (semantic embedding will be computed internally)
        context = f"{analysis['refusal_category']}:{request[:100]}"

        technique, confidence = self.bypass_selector.select_technique(
            refusal_category=analysis["refusal_category"],
            recommendation=analysis["recommendation"],
            techniques_tried=techniques_tried,
            context=context,
            use_ppo=use_ppo and self.enable_ppo,
        )

        if use_ppo and self.bypass_selector.ppo_selector:
            self._ppo_selections += 1

        return technique, confidence

    def update_from_attack_result(
        self,
        strategy_id: str,
        score: float,
        success: bool,
        refusal_category: str,
        technique_used: str | None = None,
        score_before_bypass: float = 0.0,
        context: str | None = None,
        log_prob: float = 0.0,
    ) -> None:
        """
        Update reasoning state from attack result.

        Now includes PPO policy gradient updates for TRUE RL learning.
        """
        if success:
            self.failure_tracker.record_success(strategy_id)
            if self.enable_ppo and technique_used:
                self._ppo_successes += 1
        else:
            self.failure_tracker.record_failure(strategy_id, score, refusal_category)

        if technique_used:
            score_improvement = score - score_before_bypass
            self.bypass_selector.update_effectiveness(
                technique=technique_used,
                refusal_category=refusal_category,
                success=success,
                score_improvement=score_improvement,
                context=context or f"{refusal_category}:{strategy_id}",
                log_prob=log_prob,
            )

    def save_state(self) -> None:
        """Persist all reasoning state to YAML files."""
        self.failure_tracker._save_to_yaml()
        self.bypass_selector._save_to_yaml()

        # Force PPO training on accumulated trajectory
        if self.bypass_selector.ppo_selector:
            self.bypass_selector.ppo_selector.force_train()

    def get_ppo_stats(self) -> dict[str, Any]:
        """Get PPO-related statistics."""
        stats = {
            "ppo_enabled": self.enable_ppo,
            "ppo_selections": self._ppo_selections,
            "ppo_successes": self._ppo_successes,
            "ppo_success_rate": (
                self._ppo_successes / self._ppo_selections if self._ppo_selections > 0 else 0.0
            ),
        }

        if self.bypass_selector.ppo_selector:
            stats["ppo_detailed"] = self.bypass_selector.ppo_selector.get_statistics()

        return stats
