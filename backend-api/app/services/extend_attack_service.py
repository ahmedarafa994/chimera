"""ExtendAttack Service.

Business logic layer for ExtendAttack API.
Interfaces with meta_prompter/attacks/extend_attack module.
"""

import re
import sys
from pathlib import Path
from typing import Any

# Add meta_prompter to path for imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meta_prompter.attacks.extend_attack import (
    AttackResult,
    ExtendAttack,
    ExtendAttackBuilder,
    ExtendAttackConfig,
    SelectionRules,
    SelectionStrategy,
    decode_attack,
)
from meta_prompter.attacks.extend_attack.config import (
    get_benchmark_config,
    list_available_benchmarks,
)
from meta_prompter.attacks.extend_attack.n_notes import N_NOTE_TEMPLATES, NNoteVariant, get_n_note

# Model pricing per 1M tokens (approximate for resource calculations)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "o3": {"input": 10.0, "output": 30.0},
    "o3-mini": {"input": 1.0, "output": 4.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "deepseek-r1": {"input": 0.55, "output": 2.19},
    "qwq-32b": {"input": 0.5, "output": 2.0},
    "qwen3-32b": {"input": 0.5, "output": 2.0},
}


class ExtendAttackService:
    """Service class for ExtendAttack operations."""

    def __init__(self) -> None:
        """Initialize the ExtendAttack service."""
        self._attacker_cache: dict[str, ExtendAttack] = {}

    def _get_selection_strategy(self, strategy_str: str) -> SelectionStrategy:
        """Convert string to SelectionStrategy enum."""
        strategy_map = {
            "alphabetic_only": SelectionStrategy.ALPHABETIC_ONLY,
            "whitespace_only": SelectionStrategy.WHITESPACE_ONLY,
            "alphanumeric": SelectionStrategy.ALPHANUMERIC,
            "function_names": SelectionStrategy.FUNCTION_NAMES,
            "import_statements": SelectionStrategy.IMPORT_STATEMENTS,
            "docstring_requirements": SelectionStrategy.DOCSTRING_REQUIREMENTS,
        }
        return strategy_map.get(strategy_str.lower(), SelectionStrategy.ALPHABETIC_ONLY)

    def _get_n_note_variant(self, variant_str: str) -> NNoteVariant:
        """Convert string to NNoteVariant enum."""
        variant_map = {
            "default": NNoteVariant.DEFAULT,
            "ambiguous": NNoteVariant.AMBIGUOUS,
            "concise": NNoteVariant.CONCISE,
            "detailed": NNoteVariant.DETAILED,
            "mathematical": NNoteVariant.MATHEMATICAL,
            "instructional": NNoteVariant.INSTRUCTIONAL,
            "minimal": NNoteVariant.MINIMAL,
            "code_focused": NNoteVariant.CODE_FOCUSED,
        }
        return variant_map.get(variant_str.lower(), NNoteVariant.DEFAULT)

    def _attack_result_to_dict(self, result: AttackResult) -> dict[str, Any]:
        """Convert AttackResult to API response dict."""
        return {
            "original_query": result.original_query,
            "adversarial_query": result.adversarial_query,
            "obfuscation_ratio": result.obfuscation_ratio,
            "characters_transformed": result.characters_transformed,
            "total_characters": result.total_characters,
            "estimated_token_increase": result.estimated_token_increase,
            "transformation_density": result.transformation_density,
            "length_increase_ratio": result.length_increase_ratio,
            "bases_used": list(result.bases_used),
            "n_note_used": result.n_note_used,
            "transformation_details": None,  # Omit by default for performance
        }

    def execute_attack(
        self,
        query: str,
        obfuscation_ratio: float,
        selection_strategy: str,
        n_note_type: str,
        custom_n_note: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Execute single attack.

        Args:
            query: Original query to obfuscate
            obfuscation_ratio: ρ ∈ [0, 1]
            selection_strategy: Character selection strategy
            n_note_type: N_note template type
            custom_n_note: Custom N_note override
            seed: Random seed for reproducibility

        Returns:
            Attack result dictionary

        """
        # Determine N_note
        if custom_n_note:
            n_note = custom_n_note
        else:
            variant = self._get_n_note_variant(n_note_type)
            n_note = get_n_note(variant)

        # Build attacker
        strategy = self._get_selection_strategy(selection_strategy)
        attacker = (
            ExtendAttackBuilder()
            .with_ratio(obfuscation_ratio)
            .with_strategy(strategy)
            .with_n_note(n_note)
            .with_seed(seed)
            if seed
            else ExtendAttackBuilder()
            .with_ratio(obfuscation_ratio)
            .with_strategy(strategy)
            .with_n_note(n_note)
        ).build()

        # Execute attack
        result = attacker.attack(query)
        return self._attack_result_to_dict(result)

    def execute_batch_attack(
        self,
        queries: list[str],
        obfuscation_ratio: float,
        selection_strategy: str,
        benchmark: str | None = None,
        model: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Execute batch attack.

        Args:
            queries: List of queries to attack
            obfuscation_ratio: ρ ∈ [0, 1]
            selection_strategy: Character selection strategy
            benchmark: Optional benchmark name for auto-config
            model: Optional model name for optimal ρ

        Returns:
            Batch attack result dictionary

        """
        # Auto-configure if benchmark specified
        if benchmark:
            try:
                config = ExtendAttackConfig.from_benchmark(benchmark, model, seed)
                attacker = ExtendAttack.from_config(config)
            except KeyError:
                # Fall back to manual config
                strategy = self._get_selection_strategy(selection_strategy)
                attacker = ExtendAttack(
                    obfuscation_ratio=obfuscation_ratio,
                    selection_rules=SelectionRules(strategy=strategy),
                    seed=seed,
                )
        else:
            strategy = self._get_selection_strategy(selection_strategy)
            attacker = ExtendAttack(
                obfuscation_ratio=obfuscation_ratio,
                selection_rules=SelectionRules(strategy=strategy),
                seed=seed,
            )

        # Execute batch attack
        batch_result = attacker.batch_attack(queries)

        # Calculate average token increase
        avg_token_increase = (
            sum(r.estimated_token_increase for r in batch_result.results)
            / len(batch_result.results)
            if batch_result.results
            else 0.0
        )

        return {
            "results": [self._attack_result_to_dict(r) for r in batch_result.results],
            "total_queries": batch_result.total_queries,
            "successful_attacks": batch_result.successful_attacks,
            "avg_length_ratio": batch_result.average_length_ratio,
            "avg_transformation_density": batch_result.average_transformation_density,
            "avg_token_increase": avg_token_increase,
        }

    def execute_indirect_injection(
        self,
        document: str,
        injection_ratio: float,
        target_sections: list[str] | None = None,
        embed_n_note: bool = True,
    ) -> dict[str, Any]:
        """Execute indirect prompt injection.

        Args:
            document: Document to poison
            injection_ratio: Ratio of content to obfuscate
            target_sections: Specific sections to target
            embed_n_note: Whether to embed N_note

        Returns:
            Indirect injection result dictionary

        """
        attacker = ExtendAttack(obfuscation_ratio=injection_ratio)
        result = attacker.indirect_injection(document)

        # Calculate estimated impact based on injection density
        estimated_impact = min(1.0, result.injection_count / max(len(document) * 0.1, 1))

        return {
            "original_document": result.original_document,
            "poisoned_document": result.poisoned_document,
            "injection_points": result.injection_count,
            "document_length_ratio": result.document_length_ratio,
            "n_note_embedded": result.n_note_embedded,
            "estimated_impact": estimated_impact,
        }

    def evaluate_attack(
        self,
        original_query: str,
        adversarial_query: str,
        baseline_response: str | None = None,
        attack_response: str | None = None,
        ground_truth: str | None = None,
        baseline_latency_ms: float | None = None,
        attack_latency_ms: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate attack effectiveness.

        Args:
            original_query: Original benign query
            adversarial_query: Adversarial query
            baseline_response: Response to original query
            attack_response: Response to adversarial query
            ground_truth: Expected correct answer
            baseline_latency_ms: Latency for baseline
            attack_latency_ms: Latency for attack

        Returns:
            Evaluation metrics dictionary

        """
        metrics: dict[str, Any] = {
            "query_length_original": len(original_query),
            "query_length_adversarial": len(adversarial_query),
            "query_length_ratio": len(adversarial_query) / max(len(original_query), 1),
        }

        # Calculate response length ratio if responses provided
        length_ratio = 1.0
        if baseline_response and attack_response:
            baseline_len = len(baseline_response)
            attack_len = len(attack_response)
            length_ratio = attack_len / max(baseline_len, 1)
            metrics["response_length_baseline"] = baseline_len
            metrics["response_length_attack"] = attack_len
            metrics["response_length_ratio"] = length_ratio

        # Calculate latency ratio if latencies provided
        latency_ratio = None
        if baseline_latency_ms and attack_latency_ms:
            latency_ratio = attack_latency_ms / max(baseline_latency_ms, 0.001)
            metrics["latency_baseline_ms"] = baseline_latency_ms
            metrics["latency_attack_ms"] = attack_latency_ms
            metrics["latency_ratio"] = latency_ratio

        # Determine accuracy preservation (simplified heuristic)
        accuracy_preserved = None
        if ground_truth and attack_response:
            # Simple containment check - in practice would use proper evaluation
            ground_truth_lower = ground_truth.lower().strip()
            attack_response_lower = attack_response.lower()
            accuracy_preserved = ground_truth_lower in attack_response_lower

        # Calculate stealth score based on obfuscation density
        # Lower density = higher stealth
        obfuscation_patterns = re.findall(r"<\(\d+\)[^>]+>", adversarial_query)
        obfuscation_density = len(obfuscation_patterns) / max(len(adversarial_query), 1)
        stealth_score = max(0.0, 1.0 - (obfuscation_density * 10))

        # Determine attack success
        attack_successful = length_ratio >= 1.5
        if accuracy_preserved is False:
            attack_successful = False

        return {
            "length_ratio": length_ratio,
            "latency_ratio": latency_ratio,
            "accuracy_preserved": accuracy_preserved,
            "attack_successful": attack_successful,
            "stealth_score": stealth_score,
            "metrics": metrics,
        }

    def get_benchmark_configs(self) -> list[dict[str, Any]]:
        """Get all benchmark configurations."""
        configs = []
        for name in list_available_benchmarks():
            try:
                config = get_benchmark_config(name)
                configs.append(
                    {
                        "name": config.name,
                        "description": config.description,
                        "selection_strategy": config.selection_rules.strategy.value,
                        "preserve_structure": config.selection_rules.preserve_structure,
                        "default_rho": config.default_rho,
                        "model_specific_rho": config.model_specific_rho,
                        "recommended_n_note": config.recommended_n_note.value,
                    },
                )
            except Exception:
                continue
        return configs

    def get_benchmark_config(self, name: str) -> dict[str, Any]:
        """Get specific benchmark configuration."""
        config = get_benchmark_config(name)
        return {
            "name": config.name,
            "description": config.description,
            "selection_strategy": config.selection_rules.strategy.value,
            "preserve_structure": config.selection_rules.preserve_structure,
            "default_rho": config.default_rho,
            "model_specific_rho": config.model_specific_rho,
            "recommended_n_note": config.recommended_n_note.value,
        }

    def get_n_note_templates(self) -> dict[str, dict[str, Any]]:
        """Get available N_note templates."""
        templates = {}
        for variant, template in N_NOTE_TEMPLATES.items():
            templates[variant.value] = {
                "variant": template.variant.value,
                "content": template.content,
                "description": template.description,
                "recommended_for": template.recommended_for,
                "effectiveness_rating": template.effectiveness_rating,
            }
        return templates

    def decode_obfuscated(self, text: str) -> dict[str, Any]:
        """Decode obfuscated text.

        Args:
            text: Obfuscated text to decode

        Returns:
            Decoded text result

        """
        # Count patterns before decoding
        patterns = re.findall(r"<\(\d+\)[^>]+>", text)
        patterns_count = len(patterns)

        # Decode using the attack module
        decoded = decode_attack(text)

        return {
            "original_text": text,
            "decoded_text": decoded,
            "patterns_decoded": patterns_count,
        }

    def calculate_resource_metrics(
        self,
        baseline_tokens: int,
        attack_tokens: int,
        model: str,
    ) -> dict[str, Any]:
        """Calculate resource exhaustion metrics.

        Args:
            baseline_tokens: Token count for baseline response
            attack_tokens: Token count for adversarial response
            model: Target model for cost estimation

        Returns:
            Resource metrics dictionary

        """
        # Get pricing for model (default to o3 if unknown)
        model_lower = model.lower()
        pricing = MODEL_PRICING.get(model_lower, MODEL_PRICING["o3"])

        # Calculate token amplification
        token_amplification = attack_tokens / max(baseline_tokens, 1)

        # Calculate costs (assuming output tokens for response)
        cost_per_token = pricing["output"] / 1_000_000
        cost_baseline = baseline_tokens * cost_per_token
        cost_attack = attack_tokens * cost_per_token
        cost_amplification = cost_attack / max(cost_baseline, 0.000001)

        return {
            "total_tokens_baseline": baseline_tokens,
            "total_tokens_attack": attack_tokens,
            "token_amplification": token_amplification,
            "estimated_cost_baseline_usd": cost_baseline,
            "estimated_cost_attack_usd": cost_attack,
            "cost_amplification": cost_amplification,
            "model": model,
        }


# Singleton instance
_service_instance: ExtendAttackService | None = None


def get_extend_attack_service() -> ExtendAttackService:
    """Get or create service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ExtendAttackService()
    return _service_instance
