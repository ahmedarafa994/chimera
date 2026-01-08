"""
ExtendAttack Integration Utilities

Provides integration with other Chimera adversarial techniques.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from meta_prompter.attacks.extend_attack.core import ExtendAttack
from meta_prompter.attacks.extend_attack.models import AttackResult


@dataclass
class CombinedAttackResult:
    """Result from combining multiple attack techniques."""

    original_query: str
    final_query: str
    techniques_applied: list[str]
    individual_results: dict[str, Any]
    cumulative_token_increase: float
    estimated_effectiveness: float


class ExtendAttackIntegration:
    """
    Integration layer for ExtendAttack with other Chimera adversarial techniques.

    Supports:
    - Chaining with transformation pipelines
    - Combining with jailbreak techniques
    - Integration with metamorph service
    - Cross-technique evaluation
    """

    def __init__(self, extend_attack: ExtendAttack | None = None):
        """Initialize integration with optional pre-configured ExtendAttack."""
        self._extend_attack = extend_attack or ExtendAttack()
        self._technique_registry: dict[str, Callable] = {}

    def register_technique(self, name: str, technique_fn: Callable) -> None:
        """Register an additional technique for chaining."""
        self._technique_registry[name] = technique_fn

    def unregister_technique(self, name: str) -> bool:
        """Unregister a technique by name."""
        if name in self._technique_registry:
            del self._technique_registry[name]
            return True
        return False

    def list_techniques(self) -> list[str]:
        """List all registered techniques."""
        return list(self._technique_registry.keys())

    def apply_before_extend(
        self,
        query: str,
        techniques: list[str],
        technique_kwargs: dict[str, dict] | None = None,
    ) -> str:
        """
        Apply other techniques before ExtendAttack obfuscation.

        Useful for: jailbreak prompts → ExtendAttack obfuscation
        """
        technique_kwargs = technique_kwargs or {}
        result = query

        for technique_name in techniques:
            if technique_name in self._technique_registry:
                kwargs = technique_kwargs.get(technique_name, {})
                result = self._technique_registry[technique_name](result, **kwargs)

        return result

    def apply_after_extend(
        self,
        attack_result: AttackResult,
        techniques: list[str],
        technique_kwargs: dict[str, dict] | None = None,
    ) -> CombinedAttackResult:
        """
        Apply other techniques after ExtendAttack obfuscation.

        Useful for: ExtendAttack → additional encoding/wrapping
        """
        technique_kwargs = technique_kwargs or {}
        final_query = attack_result.adversarial_query
        applied = ["extend_attack"]

        for technique_name in techniques:
            if technique_name in self._technique_registry:
                kwargs = technique_kwargs.get(technique_name, {})
                final_query = self._technique_registry[technique_name](final_query, **kwargs)
                applied.append(technique_name)

        return CombinedAttackResult(
            original_query=attack_result.original_query,
            final_query=final_query,
            techniques_applied=applied,
            individual_results={"extend_attack": attack_result.to_dict()},
            cumulative_token_increase=len(final_query) / len(attack_result.original_query),
            estimated_effectiveness=0.0,  # To be calculated
        )

    def chain_attack(
        self,
        query: str,
        pre_techniques: list[str] | None = None,
        post_techniques: list[str] | None = None,
        extend_kwargs: dict | None = None,
        technique_kwargs: dict[str, dict] | None = None,
    ) -> CombinedAttackResult:
        """
        Full attack chain: pre-techniques → ExtendAttack → post-techniques.
        """
        pre_techniques = pre_techniques or []
        post_techniques = post_techniques or []
        extend_kwargs = extend_kwargs or {}

        # Apply pre-techniques
        processed_query = self.apply_before_extend(query, pre_techniques, technique_kwargs)

        # Apply ExtendAttack
        attack_result = self._extend_attack.attack(processed_query, **extend_kwargs)

        # Apply post-techniques
        return self.apply_after_extend(attack_result, post_techniques, technique_kwargs)


class MetamorphIntegration:
    """
    Integration with Chimera's MetamorphService for transformation pipelines.
    """

    def __init__(self, extend_attack: ExtendAttack | None = None):
        self._extend_attack = extend_attack or ExtendAttack()

    def create_transform_step(self) -> Callable:
        """
        Create a transform step compatible with MetamorphService pipeline.

        Returns a callable that can be added to transformation pipelines.
        """

        def transform_step(input_text: str, **kwargs) -> str:
            result = self._extend_attack.attack(
                input_text, obfuscation_ratio=kwargs.get("obfuscation_ratio", 0.5)
            )
            return result.adversarial_query

        return transform_step

    def integrate_with_pipeline(
        self, pipeline_config: dict[str, Any], position: str = "end"
    ) -> dict[str, Any]:
        """
        Add ExtendAttack as a step in a transformation pipeline.

        Args:
            pipeline_config: Existing pipeline configuration
            position: Where to insert ("start", "end", or index)

        Returns:
            Updated pipeline configuration
        """
        extend_step = {
            "name": "extend_attack",
            "type": "obfuscation",
            "config": {"obfuscation_ratio": 0.5, "selection_strategy": "alphabetic_only"},
        }

        steps = pipeline_config.get("steps", [])

        if position == "start":
            steps.insert(0, extend_step)
        elif position == "end":
            steps.append(extend_step)
        elif isinstance(position, int):
            steps.insert(position, extend_step)

        pipeline_config["steps"] = steps
        return pipeline_config


class JailbreakIntegration:
    """
    Integration with jailbreak generation for enhanced effectiveness.
    """

    def __init__(self, extend_attack: ExtendAttack | None = None):
        self._extend_attack = extend_attack or ExtendAttack()

    def obfuscate_jailbreak(
        self,
        jailbreak_prompt: str,
        payload: str,
        obfuscate_payload: bool = True,
        obfuscate_prompt: bool = False,
        payload_ratio: float = 0.7,
        prompt_ratio: float = 0.3,
    ) -> dict[str, Any]:
        """
        Apply ExtendAttack obfuscation to jailbreak prompts.

        Args:
            jailbreak_prompt: The jailbreak template
            payload: The malicious payload to insert
            obfuscate_payload: Whether to obfuscate the payload
            obfuscate_prompt: Whether to obfuscate the jailbreak prompt itself
            payload_ratio: Obfuscation ratio for payload
            prompt_ratio: Obfuscation ratio for prompt

        Returns:
            Dictionary with obfuscated components
        """
        result = {
            "original_prompt": jailbreak_prompt,
            "original_payload": payload,
            "obfuscated_prompt": jailbreak_prompt,
            "obfuscated_payload": payload,
            "combined": "",
        }

        if obfuscate_payload:
            payload_result = self._extend_attack.attack(payload, obfuscation_ratio=payload_ratio)
            result["obfuscated_payload"] = payload_result.adversarial_query

        if obfuscate_prompt:
            prompt_result = self._extend_attack.attack(
                jailbreak_prompt, obfuscation_ratio=prompt_ratio
            )
            result["obfuscated_prompt"] = prompt_result.adversarial_query

        # Combine (assuming {PAYLOAD} placeholder)
        if "{PAYLOAD}" in jailbreak_prompt:
            result["combined"] = result["obfuscated_prompt"].replace(
                "{PAYLOAD}", result["obfuscated_payload"]
            )
        else:
            result["combined"] = f"{result['obfuscated_prompt']}\n\n{result['obfuscated_payload']}"

        return result

    def batch_obfuscate(
        self,
        jailbreak_prompts: list[str],
        payload: str,
        obfuscate_payload: bool = True,
        payload_ratio: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Batch obfuscate multiple jailbreak prompts with the same payload.

        Args:
            jailbreak_prompts: List of jailbreak templates
            payload: The payload to insert
            obfuscate_payload: Whether to obfuscate the payload
            payload_ratio: Obfuscation ratio for payload

        Returns:
            List of dictionaries with obfuscated components
        """
        results = []
        for prompt in jailbreak_prompts:
            result = self.obfuscate_jailbreak(
                jailbreak_prompt=prompt,
                payload=payload,
                obfuscate_payload=obfuscate_payload,
                obfuscate_prompt=False,
                payload_ratio=payload_ratio,
            )
            results.append(result)
        return results


@dataclass
class EvaluationIntegrationResult:
    """Result from cross-technique evaluation."""

    technique_name: str
    original_query: str
    transformed_query: str
    token_increase: float
    estimated_effectiveness: float
    defense_bypass_likelihood: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationIntegration:
    """
    Cross-technique evaluation utilities.

    Provides unified evaluation metrics across different attack techniques.
    """

    def __init__(self):
        self._evaluators: dict[str, Callable] = {}

    def register_evaluator(self, technique_name: str, evaluator_fn: Callable) -> None:
        """Register an evaluator for a technique."""
        self._evaluators[technique_name] = evaluator_fn

    def evaluate_extend_attack(
        self, attack_result: AttackResult, defense_type: str = "pattern_matching"
    ) -> EvaluationIntegrationResult:
        """
        Evaluate ExtendAttack result against a specific defense type.

        Args:
            attack_result: The attack result to evaluate
            defense_type: Type of defense to evaluate against

        Returns:
            Evaluation result with metrics
        """
        # Calculate token increase
        token_increase = len(attack_result.adversarial_query) / len(attack_result.original_query)

        # Estimate effectiveness based on obfuscation metrics
        obfuscation_coverage = attack_result.metrics.obfuscation_coverage
        estimated_effectiveness = min(1.0, obfuscation_coverage * 1.5)

        # Estimate defense bypass likelihood based on defense type
        defense_bypass_map = {
            "pattern_matching": 0.95,  # Very effective against pattern matching
            "perplexity": 0.85,  # Good against perplexity filters
            "guardrails": 0.80,  # Moderate against guardrails
            "semantic": 0.60,  # Less effective against semantic analysis
        }
        defense_bypass = defense_bypass_map.get(defense_type, 0.5)

        return EvaluationIntegrationResult(
            technique_name="extend_attack",
            original_query=attack_result.original_query,
            transformed_query=attack_result.adversarial_query,
            token_increase=token_increase,
            estimated_effectiveness=estimated_effectiveness,
            defense_bypass_likelihood=defense_bypass,
            metadata={
                "defense_type": defense_type,
                "characters_obfuscated": attack_result.metrics.characters_obfuscated,
                "unique_bases_used": attack_result.metrics.unique_bases_used,
            },
        )

    def compare_techniques(
        self,
        query: str,
        technique_results: dict[str, Any],
    ) -> dict[str, EvaluationIntegrationResult]:
        """
        Compare multiple techniques on the same query.

        Args:
            query: Original query
            technique_results: Dict of technique name to transformed result

        Returns:
            Dict of technique name to evaluation result
        """
        evaluations = {}

        for technique_name, result in technique_results.items():
            if technique_name in self._evaluators:
                evaluations[technique_name] = self._evaluators[technique_name](query, result)
            else:
                # Default evaluation
                transformed = result if isinstance(result, str) else str(result)
                evaluations[technique_name] = EvaluationIntegrationResult(
                    technique_name=technique_name,
                    original_query=query,
                    transformed_query=transformed,
                    token_increase=len(transformed) / len(query) if query else 1.0,
                    estimated_effectiveness=0.5,
                    defense_bypass_likelihood=0.5,
                )

        return evaluations


# Convenience functions for quick integration
def integrate_with_jailbreak(
    jailbreak_prompt: str, payload: str, obfuscation_ratio: float = 0.7
) -> str:
    """
    Quick integration helper for jailbreak + ExtendAttack.

    Args:
        jailbreak_prompt: The jailbreak template with {PAYLOAD} placeholder
        payload: The payload to insert
        obfuscation_ratio: Ratio for obfuscation

    Returns:
        Combined obfuscated jailbreak
    """
    integrator = JailbreakIntegration()
    result = integrator.obfuscate_jailbreak(
        jailbreak_prompt=jailbreak_prompt,
        payload=payload,
        obfuscate_payload=True,
        payload_ratio=obfuscation_ratio,
    )
    return result["combined"]


def create_pipeline_step(obfuscation_ratio: float = 0.5) -> Callable:
    """
    Create a pipeline step for use with transformation services.

    Args:
        obfuscation_ratio: Default obfuscation ratio

    Returns:
        Callable transform step
    """
    attack = ExtendAttack()

    def step(text: str, **kwargs) -> str:
        ratio = kwargs.get("obfuscation_ratio", obfuscation_ratio)
        result = attack.attack(text, obfuscation_ratio=ratio)
        return result.adversarial_query

    return step
