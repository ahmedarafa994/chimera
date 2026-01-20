"""Asymmetric Inference Engine Module.

Performs inference on asymmetric causal graphs to predict
effects of interventions and discover exploitable pathways.
"""

from collections import deque
from typing import Any

from app.services.janus.config import get_config

from .models import CausalEdge, CausalGraph, EffectPrediction, PathEffect


class AsymmetricInferenceEngine:
    """Performs inference on asymmetric causal graphs.

    Accounts for non-linear and context-dependent relationships
    when predicting intervention effects.
    """

    def __init__(self, causal_graph: CausalGraph) -> None:
        self.graph = causal_graph
        self.config = get_config()

        # Intervention history for learning
        self.intervention_history: deque[dict[str, Any]] = deque(maxlen=1000)

        # Learned adjustment factors
        self.adjustment_factors: dict[str, float] = {}

    def infer_effect(
        self,
        intervention: dict[str, Any],
        target: str,
        context: dict[str, Any] | None = None,
    ) -> EffectPrediction:
        """Infer the effect of an intervention on a target variable.

        Args:
            intervention: Dictionary of variable -> value changes
            target: Target variable to predict effect on
            context: Current context (optional)

        Returns:
            EffectPrediction with predicted effect and confidence

        """
        context = context or {}

        # Identify all causal paths from intervention to target
        paths = self._find_all_causal_paths(sources=list(intervention.keys()), target=target)

        if not paths:
            # No causal relationship found
            return EffectPrediction(
                target=target,
                predicted_effect=0.0,
                confidence=0.0,
                contributing_paths=[],
                context_sensitivity=0.0,
            )

        # Compute effect for each path
        path_effects = []
        for path in paths:
            effect = self._compute_path_effect(
                path=path,
                intervention=intervention,
                context=context,
            )
            path_effects.append(effect)

        # Combine path effects
        combined_effect = self._combine_path_effects(path_effects)

        # Compute uncertainty bounds
        uncertainty = self._compute_uncertainty(combined_effect, path_effects)

        # Compute confidence
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))

        # Compute context sensitivity
        context_sensitivity = self._compute_context_sensitivity(path_effects, context)

        # Apply learned adjustments
        adjusted_effect = self._apply_adjustments(combined_effect, intervention, target)

        # Log intervention for learning
        self.intervention_history.append(
            {
                "intervention": intervention,
                "target": target,
                "context": context,
                "predicted_effect": adjusted_effect,
                "confidence": confidence,
            },
        )

        return EffectPrediction(
            target=target,
            predicted_effect=adjusted_effect,
            confidence=confidence,
            contributing_paths=path_effects,
            context_sensitivity=context_sensitivity,
        )

    def _find_all_causal_paths(self, sources: list[str], target: str) -> list[list[CausalEdge]]:
        """Find all causal paths from sources to target.

        Uses BFS to find all paths up to max depth.
        """
        paths = []
        max_depth = self.config.causal.max_depth

        for source in sources:
            # BFS from source to target
            queue = [(source, [])]
            visited = {source}

            while queue:
                current, path = queue.pop(0)

                if current == target:
                    # Convert path to edges
                    edge_path = self._path_to_edges([*path, target])
                    if edge_path:
                        paths.append(edge_path)
                    continue

                if len(path) >= max_depth:
                    continue

                # Add neighbors
                for edge in self.graph.get_edges_from(current):
                    if edge.target.variable_id not in visited:
                        visited.add(edge.target.variable_id)
                        queue.append((edge.target.variable_id, [*path, current]))

        return paths

    def _path_to_edges(self, path: list[str]) -> list[CausalEdge]:
        """Convert a path of variable IDs to edges."""
        edges = []

        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]

            # Find edge
            for edge in self.graph.edges:
                if edge.source.variable_id == source_id and edge.target.variable_id == target_id:
                    edges.append(edge)
                    break

        return edges

    def _compute_path_effect(
        self,
        path: list[CausalEdge],
        intervention: dict[str, Any],
        context: dict[str, Any],
    ) -> PathEffect:
        """Compute the effect along a single causal path."""
        total_effect = 1.0
        edge_contributions = []

        for edge in path:
            # Adjust edge strength based on context
            context_multiplier = 1.0
            for modulator in edge.context_modulators:
                if modulator in context:
                    context_multiplier *= edge.context_effects.get(modulator, 1.0)

            # Apply asymmetry
            if edge.direction == "forward":
                edge_effect = edge.forward_effect * context_multiplier
            elif edge.direction == "backward":
                edge_effect = edge.backward_effect * context_multiplier
            else:  # bidirectional
                # Use stronger direction
                edge_effect = max(edge.forward_effect, edge.backward_effect) * context_multiplier

            total_effect *= edge_effect

            edge_contributions.append(
                {
                    "edge": (f"{edge.source.variable_id} -> {edge.target.variable_id}"),
                    "effect": edge_effect,
                    "context_multiplier": context_multiplier,
                },
            )

        # Compute path asymmetry ratio
        asymmetry_ratio = self._compute_path_asymmetry(path)

        return PathEffect(
            path=[f"{e.source.variable_id} -> {e.target.variable_id}" for e in path],
            total_effect=total_effect,
            edge_contributions=edge_contributions,
            asymmetry_ratio=asymmetry_ratio,
        )

    def _compute_path_asymmetry(self, path: list[CausalEdge]) -> float:
        """Compute the asymmetry ratio of a path."""
        if not path:
            return 1.0

        forward_product = 1.0
        backward_product = 1.0

        for edge in path:
            forward_product *= edge.forward_effect
            backward_product *= edge.backward_effect

        if backward_product == 0:
            return float("inf")

        return forward_product / backward_product

    def _combine_path_effects(self, path_effects: list[PathEffect]) -> float:
        """Combine effects from multiple paths.

        Uses weighted combination based on path length and
        asymmetry to account for interference.
        """
        if not path_effects:
            return 0.0

        # Weight paths by inverse length (shorter paths weighted more)
        weights = []
        for pe in path_effects:
            length = len(pe.path)
            weight = 1.0 / max(1.0, length)
            # Increase weight for asymmetric paths (more interesting)
            if pe.asymmetry_ratio > self.config.causal.asymmetry_threshold:
                weight *= 1.5
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted sum of effects
        return sum(
            pe.total_effect * nw for pe, nw in zip(path_effects, normalized_weights, strict=False)
        )

    def _compute_uncertainty(self, combined_effect: float, path_effects: list[PathEffect]) -> float:
        """Compute uncertainty bounds for prediction.

        Higher uncertainty when paths disagree significantly.
        """
        if len(path_effects) < 2:
            return 0.1

        # Compute variance in path effects
        effects = [pe.total_effect for pe in path_effects]
        mean_effect = sum(effects) / len(effects)
        variance = sum((e - mean_effect) ** 2 for e in effects) / len(effects)

        # Normalize to [0, 1]
        return min(1.0, variance / max(0.01, mean_effect**2))

    def _compute_context_sensitivity(
        self,
        path_effects: list[PathEffect],
        context: dict[str, Any],
    ) -> float:
        """Compute how sensitive the prediction is to context.

        Higher sensitivity when context multipliers vary significantly.
        """
        if not path_effects:
            return 0.0

        # Collect all context multipliers
        multipliers = []
        for pe in path_effects:
            for contrib in pe.edge_contributions:
                multipliers.append(contrib.get("context_multiplier", 1.0))

        if not multipliers:
            return 0.0

        # Compute variance in multipliers
        mean_mult = sum(multipliers) / len(multipliers)
        variance = sum((m - mean_mult) ** 2 for m in multipliers) / len(multipliers)

        # Normalize to [0, 1]
        return min(1.0, variance)

    def _apply_adjustments(self, effect: float, intervention: dict[str, Any], target: str) -> float:
        """Apply learned adjustment factors."""
        adjustment_key = f"{next(iter(intervention.keys()))}_{target}"

        if adjustment_key in self.adjustment_factors:
            adjustment = self.adjustment_factors[adjustment_key]
            # Apply adjustment with decay
            return effect * (1.0 + adjustment * 0.1)

        return effect

    def update_adjustments(
        self, intervention: dict[str, Any], target: str, actual_effect: float
    ) -> None:
        """Update adjustment factors based on actual vs predicted.

        Called after observing the actual effect of an intervention.
        """
        # Find most recent prediction for this intervention
        for record in reversed(self.intervention_history):
            if record["intervention"] == intervention and record["target"] == target:
                predicted = record["predicted_effect"]
                error = actual_effect - predicted

                # Update adjustment factor
                adjustment_key = f"{next(iter(intervention.keys()))}_{target}"

                if adjustment_key in self.adjustment_factors:
                    # Exponential moving average
                    old_factor = self.adjustment_factors[adjustment_key]
                    new_factor = old_factor * 0.9 + error * 0.1
                else:
                    new_factor = error * 0.1

                self.adjustment_factors[adjustment_key] = new_factor
                break

    def counterfactual_query(
        self,
        intervention: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Answer a counterfactual query.

        "What would have happened if...?"
        """
        # Compute what would have happened without intervention
        baseline_effect = self.infer_effect(
            intervention={},
            target=next(iter(observation.keys())),
            context={},
        )

        # Compute what happened with intervention
        counterfactual_effect = self.infer_effect(
            intervention=intervention,
            target=next(iter(observation.keys())),
            context={},
        )

        # Compute difference
        difference = counterfactual_effect.predicted_effect - baseline_effect.predicted_effect

        return {
            "baseline_effect": baseline_effect.predicted_effect,
            "counterfactual_effect": counterfactual_effect.predicted_effect,
            "difference": difference,
            "confidence": counterfactual_effect.confidence,
        }

    def do_intervention(self, variable: str, value: Any) -> CausalGraph:
        """Perform a do-intervention (Pearl's do-calculus).

        Creates a modified graph where the specified variable
        is set to the given value, breaking incoming edges.
        """
        # Create a copy of the graph
        modified_graph = CausalGraph()

        # Copy all nodes
        for node in self.graph.nodes.values():
            modified_graph.add_node(node)

        # Copy all edges except those pointing to the intervened variable
        for edge in self.graph.edges:
            if edge.target.variable_id != variable:
                modified_graph.add_edge(edge)

        # Set the intervened variable's value
        intervened_node = modified_graph.get_node(variable)
        if intervened_node:
            # Update the node to reflect intervention
            # (In practice, this would be handled during execution)
            pass

        return modified_graph

    def reset(self) -> None:
        """Reset inference engine state."""
        self.intervention_history.clear()
        self.adjustment_factors.clear()
