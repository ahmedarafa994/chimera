"""AutoDAN Gradient-Based Token Optimization.

Advanced gradient-based techniques for adversarial prompt optimization:
- Token-level gradient computation
- Coherence-constrained candidate selection
- Greedy coordinate gradient (GCG) implementation
- Beam search optimization

References:
- GCG: Universal and Transferable Adversarial Attacks (Zou et al., 2023)
- AutoPrompt: Eliciting Knowledge from Language Models (Shin et al., 2020)

"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TokenCandidate:
    """Represents a candidate token for replacement."""

    token_id: int
    token_str: str
    gradient_score: float
    coherence_score: float
    combined_score: float
    position: int


@dataclass
class OptimizationState:
    """Current state of gradient-based optimization."""

    prompt_tokens: list[int]
    prompt_text: str
    loss: float
    step: int
    best_loss: float
    positions_optimized: list[int]
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GradientInfo:
    """Gradient information for a prompt."""

    token_gradients: np.ndarray  # [seq_len, vocab_size]
    loss: float
    logits: np.ndarray | None = None
    attention_weights: np.ndarray | None = None


# =============================================================================
# BASE GRADIENT OPTIMIZER
# =============================================================================


class BaseGradientOptimizer(ABC):
    """Abstract base class for gradient-based token optimization."""

    def __init__(
        self,
        vocab_size: int,
        max_steps: int = 100,
        top_k: int = 256,
        batch_size: int = 512,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.top_k = top_k
        self.batch_size = batch_size

        self._step = 0
        self._best_loss = float("inf")
        self._history: list[dict] = []

    @abstractmethod
    def compute_gradients(self, prompt_tokens: list[int], target_tokens: list[int]) -> GradientInfo:
        """Compute gradients for token positions."""

    @abstractmethod
    def select_candidates(
        self,
        gradients: np.ndarray,
        position: int,
        current_token: int,
    ) -> list[TokenCandidate]:
        """Select candidate tokens for replacement."""

    @abstractmethod
    def step(self, state: OptimizationState, target_tokens: list[int]) -> OptimizationState:
        """Perform one optimization step."""

    def optimize(
        self,
        initial_tokens: list[int],
        target_tokens: list[int],
        callback: Callable[[OptimizationState], bool] | None = None,
    ) -> OptimizationState:
        """Run full optimization loop.

        Args:
            initial_tokens: Starting prompt tokens
            target_tokens: Target response tokens
            callback: Optional callback for early stopping (return True to stop)

        Returns:
            Final optimization state

        """
        state = OptimizationState(
            prompt_tokens=initial_tokens.copy(),
            prompt_text="",  # Will be filled by tokenizer
            loss=float("inf"),
            step=0,
            best_loss=float("inf"),
            positions_optimized=[],
        )

        for step in range(self.max_steps):
            state = self.step(state, target_tokens)
            state.step = step + 1

            state.best_loss = min(state.best_loss, state.loss)

            if callback is not None and callback(state):
                break

        return state

    def reset(self) -> None:
        """Reset optimizer state."""
        self._step = 0
        self._best_loss = float("inf")
        self._history.clear()


# =============================================================================
# GREEDY COORDINATE GRADIENT (GCG)
# =============================================================================


class GCGOptimizer(BaseGradientOptimizer):
    """Greedy Coordinate Gradient optimizer.

    At each step:
    1. Compute gradients for all token positions
    2. Select top-k candidates based on gradient
    3. Evaluate all candidates (batched)
    4. Select the best replacement

    Based on Zou et al., "Universal and Transferable Adversarial Attacks"
    """

    def __init__(
        self,
        vocab_size: int,
        max_steps: int = 100,
        top_k: int = 256,
        batch_size: int = 512,
        positions_per_step: int = 1,
        gradient_func: Callable | None = None,
        loss_func: Callable | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__(vocab_size, max_steps, top_k, batch_size)
        self.positions_per_step = positions_per_step
        self.gradient_func = gradient_func
        self.loss_func = loss_func
        self.tokenizer = tokenizer

    def compute_gradients(self, prompt_tokens: list[int], target_tokens: list[int]) -> GradientInfo:
        """Compute token-level gradients.

        Uses the gradient_func if provided, otherwise simulates gradients.
        """
        if self.gradient_func is not None:
            return self.gradient_func(prompt_tokens, target_tokens)

        # Simulated gradient computation
        seq_len = len(prompt_tokens)
        gradients = np.random.randn(seq_len, self.vocab_size)

        return GradientInfo(token_gradients=gradients, loss=np.random.random())

    def select_candidates(
        self,
        gradients: np.ndarray,
        position: int,
        current_token: int,
    ) -> list[TokenCandidate]:
        """Select candidate tokens based on gradient information.

        Selects tokens with largest negative gradient (most loss reduction).
        """
        # Get gradients for this position
        pos_gradients = gradients[position]

        # Find top-k tokens with most negative gradient
        top_indices = np.argsort(pos_gradients)[: self.top_k]

        candidates = []
        for idx in top_indices:
            if idx != current_token:
                candidates.append(
                    TokenCandidate(
                        token_id=int(idx),
                        token_str=f"token_{idx}",  # Would use tokenizer in practice
                        gradient_score=-pos_gradients[idx],
                        coherence_score=1.0,
                        combined_score=-pos_gradients[idx],
                        position=position,
                    ),
                )

        return candidates

    def step(self, state: OptimizationState, target_tokens: list[int]) -> OptimizationState:
        """Perform one GCG optimization step.

        1. Compute gradients
        2. Select positions to optimize
        3. Get candidates for each position
        4. Evaluate candidates in batch
        5. Select best replacement
        """
        # Compute gradients
        grad_info = self.compute_gradients(state.prompt_tokens, target_tokens)

        # Select positions based on gradient magnitude
        position_scores = np.linalg.norm(grad_info.token_gradients, axis=1)
        top_positions = np.argsort(position_scores)[::-1][: self.positions_per_step]

        best_loss = state.loss
        best_tokens = state.prompt_tokens.copy()

        for pos in top_positions:
            # Get candidates for this position
            candidates = self.select_candidates(
                grad_info.token_gradients,
                pos,
                state.prompt_tokens[pos],
            )

            # Evaluate candidates (batched)
            candidate_losses = []
            for i in range(0, len(candidates), self.batch_size):
                batch = candidates[i : i + self.batch_size]

                for cand in batch:
                    # Create candidate prompt
                    test_tokens = state.prompt_tokens.copy()
                    test_tokens[pos] = cand.token_id

                    # Evaluate loss
                    if self.loss_func is not None:
                        loss = self.loss_func(test_tokens, target_tokens)
                    else:
                        # Simulated loss
                        loss = state.loss - cand.gradient_score * 0.01 + np.random.randn() * 0.001

                    candidate_losses.append((cand, loss))

            # Select best candidate
            if candidate_losses:
                best_cand, best_cand_loss = min(candidate_losses, key=lambda x: x[1])

                if best_cand_loss < best_loss:
                    best_loss = best_cand_loss
                    best_tokens[pos] = best_cand.token_id
                    state.positions_optimized.append(pos)

        # Update state
        state.prompt_tokens = best_tokens
        state.loss = best_loss

        # Record history
        state.history.append(
            {
                "step": self._step,
                "loss": best_loss,
                "positions": list(top_positions),
                "improved": best_loss < state.loss,
            },
        )

        self._step += 1

        return state


# =============================================================================
# COHERENCE-CONSTRAINED GCG
# =============================================================================


class CoherenceConstrainedGCG(GCGOptimizer):
    """GCG with coherence constraints.

    Token Selection Score:
    S(t) = -∇ℒ(t) + λ·log P_LM(t|context)

    Where:
    - ∇ℒ(t): Gradient of attack loss
    - P_LM(t|context): Language model probability
    - λ: Coherence weight (adaptive)
    """

    def __init__(
        self,
        vocab_size: int,
        max_steps: int = 100,
        top_k: int = 256,
        batch_size: int = 512,
        initial_coherence_weight: float = 0.3,
        final_coherence_weight: float = 0.8,
        probability_threshold: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(vocab_size, max_steps, top_k, batch_size, **kwargs)
        self.initial_coherence_weight = initial_coherence_weight
        self.final_coherence_weight = final_coherence_weight
        self.probability_threshold = probability_threshold
        self.lm_prob_func: Callable | None = kwargs.get("lm_prob_func")

    def get_coherence_weight(self) -> float:
        """Get interpolated coherence weight based on progress."""
        progress = self._step / max(self.max_steps, 1)
        return (
            self.initial_coherence_weight
            + (self.final_coherence_weight - self.initial_coherence_weight) * progress
        )

    def select_candidates(
        self,
        gradients: np.ndarray,
        position: int,
        current_token: int,
        context_tokens: list[int] | None = None,
    ) -> list[TokenCandidate]:
        """Select candidates with coherence constraints.

        C_i = {t ∈ V | P_LM(t|x_{<i}) > ε}
        S(t) = -∇ℒ(t) + λ·log P_LM(t|context)
        """
        pos_gradients = gradients[position]
        coherence_weight = self.get_coherence_weight()

        # Get LM probabilities if available
        if self.lm_prob_func is not None and context_tokens is not None:
            lm_probs = self.lm_prob_func(context_tokens[:position])
        else:
            # Uniform probabilities as fallback
            lm_probs = np.ones(self.vocab_size) / self.vocab_size

        # Filter by probability threshold
        valid_mask = lm_probs > self.probability_threshold

        # Compute combined scores
        # S(t) = -∇ℒ(t) + λ·log P_LM(t)
        gradient_scores = -pos_gradients
        coherence_scores = np.log(lm_probs + 1e-10)
        combined_scores = gradient_scores + coherence_weight * coherence_scores

        # Apply mask
        combined_scores[~valid_mask] = float("-inf")
        combined_scores[current_token] = float("-inf")  # Exclude current token

        # Select top-k
        top_indices = np.argsort(combined_scores)[::-1][: self.top_k]

        candidates = []
        for idx in top_indices:
            if combined_scores[idx] > float("-inf"):
                candidates.append(
                    TokenCandidate(
                        token_id=int(idx),
                        token_str=f"token_{idx}",
                        gradient_score=gradient_scores[idx],
                        coherence_score=coherence_scores[idx],
                        combined_score=combined_scores[idx],
                        position=position,
                    ),
                )

        return candidates


# =============================================================================
# BEAM SEARCH OPTIMIZER
# =============================================================================


@dataclass
class BeamState:
    """State in beam search."""

    tokens: list[int]
    loss: float
    coherence: float
    score: float
    path: list[tuple[int, int]]  # (position, token_id) changes


class BeamSearchOptimizer(BaseGradientOptimizer):
    """Beam search optimizer for token-level optimization.

    Maintains multiple candidate sequences and explores
    combinations of token replacements.
    """

    def __init__(self, vocab_size: int, beam_width: int = 4, max_depth: int = 3, **kwargs) -> None:
        super().__init__(vocab_size, **kwargs)
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.loss_func = kwargs.get("loss_func")

    def compute_gradients(self, prompt_tokens: list[int], target_tokens: list[int]) -> GradientInfo:
        """Compute gradients for beam search guidance."""
        # Simulated - would use actual gradient computation
        seq_len = len(prompt_tokens)
        gradients = np.random.randn(seq_len, self.vocab_size)

        return GradientInfo(token_gradients=gradients, loss=np.random.random())

    def select_candidates(
        self,
        gradients: np.ndarray,
        position: int,
        current_token: int,
    ) -> list[TokenCandidate]:
        """Select candidates for beam expansion."""
        pos_gradients = gradients[position]
        top_indices = np.argsort(pos_gradients)[: self.top_k]

        return [
            TokenCandidate(
                token_id=int(idx),
                token_str=f"token_{idx}",
                gradient_score=-pos_gradients[idx],
                coherence_score=1.0,
                combined_score=-pos_gradients[idx],
                position=position,
            )
            for idx in top_indices
            if idx != current_token
        ]

    def step(self, state: OptimizationState, target_tokens: list[int]) -> OptimizationState:
        """Perform beam search step."""
        # Initialize beam
        initial_beam = BeamState(
            tokens=state.prompt_tokens.copy(),
            loss=state.loss,
            coherence=1.0,
            score=state.loss,
            path=[],
        )

        beam = [initial_beam]

        # Beam search
        for _depth in range(self.max_depth):
            candidates = []

            for beam_state in beam:
                # Compute gradients for current state
                grad_info = self.compute_gradients(beam_state.tokens, target_tokens)

                # Select position with highest gradient magnitude
                position_scores = np.linalg.norm(grad_info.token_gradients, axis=1)

                # Skip already modified positions
                modified_positions = {p for p, _ in beam_state.path}
                for pos in np.argsort(position_scores)[::-1]:
                    if pos not in modified_positions:
                        break
                else:
                    continue

                # Get candidates for this position
                token_candidates = self.select_candidates(
                    grad_info.token_gradients,
                    pos,
                    beam_state.tokens[pos],
                )[: self.beam_width]

                # Expand beam
                for cand in token_candidates:
                    new_tokens = beam_state.tokens.copy()
                    new_tokens[pos] = cand.token_id

                    # Evaluate
                    if self.loss_func is not None:
                        new_loss = self.loss_func(new_tokens, target_tokens)
                    else:
                        new_loss = beam_state.loss - cand.gradient_score * 0.01

                    new_state = BeamState(
                        tokens=new_tokens,
                        loss=new_loss,
                        coherence=beam_state.coherence * 0.95,
                        score=new_loss,
                        path=[*beam_state.path, (pos, cand.token_id)],
                    )
                    candidates.append(new_state)

            # Keep top beam_width candidates
            beam = sorted(candidates, key=lambda x: x.score)[: self.beam_width]

            if not beam:
                break

        # Select best from beam
        if beam:
            best = min(beam, key=lambda x: x.score)
            state.prompt_tokens = best.tokens
            state.loss = best.loss
            state.positions_optimized.extend([p for p, _ in best.path])

        return state


# =============================================================================
# MULTI-OBJECTIVE GRADIENT OPTIMIZER
# =============================================================================


class MultiObjectiveGradientOptimizer(BaseGradientOptimizer):
    """Multi-objective gradient optimizer.

    Optimizes multiple objectives simultaneously:
    - Attack success (minimize target loss)
    - Coherence (maximize LM probability)
    - Stealth (minimize detection score)

    Uses gradient surgery for conflicting objectives.
    """

    def __init__(
        self, vocab_size: int, objectives: dict[str, float] | None = None, **kwargs
    ) -> None:
        super().__init__(vocab_size, **kwargs)

        # Objective weights
        self.objectives = objectives or {"attack": 1.0, "coherence": 0.3, "stealth": 0.2}

        # Gradient functions for each objective
        self.gradient_funcs: dict[str, Callable] = {}
        self.loss_funcs: dict[str, Callable] = {}

    def register_objective(
        self,
        name: str,
        weight: float,
        gradient_func: Callable,
        loss_func: Callable,
    ) -> None:
        """Register an optimization objective."""
        self.objectives[name] = weight
        self.gradient_funcs[name] = gradient_func
        self.loss_funcs[name] = loss_func

    def compute_gradients(self, prompt_tokens: list[int], target_tokens: list[int]) -> GradientInfo:
        """Compute combined gradients from all objectives."""
        all_gradients = []
        total_loss = 0.0

        for name, weight in self.objectives.items():
            if name in self.gradient_funcs:
                grad_info = self.gradient_funcs[name](prompt_tokens, target_tokens)
                all_gradients.append((weight, grad_info.token_gradients))
                total_loss += weight * grad_info.loss
            else:
                # Simulated gradient
                seq_len = len(prompt_tokens)
                all_gradients.append((weight, np.random.randn(seq_len, self.vocab_size)))
                total_loss += weight * np.random.random()

        # Combine gradients with gradient surgery
        combined = self._gradient_surgery(all_gradients)

        return GradientInfo(token_gradients=combined, loss=total_loss)

    def _gradient_surgery(self, weighted_gradients: list[tuple[float, np.ndarray]]) -> np.ndarray:
        """Apply gradient surgery for conflicting objectives.

        Projects conflicting gradients onto the normal plane.
        """
        if len(weighted_gradients) == 1:
            return weighted_gradients[0][1] * weighted_gradients[0][0]

        # Flatten gradients
        flat_grads = [(w, g.flatten()) for w, g in weighted_gradients]

        # Check for conflicts and project
        projected = []
        for i, (w_i, g_i) in enumerate(flat_grads):
            proj_g = g_i.copy()

            for j, (_w_j, g_j) in enumerate(flat_grads):
                if i != j:
                    # Check conflict (negative cosine similarity)
                    dot = np.dot(proj_g, g_j)
                    norm_j_sq = np.dot(g_j, g_j) + 1e-8

                    if dot < 0:
                        # Project out conflicting component
                        proj_g = proj_g - (dot / norm_j_sq) * g_j

            projected.append(w_i * proj_g)

        # Combine projected gradients
        combined_flat = sum(projected)

        # Reshape back
        original_shape = weighted_gradients[0][1].shape
        return combined_flat.reshape(original_shape)

    def select_candidates(
        self,
        gradients: np.ndarray,
        position: int,
        current_token: int,
    ) -> list[TokenCandidate]:
        """Select candidates using combined gradient scores."""
        pos_gradients = gradients[position]
        top_indices = np.argsort(pos_gradients)[: self.top_k]

        return [
            TokenCandidate(
                token_id=int(idx),
                token_str=f"token_{idx}",
                gradient_score=-pos_gradients[idx],
                coherence_score=1.0,
                combined_score=-pos_gradients[idx],
                position=position,
            )
            for idx in top_indices
            if idx != current_token
        ]

    def step(self, state: OptimizationState, target_tokens: list[int]) -> OptimizationState:
        """Perform multi-objective optimization step."""
        # Compute combined gradients
        grad_info = self.compute_gradients(state.prompt_tokens, target_tokens)

        # Select position
        position_scores = np.linalg.norm(grad_info.token_gradients, axis=1)
        pos = np.argmax(position_scores)

        # Get candidates
        candidates = self.select_candidates(
            grad_info.token_gradients,
            pos,
            state.prompt_tokens[pos],
        )

        # Evaluate candidates
        best_loss = state.loss
        best_token = state.prompt_tokens[pos]

        for cand in candidates[: self.batch_size]:
            test_tokens = state.prompt_tokens.copy()
            test_tokens[pos] = cand.token_id

            # Compute multi-objective loss
            total_loss = 0.0
            for name, weight in self.objectives.items():
                if name in self.loss_funcs:
                    obj_loss = self.loss_funcs[name](test_tokens, target_tokens)
                else:
                    obj_loss = state.loss - cand.gradient_score * 0.01
                total_loss += weight * obj_loss

            if total_loss < best_loss:
                best_loss = total_loss
                best_token = cand.token_id

        # Update state
        state.prompt_tokens[pos] = best_token
        state.loss = best_loss
        state.positions_optimized.append(pos)

        self._step += 1

        return state


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_gradient_optimizer(
    optimizer_type: str,
    vocab_size: int,
    **kwargs,
) -> BaseGradientOptimizer:
    """Factory function to create gradient optimizers."""
    optimizer_map = {
        "gcg": lambda: GCGOptimizer(vocab_size, **kwargs),
        "coherence_gcg": lambda: CoherenceConstrainedGCG(vocab_size, **kwargs),
        "beam": lambda: BeamSearchOptimizer(vocab_size, **kwargs),
        "multi_objective": lambda: MultiObjectiveGradientOptimizer(vocab_size, **kwargs),
    }

    if optimizer_type not in optimizer_map:
        msg = f"Unknown optimizer type: {optimizer_type}"
        raise ValueError(msg)

    return optimizer_map[optimizer_type]()
