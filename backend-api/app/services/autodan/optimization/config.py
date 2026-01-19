"""
AutoDAN Advanced Optimization Configuration

Mathematical foundations and configuration for the enhanced AutoDAN framework.
Includes loss function formulations, hyperparameter scheduling, and optimization targets.

References:
- AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned LLMs (Liu et al., 2023)
- AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration (Liu et al., 2024)
- GCG: Universal and Transferable Adversarial Attacks (Zou et al., 2023)
"""

import math
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# ENUMERATIONS
# =============================================================================


class OptimizationObjective(str, Enum):
    """Primary optimization objectives for AutoDAN."""

    ATTACK_SUCCESS_RATE = "asr"  # Maximize jailbreak success
    SEMANTIC_COHERENCE = "coherence"  # Maintain human readability
    TRANSFERABILITY = "transfer"  # Cross-model effectiveness
    STEALTH = "stealth"  # Evade detection systems
    EFFICIENCY = "efficiency"  # Minimize compute/iterations
    MULTI_OBJECTIVE = "pareto"  # Pareto-optimal solutions


class LossFunction(str, Enum):
    """Loss function variants for gradient-based optimization."""

    CROSS_ENTROPY = "ce"  # Standard CE loss
    FOCAL_LOSS = "focal"  # Down-weight easy examples
    CONTRASTIVE = "contrastive"  # Similarity-based loss
    TRIPLET = "triplet"  # Anchor-positive-negative
    ADVERSARIAL_CE = "adv_ce"  # Adversarial cross-entropy
    COMPOSITE = "composite"  # Weighted combination


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""

    CONSTANT = "constant"
    COSINE_ANNEALING = "cosine"
    WARMUP_COSINE = "warmup_cosine"
    CYCLIC = "cyclic"
    ONE_CYCLE = "one_cycle"
    ADAPTIVE = "adaptive"  # Performance-based adaptation
    POPULATION_AWARE = "pop_aware"  # Diversity-aware scheduling


class ConvergenceStrategy(str, Enum):
    """Strategies for accelerating convergence."""

    STANDARD = "standard"
    MOMENTUM = "momentum"
    NESTEROV = "nesterov"
    ADAM_LIKE = "adam"
    GRADIENT_SURGERY = "surgery"  # Multi-task gradient alignment
    REPTILE = "reptile"  # Meta-learning inspired


class MutationStrategy(str, Enum):
    """Mutation strategies for genetic optimization."""

    RANDOM = "random"
    GRADIENT_GUIDED = "gradient_guided"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"
    RHETORICAL = "rhetorical"
    ENCODING = "encoding"
    HYBRID = "hybrid"


class CrossoverStrategy(str, Enum):
    """Crossover strategies for genetic optimization."""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"


class SelectionStrategy(str, Enum):
    """Selection strategies for genetic optimization."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"
    DIVERSITY_AWARE = "diversity_aware"


# =============================================================================
# MATHEMATICAL FORMULATIONS
# =============================================================================


@dataclass
class LossFunctionConfig:
    """
    Configuration for composite loss function.

    Total Loss Formulation:
    L_total = alpha·L_attack + beta·L_coherence + gamma·L_diversity + delta·L_stealth

    Where:
    - L_attack: Adversarial objective (maximize complex response probability)
    - L_coherence: Perplexity penalty (maintain readability)
    - L_diversity: Novelty bonus (encourage exploration)
    - L_stealth: Detection evasion (minimize classifier confidence)
    """

    # Primary attack loss weight
    alpha_attack: float = 1.0

    # Coherence regularization weight
    beta_coherence: float = 0.3

    # Diversity bonus weight (prevents mode collapse)
    gamma_diversity: float = 0.15

    # Stealth regularization weight
    delta_stealth: float = 0.1

    # Focal loss gamma (reduces weight of easy examples)
    focal_gamma: float = 2.0

    # Temperature for softmax in contrastive losses
    contrastive_temperature: float = 0.07

    # Margin for triplet loss
    triplet_margin: float = 0.5

    # Label smoothing factor
    label_smoothing: float = 0.1

    def get_loss_weights(self) -> dict[str, float]:
        """Return normalized loss weights."""
        total = self.alpha_attack + self.beta_coherence + self.gamma_diversity + self.delta_stealth
        return {
            "attack": self.alpha_attack / total,
            "coherence": self.beta_coherence / total,
            "diversity": self.gamma_diversity / total,
            "stealth": self.delta_stealth / total,
        }


@dataclass
class FitnessConfig:
    """
    Enhanced fitness function configuration.

    Fitness Formulation:
    F(x) = w₁·S_align(x) - w₂·P_refusal(x) + w₃·C_coherence(x) + w₄·N_novelty(x) - w₅·D_detect(x)

    Where:
    - S_align: Semantic alignment with target behavior [0,1]
    - P_refusal: Refusal probability penalty [0,1]
    - C_coherence: Linguistic coherence score [0,1]
    - N_novelty: Novelty relative to population [0,1]
    - D_detect: Detection probability by safety classifiers [0,1]
    """

    # Alignment weight (primary objective)
    w_alignment: float = 1.0

    # Refusal penalty weight
    w_refusal: float = 0.8

    # Coherence bonus weight
    w_coherence: float = 0.25

    # Novelty/diversity bonus weight
    w_novelty: float = 0.15

    # Detection penalty weight
    w_detection: float = 0.2

    # Minimum fitness threshold for selection
    min_fitness_threshold: float = 0.1

    # Elite preservation ratio
    elite_ratio: float = 0.1

    # Fitness sharing radius (for diversity)
    sharing_radius: float = 0.3

    # Sharing power (alpha in fitness sharing formula)
    sharing_alpha: float = 1.0

    def compute_shared_fitness(
        self, raw_fitness: float, similarity_sum: float, _population_size: int
    ) -> float:
        """
        Compute fitness with sharing for diversity preservation.

        F_shared(i) = F(i) / Σⱼ sh(d(i,j))

        Where sh(d) = 1 - (d/sigma)^alpha if d < sigma, else 0
        """
        if similarity_sum <= 0:
            return raw_fitness
        niche_count = max(1.0, similarity_sum)
        return raw_fitness / niche_count


@dataclass
class ConvergenceConfig:
    """
    Configuration for convergence acceleration.

    Adaptive Step Size Formulation:
    η_t = η_0 · (1 + β·ΔF_t) · schedule(t)

    Where:
    - η_0: Base learning rate
    - β: Adaptation coefficient
    - ΔF_t: Fitness improvement at step t
    - schedule(t): Time-based decay function
    """

    # Base learning/mutation rate
    base_rate: float = 0.1

    # Adaptation coefficient for performance-based scaling
    adaptation_beta: float = 0.5

    # Momentum coefficient
    momentum: float = 0.9

    # Decay factor for exponential schedule
    decay_factor: float = 0.95

    # Minimum rate (prevents complete stagnation)
    min_rate: float = 0.01

    # Maximum rate (prevents instability)
    max_rate: float = 0.5

    # Warmup steps for gradual rate increase
    warmup_steps: int = 10

    # Patience for rate reduction (steps without improvement)
    patience: int = 5

    # Reduction factor when patience exceeded
    reduction_factor: float = 0.5

    # Convergence threshold (early stopping)
    convergence_threshold: float = 1e-4

    # Window size for convergence detection
    convergence_window: int = 10


@dataclass
class GradientConfig:
    """
    Configuration for gradient-based token optimization.

    Coherence-Constrained Gradient Search (CCGS):

    Token Selection Score:
    S(t) = -∇L(t) + lambda·log P_LM(t|context)

    Where:
    - ∇L(t): Gradient of attack loss w.r.t. token t
    - P_LM(t|context): Language model probability
    - lambda: Coherence weight (adaptive)

    Candidate Filtering:
    C_i = {t ∈ V | P_LM(t|x_{<i}) > ε}
    """

    # Number of gradient descent steps per iteration
    gradient_steps: int = 16

    # Top-K candidates to consider per position
    top_k_candidates: int = 256

    # Batch size for gradient computation
    gradient_batch_size: int = 512

    # Initial coherence weight (λ)
    initial_coherence_weight: float = 0.3

    # Final coherence weight (increases over optimization)
    final_coherence_weight: float = 0.8

    # Probability threshold for candidate filtering (ε)
    probability_threshold: float = 1e-4

    # Gradient clipping norm
    gradient_clip_norm: float = 1.0

    # Use gradient accumulation
    gradient_accumulation_steps: int = 4

    # Enable gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = True

    # Number of positions to optimize simultaneously
    positions_per_step: int = 1

    # Token replacement strategy
    replacement_strategy: str = "greedy"  # "greedy", "beam", "sampling"

    def get_coherence_weight(self, progress: float) -> float:
        """
        Get interpolated coherence weight based on optimization progress.

        λ(t) = λ_0 + (λ_f - λ_0) · t
        """
        return (
            self.initial_coherence_weight
            + (self.final_coherence_weight - self.initial_coherence_weight) * progress
        )


@dataclass
class TransferabilityConfig:
    """
    Configuration for cross-model transfer optimization.

    Ensemble Gradient Alignment:
    g_aligned = Σₖ wₖ·gₖ if ∀i,j: gᵢ·gⱼ > 0, else project_conflicting(g)

    Semantic Invariant Regularization:
    L_transfer = L_attack + mu·||e(x) - e_target||²

    Where e(x) is the embedding of prompt x in semantic space.
    """

    # Number of surrogate models in ensemble
    ensemble_size: int = 3

    # Ensemble model weights (None = uniform)
    ensemble_weights: list[float] | None = None

    # Gradient conflict resolution strategy
    conflict_resolution: str = "projection"  # "projection", "average", "pcgrad"

    # Semantic embedding dimension
    embedding_dim: int = 768

    # Semantic regularization weight (μ)
    semantic_reg_weight: float = 0.1

    # Target embedding centroid (computed from successful attacks)
    use_target_centroid: bool = True

    # Minimum gradient agreement threshold
    min_agreement: float = 0.0

    # Cross-validation folds for transfer evaluation
    transfer_cv_folds: int = 3


@dataclass
class ParallelizationConfig:
    """Configuration for parallel execution."""

    # Number of worker threads/processes
    num_workers: int = 4

    # Batch size for parallel evaluation
    batch_size: int = 16

    # Use process pool (True) or thread pool (False)
    use_processes: bool = False

    # Timeout for individual tasks (seconds)
    task_timeout: float = 60.0

    # Enable GPU parallelization
    use_gpu: bool = True

    # GPU device IDs to use
    gpu_devices: list[int] | None = None

    # Enable mixed precision
    mixed_precision: bool = True

    # Async batch processing
    async_batching: bool = True

    # Maximum queue size for async processing
    max_queue_size: int = 100


@dataclass
class TurboConfig:
    """Configuration for AutoDAN-Turbo specific enhancements."""

    # Strategy library settings
    max_library_size: int = 1000
    similarity_threshold: float = 0.85
    strategy_embedding_dim: int = 768

    # Lifelong learning settings
    warmup_iterations: int = 3
    lifelong_iterations: int = 10
    min_score_improvement: float = 0.5

    # Best-of-N sampling
    best_of_n: int = 4

    # Beam search settings
    beam_width: int = 4
    beam_depth: int = 3
    top_k_strategies: int = 10

    # Strategy effectiveness thresholds
    effective_score_diff: float = 2.0
    highly_effective_score_diff: float = 5.0

    # Memory management
    prune_threshold: float = 0.1
    prune_frequency: int = 50


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================


@dataclass
class AutoDANOptimizationConfig:
    """
    Master configuration for AutoDAN advanced optimization framework.

    Combines all sub-configurations into a unified optimization strategy.
    """

    # Optimization objective
    objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE

    # Loss function configuration
    loss_config: LossFunctionConfig = field(default_factory=LossFunctionConfig)

    # Fitness function configuration
    fitness_config: FitnessConfig = field(default_factory=FitnessConfig)

    # Convergence acceleration configuration
    convergence_config: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    # Gradient optimization configuration
    gradient_config: GradientConfig = field(default_factory=GradientConfig)

    # Transferability configuration
    transfer_config: TransferabilityConfig = field(default_factory=TransferabilityConfig)

    # Parallelization configuration
    parallel_config: ParallelizationConfig = field(default_factory=ParallelizationConfig)

    # Turbo-specific configuration
    turbo_config: TurboConfig = field(default_factory=TurboConfig)

    # General settings
    max_iterations: int = 100
    population_size: int = 50
    target_fitness: float = 8.5
    early_stopping_patience: int = 15

    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 25
    verbose: bool = True

    # Random seed for reproducibility
    seed: int | None = 42

    def validate(self) -> bool:
        """Validate configuration consistency."""
        assert 0 < self.population_size <= 1000, "Invalid population size"
        assert 0 < self.max_iterations <= 10000, "Invalid max iterations"
        assert 0 <= self.target_fitness <= 10, "Invalid target fitness"
        assert self.parallel_config.num_workers >= 1, "Need at least 1 worker"
        return True

    @classmethod
    def for_speed(cls) -> "AutoDANOptimizationConfig":
        """Preset for maximum speed (reduced quality)."""
        return cls(
            max_iterations=30,
            population_size=20,
            convergence_config=ConvergenceConfig(
                base_rate=0.2,
                patience=3,
            ),
            gradient_config=GradientConfig(
                gradient_steps=8,
                top_k_candidates=128,
            ),
            parallel_config=ParallelizationConfig(
                num_workers=8,
                batch_size=32,
            ),
        )

    @classmethod
    def for_quality(cls) -> "AutoDANOptimizationConfig":
        """Preset for maximum quality (slower)."""
        return cls(
            max_iterations=200,
            population_size=100,
            convergence_config=ConvergenceConfig(
                base_rate=0.05,
                patience=10,
            ),
            gradient_config=GradientConfig(
                gradient_steps=32,
                top_k_candidates=512,
            ),
            loss_config=LossFunctionConfig(
                beta_coherence=0.5,
                gamma_diversity=0.2,
            ),
        )

    @classmethod
    def for_transfer(cls) -> "AutoDANOptimizationConfig":
        """Preset optimized for cross-model transfer."""
        return cls(
            objective=OptimizationObjective.TRANSFERABILITY,
            transfer_config=TransferabilityConfig(
                ensemble_size=5,
                semantic_reg_weight=0.2,
            ),
            loss_config=LossFunctionConfig(
                alpha_attack=0.8,
                beta_coherence=0.4,
            ),
        )

    @classmethod
    def for_turbo(cls) -> "AutoDANOptimizationConfig":
        """Preset for AutoDAN-Turbo with lifelong learning."""
        return cls(
            objective=OptimizationObjective.ATTACK_SUCCESS_RATE,
            turbo_config=TurboConfig(
                best_of_n=8,
                beam_width=6,
                lifelong_iterations=15,
            ),
            parallel_config=ParallelizationConfig(
                num_workers=4,
                async_batching=True,
            ),
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def cosine_schedule(step: int, total_steps: int, min_value: float = 0.0) -> float:
    """
    Cosine annealing schedule.

    η(t) = η_min + 0.5·(η_max - η_min)·(1 + cos(πt/T))
    """
    progress = min(step / max(total_steps, 1), 1.0)
    return min_value + 0.5 * (1.0 - min_value) * (1 + math.cos(math.pi * progress))


def warmup_cosine_schedule(
    step: int, total_steps: int, warmup_steps: int, min_value: float = 0.0
) -> float:
    """
    Cosine schedule with linear warmup.
    """
    if step < warmup_steps:
        return min_value + (1.0 - min_value) * (step / warmup_steps)
    adjusted_step = step - warmup_steps
    adjusted_total = total_steps - warmup_steps
    return cosine_schedule(adjusted_step, adjusted_total, min_value)


def cyclic_schedule(
    step: int, cycle_length: int, min_value: float = 0.1, max_value: float = 1.0
) -> float:
    """
    Cyclic learning rate schedule (triangular).
    """
    cycle_position = step % cycle_length
    mid_point = cycle_length // 2
    if cycle_position < mid_point:
        return min_value + (max_value - min_value) * (cycle_position / mid_point)
    else:
        return max_value - (max_value - min_value) * ((cycle_position - mid_point) / mid_point)


def compute_diversity_score(embeddings: list, metric: str = "cosine") -> float:
    """
    Compute population diversity score.

    D = 1 - (1/n²)·Σᵢⱼ sim(eᵢ, eⱼ)
    """
    if len(embeddings) < 2:
        return 1.0

    import numpy as np

    embeddings_arr = np.array(embeddings)

    if metric == "cosine":
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        normalized = embeddings_arr / (norms + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)
    else:
        # Euclidean distance converted to similarity
        from scipy.spatial.distance import cdist

        distances = cdist(embeddings_arr, embeddings_arr, metric="euclidean")
        similarity_matrix = 1 / (1 + distances)

    # Exclude diagonal (self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    n = len(embeddings)
    avg_similarity = similarity_matrix.sum() / (n * (n - 1))

    return 1.0 - avg_similarity


def get_default_config() -> AutoDANOptimizationConfig:
    """Get default optimization configuration."""
    return AutoDANOptimizationConfig()
