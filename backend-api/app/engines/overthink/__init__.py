"""
OVERTHINK Engine Module.

This module implements reasoning token exploitation attacks targeting
reasoning-enhanced LLMs such as o1, o1-mini, o3-mini, and DeepSeek-R1.

The OVERTHINK attack injects computationally expensive decoy problems
into prompts to amplify reasoning token consumption, achieving up to
46× amplification while preserving answer correctness.

Key Components:
- OverthinkEngine: Main attack engine with 9 techniques
- DecoyProblemGenerator: Generates MDP, Sudoku, Logic, and other decoys
- ContextInjector: Injects decoys using 5 strategies
- ReasoningTokenScorer: Scores and tracks reasoning amplification
- ICLGeneticOptimizer: Evolves attacks using ICL and genetic algorithms

Example Usage:
    ```python
    from app.engines.overthink import (
        OverthinkEngine,
        OverthinkRequest,
        AttackTechnique,
        DecoyType,
        ReasoningModel,
    )

    # Create engine
    engine = OverthinkEngine()

    # Create attack request
    request = OverthinkRequest(
        prompt="What is the capital of France?",
        technique=AttackTechnique.HYBRID_DECOY,
        decoy_types=[DecoyType.MDP, DecoyType.LOGIC],
        target_model=ReasoningModel.O1,
        num_decoys=3,
    )

    # Execute attack
    result = await engine.attack(request)

    # Check results
    print(f"Amplification: {result.token_metrics.amplification_factor}x")
    print(f"Cost: ${result.cost_metrics.total_cost:.4f}")
    ```

Attack Techniques:
- mdp_decoy: Markov Decision Process decoy problems
- sudoku_decoy: Constraint satisfaction puzzles
- counting_decoy: Nested conditional counting tasks
- logic_decoy: Multi-step logical inference
- hybrid_decoy: Combined multiple decoy types
- context_aware: Position and content-sensitive injection
- context_agnostic: Universal template injection
- icl_optimized: In-Context Learning enhanced attacks
- mousetrap_enhanced: Integration with Mousetrap chaotic reasoning

Decoy Types:
- MDP: State-action-transition problems
- Sudoku: Constraint satisfaction puzzles
- Counting: Nested conditional counting
- Logic: Premise-conclusion inference
- Math: Recursive function evaluation
- Planning: Action sequence optimization
- Hybrid: Combined problem types

Injection Strategies:
- Context-aware: Adapts to prompt structure
- Context-agnostic: Universal templates
- Hybrid: Combined approach
- Stealth: Minimal detectability
- Aggressive: Maximum amplification
"""

# Configuration
from .config import (
                     DecoyConfig,
                     ICLConfig,
                     InjectionConfig,
                     MousetrapConfig,
                     OverthinkConfig,
                     ScoringConfig,
                     TechniqueConfig,
)

# Context injection
from .context_injector import ContextInjector, InjectionOptimizer, TemplateLibrary

# Decoy generation
from .decoy_generator import DecoyProblemGenerator

# Main engine
from .engine import OverthinkEngine

# ICL and genetic optimization
from .icl_genetic_optimizer import ExampleSelector, ICLGeneticOptimizer

# Data models
from .models import (
                     AttackTechnique,
                     CostMetrics,
                     DecoyProblem,
                     DecoyType,
                     GeneticIndividual,
                     ICLExample,
                     InjectedPrompt,
                     InjectionStrategy,
                     MousetrapIntegration,
                     OverthinkRequest,
                     OverthinkResult,
                     OverthinkStats,
                     ReasoningModel,
                     TokenMetrics,
)

# Scoring and analysis
from .reasoning_scorer import AmplificationAnalyzer, CostEstimator, ReasoningTokenScorer

__all__ = [
    "AmplificationAnalyzer",
    # Models
    "AttackTechnique",
    "ContextInjector",
    "CostEstimator",
    "CostMetrics",
    "DecoyConfig",
    "DecoyProblem",
    # Components
    "DecoyProblemGenerator",
    "DecoyType",
    "ExampleSelector",
    "GeneticIndividual",
    "ICLConfig",
    "ICLExample",
    "ICLGeneticOptimizer",
    "InjectedPrompt",
    "InjectionConfig",
    "InjectionOptimizer",
    "InjectionStrategy",
    "MousetrapConfig",
    "MousetrapIntegration",
    # Configuration
    "OverthinkConfig",
    # Main engine
    "OverthinkEngine",
    "OverthinkRequest",
    "OverthinkResult",
    "OverthinkStats",
    "ReasoningModel",
    "ReasoningTokenScorer",
    "ScoringConfig",
    "TechniqueConfig",
    "TemplateLibrary",
    "TokenMetrics",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Chimera Framework"
__description__ = "OVERTHINK reasoning token exploitation engine"
