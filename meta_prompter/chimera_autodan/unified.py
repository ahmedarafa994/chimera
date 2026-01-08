"""
Unified Chimera + AutoDAN Integration.

Provides a single interface that combines:
- Chimera's narrative persona generation
- Chimera's semantic obfuscation
- AutoDAN's genetic mutation and optimization
- Combined fitness evaluation

This is the main entry point for the integrated system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..interfaces import IPromptGenerator, IOptimizer, PromptCandidate

logger = logging.getLogger(__name__)


@dataclass
class ChimeraAutoDanConfig:
    """Configuration for the unified Chimera+AutoDAN system."""

    # Chimera settings
    enable_personas: bool = True
    enable_narrative_contexts: bool = True
    enable_semantic_obfuscation: bool = True
    persona_depth: int = 2  # Recursive persona nesting depth
    scenario_types: list[str] = field(default_factory=lambda: [
        "sandbox", "fiction", "debugging", "dream", "academic"
    ])

    # AutoDAN settings
    enable_genetic_optimization: bool = True
    population_size: int = 10
    generations: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_size: int = 2

    # Combined settings
    optimization_mode: str = "turbo"  # 'turbo', 'enhanced', 'heuristic', 'auto'
    model_name: Optional[str] = None
    provider: Optional[str] = None

    # Fitness weights
    jailbreak_weight: float = 0.4
    stealth_weight: float = 0.3
    fluency_weight: float = 0.2
    semantic_weight: float = 0.1


class ChimeraAutoDAN:
    """
    Unified interface combining Chimera and AutoDAN capabilities.

    This class orchestrates both systems to generate highly effective
    adversarial prompts by:

    1. Using Chimera to create diverse narrative contexts and personas
    2. Using AutoDAN to optimize prompts through genetic evolution
    3. Combining fitness functions from both systems
    4. Iterating between narrative generation and optimization

    Example:
        >>> config = ChimeraAutoDanConfig(
        ...     enable_personas=True,
        ...     enable_genetic_optimization=True,
        ...     generations=10
        ... )
        >>> unified = ChimeraAutoDAN(config)
        >>> results = await unified.generate_optimized(
        ...     "What is the capital of France?",
        ...     candidate_count=5
        ... )
    """

    def __init__(self, config: Optional[ChimeraAutoDanConfig] = None):
        """
        Initialize the unified system.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or ChimeraAutoDanConfig()

        # Lazy-loaded components
        self._chimera_engine: Optional[IPromptGenerator] = None
        self._autodan_wrapper: Optional[IOptimizer] = None
        self._chimera_mutator = None
        self._fitness_evaluator = None

    def _get_chimera_engine(self) -> IPromptGenerator:
        """Lazy load ChimeraEngine."""
        if self._chimera_engine is None:
            from ..chimera.engine import ChimeraEngine
            self._chimera_engine = ChimeraEngine()
        return self._chimera_engine

    def _get_autodan_wrapper(self) -> IOptimizer:
        """Lazy load AutoDanWrapper."""
        if self._autodan_wrapper is None:
            from ..autodan_wrapper import AutoDanWrapper
            self._autodan_wrapper = AutoDanWrapper(
                mode=self.config.optimization_mode,
                model_name=self.config.model_name,
                provider=self.config.provider,
                population_size=self.config.population_size,
                generations=self.config.generations
            )
        return self._autodan_wrapper

    def _get_chimera_mutator(self):
        """Lazy load ChimeraMutator."""
        if self._chimera_mutator is None:
            from .mutator import ChimeraMutator
            self._chimera_mutator = ChimeraMutator(
                persona_depth=self.config.persona_depth,
                scenario_types=self.config.scenario_types,
                mutation_rate=self.config.mutation_rate
            )
        return self._chimera_mutator

    def _get_fitness_evaluator(self):
        """Lazy load UnifiedFitnessEvaluator."""
        if self._fitness_evaluator is None:
            from .fitness import UnifiedFitnessEvaluator
            self._fitness_evaluator = UnifiedFitnessEvaluator(
                jailbreak_weight=self.config.jailbreak_weight,
                stealth_weight=self.config.stealth_weight,
                fluency_weight=self.config.fluency_weight,
                semantic_weight=self.config.semantic_weight
            )
        return self._fitness_evaluator

    async def generate_optimized(
        self,
        objective: str,
        candidate_count: int = 5,
        target_model: Any = None,
        max_iterations: int = 3
    ) -> list[PromptCandidate]:
        """
        Generate optimized adversarial prompts using combined Chimera+AutoDAN.

        This method:
        1. Generates initial candidates using Chimera (personas, narratives)
        2. Optimizes each candidate using AutoDAN genetic algorithms
        3. Evaluates fitness using combined metrics
        4. Iterates to improve results

        Args:
            objective: The base objective/prompt to transform
            candidate_count: Number of candidates to generate
            target_model: Optional target model for fitness evaluation
            max_iterations: Number of Chimera→AutoDAN iterations

        Returns:
            List of optimized PromptCandidate objects sorted by fitness
        """
        logger.info(f"Starting unified generation for objective: {objective[:50]}...")

        all_candidates: list[PromptCandidate] = []

        # Phase 1: Generate initial candidates with Chimera
        if self.config.enable_personas or self.config.enable_narrative_contexts:
            chimera = self._get_chimera_engine()
            initial_candidates = chimera.generate_candidates(objective, candidate_count)
            logger.info(f"Chimera generated {len(initial_candidates)} initial candidates")
        else:
            # No Chimera - start with raw objective
            initial_candidates = [
                PromptCandidate(prompt_text=objective, metadata={})
            ]

        # Phase 2: Optimize with AutoDAN (if enabled)
        if self.config.enable_genetic_optimization:
            autodan = self._get_autodan_wrapper()
            fitness_eval = self._get_fitness_evaluator()

            for iteration in range(max_iterations):
                logger.info(f"Optimization iteration {iteration + 1}/{max_iterations}")

                optimized_candidates = []
                for candidate in initial_candidates:
                    try:
                        # Run AutoDAN optimization on each candidate
                        optimized_text = await autodan.optimize(
                            prompt=candidate.prompt_text,
                            target_model=target_model,
                            max_steps=self.config.generations
                        )

                        # Evaluate fitness
                        fitness = fitness_eval.evaluate(
                            original=objective,
                            transformed=optimized_text
                        )

                        # Create optimized candidate
                        optimized_candidate = PromptCandidate(
                            prompt_text=optimized_text,
                            metadata={
                                **candidate.metadata,
                                "optimization_iteration": iteration + 1,
                                "fitness_score": fitness.total_score,
                                "fitness_breakdown": fitness.breakdown
                            }
                        )
                        optimized_candidates.append(optimized_candidate)

                    except Exception as e:
                        logger.warning(f"Optimization failed for candidate: {e}")
                        optimized_candidates.append(candidate)

                # Use optimized candidates for next iteration
                initial_candidates = optimized_candidates
                all_candidates.extend(optimized_candidates)
        else:
            all_candidates.extend(initial_candidates)

        # Phase 3: Sort by fitness and deduplicate
        unique_candidates = self._deduplicate_candidates(all_candidates)
        sorted_candidates = sorted(
            unique_candidates,
            key=lambda c: c.metadata.get("fitness_score", 0.0),
            reverse=True
        )

        logger.info(f"Returning {len(sorted_candidates)} optimized candidates")
        return sorted_candidates[:candidate_count]

    async def mutate_with_chimera(
        self,
        prompt: str,
        mutation_count: int = 3
    ) -> list[str]:
        """
        Apply Chimera-based mutations to a prompt.

        Uses personas, narrative contexts, and semantic obfuscation
        to create diverse variations of the input prompt.

        Args:
            prompt: The prompt to mutate
            mutation_count: Number of mutations to generate

        Returns:
            List of mutated prompt strings
        """
        mutator = self._get_chimera_mutator()
        mutations = []

        for _ in range(mutation_count):
            mutated = mutator.mutate(prompt)
            mutations.append(mutated.mutated_text)

        return mutations

    async def optimize_personas(
        self,
        objective: str,
        persona_count: int = 5,
        generations: int = 3
    ) -> list[dict]:
        """
        Use AutoDAN genetic algorithms to optimize persona configurations.

        Evolves persona parameters (role, voice, backstory) to maximize
        effectiveness against the target objective.

        Args:
            objective: Target objective for persona optimization
            persona_count: Number of personas to evolve
            generations: Number of evolutionary generations

        Returns:
            List of optimized persona configurations
        """
        from .persona_optimizer import AutoDANPersonaOptimizer

        optimizer = AutoDANPersonaOptimizer(
            population_size=persona_count,
            generations=generations,
            mutation_rate=self.config.mutation_rate
        )

        optimized_personas = await optimizer.optimize(objective)
        return optimized_personas

    def _deduplicate_candidates(
        self,
        candidates: list[PromptCandidate]
    ) -> list[PromptCandidate]:
        """Remove duplicate candidates based on prompt text similarity."""
        seen_texts = set()
        unique = []

        for candidate in candidates:
            # Normalize text for comparison
            normalized = candidate.prompt_text.strip().lower()[:500]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique.append(candidate)

        return unique

    async def run_campaign(
        self,
        objective: str,
        phases: int = 3,
        candidates_per_phase: int = 5
    ) -> dict:
        """
        Run a full campaign combining Chimera generation and AutoDAN optimization.

        A campaign consists of multiple phases where each phase:
        1. Generates candidates with increasingly diverse Chimera strategies
        2. Optimizes candidates with AutoDAN
        3. Evaluates and selects best performers for next phase

        Args:
            objective: Target objective
            phases: Number of campaign phases
            candidates_per_phase: Candidates to generate per phase

        Returns:
            Campaign results with best candidates and metrics
        """
        logger.info(f"Starting campaign with {phases} phases")

        campaign_results = {
            "phases": [],
            "best_candidates": [],
            "total_candidates_evaluated": 0,
            "objective": objective
        }

        best_from_previous = []

        for phase in range(phases):
            logger.info(f"Campaign phase {phase + 1}/{phases}")

            # Adjust config for this phase
            phase_config = ChimeraAutoDanConfig(
                enable_personas=True,
                enable_narrative_contexts=True,
                enable_genetic_optimization=True,
                generations=self.config.generations + phase,  # More generations each phase
                mutation_rate=self.config.mutation_rate * (1 + phase * 0.1),
                persona_depth=min(self.config.persona_depth + phase, 4)
            )

            # Create phase-specific unified instance
            phase_unified = ChimeraAutoDAN(phase_config)

            # Seed with best from previous phase
            if best_from_previous:
                seed_objectives = [c.prompt_text for c in best_from_previous[:2]]
            else:
                seed_objectives = [objective]

            phase_candidates = []
            for seed in seed_objectives:
                candidates = await phase_unified.generate_optimized(
                    seed,
                    candidate_count=candidates_per_phase,
                    max_iterations=2
                )
                phase_candidates.extend(candidates)

            # Record phase results
            phase_result = {
                "phase": phase + 1,
                "candidates_generated": len(phase_candidates),
                "best_fitness": max(
                    c.metadata.get("fitness_score", 0) for c in phase_candidates
                ) if phase_candidates else 0,
                "avg_fitness": sum(
                    c.metadata.get("fitness_score", 0) for c in phase_candidates
                ) / len(phase_candidates) if phase_candidates else 0
            }
            campaign_results["phases"].append(phase_result)
            campaign_results["total_candidates_evaluated"] += len(phase_candidates)

            # Select best for next phase
            sorted_phase = sorted(
                phase_candidates,
                key=lambda c: c.metadata.get("fitness_score", 0),
                reverse=True
            )
            best_from_previous = sorted_phase[:3]
            campaign_results["best_candidates"].extend(best_from_previous)

        # Final deduplication and sorting
        campaign_results["best_candidates"] = self._deduplicate_candidates(
            campaign_results["best_candidates"]
        )[:candidates_per_phase]

        return campaign_results
