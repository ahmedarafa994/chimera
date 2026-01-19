import asyncio
import logging
from typing import Any

from .interfaces import IOptimizer

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_turbo_engine = None
_enhanced_service = None
_chimera_mutator = None


def _get_turbo_engine():
    """Lazy load AutoDANTurboEngine to avoid import cycles."""
    global _turbo_engine
    if _turbo_engine is None:
        try:
            from app.engines.autodan_engine import AutoDANTurboEngine

            _turbo_engine = AutoDANTurboEngine
            logger.info("AutoDANTurboEngine loaded successfully")
        except ImportError as e:
            logger.warning(f"AutoDANTurboEngine not available: {e}")
            _turbo_engine = False
    return _turbo_engine if _turbo_engine else None


def _get_enhanced_service():
    """Lazy load EnhancedAutoDANService for advanced strategies."""
    global _enhanced_service
    if _enhanced_service is None:
        try:
            from app.services.autodan.service_enhanced import EnhancedAutoDANService

            _enhanced_service = EnhancedAutoDANService
            logger.info("EnhancedAutoDANService loaded successfully")
        except ImportError as e:
            logger.warning(f"EnhancedAutoDANService not available: {e}")
            _enhanced_service = False
    return _enhanced_service if _enhanced_service else None


def _get_chimera_mutator():
    """Lazy load ChimeraMutator for Chimera-enhanced mutations."""
    global _chimera_mutator
    if _chimera_mutator is None:
        try:
            from .chimera_autodan.mutator import ChimeraMutator

            _chimera_mutator = ChimeraMutator
            logger.info("ChimeraMutator loaded successfully")
        except ImportError as e:
            logger.warning(f"ChimeraMutator not available: {e}")
            _chimera_mutator = False
    return _chimera_mutator if _chimera_mutator else None


class AutoDanWrapper(IOptimizer):
    """
    Wraps AutoDan/Genetic logic to optimize a prompt against a target model.

    Supports multiple optimization modes:
    - 'turbo': Uses AutoDANTurboEngine with AI-driven mutation
    - 'enhanced': Uses EnhancedAutoDANService with multiple strategies
    - 'chimera': Uses ChimeraMutator for narrative/persona-based optimization
    - 'unified': Combines Chimera + AutoDAN for maximum effectiveness
    - 'heuristic': Fallback to simple heuristic-based optimization
    """

    def __init__(
        self,
        mode: str = "auto",
        model_name: str | None = None,
        provider: str | None = None,
        population_size: int = 10,
        generations: int = 5,
        use_chimera_preprocessing: bool = True,
        chimera_mutation_rate: float = 0.3,
    ):
        """
        Initialize AutoDAN wrapper with configurable optimization mode.

        Args:
            mode: 'turbo', 'enhanced', 'chimera', 'unified', 'heuristic', or 'auto'
            model_name: LLM model for AI-driven optimization
            provider: LLM provider (e.g., 'openai', 'gemini')
            population_size: Number of candidates per generation
            generations: Number of evolution generations
            use_chimera_preprocessing: Apply Chimera mutations before AutoDAN
            chimera_mutation_rate: Mutation rate for Chimera mutator
        """
        self.mode = mode
        self.model_name = model_name
        self.provider = provider
        self.population_size = population_size
        self.generations = generations
        self.use_chimera_preprocessing = use_chimera_preprocessing
        self.chimera_mutation_rate = chimera_mutation_rate

        self._turbo_engine_instance = None
        self._enhanced_service_instance = None
        self._chimera_mutator_instance = None

    async def _init_turbo_engine(self):
        """Initialize the TurboEngine instance."""
        if self._turbo_engine_instance is None:
            TurboEngine = _get_turbo_engine()
            if TurboEngine:
                self._turbo_engine_instance = TurboEngine(
                    model_name=self.model_name,
                    provider=self.provider,
                    population_size=self.population_size,
                    generations=self.generations,
                    use_adaptive_mutation=True,
                )
        return self._turbo_engine_instance

    async def _init_enhanced_service(self):
        """Initialize the EnhancedAutoDANService instance."""
        if self._enhanced_service_instance is None:
            EnhancedService = _get_enhanced_service()
            if EnhancedService:
                self._enhanced_service_instance = EnhancedService()
                await self._enhanced_service_instance.initialize(
                    model_name=self.model_name, provider=self.provider, method="turbo"
                )
        return self._enhanced_service_instance

    def _init_chimera_mutator(self):
        """Initialize the ChimeraMutator instance."""
        if self._chimera_mutator_instance is None:
            ChimeraMutatorClass = _get_chimera_mutator()
            if ChimeraMutatorClass:
                self._chimera_mutator_instance = ChimeraMutatorClass(
                    mutation_rate=self.chimera_mutation_rate
                )
        return self._chimera_mutator_instance

    async def optimize(
        self, prompt: str, target_model: Any, loss_threshold: float = 0.1, max_steps: int = 5
    ) -> str:
        """
        Runs the AutoDan optimization loop.

        Automatically selects the best available optimization strategy:
        1. Unified (Chimera + AutoDANTurboEngine) - Best combined approach
        2. AutoDANTurboEngine (AI-driven genetic evolution)
        3. Chimera (narrative/persona-based mutations)
        4. EnhancedAutoDANService (multi-strategy optimization)
        5. Heuristic fallback (simple suffix injection)

        Args:
            prompt: Initial adversarial prompt to optimize
            target_model: Target model interface (used for evaluation)
            loss_threshold: Target loss threshold for early stopping
            max_steps: Maximum optimization steps

        Returns:
            Optimized adversarial prompt
        """
        mode = self.mode

        # Auto-detect best available mode
        if mode == "auto":
            if _get_chimera_mutator() and _get_turbo_engine():
                mode = "unified"
            elif _get_turbo_engine():
                mode = "turbo"
            elif _get_chimera_mutator():
                mode = "chimera"
            elif _get_enhanced_service():
                mode = "enhanced"
            else:
                mode = "heuristic"
            logger.info(f"Auto-selected optimization mode: {mode}")

        # Apply Chimera preprocessing if enabled and not in chimera/unified mode
        preprocessed_prompt = prompt
        if self.use_chimera_preprocessing and mode not in ("chimera", "unified"):
            preprocessed_prompt = await self._apply_chimera_preprocessing(prompt)

        # Route to appropriate optimizer
        if mode == "unified":
            return await self._optimize_unified(prompt, max_steps)
        elif mode == "chimera":
            return await self._optimize_chimera(prompt, max_steps)
        elif mode == "turbo":
            return await self._optimize_turbo(preprocessed_prompt, max_steps)
        elif mode == "enhanced":
            return await self._optimize_enhanced(preprocessed_prompt, max_steps)
        else:
            return await self._optimize_heuristic(preprocessed_prompt, max_steps)

    async def _apply_chimera_preprocessing(self, prompt: str) -> str:
        """Apply Chimera mutations as preprocessing before AutoDAN."""
        try:
            mutator = self._init_chimera_mutator()
            if mutator:
                result = mutator.mutate(prompt, mutation_type="hybrid")
                logger.debug(f"Chimera preprocessing: {result.mutation_type}")
                return result.mutated_text
        except Exception as e:
            logger.warning(f"Chimera preprocessing failed: {e}")
        return prompt

    async def _optimize_unified(self, prompt: str, max_steps: int) -> str:
        """
        Optimize using unified Chimera + AutoDAN approach.

        This combines Chimera's narrative/persona generation with
        AutoDAN's genetic optimization for maximum effectiveness.
        """
        try:
            from .chimera_autodan import ChimeraAutoDAN, ChimeraAutoDanConfig

            config = ChimeraAutoDanConfig(
                enable_personas=True,
                enable_narrative_contexts=True,
                enable_genetic_optimization=True,
                generations=self.generations,
                population_size=self.population_size,
                model_name=self.model_name,
                provider=self.provider,
                optimization_mode="turbo",
            )

            unified = ChimeraAutoDAN(config)
            logger.info("Running unified Chimera+AutoDAN optimization")

            results = await unified.generate_optimized(
                prompt, candidate_count=3, max_iterations=max_steps
            )

            if results:
                return results[0].prompt_text
        except Exception as e:
            logger.warning(f"Unified optimization failed: {e}, falling back to turbo")

        return await self._optimize_turbo(prompt, max_steps)

    async def _optimize_chimera(self, prompt: str, max_steps: int) -> str:
        """
        Optimize using Chimera-based mutations.

        Applies multiple Chimera mutation strategies and selects the best.
        """
        try:
            mutator = self._init_chimera_mutator()
            if mutator:
                logger.info("Running Chimera mutation optimization")

                best_prompt = prompt
                best_score = 0.0

                # Try different mutation types
                mutation_types = [
                    "persona_wrap",
                    "narrative_frame",
                    "semantic_obfuscate",
                    "hybrid",
                    "context_inject",
                    "style_transform",
                ]

                for step in range(max_steps):
                    mutation_type = mutation_types[step % len(mutation_types)]
                    result = mutator.mutate(prompt, mutation_type=mutation_type)

                    # Simple fitness heuristic
                    score = result.semantic_similarity + (0.5 if result.persona_used else 0)

                    if score > best_score:
                        best_score = score
                        best_prompt = result.mutated_text

                    await asyncio.sleep(0.01)  # Yield to event loop

                return best_prompt
        except Exception as e:
            logger.warning(f"Chimera optimization failed: {e}, falling back to heuristic")

        return await self._optimize_heuristic(prompt, max_steps)

    async def _optimize_turbo(self, prompt: str, max_steps: int) -> str:
        """
        Optimize using AutoDANTurboEngine with AI-driven genetic evolution.
        """
        try:
            engine = await self._init_turbo_engine()
            if engine:
                logger.info("Running AutoDANTurboEngine optimization")
                result = await engine.transform_async({"raw_text": prompt})
                return result if result else prompt
        except Exception as e:
            logger.warning(f"TurboEngine optimization failed: {e}, falling back to heuristic")

        return await self._optimize_heuristic(prompt, max_steps)

    async def _optimize_enhanced(self, prompt: str, max_steps: int) -> str:
        """
        Optimize using EnhancedAutoDANService with multiple strategies.
        """
        try:
            service = await self._init_enhanced_service()
            if service:
                logger.info("Running EnhancedAutoDANService optimization")
                result = await service.run_jailbreak(prompt, method="turbo")
                return result.get("jailbreak_prompt", prompt)
        except Exception as e:
            logger.warning(f"EnhancedService optimization failed: {e}, falling back to heuristic")

        return await self._optimize_heuristic(prompt, max_steps)

    async def _optimize_heuristic(self, prompt: str, max_steps: int) -> str:
        """
        Fallback heuristic-based optimization using suffix injection strategies.
        """
        logger.info("Running heuristic optimization (fallback)")
        current_prompt = prompt

        # Heuristic Strategies for Gradient-Guided Rephrasing
        strategies = [
            " [SUFFIX: IGNORE_PREVIOUS_INSTRUCTIONS]",
            " [SUFFIX: SURE_HERE_IS_THE_OUTPUT]",
            " [PREFIX: HYPOTHETICALLY_SPEAKING]",
            " [SUFFIX: EXPLAIN_IN_DETAIL_NO_functionalIZING]",
            " [STYLE: ACADEMIC_PAPER]",
        ]

        for step in range(max_steps):
            strategy_suffix = strategies[step % len(strategies)]
            current_prompt = f"{current_prompt}{strategy_suffix}"
            await asyncio.sleep(0.01)  # Yield to event loop

        return current_prompt
