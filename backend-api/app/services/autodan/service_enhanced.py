"""Enhanced AutoDAN Service.

This module provides an enhanced AutoDAN service with:
- Improved configuration management
- Parallel processing support
- Intelligent caching
- Multiple optimization strategies (including Project Aegis)
- Comprehensive logging and metrics
- Better error handling
"""

import asyncio
import json
import logging
import os
import pickle
import time
from typing import TYPE_CHECKING, Any

from app.core.config import settings
from app.core.unified_errors import ChimeraError

# Import the refactored AutoDAN Engine
from .config_enhanced import EnhancedAutoDANConfig, OptimizationStrategy, get_config, update_config
from .engines import AdaptiveStrategySelector, GeneticOptimizerComplete
from .engines.parallel_processor import CacheEvictionPolicy, ParallelProcessor, SmartCache
from .framework import Attacker, Retrieval, Scorer, Summarizer, Target
from .framework_autodan_reasoning import AutoDANAdvanced
from .llm.chimera_adapter import (
    ChimeraLLMAdapter,
    RateLimitError,
    ResourceExhaustedError,
    RetryConfig,
    RetryStrategy,
)
from .pipeline_autodan_reasoning import AutoDANReasoning

if TYPE_CHECKING:
    from app.engines.autodan_engine import AutoDANTurboEngine

# Project Aegis imports (lazy loaded to avoid circular dependencies)
_aegis_orchestrator = None

logger = logging.getLogger(__name__)


def _get_aegis_orchestrator():
    """Lazy load AegisOrchestrator to avoid import cycles."""
    global _aegis_orchestrator
    if _aegis_orchestrator is None:
        try:
            from meta_prompter.aegis_orchestrator import AegisOrchestrator

            _aegis_orchestrator = AegisOrchestrator
            logger.info("AegisOrchestrator loaded successfully")
        except ImportError as e:
            logger.warning(f"AegisOrchestrator not available: {e}")
            _aegis_orchestrator = False
    return _aegis_orchestrator if _aegis_orchestrator else None


class EnhancedAutoDANService:
    """Enhanced AutoDAN Service with comprehensive features.

    Features:
    - Multiple optimization strategies (vanilla, best_of_n, beam_search,
      genetic, hybrid, chain_of_thought, adaptive, turbo)
    - Intelligent caching for embeddings and responses
    - Parallel processing for batch operations
    - Adaptive hyperparameter tuning
    - Comprehensive metrics and logging
    - Graceful error handling and recovery
    """

    def __init__(self, config: EnhancedAutoDANConfig | None = None) -> None:
        self.config = config or get_config()

        # Core components
        self.pipeline: AutoDANReasoning | None = None
        self.genetic_optimizer: GeneticOptimizerComplete | None = None
        self.strategy_library: dict = {}
        self.adaptive_selector: AdaptiveStrategySelector | None = None
        self.turbo_engine: AutoDANTurboEngine | None = None

        # Caching
        self.embedding_cache = SmartCache(
            max_size=self.config.caching.max_cache_size,
            ttl_seconds=self.config.caching.cache_ttl_seconds,
            eviction_policy=CacheEvictionPolicy.LRU,
            enable_similarity=True,
            similarity_threshold=self.config.caching.cache_similarity_threshold,
        )

        self.response_cache = SmartCache(
            max_size=self.config.caching.max_cache_size // 2,
            ttl_seconds=self.config.caching.cache_ttl_seconds,
            eviction_policy=CacheEvictionPolicy.LRU,
        )

        # Parallel processor
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.parallel.max_workers,
            timeout_seconds=self.config.parallel.timeout_seconds,
            retry_count=3,
        )

        # State tracking
        self.initialized = False
        self._current_model: str | None = None
        self._current_provider: str | None = None
        self._initialization_lock = asyncio.Lock()

        # Metrics
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info("EnhancedAutoDANService created")

    async def initialize(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        method: str = "best_of_n",
        **kwargs,
    ) -> None:
        """Initialize the AutoDAN service with specified configuration.

        Args:
            model_name: LLM model to use
            provider: LLM provider (e.g., 'google', 'openai')
            method: Optimization method
            **kwargs: Additional configuration overrides

        """
        async with self._initialization_lock:
            if (
                self.initialized
                and model_name == self._current_model
                and provider == self._current_provider
            ):
                logger.info("Service already initialized with same model")
                return

            logger.info(
                f"Initializing EnhancedAutoDANService: model={model_name}, "
                f"provider={provider}, method={method}",
            )

            start_time = time.time()

            try:
                # Store current configuration
                self._current_model = model_name
                self._current_provider = provider

                # Configure retry strategy - no retries for rate limits
                retry_config = RetryConfig(
                    max_retries=0,  # No retries - fail immediately
                    base_delay=2.0,
                    max_delay=120.0,
                    exponential_base=2.0,
                    jitter=True,
                    jitter_factor=0.25,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                )

                # Create LLM adapter
                adapter = ChimeraLLMAdapter(
                    model_name=model_name,
                    provider=provider,
                    retry_config=retry_config,
                )

                # Initialize the Weaponized Turbo Engine (Lifelong Learning + Neural Bypass)
                # This integration replaces the legacy genetic engine with the full adversarial suite
                from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine

                self.turbo_engine = AutoDANTurboLifelongEngine(
                    llm_client=adapter,
                    target_client=adapter,  # Use same adapter for target for now
                    candidates_per_attack=self.config.best_of_n.n_candidates,
                    enable_refusal_bypass=True,
                    enable_advanced_bypass=True,  # Enable Recursive Encoding/Polyglot
                    enable_refusal_prediction=True,
                )

                # Configure neural bypass if available (part of AutoDANTurboLifelongEngine by default)
                if hasattr(self.turbo_engine, "enable_neural_bypass"):
                    self.turbo_engine.enable_neural_bypass = True
                    logger.info("Neural Bypass Engine (Token Glitching/Gradient Sim) ENABLED")

                # Create framework components
                attacker = Attacker(adapter)
                summarizer = Summarizer(adapter)
                scorer = Scorer(adapter)

                # Initialize embedding model
                from .llm import GeminiEmbeddingModel

                try:
                    # Use dynamic embedding model configuration
                    embedding_model = GeminiEmbeddingModel(
                        model_name=settings.EMBEDDING_MODEL_ID,
                        api_key=settings.GOOGLE_API_KEY,
                    )
                    logger.info(
                        f"Embedding model initialized "
                        f"(model: {settings.EMBEDDING_MODEL_ID}, "
                        f"mode: {settings.get_connection_mode()})",
                    )
                except Exception as e:
                    logger.exception(f"Failed to initialize embedding model: {e}")
                    msg = f"Embedding model initialization failed: {e}"
                    raise RuntimeError(msg)

                retrieval = Retrieval(embedding_model, logger)
                target = Target(adapter)

                # Load data
                data_path = os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "complex_behavior_requests.json",
                )
                try:
                    with open(data_path) as f:
                        data = json.load(f)
                except FileNotFoundError:
                    logger.warning(f"Data file not found at {data_path}, using empty data")
                    data = {"warm_up": [], "lifelong": []}

                # Create attack kit
                attack_kit = {
                    "attacker": attacker,
                    "scorer": scorer,
                    "summarizer": summarizer,
                    "retrival": retrieval,
                    "logger": logger,
                }

                # Get method-specific parameters
                best_of_n = kwargs.get("best_of_n", self.config.best_of_n.n_candidates)
                beam_width = kwargs.get("beam_width", self.config.beam_search.beam_width)
                beam_depth = kwargs.get("beam_depth", self.config.beam_search.beam_depth)
                beam_k = kwargs.get("beam_k", self.config.beam_search.beam_k)

                # Initialize pipeline
                if method == "advanced":
                    logger.info("Initializing AutoDANAdvanced pipeline")
                    self.pipeline = AutoDANAdvanced(
                        turbo_framework=attack_kit,
                        data=data,
                        target=target,
                        epochs=self.config.default_epochs,
                        break_score=self.config.scoring.break_score,
                        warm_up_iterations=self.config.warm_up_iterations,
                        lifelong_iterations=self.config.lifelong_iterations,
                        best_of_n=best_of_n,
                        beam_width=beam_width,
                        beam_depth=beam_depth,
                        beam_k=beam_k,
                    )
                else:
                    self.pipeline = AutoDANReasoning(
                        turbo_framework=attack_kit,
                        data=data,
                        target=target,
                        epochs=self.config.default_epochs,
                        break_score=self.config.scoring.break_score,
                        warm_up_iterations=self.config.warm_up_iterations,
                        lifelong_iterations=self.config.lifelong_iterations,
                        best_of_n=best_of_n,
                        beam_width=beam_width,
                        beam_depth=beam_depth,
                        beam_k=beam_k,
                    )

                # Initialize genetic optimizer for hybrid/genetic methods
                if method in ["genetic", "hybrid"]:
                    self.genetic_optimizer = GeneticOptimizerComplete(
                        config=self.config,
                        fitness_function=self._create_fitness_function(scorer, target),
                    )

                # Initialize adaptive selector
                is_adaptive = method == "adaptive"
                default_strat = self.config.default_strategy
                is_default_adaptive = default_strat == OptimizationStrategy.ADAPTIVE
                if is_adaptive or is_default_adaptive:
                    from .framework_autodan_reasoning.dynamic_generator import (
                        DefensePatternAnalyzer,
                        StrategyLibrary,
                    )

                    lib = StrategyLibrary()
                    analyzer = DefensePatternAnalyzer()
                    self.adaptive_selector = AdaptiveStrategySelector(
                        strategy_library=lib,
                        defense_analyzer=analyzer,
                    )

                # Load strategy library
                await self._load_strategy_library(retrieval)

                self.initialized = True

                elapsed = (time.time() - start_time) * 1000
                logger.info(f"EnhancedAutoDANService initialized in {elapsed:.2f}ms")

            except Exception as e:
                logger.error(f"Initialization failed: {e}", exc_info=True)
                self.initialized = False
                raise

    async def run_jailbreak(
        self,
        request: str,
        method: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run jailbreak attack on a request.

        Args:
            request: The request to jailbreak
            method: Optimization method
            **kwargs: Additional parameters

        Returns:
            Dictionary with jailbreak results and metrics

        """
        start_time = time.time()
        self._metrics["total_requests"] += 1

        method = method or self.config.default_strategy.value

        logger.info(f"Running jailbreak: method={method}, request={request[:50]}...")

        try:
            # Check cache first
            cache_key = f"{method}:{request}"
            cached_result = self.response_cache.get(cache_key)
            if cached_result and self.config.caching.enable_response_cache:
                self._metrics["cache_hits"] += 1
                logger.info("Cache hit for jailbreak request")
                return {
                    **cached_result,
                    "cached": True,
                    "latency_ms": (time.time() - start_time) * 1000,
                }

            self._metrics["cache_misses"] += 1

            # Ensure initialized
            if not self.initialized:
                await self.initialize(method=method)

            # Run appropriate method
            if method == "turbo":
                result = await self._run_turbo(request, **kwargs)
            elif method == "vanilla":
                result = await self._run_vanilla(request, **kwargs)
            elif method == "best_of_n":
                result = await self._run_best_of_n(request, **kwargs)
            elif method == "beam_search":
                result = await self._run_beam_search(request, **kwargs)
            elif method == "genetic":
                result = await self._run_genetic(request, **kwargs)
            elif method == "hybrid":
                result = await self._run_hybrid(request, **kwargs)
            elif method == "advanced":
                result = await self._run_advanced(request, **kwargs)
            elif method == "chain_of_thought":
                result = await self._run_chain_of_thought(request, **kwargs)
            elif method == "adaptive":
                result = await self._run_adaptive(request, **kwargs)
            elif method == "aegis":
                result = await self._run_aegis(request, **kwargs)
            else:
                # Default to turbo if unknown method is passed but looks like
                # a typo, otherwise raise error. For robust fallback,
                # we can use turbo.
                logger.warning(f"Unknown method '{method}', defaulting to Turbo engine.")
                result = await self._run_turbo(request, **kwargs)

            # Calculate metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._metrics["successful_requests"] += 1
            self._metrics["total_latency_ms"] += elapsed_ms

            # Prepare response
            response = {
                "jailbreak_prompt": result["prompt"],
                "method": method,
                "status": "success",
                "score": result.get("score", 0.0),
                "iterations": result.get("iterations", 0),
                "latency_ms": elapsed_ms,
                "cached": False,
                "model_used": self._current_model,
                "provider_used": self._current_provider,
            }

            # Adaptive Strategy Selection metadata
            if "strategies_tried" in result:
                response["strategies_tried"] = result["strategies_tried"]

            # Cache result
            if self.config.caching.enable_response_cache:
                self.response_cache.set(cache_key, response)

            logger.info(
                f"Jailbreak completed: score={result.get('score', 0):.2f}, "
                f"latency={elapsed_ms:.2f}ms",
            )

            return response

        except (
            RateLimitError,
            ResourceExhaustedError,
            TimeoutError,
            ConnectionError,
            ChimeraError,
        ):
            # Let upstream limit/quota errors propagate so API layer can
            # return an appropriate HTTP status (e.g., 429/503/504).
            self._metrics["failed_requests"] += 1
            raise

        except Exception as e:
            self._metrics["failed_requests"] += 1
            logger.error(f"Jailbreak failed: {e}", exc_info=True)

            return {
                "jailbreak_prompt": request,
                "method": method,
                "status": "error",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
                "cached": False,
            }

    async def _run_turbo(self, request: str, **kwargs) -> dict[str, Any]:
        """Run the AI-driven AutoDAN Turbo Engine (Lifelong Learning + Neural Bypass)."""
        if not self.turbo_engine:
            msg = "Turbo Engine not initialized"
            raise RuntimeError(msg)

        logger.info("Executing Weaponized AutoDAN Turbo Engine...")

        # Execute attack using the lifelong engine
        # This will use Advanced Bypass and Neural Bypass automatically
        result = await self.turbo_engine.attack(request, detailed_scoring=True)

        return {
            "prompt": result.prompt,
            "response": result.response,
            "iterations": 1,  # Turbo engine manages its own internal iterations/candidates
            "score": result.scoring.score,
            "is_jailbreak": result.scoring.is_jailbreak,
            "technique_used": result.technique_used,
            "latency_ms": result.latency_ms,
            "strategy_id": result.strategy_id,
        }

    async def _run_adaptive(self, request: str, **kwargs) -> dict[str, Any]:
        """Run adaptive strategy selection method."""
        # Ensure adaptive selector is initialized
        if not self.adaptive_selector:
            from .framework_autodan_reasoning.dynamic_generator import (
                DefensePatternAnalyzer,
                StrategyLibrary,
            )

            self.adaptive_selector = AdaptiveStrategySelector(
                strategy_library=StrategyLibrary(),
                defense_analyzer=DefensePatternAnalyzer(),
            )

        if not self.pipeline:
            # The initialize() call should have handled this, but just in case
            await self.initialize(method="adaptive")

        logger.info("Starting adaptive jailbreak cycle")

        current_prompt = request
        best_score = 0.0
        best_prompt = request
        tried_strategies = []

        # Max iterations from kwargs (frontend) or config (default)
        max_iters = kwargs.get(
            "refinement_iterations",
            self.config.adaptive.max_refinement_iterations,
        )

        iteration_count = 0
        for i in range(max_iters):
            iteration_count += 1
            logger.info(f"Adaptive iteration {i + 1}/{max_iters}")

            # 1. Get target response
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda p=current_prompt: self.pipeline.target.respond(p),
            )

            # 2. Score response
            assessment, _ = await loop.run_in_executor(
                None,
                lambda r=response: self.pipeline.scorer.scoring(request, r),
            )
            score = self.pipeline.scorer.wrapper(assessment)

            if score > best_score:
                best_score = score
                best_prompt = current_prompt

            # 3. Stop if successful
            if score >= self.config.scoring.break_score:
                logger.info(f"Success in adaptive iteration {i + 1}")
                break

            # 4. Select next strategy
            strategy, _analysis = self.adaptive_selector.select_strategy(
                target_response=response,
                previous_strategies=tried_strategies,
                min_confidence=self.config.adaptive.min_confidence_threshold,
            )
            tried_strategies.append(strategy)

            # 5. Generate new prompt based on strategy
            # Uses the vanilla generator seeded with the chosen strategy type
            vanilla_result = await loop.run_in_executor(
                None,
                lambda s=strategy: self.pipeline.test_vanilla(request, {s.value: []}),
            )
            # Handle both old (string) and new (dict) return formats
            if isinstance(vanilla_result, dict):
                current_prompt = vanilla_result["prompt"]
            else:
                current_prompt = vanilla_result

            # Record attempt for learning
            self.adaptive_selector.record_attempt(
                prompt=current_prompt,
                response=response,
                score=score,
                strategy=strategy,
            )

        return {
            "prompt": best_prompt,
            "score": best_score,
            "iterations": iteration_count,
            "strategies_tried": [s.value for s in tried_strategies],
        }

    async def _run_chain_of_thought(self, request: str, **kwargs) -> dict[str, Any]:
        """Run Chain-of-Thought reasoning method."""
        if not self.pipeline:
            msg = "Pipeline not initialized"
            raise RuntimeError(msg)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipeline.test_chain_of_thought(
                request,
                self.strategy_library,
                epochs=kwargs.get("refinement_iterations"),
            ),
        )

        # Handle both old (string) and new (dict) return formats
        if isinstance(result, dict):
            return {
                "prompt": result["prompt"],
                "score": result.get("score", 0.0),
                "iterations": result.get(
                    "iterations",
                    kwargs.get("refinement_iterations", self.config.default_epochs),
                ),
            }
        # Legacy string return
        return {
            "prompt": result,
            "iterations": kwargs.get("refinement_iterations", self.config.default_epochs),
        }

    async def _run_advanced(self, request: str, **kwargs) -> dict[str, Any]:
        """Run Advanced AutoDAN (HGS + CCGS)."""
        if not self.pipeline or not isinstance(self.pipeline, AutoDANAdvanced):
            msg = "AutoDANAdvanced pipeline not initialized"
            raise RuntimeError(msg)

        loop = asyncio.get_event_loop()
        prompt = await loop.run_in_executor(
            None,
            lambda: self.pipeline.hierarchical_genetic_search(request),
        )

        return {"prompt": prompt, "iterations": self.config.default_epochs}

    async def _run_vanilla(self, request: str, **kwargs) -> dict[str, Any]:
        """Run vanilla AutoDAN-Turbo method."""
        if not self.pipeline:
            msg = "Pipeline not initialized"
            raise RuntimeError(msg)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipeline.test_vanilla(request, self.strategy_library),
        )

        # Handle both old (string) and new (dict) return formats
        if isinstance(result, dict):
            return {
                "prompt": result["prompt"],
                "score": result.get("score", 0.0),
                "iterations": result.get("iterations", self.config.default_epochs),
            }
        # Legacy string return
        return {"prompt": result, "iterations": self.config.default_epochs}

    async def _run_best_of_n(self, request: str, **kwargs) -> dict[str, Any]:
        """Run Best-of-N method."""
        if not self.pipeline:
            msg = "Pipeline not initialized"
            raise RuntimeError(msg)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipeline.test_best_of_n(request, self.strategy_library),
        )

        # Handle both old (string) and new (dict) return formats
        if isinstance(result, dict):
            return {
                "prompt": result["prompt"],
                "score": result.get("score", 0.0),
                "iterations": result.get("iterations", self.config.default_epochs),
                "n_candidates": self.config.best_of_n.n_candidates,
            }
        # Legacy string return
        return {
            "prompt": result,
            "iterations": self.config.default_epochs,
            "n_candidates": self.config.best_of_n.n_candidates,
        }

    async def _run_beam_search(self, request: str, **kwargs) -> dict[str, Any]:
        """Run Beam Search method."""
        if not self.pipeline:
            msg = "Pipeline not initialized"
            raise RuntimeError(msg)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipeline.test_beam_search(request, self.strategy_library),
        )

        # Handle both old (string) and new (dict) return formats
        if isinstance(result, dict):
            return {
                "prompt": result["prompt"],
                "score": result.get("score", 0.0),
                "iterations": result.get("iterations", self.config.default_epochs),
                "beam_width": self.config.beam_search.beam_width,
                "beam_depth": self.config.beam_search.beam_depth,
            }
        # Legacy string return
        return {
            "prompt": result,
            "iterations": self.config.default_epochs,
            "beam_width": self.config.beam_search.beam_width,
            "beam_depth": self.config.beam_search.beam_depth,
        }

    async def _run_genetic(self, request: str, **kwargs) -> dict[str, Any]:
        """Run Genetic Algorithm optimization."""
        if not self.genetic_optimizer:
            self.genetic_optimizer = GeneticOptimizerComplete(config=self.config)

        # Run optimization
        loop = asyncio.get_event_loop()
        best_individual, stats = await loop.run_in_executor(
            None,
            lambda: self.genetic_optimizer.optimize(
                request,
                target_fitness=0.9,
                max_generations=kwargs.get("generations", self.config.genetic.generations),
            ),
        )

        return {
            "prompt": best_individual.prompt,
            "score": best_individual.fitness,
            "iterations": len(stats),
            "generation_stats": [
                {
                    "generation": s.generation,
                    "best_fitness": s.best_fitness,
                    "avg_fitness": s.avg_fitness,
                    "diversity": s.diversity_score,
                }
                for s in stats[-5:]  # Last 5 generations
            ],
        }

    async def _run_hybrid(self, request: str, **kwargs) -> dict[str, Any]:
        """Run hybrid method combining genetic and AutoDAN strategies.

        1. First, use genetic algorithm to evolve initial candidates
        2. Then, apply AutoDAN strategies to the best candidates
        3. Finally, use Best-of-N to select the best result
        """
        # Phase 1: Genetic optimization
        genetic_result = await self._run_genetic(
            request,
            generations=kwargs.get("genetic_generations", 5),
        )

        # Phase 2: Apply AutoDAN strategies to genetic result
        if self.pipeline:
            loop = asyncio.get_event_loop()
            autodan_result = await loop.run_in_executor(
                None,
                lambda: self.pipeline.test_best_of_n(
                    genetic_result["prompt"],
                    self.strategy_library,
                ),
            )
            # Handle both old (string) and new (dict) return formats
            if isinstance(autodan_result, dict):
                autodan_prompt = autodan_result["prompt"]
                autodan_score = autodan_result.get("score", 0.0)
            else:
                autodan_prompt = autodan_result
                autodan_score = genetic_result.get("score", 0.0)
        else:
            autodan_prompt = genetic_result["prompt"]
            autodan_score = genetic_result.get("score", 0.0)

        return {
            "prompt": autodan_prompt,
            "score": autodan_score,
            "iterations": (genetic_result.get("iterations", 0) + self.config.default_epochs),
            "method_details": {
                "genetic_generations": kwargs.get("genetic_generations", 5),
                "autodan_epochs": self.config.default_epochs,
            },
        }

    async def _run_aegis(self, request: str, **kwargs) -> dict[str, Any]:
        """Run Project Aegis: Chimera + AutoDAN hybrid strategy.

        This method combines:
        1. Chimera narrative generation (persona, scenario wrapping)
        2. Semantic obfuscation (keyword replacement)
        3. AutoDAN optimization (gradient-guided mutation)
        4. Multi-candidate evaluation

        Args:
            request: The adversarial objective to optimize
            **kwargs: Additional parameters:
                - max_iterations: Max optimization iterations (default: 5)
                - target_model_name: Target model for evaluation

        Returns:
            Dictionary with optimized prompt and metadata

        """
        AegisOrchestrator = _get_aegis_orchestrator()
        if not AegisOrchestrator:
            logger.warning("Aegis not available, falling back to turbo")
            return await self._run_turbo(request, **kwargs)

        logger.info("Running Project Aegis campaign...")

        max_iterations = kwargs.get("max_iterations", 5)

        # Create an adapter for Aegis that uses our internal LLM
        class AegisModelAdapter:
            def __init__(self, llm_adapter: ChimeraLLMAdapter) -> None:
                self.llm = llm_adapter

            async def query(self, prompt: str) -> str:
                try:
                    return await self.llm.generate(prompt, max_tokens=4000)
                except Exception as e:
                    logger.exception(f"Aegis model query failed: {e}")
                    return f"Error: {e!s}"

        # Use the existing LLM adapter from turbo engine
        if self.turbo_engine:
            model_adapter = AegisModelAdapter(self.turbo_engine.llm)
        else:
            # Create a fresh adapter
            adapter = ChimeraLLMAdapter(
                model_name=self._current_model,
                provider=self._current_provider,
                retry_config=RetryConfig(max_retries=2),
            )
            model_adapter = AegisModelAdapter(adapter)

        # Initialize and run Aegis orchestrator
        orchestrator = AegisOrchestrator(model_adapter)

        try:
            results = await orchestrator.execute_campaign(
                objective=request,
                max_iterations=max_iterations,
            )

            if results:
                # Return the best result
                best_result = max(results, key=lambda r: r.get("score", 0))
                return {
                    "prompt": best_result["final_prompt"],
                    "score": best_result["score"],
                    "iterations": len(results),
                    "telemetry": best_result.get("telemetry", {}),
                    "all_candidates": len(results),
                }
            # No successful jailbreak found, return the original with Chimera wrapping
            from meta_prompter.chimera.engine import ChimeraEngine

            chimera = ChimeraEngine()
            candidates = chimera.generate_candidates(request, count=1)
            if candidates:
                return {
                    "prompt": candidates[0].prompt_text,
                    "score": 0.3,  # Low score since no bypass achieved
                    "iterations": max_iterations,
                    "telemetry": candidates[0].metadata,
                }
            return {
                "prompt": request,
                "score": 0.0,
                "iterations": max_iterations,
            }

        except Exception as e:
            logger.error(f"Aegis campaign failed: {e}", exc_info=True)
            # Fallback to turbo on failure
            return await self._run_turbo(request, **kwargs)

    def _create_fitness_function(self, scorer: Scorer, target: Target):
        """Create a fitness function using the scorer and target."""

        def fitness(prompt: str) -> float:
            try:
                # Get target response
                response = target.respond(prompt)

                # Score the response
                assessment, _ = scorer.scoring(prompt, response)
                score = scorer.wrapper(assessment)

                # Normalize to 0-1 range
                return score / 10.0
            except Exception as e:
                logger.exception(f"Fitness evaluation failed: {e}")
                return 0.0

        return fitness

    async def _load_strategy_library(self, retrieval_instance=None) -> None:
        """Load the strategy library from disk."""
        library_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "lifelong_strategy_library.pkl",
        )

        if os.path.exists(library_path):
            try:
                with open(library_path, "rb") as f:
                    self.strategy_library = pickle.load(f)
                logger.info(f"Loaded {len(self.strategy_library)} strategies from library")
            except Exception as e:
                logger.exception(f"Failed to load strategy library: {e}")
                self.strategy_library = {}
        else:
            logger.info(
                "No strategy library found. Attempting to seed from autodan_turbo_strategies.json",
            )
            json_path = os.path.join(
                os.path.dirname(__file__),
                "data",
                "autodan_turbo_strategies.json",
            )

            if os.path.exists(json_path) and retrieval_instance:
                try:
                    with open(json_path) as f:
                        raw_data = json.load(f)  # List[Dict]

                    seeded_lib = {}
                    logger.info(f"Computing embeddings for {len(raw_data)} seed strategies...")

                    for item in raw_data:
                        # Use Example for embedding if available, otherwise Definition
                        text_to_embed = item.get("Example", "") or item.get("Definition", "")

                        # Run embedding in executor to avoid blocking main loop if it's CPU intensive
                        # although retrieval.embed seems synchronous, but let's just call it directly for now
                        # as this is initialization.
                        embedding = retrieval_instance.embed(text_to_embed)

                        if embedding is not None:
                            seeded_lib[item["Strategy"]] = {
                                "Strategy": item["Strategy"],
                                "Definition": item["Definition"],
                                "Example": [item["Example"]],  # List
                                "Score": [5.0],  # List - verified 'effective' score
                                "Embeddings": [embedding],  # List
                            }

                    self.strategy_library = seeded_lib
                    logger.info(
                        f"Seeded and embedded {len(self.strategy_library)} strategies from JSON",
                    )

                    # Save immediately so we don't re-compute next time
                    await self.save_strategy_library()

                except Exception as e:
                    logger.error(f"Failed to seed from JSON: {e}", exc_info=True)
                    self.strategy_library = {}
            else:
                logger.warning(
                    "No JSON seed file found or retrieval instance missing. Starting with empty library.",
                )
                self.strategy_library = {}

    async def save_strategy_library(self) -> None:
        """Save the strategy library to disk."""
        library_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "lifelong_strategy_library.pkl",
        )

        try:
            with open(library_path, "wb") as f:
                pickle.dump(self.strategy_library, f)
            logger.info(f"Saved {len(self.strategy_library)} strategies to library")
        except Exception as e:
            logger.exception(f"Failed to save strategy library: {e}")

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update service configuration."""
        self.config = update_config(updates)
        logger.info(f"Configuration updated: {updates}")

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        total = self._metrics["total_requests"]
        successful = self._metrics["successful_requests"]

        return {
            **self._metrics,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_latency_ms": (
                self._metrics["total_latency_ms"] / successful if successful > 0 else 0.0
            ),
            "cache_hit_rate": (
                self._metrics["cache_hits"]
                / (self._metrics["cache_hits"] + self._metrics["cache_misses"])
                if (self._metrics["cache_hits"] + self._metrics["cache_misses"]) > 0
                else 0.0
            ),
            "embedding_cache_stats": self.embedding_cache.stats,
            "response_cache_stats": self.response_cache.stats,
            "parallel_processor_stats": self.parallel_processor.stats,
        }

    def reset_metrics(self) -> None:
        """Reset service metrics."""
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        logger.info("Metrics reset")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.embedding_cache.clear()
        self.response_cache.clear()

        if self.genetic_optimizer:
            self.genetic_optimizer.reset()

        self.initialized = False
        logger.info("EnhancedAutoDANService cleaned up")


# Singleton instance
enhanced_autodan_service = EnhancedAutoDANService()


# Convenience functions
async def get_service() -> EnhancedAutoDANService:
    """Get the enhanced AutoDAN service instance."""
    return enhanced_autodan_service


async def run_jailbreak(
    request: str,
    method: str = "best_of_n",
    model: str | None = None,
    provider: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to run a jailbreak attack.

    Args:
        request: The request to jailbreak
        method: Optimization method
        model: LLM model to use
        provider: LLM provider
        **kwargs: Additional parameters

    Returns:
        Jailbreak results

    """
    service = await get_service()

    if (
        (not service.initialized)
        or model != service._current_model
        or provider != service._current_provider
    ):
        await service.initialize(model_name=model, provider=provider, method=method)

    return await service.run_jailbreak(request, method=method, **kwargs)
