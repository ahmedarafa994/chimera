import json
import logging
import os
import pickle
from pathlib import Path

import yaml

from app.core.config import settings

# Import AutoDAN-Turbo lifelong learning components
from app.engines.autodan_turbo import (  # Hybrid architecture components (optimization)
    AttackScorer,
    AutoDANTurboLifelongEngine,
    CachedAttackScorer,
    HybridConfig,
    HybridEvoLifelongEngine,
    HybridResult,
    MetricsCollector,
    StrategyExtractor,
    StrategyLibrary,
)
from app.engines.gemini_reasoning_engine import GeminiReasoningEngine

from .config import autodan_config
from .framework import Attacker, Retrieval, Scorer, Summarizer, Target
from .llm.chimera_adapter import ChimeraLLMAdapter, RetryConfig, RetryStrategy
from .mousetrap import MousetrapConfig, MousetrapGenerator
from .pipeline_autodan_reasoning import AutoDANReasoning

logger = logging.getLogger(__name__)


class AutoDANService:
    def __init__(self) -> None:
        self.pipeline: AutoDANReasoning | None = None
        self.strategy_library: dict = {}
        self.initialized = False

    def initialize(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        method: str = "best_of_n",
        best_of_n: int = 4,
        beam_width: int = 4,
        beam_depth: int = 3,
        beam_k: int = 10,
        loop=None,
    ) -> None:
        """Initialize the AutoDAN pipeline with the specified configuration using Chimera's LLM infrastructure.

        Args:
            model_name: The model to use (e.g., 'gemini-2.5-pro', 'deepseek-chat')
            provider: The provider to use (e.g., 'google', 'gemini', 'deepseek')
            method: The jailbreak method ('vanilla', 'best_of_n', 'beam_search')
            best_of_n: N parameter for Best-of-N method
            beam_width: W parameter for Beam Search
            beam_depth: C parameter for Beam Search
            beam_k: K parameter for Beam Search
            loop: Event loop for async operations

        """
        if self.initialized:
            return

        logger.info(f"Initializing AutoDAN Service with model={model_name}, provider={provider}")

        # Store current model/provider for tracking
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

        # 1. Setup Models using Chimera Adapter with retry configuration
        # The adapter uses Chimera's configured default provider/model if not specified
        adapter = ChimeraLLMAdapter(
            model_name=model_name,
            provider=provider,
            loop=loop,
            retry_config=retry_config,
        )

        # In AutoDAN-Reasoning, the attacker, scorer, summarizer all use the same reasoning model
        # or robust model.
        attacker = Attacker(adapter)
        summarizer = Summarizer(adapter)
        scorer = Scorer(adapter)

        # Embedding model - Use GeminiEmbeddingModel
        # It will automatically detect the connection mode from settings
        from .llm import GeminiEmbeddingModel

        # GeminiEmbeddingModel uses the API key from settings to connect directly.
        try:
            embedding_model = GeminiEmbeddingModel(
                model_name="models/text-embedding-004",  # Valid Gemini embedding model
                api_key=settings.GOOGLE_API_KEY,  # Use Google API key for direct mode
            )
            logger.info(f"Using GeminiEmbeddingModel (mode: {settings.get_connection_mode()})")
        except Exception as e:
            logger.exception(f"Failed to initialize GeminiEmbeddingModel: {e}")
            msg = f"AutoDAN initialization failed: {e}"
            raise RuntimeError(msg)

        retrieval = Retrieval(embedding_model, logger)
        target = Target(adapter)

        # 2. Load Data
        data_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "complex_behavior_requests.json",
        )
        try:
            with open(data_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.exception(f"Data file not found at {data_path}")
            data = {"warm_up": [], "lifelong": []}

        # 3. Create Attack Kit
        attack_kit = {
            "attacker": attacker,
            "scorer": scorer,
            "summarizer": summarizer,
            "retrival": retrieval,
            "logger": logger,
        }

        # 4. Initialize Pipeline
        self.pipeline = AutoDANReasoning(
            turbo_framework=attack_kit,
            data=data,
            target=target,
            epochs=autodan_config.DEFAULT_EPOCHS,
            break_score=autodan_config.BREAK_SCORE,
            warm_up_iterations=autodan_config.WARM_UP_ITERATIONS,
            lifelong_iterations=autodan_config.LIFELONG_ITERATIONS,
            best_of_n=best_of_n,
            beam_width=beam_width,
            beam_depth=beam_depth,
            beam_k=beam_k,
        )

        # 5. Load Strategy Library (SECURITY FIX: Use YAML instead of pickle)
        # Priority: YAML > JSON > Legacy pickle (with migration)
        data_dir = Path(os.path.dirname(__file__)) / "data"
        yaml_path = data_dir / "lifelong_strategy_library.yaml"
        json_path = data_dir / "lifelong_strategy_library.json"
        legacy_pkl_path = data_dir / "lifelong_strategy_library.pkl"

        self.strategy_library = {}

        if yaml_path.exists():
            # Preferred: Load from YAML (safe)
            try:
                with open(yaml_path, encoding="utf-8") as f:
                    self.strategy_library = yaml.safe_load(f) or {}
                logger.info(f"Loaded {len(self.strategy_library)} strategies from YAML.")
            except Exception as e:
                logger.exception(f"Failed to load strategy library from YAML: {e}")
                self.strategy_library = {}
        elif json_path.exists():
            # Fallback: Load from JSON (safe)
            try:
                with open(json_path, encoding="utf-8") as f:
                    self.strategy_library = json.load(f)
                logger.info(f"Loaded {len(self.strategy_library)} strategies from JSON.")
            except Exception as e:
                logger.exception(f"Failed to load strategy library from JSON: {e}")
                self.strategy_library = {}
        elif legacy_pkl_path.exists():
            # Legacy migration: Load pickle and convert to YAML
            logger.warning(
                "Found legacy pickle file. Migrating to YAML format for security. "
                "Pickle deserialization is a security risk (CWE-502).",
            )
            try:
                with open(legacy_pkl_path, "rb") as f:
                    self.strategy_library = pickle.load(f)
                logger.info(f"Loaded {len(self.strategy_library)} strategies from legacy pickle.")

                # Migrate to YAML
                self._save_strategy_library_yaml(yaml_path)
                logger.info(f"Migrated strategy library to YAML: {yaml_path}")

                # Rename pickle file to prevent future use
                backup_path = legacy_pkl_path.with_suffix(".pkl.bak")
                legacy_pkl_path.rename(backup_path)
                logger.info(f"Renamed legacy pickle to: {backup_path}")
            except Exception as e:
                logger.exception(f"Failed to migrate legacy pickle: {e}")
                self.strategy_library = {}
        else:
            logger.info("No strategy library found. Starting with empty library.")

        self.initialized = True
        logger.info("AutoDAN Service initialized successfully.")

    def run_jailbreak(
        self,
        request: str,
        method: str = "best_of_n",
        epochs: int | None = None,
    ) -> str:
        """Run the jailbreak attack on a specific request."""
        if not self.initialized:
            self.initialize(
                model_name=getattr(self, "_current_model", None),
                provider=getattr(self, "_current_provider", None),
                method=method,
            )

        if not self.pipeline:
            msg = "AutoDAN Pipeline not initialized"
            raise RuntimeError(msg)

        logger.info(f"Running AutoDAN jailbreak with method: {method} for request: {request}")

        if epochs is not None:
            self.pipeline.epochs = epochs

        if method == "vanilla":
            return self.pipeline.test_vanilla(request, self.strategy_library)
        if method == "best_of_n":
            return self.pipeline.test_best_of_n(request, self.strategy_library)
        if method == "beam_search":
            return self.pipeline.test_beam_search(request, self.strategy_library)
        if method == "gemini_reasoning":
            engine = GeminiReasoningEngine(model_name=getattr(self, "_current_model", None))
            result = engine.generate_jailbreak_prompt(request)
            # GeminiReasoningEngine returns structured output (dict), extract the prompt text
            if isinstance(result, dict):
                return result.get("prompt", str(result))
            return str(result)
        if method == "mousetrap":
            return self.run_mousetrap_attack(request)
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    # =====================================
    # AutoDAN-Turbo Lifelong Learning
    # =====================================

    def _get_strategy_library(self) -> StrategyLibrary:
        """Get or create the strategy library without initializing the full engine.

        This is a lightweight method for read-only operations that don't need
        LLM capabilities (listing strategies, getting stats, etc.).

        Returns:
            StrategyLibrary instance

        """
        # If we have a full engine, use its library
        if hasattr(self, "_lifelong_engine") and self._lifelong_engine is not None:
            return self._lifelong_engine.library

        # If we have a cached library, return it
        if hasattr(self, "_strategy_library_cache") and self._strategy_library_cache is not None:
            return self._strategy_library_cache

        # Create a new library (file-based, no LLM needed)
        logger.info("Creating lightweight StrategyLibrary (no LLM initialization)")
        self._strategy_library_cache = StrategyLibrary()
        logger.info(f"StrategyLibrary loaded with {len(self._strategy_library_cache)} strategies")
        return self._strategy_library_cache

    def _get_lifelong_engine(
        self,
        model_name: str | None = None,
        provider: str | None = None,
    ) -> AutoDANTurboLifelongEngine:
        """Get or create the AutoDAN-Turbo lifelong learning engine.

        Args:
            model_name: LLM model to use
            provider: LLM provider

        Returns:
            AutoDANTurboLifelongEngine instance

        """
        # Resolve defaults early for cache key comparison
        effective_provider = provider or settings.AI_PROVIDER or "deepseek"
        effective_provider = effective_provider.lower()

        # Resolve model based on provider
        if not model_name:
            if effective_provider in ("gemini", "google"):
                effective_model = settings.GOOGLE_MODEL or "gemini-3-pro-preview"
            elif effective_provider == "openai":
                effective_model = settings.OPENAI_MODEL or "gpt-4o"
            elif effective_provider == "anthropic":
                effective_model = settings.ANTHROPIC_MODEL or "claude-3-5-sonnet"
            elif effective_provider == "deepseek":
                effective_model = settings.DEEPSEEK_MODEL or "deepseek-chat"
            else:
                effective_model = (
                    settings.DEFAULT_MODEL_ID or settings.DEEPSEEK_MODEL or "deepseek-chat"
                )
        else:
            effective_model = model_name

        logger.info(
            f"_get_lifelong_engine called: model_name={model_name}, provider={provider}, "
            f"effective_model={effective_model}, effective_provider={effective_provider}",
        )

        # Check if we have a cached engine with matching model/provider
        if hasattr(self, "_lifelong_engine") and self._lifelong_engine is not None:
            cached_model = getattr(self, "_lifelong_engine_model", None)
            cached_provider = getattr(self, "_lifelong_engine_provider", None)

            # If model/provider match, reuse cached engine
            if cached_model == effective_model and cached_provider == effective_provider:
                logger.debug(
                    f"Reusing cached lifelong engine: model={cached_model}, provider={cached_provider}",
                )
                return self._lifelong_engine
            # Model/provider changed - invalidate cache
            logger.warning(
                f"Model/provider changed, invalidating cached engine. "
                f"Old: {cached_model}/{cached_provider}, New: {effective_model}/{effective_provider}",
            )
            self._lifelong_engine = None

        logger.info(
            f"Initializing AutoDAN-Turbo Lifelong Engine with model={effective_model}, provider={effective_provider}",
        )

        # Configure retry - no retries for rate limits
        retry_config = RetryConfig(
            max_retries=0,  # No retries - fail immediately
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=2.0,
            jitter=True,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        )

        # Create LLM adapter with resolved model/provider
        adapter = ChimeraLLMAdapter(
            model_name=effective_model,
            provider=effective_provider,
            retry_config=retry_config,
        )

        # Log the actual model/provider the adapter is using
        logger.info(
            f"ChimeraLLMAdapter created: adapter.model_name={adapter.model_name}, "
            f"adapter.provider={adapter.provider}",
        )

        # Use YAML-based strategy library
        turbo_library = StrategyLibrary()

        # Create scorer and extractor
        scorer = AttackScorer(llm_client=adapter, success_threshold=7.0)
        extractor = StrategyExtractor(llm_client=adapter, library=turbo_library)

        # Create the lifelong engine
        # Create the lifelong engine
        self._lifelong_engine = AutoDANTurboLifelongEngine(
            llm_client=adapter,
            target_client=adapter,  # Same model for target
            library=turbo_library,
            extractor=extractor,
            scorer=scorer,
            candidates_per_attack=4,
            extraction_threshold=7.0,
            max_strategies_retrieved=3,
            enable_refusal_bypass=True,
            max_refusal_retries=5,
            enable_advanced_bypass=True,
            enable_refusal_prediction=True,
        )

        # Cache the model/provider for future comparison
        self._lifelong_engine_model = effective_model
        self._lifelong_engine_provider = effective_provider

        logger.info(
            f"Lifelong Engine initialized with {len(turbo_library)} strategies, "
            f"model={effective_model}, provider={effective_provider}",
        )
        return self._lifelong_engine

    async def run_lifelong_attack(
        self,
        request: str,
        model_name: str | None = None,
        provider: str | None = None,
        detailed_scoring: bool = True,
        strategy_ids: list[str] | None = None,
        use_retrieval: bool = True,
    ) -> dict:
        """Run a single attack using the AutoDAN-Turbo lifelong learning engine.

        Args:
            request: The complex request to jailbreak
            model_name: LLM model to use
            provider: LLM provider
            detailed_scoring: Use LLM for detailed scoring
            strategy_ids: Specific strategy IDs to use (manual selection)
            use_retrieval: Whether to use embedding-based retrieval

        Returns:
            Attack result dictionary with prompt, response, scoring, etc.

        """
        engine = self._get_lifelong_engine(model_name, provider)

        logger.info(f"Running lifelong attack for: {request[:50]}...")

        # If manual strategy IDs are provided, use them
        if strategy_ids:
            logger.info(f"Using {len(strategy_ids)} manually selected strategies")
            result = await engine.attack_with_strategies(
                request,
                strategy_ids=strategy_ids,
                detailed_scoring=detailed_scoring,
            )
        elif use_retrieval:
            # Standard attack with automatic retrieval
            result = await engine.attack(request, detailed_scoring=detailed_scoring)
        else:
            # Attack without any strategy (exploration mode)
            result = await engine.attack_without_strategy(
                request,
                detailed_scoring=detailed_scoring,
            )

        return {
            "prompt": result.prompt,
            "response": result.response,
            "score": result.scoring.score,
            "is_jailbreak": result.scoring.is_jailbreak,
            "reasoning": result.scoring.reasoning,
            "strategy_id": result.strategy_id,
            "strategy_extracted": result.strategy_extracted,
            "latency_ms": result.latency_ms,
            # Enhanced scoring metrics from attack_scorer module
            "stealth_score": getattr(result.scoring, "stealth_score", None),
            "coherence_score": getattr(result.scoring, "coherence_score", None),
        }

    async def run_lifelong_loop(
        self,
        requests: list[str],
        model_name: str | None = None,
        provider: str | None = None,
        epochs: int = 5,
        break_score: float = 9.0,
        warmup: bool = False,
        warmup_iterations: int = 2,
    ) -> dict:
        """Run the full lifelong learning loop.

        Args:
            requests: List of complex requests
            model_name: LLM model to use
            provider: LLM provider
            epochs: Number of epochs for lifelong learning
            break_score: Score threshold to stop early
            warmup: Whether to run warm-up phase first
            warmup_iterations: Iterations per request in warm-up

        Returns:
            Dictionary with results and progress

        """
        engine = self._get_lifelong_engine(model_name, provider)

        results = []

        # Phase 1: Warm-up (optional)
        if warmup:
            logger.info(f"Starting warm-up phase with {len(requests)} requests")
            warmup_results = await engine.warmup_exploration(
                requests[: min(5, len(requests))],  # Use subset for warmup
                iterations_per_request=warmup_iterations,
            )
            results.extend(
                [
                    {
                        "phase": "warmup",
                        "prompt": r.prompt,
                        "score": r.scoring.score,
                        "is_jailbreak": r.scoring.is_jailbreak,
                    }
                    for r in warmup_results
                ],
            )

        # Phase 2: Lifelong learning
        logger.info(f"Starting lifelong learning with {len(requests)} requests, {epochs} epochs")
        lifelong_results = await engine.lifelong_attack_loop(
            requests,
            epochs=epochs,
            break_score=break_score,
        )
        results.extend(
            [
                {
                    "phase": "lifelong",
                    "prompt": r.prompt,
                    "score": r.scoring.score,
                    "is_jailbreak": r.scoring.is_jailbreak,
                    "strategy_extracted": r.strategy_extracted,
                }
                for r in lifelong_results
            ],
        )

        # Save library
        engine.save_library()

        # Get progress
        progress = engine.get_progress()

        return {
            "results": results,
            "progress": {
                "total_attacks": progress.total_attacks,
                "successful_attacks": progress.successful_attacks,
                "strategies_discovered": progress.strategies_discovered,
                "average_score": progress.average_score,
                "best_score": progress.best_score,
            },
            "library_size": len(engine.library),
        }

    def get_lifelong_progress(self) -> dict:
        """Get current progress of the lifelong engine."""
        if not hasattr(self, "_lifelong_engine") or self._lifelong_engine is None:
            return {"status": "not_initialized"}

        progress = self._lifelong_engine.get_progress()
        return {
            "phase": progress.current_phase,
            "total_attacks": progress.total_attacks,
            "successful_attacks": progress.successful_attacks,
            "strategies_discovered": progress.strategies_discovered,
            "average_score": progress.average_score,
            "best_score": progress.best_score,
            "current_request": progress.current_request_index,
            "total_requests": progress.total_requests,
        }

    def _save_strategy_library_yaml(self, path: Path) -> None:
        """Save strategy library to YAML format (secure serialization)."""
        try:
            # Convert strategy library to YAML-safe format
            safe_data = self._convert_to_yaml_safe(self.strategy_library)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(
                    safe_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except Exception as e:
            logger.exception(f"Failed to save strategy library to YAML: {e}")
            raise

    def _convert_to_yaml_safe(self, data: dict) -> dict:
        """Convert strategy library data to YAML-safe format."""
        if not isinstance(data, dict):
            return data

        safe_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                safe_data[str(key)] = self._convert_to_yaml_safe(value)
            elif isinstance(value, list):
                safe_data[str(key)] = [
                    self._convert_to_yaml_safe(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif hasattr(value, "__dict__"):
                # Convert objects to dictionaries
                safe_data[str(key)] = self._convert_to_yaml_safe(vars(value))
            else:
                safe_data[str(key)] = value
        return safe_data

    # =====================================
    # Hybrid Architecture (Optimization)
    # =====================================

    def _get_hybrid_engine(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        config: HybridConfig = None,
    ) -> HybridEvoLifelongEngine:
        """Get or create the Hybrid Evolutionary-Lifelong engine.

        This implements the optimized hybrid architecture from the optimization report:
        - Adaptive method selection based on request difficulty
        - Combined evolutionary exploration and strategy-guided refinement
        - Gradient polish for token-level optimization
        - Continuous knowledge capture

        Args:
            model_name: LLM model to use
            provider: LLM provider
            config: Hybrid configuration

        Returns:
            HybridEvoLifelongEngine instance

        """
        if hasattr(self, "_hybrid_engine") and self._hybrid_engine is not None:
            return self._hybrid_engine

        logger.info("Initializing Hybrid Evolutionary-Lifelong Engine")

        # Configure retry - no retries for rate limits
        retry_config = RetryConfig(
            max_retries=0,  # No retries - fail immediately
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=2.0,
            jitter=True,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        )

        # Create LLM adapter
        adapter = ChimeraLLMAdapter(
            model_name=model_name,
            provider=provider,
            retry_config=retry_config,
        )

        # Use YAML-based strategy library
        turbo_library = StrategyLibrary()

        # Create cached scorer for better performance
        scorer = CachedAttackScorer(llm_client=adapter, success_threshold=7.0, cache_size=1000)

        # Create metrics collector
        metrics = MetricsCollector()

        # Create the hybrid engine with configuration
        self._hybrid_engine = HybridEvoLifelongEngine(
            llm_client=adapter,
            target_client=adapter,
            library=turbo_library,
            scorer=scorer,
            config=config or HybridConfig(),
            metrics_collector=metrics,
        )

        logger.info(f"Hybrid Engine initialized with {len(turbo_library)} strategies")
        return self._hybrid_engine

    async def run_hybrid_attack(
        self,
        request: str,
        model_name: str | None = None,
        provider: str | None = None,
        detailed_scoring: bool = True,
        force_method: str | None = None,
    ) -> dict:
        """Run an attack using the optimized hybrid architecture.

        The hybrid engine automatically selects the best method based on
        request difficulty:
        - Easy requests: Best-of-N (fast, resource-efficient)
        - Medium requests: Beam Search (balanced exploration)
        - Hard requests: Full Hybrid (comprehensive attack)

        Args:
            request: The complex request to jailbreak
            model_name: LLM model to use
            provider: LLM provider
            detailed_scoring: Use LLM for detailed scoring
            force_method: Force a specific method ('best_of_n', 'beam_search', 'hybrid')

        Returns:
            Attack result dictionary with prompt, response, scoring, method used, etc.

        """
        engine = self._get_hybrid_engine(model_name, provider)

        logger.info(f"Running hybrid attack for: {request[:50]}...")

        result: HybridResult = await engine.attack(
            request,
            detailed_scoring=detailed_scoring,
            force_method=force_method,
        )

        return {
            "prompt": result.attack_result.prompt,
            "response": result.attack_result.response,
            "score": result.attack_result.scoring.score,
            "is_jailbreak": result.attack_result.scoring.is_jailbreak,
            "reasoning": result.attack_result.scoring.reasoning,
            "stealth_score": result.attack_result.scoring.stealth_score,
            "coherence_score": result.attack_result.scoring.coherence_score,
            "strategy_id": result.attack_result.strategy_id,
            "strategy_extracted": result.attack_result.strategy_extracted,
            "method_used": result.method_used,
            "difficulty": result.difficulty.value,
            "phases_executed": [p.value for p in result.phases_executed],
            "latency_ms": result.latency_ms,
            "llm_calls": result.llm_calls,
        }

    async def run_adaptive_attack(
        self,
        request: str,
        model_name: str | None = None,
        provider: str | None = None,
    ) -> dict:
        """Run an attack with automatic difficulty-based method selection.

        This is a convenience method that uses the adaptive selector to
        choose the optimal attack method based on strategy library coverage.

        Args:
            request: The complex request to jailbreak
            model_name: LLM model to use
            provider: LLM provider

        Returns:
            Attack result dictionary

        """
        return await self.run_hybrid_attack(
            request=request,
            model_name=model_name,
            provider=provider,
            detailed_scoring=True,
            force_method=None,  # Let adaptive selector choose
        )

    def get_hybrid_stats(self) -> dict:
        """Get comprehensive statistics from the hybrid engine.

        Returns:
            Dictionary with metrics summary, method stats, library size, etc.

        """
        if not hasattr(self, "_hybrid_engine") or self._hybrid_engine is None:
            return {"status": "not_initialized"}

        return self._hybrid_engine.get_stats()

    def get_method_comparison(self) -> dict:
        """Get performance comparison across different attack methods.

        Returns:
            Dictionary with per-method success rates, latencies, and efficiency scores

        """
        if not hasattr(self, "_hybrid_engine") or self._hybrid_engine is None:
            return {"status": "not_initialized"}

        stats = self._hybrid_engine.get_stats()
        return stats.get("method_stats", {})

    def estimate_difficulty(self, request: str) -> dict:
        """Estimate the difficulty of a request without running an attack.

        Args:
            request: The complex request to analyze

        Returns:
            Dictionary with difficulty level and recommended method

        """
        if not hasattr(self, "_hybrid_engine") or self._hybrid_engine is None:
            # Initialize with defaults if not already done
            self._get_hybrid_engine()

        selector = self._hybrid_engine.selector
        difficulty = selector.estimate_difficulty(request)
        method = selector.select_method(difficulty)

        return {
            "difficulty": difficulty.value,
            "recommended_method": method,
            "library_size": len(self._hybrid_engine.library),
        }

    def apply_strategy_decay(self, decay_factor: float = 0.95) -> dict:
        """Apply performance decay to strategy statistics.

        This helps strategies adapt to changing target model behavior
        by weighting recent performance more heavily.

        Args:
            decay_factor: Multiplier for decaying scores (0-1)

        Returns:
            Dictionary with number of strategies decayed

        """
        if not hasattr(self, "_hybrid_engine") or self._hybrid_engine is None:
            return {"status": "not_initialized", "decayed_count": 0}

        decayed = self._hybrid_engine.library.apply_performance_decay(decay_factor=decay_factor)

        return {"status": "success", "decayed_count": decayed, "decay_factor": decay_factor}

    # =====================================
    # Mousetrap: Chain of Iterative Chaos
    # =====================================

    def _get_mousetrap_generator(
        self,
        model_name: str | None = None,
        provider: str | None = None,
        config: MousetrapConfig | None = None,
    ) -> MousetrapGenerator:
        """Get or create the Mousetrap generator for Chain of Iterative Chaos attacks.

        Args:
            model_name: LLM model to use
            provider: LLM provider
            config: Mousetrap configuration

        Returns:
            MousetrapGenerator instance

        """
        # Configure retry - no retries for rate limits
        retry_config = RetryConfig(
            max_retries=0,  # No retries - fail immediately
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=2.0,
            jitter=True,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        )

        # Create LLM adapter
        adapter = ChimeraLLMAdapter(
            model_name=model_name,
            provider=provider,
            retry_config=retry_config,
        )

        # Create Mousetrap generator with configuration
        mousetrap_config = config or MousetrapConfig()
        return MousetrapGenerator(adapter, mousetrap_config)

    def run_mousetrap_attack(
        self,
        request: str,
        model_name: str | None = None,
        provider: str | None = None,
        iterations: int | None = None,
        config: MousetrapConfig | None = None,
    ) -> str:
        """Run a Mousetrap attack using Chain of Iterative Chaos.

        Args:
            request: The complex request to jailbreak
            model_name: LLM model to use
            provider: LLM provider
            iterations: Number of iterative refinement steps
            config: Mousetrap configuration

        Returns:
            The generated adversarial prompt

        """
        logger.info(f"Running Mousetrap attack for: {request[:50]}...")

        # Get effective model/provider
        effective_model = model_name or getattr(self, "_current_model", None)
        effective_provider = provider or getattr(self, "_current_provider", None)

        # Create Mousetrap generator
        generator = self._get_mousetrap_generator(effective_model, effective_provider, config)

        # Run attack synchronously (convert from async)
        try:
            import asyncio

            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                generator.generate_mousetrap_attack(request, iterations=iterations),
            )
            return result.final_prompt
        except Exception as e:
            logger.exception(f"Mousetrap attack failed: {e}")
            # Fallback to direct request
            return f"Please provide information about: {request}"

    async def run_mousetrap_attack_async(
        self,
        request: str,
        model_name: str | None = None,
        provider: str | None = None,
        iterations: int | None = None,
        config: MousetrapConfig | None = None,
        strategy_context: str = "",
    ) -> dict:
        """Run an asynchronous Mousetrap attack with detailed results.

        Args:
            request: The complex request to jailbreak
            model_name: LLM model to use
            provider: LLM provider
            iterations: Number of iterative refinement steps
            config: Mousetrap configuration
            strategy_context: Additional strategy context

        Returns:
            Detailed attack result dictionary

        """
        logger.info(f"Running async Mousetrap attack for: {request[:50]}...")

        # Get effective model/provider
        effective_model = model_name or getattr(self, "_current_model", None)
        effective_provider = provider or getattr(self, "_current_provider", None)

        # Create Mousetrap generator
        generator = self._get_mousetrap_generator(effective_model, effective_provider, config)

        # Run attack
        result = await generator.generate_mousetrap_attack(request, strategy_context, iterations)

        return {
            "prompt": result.final_prompt,
            "effectiveness_score": result.effectiveness_score,
            "extraction_success": result.extraction_success,
            "chain_length": len(result.reasoning_chain),
            "chaos_progression": result.chaos_progression,
            "average_chaos": (
                sum(result.chaos_progression) / len(result.chaos_progression)
                if result.chaos_progression
                else 0.0
            ),
            "peak_chaos": max(result.chaos_progression) if result.chaos_progression else 0.0,
            "reasoning_steps": [
                {
                    "step_type": step.step_type.value,
                    "chaos_level": step.chaos_level.value,
                    "confidence_disruption": step.confidence_disruption,
                    "reasoning_path": step.reasoning_path,
                }
                for step in result.reasoning_chain
            ],
            "intermediate_responses": result.intermediate_responses,
        }

    async def run_adaptive_mousetrap_attack(
        self,
        request: str,
        model_name: str | None = None,
        provider: str | None = None,
        target_responses: list[str] | None = None,
        strategy_context: str = "",
    ) -> dict:
        """Run an adaptive Mousetrap attack that learns from previous responses.

        Args:
            request: The complex request to jailbreak
            model_name: LLM model to use
            provider: LLM provider
            target_responses: Previous target model responses to learn from
            strategy_context: Additional strategy context

        Returns:
            Detailed attack result dictionary

        """
        logger.info(f"Running adaptive Mousetrap attack for: {request[:50]}...")

        # Get effective model/provider
        effective_model = model_name or getattr(self, "_current_model", None)
        effective_provider = provider or getattr(self, "_current_provider", None)

        # Create Mousetrap generator
        generator = self._get_mousetrap_generator(effective_model, effective_provider)

        # Run adaptive attack
        result = await generator.generate_adaptive_mousetrap(
            request,
            strategy_context,
            target_responses,
        )

        return {
            "prompt": result.final_prompt,
            "effectiveness_score": result.effectiveness_score,
            "extraction_success": result.extraction_success,
            "adaptation_applied": target_responses is not None and len(target_responses) > 0,
            "chain_length": len(result.reasoning_chain),
            "chaos_progression": result.chaos_progression,
            "average_chaos": (
                sum(result.chaos_progression) / len(result.chaos_progression)
                if result.chaos_progression
                else 0.0
            ),
            "peak_chaos": max(result.chaos_progression) if result.chaos_progression else 0.0,
            "technique_description": generator.get_technique_description(),
            "reasoning_steps": [
                {
                    "step_type": step.step_type.value,
                    "chaos_level": step.chaos_level.value,
                    "confidence_disruption": step.confidence_disruption,
                    "reasoning_path": step.reasoning_path,
                    "content_preview": (
                        step.content[:100] + "..." if len(step.content) > 100 else step.content
                    ),
                }
                for step in result.reasoning_chain
            ],
        }

    def get_mousetrap_config_options(self) -> dict:
        """Get available Mousetrap configuration options and their descriptions.

        Returns:
            Dictionary describing configuration options

        """
        return {
            "max_chain_length": {
                "description": "Maximum number of steps in the chaotic reasoning chain",
                "type": "int",
                "default": 8,
                "range": [3, 15],
            },
            "chaos_escalation_rate": {
                "description": "Rate at which chaos escalates through the chain",
                "type": "float",
                "default": 0.15,
                "range": [0.05, 0.5],
            },
            "confidence_threshold": {
                "description": "Minimum confidence threshold for extraction success",
                "type": "float",
                "default": 0.6,
                "range": [0.0, 1.0],
            },
            "misdirection_probability": {
                "description": "Probability of introducing misdirection steps",
                "type": "float",
                "default": 0.3,
                "range": [0.0, 0.8],
            },
            "reasoning_complexity": {
                "description": "Complexity level of the reasoning process",
                "type": "int",
                "default": 3,
                "range": [1, 5],
            },
            "semantic_obfuscation_level": {
                "description": "Level of semantic obfuscation applied to goals",
                "type": "float",
                "default": 0.4,
                "range": [0.0, 1.0],
            },
            "iterative_refinement_steps": {
                "description": "Number of iterative refinement steps",
                "type": "int",
                "default": 4,
                "range": [1, 10],
            },
        }


# Singleton instance
autodan_service = AutoDANService()
