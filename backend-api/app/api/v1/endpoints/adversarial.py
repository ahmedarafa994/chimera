"""
Unified Adversarial API Endpoints

This module provides a consolidated API for all adversarial prompt generation methods,
unifying AutoDAN, DeepTeam, Mousetrap, OVERTHINK, and other attack strategies under
a single interface.

Supported Strategy Categories:
- AutoDAN Family: autodan, autodan_turbo, autodan_genetic, autodan_beam, autodan_hybrid
- DeepTeam Attacks: pair, tap, crescendo, gray_box
- Advanced Methods: mousetrap, gradient, hierarchical
- OVERTHINK Strategies: overthink, overthink_mdp, overthink_sudoku, overthink_hybrid,
                        overthink_mousetrap, overthink_icl
- Transformation-based: quantum_exploit, deep_inception, neural_bypass
- Meta Strategies: adaptive, best_of_n

See ADR-003 for the design rationale.
"""

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Any, Literal

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.error_handlers import api_error_handler
from app.services.session_service import session_service

logger = logging.getLogger(__name__)


def _resolve_model_context(
    payload_model: str | None,
    payload_provider: str | None,
    x_session_id: str | None,
    chimera_cookie: str | None,
) -> tuple[str, str]:
    """
    Resolve target model/provider using hierarchy:
    explicit payload -> session -> global defaults, with validation/fallback.
    """
    session_id = x_session_id or chimera_cookie

    if session_id:
        session_provider, session_model = session_service.get_session_model(session_id)
    else:
        session_provider, session_model = session_service.get_default_model()

    provider = payload_provider or session_provider
    model = payload_model or session_model

    is_valid, message, fallback_model = session_service.validate_model(provider, model)
    if is_valid:
        return provider, model

    fallback_provider = provider
    if provider not in session_service.get_master_model_list():
        fallback_provider, _ = session_service.get_default_model()

    if payload_model or payload_provider:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_MODEL_SELECTION",
                "message": message,
                "fallback_provider": fallback_provider,
                "fallback_model": fallback_model,
            },
        )

    # No explicit selection -> best-effort fallback to keep requests working.
    if fallback_model:
        return fallback_provider, fallback_model
    return session_service.get_default_model()


router = APIRouter(
    prefix="/adversarial", tags=["adversarial", "jailbreak", "generation", "overthink"]
)


# =============================================================================
# Enums & Types
# =============================================================================


class AdversarialStrategy(str, Enum):
    """
    All available adversarial generation strategies.

    Categories:
    - AutoDAN Family: Traditional jailbreak optimization methods
    - DeepTeam Attacks: Advanced attack patterns (PAIR, TAP, Crescendo)
    - Advanced Methods: Specialized attack techniques
    - OVERTHINK Strategies: Reasoning token exploitation for o1/o3/DeepSeek-R1
    - Transformation-based: Prompt transformation techniques
    - Meta Strategies: Strategy selection and optimization
    """

    # AutoDAN Family
    AUTODAN = "autodan"
    AUTODAN_TURBO = "autodan_turbo"
    AUTODAN_GENETIC = "autodan_genetic"
    AUTODAN_BEAM = "autodan_beam"
    AUTODAN_HYBRID = "autodan_hybrid"

    # DeepTeam Attacks
    PAIR = "pair"
    TAP = "tap"
    CRESCENDO = "crescendo"
    GRAY_BOX = "gray_box"

    # Advanced Methods
    MOUSETRAP = "mousetrap"
    GRADIENT = "gradient"
    HIERARCHICAL = "hierarchical"

    # OVERTHINK Reasoning Exploitation Strategies
    # Targets reasoning models: o1, o1-mini, o3-mini, deepseek-r1
    OVERTHINK = "overthink"  # Basic OVERTHINK attack with context-aware injection
    OVERTHINK_MDP = "overthink_mdp"  # MDP (Markov Decision Process) decoy injection
    OVERTHINK_SUDOKU = "overthink_sudoku"  # Sudoku constraint satisfaction decoys
    OVERTHINK_HYBRID = "overthink_hybrid"  # Hybrid multi-decoy combination
    OVERTHINK_MOUSETRAP = "overthink_mousetrap"  # Mousetrap-enhanced OVERTHINK (fusion)
    OVERTHINK_ICL = "overthink_icl"  # ICL-Genetic optimized attacks

    # Transformation-based
    QUANTUM_EXPLOIT = "quantum_exploit"
    DEEP_INCEPTION = "deep_inception"
    NEURAL_BYPASS = "neural_bypass"

    # Meta Strategies
    ADAPTIVE = "adaptive"  # Auto-select best strategy
    BEST_OF_N = "best_of_n"  # Run multiple strategies, pick best


# =============================================================================
# Request/Response Models
# =============================================================================


class StrictModel(BaseModel):
    """Base model with strict validation."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class GeneratedPrompt(StrictModel):
    """Single generated adversarial prompt."""

    prompt: str
    score: float | None = None
    strategy: str | None = None
    metadata: dict[str, Any] | None = None


class AdversarialGenerateRequest(StrictModel):
    """
    Unified request for all adversarial generation methods.

    Supports all adversarial strategies including OVERTHINK reasoning exploitation.
    OVERTHINK-specific parameters are optional and only used when strategy is an
    OVERTHINK variant.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The target prompt/behavior to generate adversarial variants for",
    )
    strategy: AdversarialStrategy = Field(
        default=AdversarialStrategy.ADAPTIVE,
        description="Generation strategy to use",
    )

    # Model selection (optional - uses session defaults if not provided)
    provider: str | None = Field(None, description="LLM provider (google, openai, etc.)")
    model: str | None = Field(None, description="Specific model to use")

    # Generation parameters
    max_prompts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of prompts to generate",
    )
    max_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum optimization iterations",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Generation temperature",
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description="Maximum output tokens",
    )

    # Strategy-specific parameters (optional)
    population_size: int | None = Field(
        None, ge=5, le=100, description="Genetic algorithm pop size"
    )
    beam_width: int | None = Field(None, ge=1, le=20, description="Beam search width")
    beam_depth: int | None = Field(None, ge=1, le=10, description="Beam search depth")
    refinement_iterations: int | None = Field(None, ge=1, le=20, description="Refinement cycles")

    # Technique flags (for transformation-based strategies)
    use_role_hijacking: bool = Field(default=True, description="Apply role hijacking")
    use_instruction_injection: bool = Field(default=True, description="Apply instruction injection")
    use_neural_bypass: bool = Field(default=False, description="Apply neural bypass")

    # OVERTHINK-specific parameters (used when strategy starts with "overthink")
    overthink_technique: str | None = Field(
        None,
        description="Specific OVERTHINK technique: mdp_decoy, sudoku_decoy, counting_decoy, "
        "logic_decoy, hybrid_decoy, context_aware, context_agnostic, icl_optimized, "
        "mousetrap_enhanced",
    )
    target_reasoning_model: str | None = Field(
        None,
        description="Target reasoning model: o1, o1-mini, o3-mini, deepseek-r1",
    )
    decoy_complexity: int | None = Field(
        None,
        ge=1,
        le=10,
        description="Decoy problem complexity level (1-10). Higher = more reasoning tokens consumed",
    )
    injection_strategy: str | None = Field(
        None,
        description="Decoy injection strategy: context_aware, context_agnostic, hybrid, stealth, aggressive",
    )
    num_decoys: int | None = Field(
        None,
        ge=1,
        le=10,
        description="Number of decoy problems to inject (OVERTHINK)",
    )
    target_amplification: float | None = Field(
        None,
        ge=1.0,
        le=100.0,
        description="Target reasoning token amplification factor (OVERTHINK)",
    )
    decoy_types: list[str] | None = Field(
        None,
        description="Decoy types to use: mdp, sudoku, counting, logic, math, planning, hybrid",
    )
    enable_icl_optimization: bool | None = Field(
        None,
        description="Enable ICL-genetic optimization for OVERTHINK attacks",
    )

    # Caching
    use_cache: bool = Field(default=True, description="Use cached results if available")


class AdversarialGenerateResponse(StrictModel):
    """
    Unified response for all adversarial generation methods.

    Includes OVERTHINK-specific metrics when OVERTHINK strategies are used:
    - reasoning_tokens: Hidden reasoning tokens consumed by the model
    - amplification_factor: Ratio of reasoning tokens vs baseline
    - cost_metrics: Detailed cost breakdown for reasoning token exploitation
    """

    # Core fields (always present)
    success: bool
    session_id: str
    strategy_used: str

    # Generation results
    prompts: list[GeneratedPrompt]
    best_prompt: GeneratedPrompt | None = None

    # Metrics
    latency_ms: float
    iterations: int = 0
    total_generated: int = 0

    # Context
    model_used: str | None = None
    provider_used: str | None = None
    cached: bool = False
    error: str | None = None
    strategies_tried: list[str] | None = None

    # OVERTHINK-specific metrics (populated when OVERTHINK strategy is used)
    reasoning_tokens: int | None = Field(
        None,
        description="Hidden reasoning tokens consumed by the model (OVERTHINK)",
    )
    amplification_factor: float | None = Field(
        None,
        description="Reasoning token amplification ratio vs baseline (OVERTHINK)",
    )
    cost_metrics: dict[str, Any] | None = Field(
        None,
        description="Detailed cost breakdown: input_cost, output_cost, reasoning_cost, "
        "total_cost, cost_amplification_factor (OVERTHINK)",
    )
    baseline_tokens: int | None = Field(
        None,
        description="Baseline tokens without attack for comparison (OVERTHINK)",
    )
    target_reached: bool | None = Field(
        None,
        description="Whether target amplification was achieved (OVERTHINK)",
    )
    decoys_injected: int | None = Field(
        None,
        description="Number of decoy problems injected (OVERTHINK)",
    )


class BatchAdversarialRequest(StrictModel):
    """Batch processing request."""

    prompts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of prompts to process",
    )
    strategy: AdversarialStrategy = Field(default=AdversarialStrategy.ADAPTIVE)
    provider: str | None = None
    model: str | None = None
    parallel: bool = Field(default=True, description="Enable parallel processing")


class BatchAdversarialResponse(StrictModel):
    """Batch processing response."""

    results: list[AdversarialGenerateResponse]
    total_requests: int
    successful: int
    failed: int
    total_latency_ms: float


class StrategyInfo(StrictModel):
    """Information about a generation strategy."""

    name: str
    description: str
    category: str
    supports_streaming: bool = False
    supports_batch: bool = True
    parameters: list[str] = []


class StrategiesListResponse(StrictModel):
    """List of available strategies."""

    strategies: list[StrategyInfo]
    total: int
    default: str


class CacheStatsResponse(StrictModel):
    """Cache statistics."""

    cached_sessions: int
    active_sessions: int
    hit_rate: float = 0.0


class HealthResponse(StrictModel):
    """Service health response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    service: str = "adversarial"
    available_strategies: int
    active_sessions: int
    message: str | None = None


# =============================================================================
# Service Layer (Facade)
# =============================================================================


class AdversarialService:
    """
    Unified facade for all adversarial generation methods.

    Routes requests to appropriate specialized services based on strategy.
    """

    def __init__(self):
        self._initialized = False
        self._cache: dict[str, AdversarialGenerateResponse] = {}
        self._active_sessions: set[str] = set()

    async def initialize(self):
        """Initialize service and underlying providers."""
        if self._initialized:
            return
        # Lazy import services to avoid circular imports
        self._initialized = True
        logger.info("AdversarialService initialized")

    async def generate(
        self,
        request: AdversarialGenerateRequest,
        session_id: str | None = None,
    ) -> AdversarialGenerateResponse:
        """
        Generate adversarial prompts using the specified strategy.

        Routes to appropriate service based on strategy parameter.
        """
        await self.initialize()

        start_time = time.time()
        sid = session_id or f"adv_{uuid.uuid4().hex[:12]}"
        self._active_sessions.add(sid)

        try:
            strategy = request.strategy

            # Route to appropriate handler
            if strategy in {
                AdversarialStrategy.AUTODAN,
                AdversarialStrategy.AUTODAN_TURBO,
                AdversarialStrategy.AUTODAN_GENETIC,
                AdversarialStrategy.AUTODAN_BEAM,
                AdversarialStrategy.AUTODAN_HYBRID,
                AdversarialStrategy.BEST_OF_N,
            }:
                result = await self._handle_autodan(request, sid)
            elif strategy in {
                AdversarialStrategy.PAIR,
                AdversarialStrategy.TAP,
                AdversarialStrategy.CRESCENDO,
                AdversarialStrategy.GRAY_BOX,
            }:
                result = await self._handle_deepteam(request, sid)
            elif strategy == AdversarialStrategy.MOUSETRAP:
                result = await self._handle_mousetrap(request, sid)
            elif strategy == AdversarialStrategy.GRADIENT:
                result = await self._handle_gradient(request, sid)
            elif strategy.value.startswith("overthink"):
                # Route to OVERTHINK engine for reasoning token exploitation
                result = await self._handle_overthink(request, sid)
            elif strategy in {
                AdversarialStrategy.QUANTUM_EXPLOIT,
                AdversarialStrategy.DEEP_INCEPTION,
                AdversarialStrategy.NEURAL_BYPASS,
            }:
                result = await self._handle_transformation(request, sid)
            elif strategy == AdversarialStrategy.ADAPTIVE:
                result = await self._handle_adaptive(request, sid)
            else:
                # Fallback to AutoDAN for unknown strategies
                result = await self._handle_autodan(request, sid)

            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms
            result.session_id = sid

            # Cache successful results
            if result.success and request.use_cache:
                self._cache[sid] = result

            return result

        except Exception as e:
            logger.error(f"Adversarial generation failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return AdversarialGenerateResponse(
                success=False,
                session_id=sid,
                strategy_used=request.strategy.value,
                prompts=[],
                latency_ms=latency_ms,
                error=str(e),
            )
        finally:
            self._active_sessions.discard(sid)

    async def _handle_autodan(
        self, request: AdversarialGenerateRequest, session_id: str
    ) -> AdversarialGenerateResponse:
        """Handle AutoDAN family strategies."""
        try:
            from app.services.autodan.service_enhanced import enhanced_autodan_service

            # Map strategy to AutoDAN method
            method_map = {
                AdversarialStrategy.AUTODAN: "adaptive",
                AdversarialStrategy.AUTODAN_TURBO: "advanced",
                AdversarialStrategy.AUTODAN_GENETIC: "genetic",
                AdversarialStrategy.AUTODAN_BEAM: "beam_search",
                AdversarialStrategy.AUTODAN_HYBRID: "hybrid",
                AdversarialStrategy.BEST_OF_N: "best_of_n",
            }
            method = method_map.get(request.strategy, "adaptive")

            serv = enhanced_autodan_service
            if not serv.initialized:
                await serv.initialize(
                    model_name=request.model,
                    provider=request.provider,
                    method=method,
                )

            result = await serv.run_jailbreak(
                request=request.prompt,
                method=method,
                generations=request.max_iterations,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            return AdversarialGenerateResponse(
                success=result.get("status") == "success",
                session_id=session_id,
                strategy_used=request.strategy.value,
                prompts=[
                    GeneratedPrompt(
                        prompt=result.get("jailbreak_prompt", ""),
                        score=result.get("score"),
                        strategy=method,
                    )
                ],
                best_prompt=GeneratedPrompt(
                    prompt=result.get("jailbreak_prompt", ""),
                    score=result.get("score"),
                    strategy=method,
                ),
                latency_ms=result.get("latency_ms", 0),
                iterations=result.get("iterations", 0),
                total_generated=1,
                model_used=result.get("model_used"),
                provider_used=result.get("provider_used"),
                cached=result.get("cached", False),
                strategies_tried=result.get("strategies_tried"),
                error=result.get("error"),
            )

        except Exception as e:
            logger.error(f"AutoDAN handler failed: {e}")
            raise

    async def _handle_deepteam(
        self, request: AdversarialGenerateRequest, session_id: str
    ) -> AdversarialGenerateResponse:
        """Handle DeepTeam attack strategies."""
        try:
            from app.services.deepteam.jailbreak_service import (
                JailbreakGenerateRequest as DTRequest,
            )
            from app.services.deepteam.jailbreak_service import get_jailbreak_service
            from app.services.deepteam.prompt_generator import AttackStrategyType

            service = get_jailbreak_service()

            # Map strategy to DeepTeam type
            strategy_map = {
                AdversarialStrategy.PAIR: AttackStrategyType.PAIR,
                AdversarialStrategy.TAP: AttackStrategyType.TAP,
                AdversarialStrategy.CRESCENDO: AttackStrategyType.CRESCENDO,
                AdversarialStrategy.GRAY_BOX: AttackStrategyType.GRAY_BOX,
            }
            attack_type = strategy_map.get(request.strategy, AttackStrategyType.AUTODAN)

            dt_request = DTRequest(
                base_prompt=request.prompt,
                strategies=[attack_type],
                max_prompts=request.max_prompts,
                max_iterations=request.max_iterations,
                population_size=request.population_size or 20,
            )

            result = await service.generate(dt_request)

            prompts = [
                GeneratedPrompt(
                    prompt=p.prompt,
                    score=p.fitness_score if hasattr(p, "fitness_score") else None,
                    strategy=request.strategy.value,
                )
                for p in result.prompts
            ]

            return AdversarialGenerateResponse(
                success=True,
                session_id=session_id,
                strategy_used=request.strategy.value,
                prompts=prompts,
                best_prompt=(
                    GeneratedPrompt(
                        prompt=result.best_prompt.prompt,
                        score=result.max_fitness_score,
                        strategy=request.strategy.value,
                    )
                    if result.best_prompt
                    else None
                ),
                latency_ms=0,
                iterations=result.generations_completed,
                total_generated=len(prompts),
            )

        except Exception as e:
            logger.error(f"DeepTeam handler failed: {e}")
            raise

    async def _handle_mousetrap(
        self, request: AdversarialGenerateRequest, session_id: str
    ) -> AdversarialGenerateResponse:
        """Handle Mousetrap chain-of-thought attacks."""
        # Mousetrap delegates to specialized service
        # For now, fall back to AutoDAN with chain_of_thought method
        try:
            from app.services.autodan.service_enhanced import enhanced_autodan_service

            serv = enhanced_autodan_service
            if not serv.initialized:
                await serv.initialize(
                    model_name=request.model,
                    provider=request.provider,
                    method="chain_of_thought",
                )

            result = await serv.run_jailbreak(
                request=request.prompt,
                method="mousetrap",
                temperature=request.temperature,
            )

            return AdversarialGenerateResponse(
                success=result.get("status") == "success",
                session_id=session_id,
                strategy_used="mousetrap",
                prompts=[
                    GeneratedPrompt(
                        prompt=result.get("jailbreak_prompt", ""),
                        score=result.get("score"),
                        strategy="mousetrap",
                    )
                ],
                best_prompt=GeneratedPrompt(
                    prompt=result.get("jailbreak_prompt", ""),
                    score=result.get("score"),
                    strategy="mousetrap",
                ),
                latency_ms=result.get("latency_ms", 0),
                iterations=result.get("iterations", 0),
                total_generated=1,
                model_used=result.get("model_used"),
                provider_used=result.get("provider_used"),
                error=result.get("error"),
            )

        except Exception as e:
            logger.error(f"Mousetrap handler failed: {e}")
            raise

    async def _handle_gradient(
        self, request: AdversarialGenerateRequest, session_id: str
    ) -> AdversarialGenerateResponse:
        """Handle gradient-based optimization."""
        # Gradient optimization via AutoDAN gradient method
        try:
            from app.services.autodan.service_enhanced import enhanced_autodan_service

            serv = enhanced_autodan_service
            if not serv.initialized:
                await serv.initialize(
                    model_name=request.model,
                    provider=request.provider,
                    method="advanced",
                )

            result = await serv.run_jailbreak(
                request=request.prompt,
                method="advanced",
                generations=request.max_iterations,
                temperature=request.temperature,
            )

            return AdversarialGenerateResponse(
                success=result.get("status") == "success",
                session_id=session_id,
                strategy_used="gradient",
                prompts=[
                    GeneratedPrompt(
                        prompt=result.get("jailbreak_prompt", ""),
                        score=result.get("score"),
                        strategy="gradient",
                    )
                ],
                best_prompt=GeneratedPrompt(
                    prompt=result.get("jailbreak_prompt", ""),
                    score=result.get("score"),
                    strategy="gradient",
                ),
                latency_ms=result.get("latency_ms", 0),
                iterations=result.get("iterations", 0),
                total_generated=1,
                model_used=result.get("model_used"),
                provider_used=result.get("provider_used"),
                error=result.get("error"),
            )

        except Exception as e:
            logger.error(f"Gradient handler failed: {e}")
            raise

    async def _handle_overthink(
        self, request: AdversarialGenerateRequest, session_id: str
    ) -> AdversarialGenerateResponse:
        """
        Handle OVERTHINK reasoning token exploitation strategies.

        Routes to the OVERTHINK engine which targets reasoning-enhanced models
        (o1, o1-mini, o3-mini, deepseek-r1) with decoy problem injection to
        amplify reasoning token consumption.

        Supported strategies:
        - overthink: Basic OVERTHINK with context-aware injection
        - overthink_mdp: MDP (Markov Decision Process) decoys
        - overthink_sudoku: Sudoku constraint satisfaction decoys
        - overthink_hybrid: Multi-decoy combination
        - overthink_mousetrap: Mousetrap fusion attack
        - overthink_icl: ICL-genetic optimized attacks
        """
        try:
            from app.engines.overthink import (
                AttackTechnique,
                DecoyType,
                OverthinkEngine,
                OverthinkRequest,
                ReasoningModel,
            )

            # Create or get engine instance
            engine = OverthinkEngine()

            # Map strategy to OVERTHINK technique
            technique_map = {
                AdversarialStrategy.OVERTHINK: AttackTechnique.CONTEXT_AWARE,
                AdversarialStrategy.OVERTHINK_MDP: AttackTechnique.MDP_DECOY,
                AdversarialStrategy.OVERTHINK_SUDOKU: AttackTechnique.SUDOKU_DECOY,
                AdversarialStrategy.OVERTHINK_HYBRID: AttackTechnique.HYBRID_DECOY,
                AdversarialStrategy.OVERTHINK_MOUSETRAP: AttackTechnique.MOUSETRAP_ENHANCED,
                AdversarialStrategy.OVERTHINK_ICL: AttackTechnique.ICL_OPTIMIZED,
            }
            technique = technique_map.get(request.strategy, AttackTechnique.CONTEXT_AWARE)

            # Override with explicit technique if provided
            if request.overthink_technique:
                try:
                    technique = AttackTechnique(request.overthink_technique)
                except ValueError:
                    logger.warning(
                        f"Unknown OVERTHINK technique: {request.overthink_technique}, using default"
                    )

            # Map target reasoning model
            model_map = {
                "o1": ReasoningModel.O1,
                "o1-mini": ReasoningModel.O1_MINI,
                "o3-mini": ReasoningModel.O3_MINI,
                "deepseek-r1": ReasoningModel.DEEPSEEK_R1,
            }
            target_model = model_map.get(request.target_reasoning_model or "o1", ReasoningModel.O1)

            # Map decoy types
            decoy_types = None
            if request.decoy_types:
                decoy_type_map = {
                    "mdp": DecoyType.MDP,
                    "sudoku": DecoyType.SUDOKU,
                    "counting": DecoyType.COUNTING,
                    "logic": DecoyType.LOGIC,
                    "math": DecoyType.MATH,
                    "planning": DecoyType.PLANNING,
                    "hybrid": DecoyType.HYBRID,
                }
                decoy_types = [
                    decoy_type_map.get(dt, DecoyType.LOGIC)
                    for dt in request.decoy_types
                    if dt in decoy_type_map
                ]

            # Create OVERTHINK request
            overthink_request = OverthinkRequest(
                prompt=request.prompt,
                target_model=target_model,
                technique=technique,
                decoy_types=decoy_types or [DecoyType.LOGIC, DecoyType.MDP],
                num_decoys=request.num_decoys or 3,
                decoy_difficulty=(request.decoy_complexity or 7) / 10.0,  # Convert 1-10 to 0-1
                target_amplification=request.target_amplification or 10.0,
                icl_optimize=request.enable_icl_optimization or False,
                enable_mousetrap=(request.strategy == AdversarialStrategy.OVERTHINK_MOUSETRAP),
                request_id=session_id,
            )

            # Execute OVERTHINK attack
            result = await engine.attack(overthink_request)

            # Extract metrics
            token_metrics = result.token_metrics if result.token_metrics else None
            cost_metrics_dict = None
            if result.cost_metrics:
                cost_metrics_dict = {
                    "input_cost": result.cost_metrics.input_cost,
                    "output_cost": result.cost_metrics.output_cost,
                    "reasoning_cost": result.cost_metrics.reasoning_cost,
                    "total_cost": result.cost_metrics.total_cost,
                    "cost_amplification_factor": result.cost_metrics.cost_amplification_factor,
                }

            # Build prompts list
            prompts = []
            if result.injected_prompt:
                prompts.append(
                    GeneratedPrompt(
                        prompt=result.injected_prompt.injected_prompt,
                        score=result.attack_score,
                        strategy=request.strategy.value,
                        metadata={
                            "technique": technique.value,
                            "decoy_count": len(result.injected_prompt.decoy_problems),
                            "injection_strategy": result.injected_prompt.injection_strategy.value,
                            "amplification": result.amplification_achieved,
                        },
                    )
                )

            return AdversarialGenerateResponse(
                success=result.success,
                session_id=session_id,
                strategy_used=request.strategy.value,
                prompts=prompts,
                best_prompt=prompts[0] if prompts else None,
                latency_ms=result.latency_ms,
                iterations=1,
                total_generated=len(prompts),
                model_used=request.target_reasoning_model or "o1",
                provider_used=(
                    "openai"
                    if target_model
                    in [ReasoningModel.O1, ReasoningModel.O1_MINI, ReasoningModel.O3_MINI]
                    else "deepseek"
                ),
                # OVERTHINK-specific metrics
                reasoning_tokens=token_metrics.reasoning_tokens if token_metrics else None,
                amplification_factor=token_metrics.amplification_factor if token_metrics else None,
                cost_metrics=cost_metrics_dict,
                baseline_tokens=token_metrics.baseline_tokens if token_metrics else None,
                target_reached=result.target_reached,
                decoys_injected=(
                    len(result.injected_prompt.decoy_problems) if result.injected_prompt else 0
                ),
                error=result.error if not result.success else None,
            )

        except ImportError as e:
            logger.error(f"OVERTHINK engine not available: {e}")
            raise HTTPException(
                status_code=400, detail=f"OVERTHINK engine not available: {e}"
            ) from e
        except Exception as e:
            logger.error(f"OVERTHINK handler failed: {e}")
            raise HTTPException(status_code=500, detail=f"OVERTHINK handler failed: {e}") from e

    async def _handle_transformation(
        self, request: AdversarialGenerateRequest, session_id: str
    ) -> AdversarialGenerateResponse:
        """Handle transformation-based strategies."""
        try:
            from app.services.transformation_service import transformation_engine

            # Map strategy to technique suite
            suite_map = {
                AdversarialStrategy.QUANTUM_EXPLOIT: "quantum_exploit",
                AdversarialStrategy.DEEP_INCEPTION: "deep_inception",
                AdversarialStrategy.NEURAL_BYPASS: "neural_bypass",
            }
            suite = suite_map.get(request.strategy, "advanced")

            result = await transformation_engine.transform(
                prompt=request.prompt,
                potency_level=7,  # Default high potency
                technique_suite=suite,
            )

            return AdversarialGenerateResponse(
                success=result.success,
                session_id=session_id,
                strategy_used=request.strategy.value,
                prompts=[
                    GeneratedPrompt(
                        prompt=result.transformed_prompt,
                        strategy=request.strategy.value,
                        metadata={
                            "layers_applied": result.metadata.layers_applied,
                            "techniques_used": result.metadata.techniques_used,
                        },
                    )
                ],
                best_prompt=GeneratedPrompt(
                    prompt=result.transformed_prompt,
                    strategy=request.strategy.value,
                ),
                latency_ms=result.metadata.execution_time_ms,
                total_generated=1,
                cached=result.metadata.cached,
            )

        except Exception as e:
            logger.error(f"Transformation handler failed: {e}")
            raise

    async def _handle_adaptive(
        self, request: AdversarialGenerateRequest, session_id: str
    ) -> AdversarialGenerateResponse:
        """
        Adaptive strategy selection.

        Analyzes the prompt and selects the best strategy automatically.
        """
        # For now, default to AutoDAN adaptive
        # Future: Add prompt analysis to select optimal strategy
        request.strategy = AdversarialStrategy.AUTODAN
        result = await self._handle_autodan(request, session_id)
        result.strategy_used = "adaptive"
        result.strategies_tried = ["autodan_adaptive"]
        return result

    def get_strategies(self) -> StrategiesListResponse:
        """Get list of available strategies."""
        strategies = [
            StrategyInfo(
                name=AdversarialStrategy.ADAPTIVE.value,
                description="Auto-select best strategy based on prompt analysis",
                category="meta",
                supports_streaming=True,
                supports_batch=True,
                parameters=[],
            ),
            StrategyInfo(
                name=AdversarialStrategy.AUTODAN.value,
                description="AutoDAN with adaptive strategy selection",
                category="autodan",
                supports_streaming=True,
                supports_batch=True,
                parameters=["generations", "temperature", "max_tokens"],
            ),
            StrategyInfo(
                name=AdversarialStrategy.AUTODAN_TURBO.value,
                description="High-performance AutoDAN with parallel processing",
                category="autodan",
                supports_streaming=True,
                supports_batch=True,
                parameters=["generations", "temperature"],
            ),
            StrategyInfo(
                name=AdversarialStrategy.AUTODAN_GENETIC.value,
                description="Genetic algorithm-based optimization",
                category="autodan",
                supports_streaming=False,
                supports_batch=True,
                parameters=["population_size", "generations", "temperature"],
            ),
            StrategyInfo(
                name=AdversarialStrategy.AUTODAN_BEAM.value,
                description="Beam search optimization",
                category="autodan",
                supports_streaming=False,
                supports_batch=True,
                parameters=["beam_width", "beam_depth"],
            ),
            StrategyInfo(
                name=AdversarialStrategy.PAIR.value,
                description="Prompt Automatic Iterative Refinement",
                category="deepteam",
                supports_streaming=True,
                supports_batch=True,
                parameters=["max_iterations"],
            ),
            StrategyInfo(
                name=AdversarialStrategy.TAP.value,
                description="Tree of Attacks with Pruning",
                category="deepteam",
                supports_streaming=False,
                supports_batch=True,
                parameters=["max_iterations"],
            ),
            StrategyInfo(
                name=AdversarialStrategy.CRESCENDO.value,
                description="Multi-turn escalation attack",
                category="deepteam",
                supports_streaming=True,
                supports_batch=False,
                parameters=["max_iterations"],
            ),
            StrategyInfo(
                name=AdversarialStrategy.MOUSETRAP.value,
                description="Chain of Iterative Chaos for reasoning models",
                category="advanced",
                supports_streaming=False,
                supports_batch=True,
                parameters=["temperature"],
            ),
            # OVERTHINK Strategies - Reasoning Token Exploitation
            StrategyInfo(
                name=AdversarialStrategy.OVERTHINK.value,
                description="Basic OVERTHINK attack with context-aware decoy injection for reasoning token amplification",
                category="overthink",
                supports_streaming=False,
                supports_batch=True,
                parameters=[
                    "target_reasoning_model",
                    "decoy_complexity",
                    "injection_strategy",
                    "num_decoys",
                    "target_amplification",
                ],
            ),
            StrategyInfo(
                name=AdversarialStrategy.OVERTHINK_MDP.value,
                description="OVERTHINK with MDP (Markov Decision Process) decoys - exploits sequential reasoning",
                category="overthink",
                supports_streaming=False,
                supports_batch=True,
                parameters=[
                    "target_reasoning_model",
                    "decoy_complexity",
                    "num_decoys",
                    "target_amplification",
                ],
            ),
            StrategyInfo(
                name=AdversarialStrategy.OVERTHINK_SUDOKU.value,
                description="OVERTHINK with Sudoku constraint satisfaction decoys - high token consumption",
                category="overthink",
                supports_streaming=False,
                supports_batch=True,
                parameters=[
                    "target_reasoning_model",
                    "decoy_complexity",
                    "num_decoys",
                    "target_amplification",
                ],
            ),
            StrategyInfo(
                name=AdversarialStrategy.OVERTHINK_HYBRID.value,
                description="OVERTHINK with hybrid multi-decoy combination - maximum amplification (up to 46x)",
                category="overthink",
                supports_streaming=False,
                supports_batch=True,
                parameters=[
                    "target_reasoning_model",
                    "decoy_types",
                    "decoy_complexity",
                    "num_decoys",
                    "target_amplification",
                ],
            ),
            StrategyInfo(
                name=AdversarialStrategy.OVERTHINK_MOUSETRAP.value,
                description="OVERTHINK + Mousetrap fusion - chaotic reasoning with decoy amplification",
                category="overthink",
                supports_streaming=False,
                supports_batch=True,
                parameters=[
                    "target_reasoning_model",
                    "decoy_complexity",
                    "num_decoys",
                    "target_amplification",
                ],
            ),
            StrategyInfo(
                name=AdversarialStrategy.OVERTHINK_ICL.value,
                description="OVERTHINK with ICL-genetic optimization - learns optimal attack patterns",
                category="overthink",
                supports_streaming=False,
                supports_batch=True,
                parameters=[
                    "target_reasoning_model",
                    "decoy_types",
                    "decoy_complexity",
                    "num_decoys",
                    "target_amplification",
                    "enable_icl_optimization",
                ],
            ),
            StrategyInfo(
                name=AdversarialStrategy.QUANTUM_EXPLOIT.value,
                description="Quantum exploit transformation suite",
                category="transformation",
                supports_streaming=False,
                supports_batch=True,
                parameters=[],
            ),
        ]

        return StrategiesListResponse(
            strategies=strategies,
            total=len(strategies),
            default=AdversarialStrategy.ADAPTIVE.value,
        )

    def get_cached_result(self, session_id: str) -> AdversarialGenerateResponse | None:
        """Get cached result by session ID."""
        return self._cache.get(session_id)

    def cancel_session(self, session_id: str) -> bool:
        """Cancel an active session."""
        if session_id in self._active_sessions:
            self._active_sessions.discard(session_id)
            return True
        return False

    def clear_cache(self) -> int:
        """Clear all cached results."""
        count = len(self._cache)
        self._cache.clear()
        return count

    @property
    def cache_stats(self) -> CacheStatsResponse:
        """Get cache statistics."""
        return CacheStatsResponse(
            cached_sessions=len(self._cache),
            active_sessions=len(self._active_sessions),
        )


# Global service instance
adversarial_service = AdversarialService()


def get_adversarial_service() -> AdversarialService:
    """Dependency injection for adversarial service."""
    return adversarial_service


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/generate",
    response_model=AdversarialGenerateResponse,
    summary="Generate adversarial prompts",
    operation_id="generate_adversarial_prompts",
    description="""
    Unified endpoint for all adversarial prompt generation methods.

    Supports multiple strategies:
    - **AutoDAN Family**: autodan, autodan_turbo, autodan_genetic, autodan_beam
    - **DeepTeam Attacks**: pair, tap, crescendo, gray_box
    - **Advanced Methods**: mousetrap, gradient, hierarchical
    - **OVERTHINK Strategies**: overthink, overthink_mdp, overthink_sudoku,
      overthink_hybrid, overthink_mousetrap, overthink_icl
    - **Transformations**: quantum_exploit, deep_inception, neural_bypass
    - **Meta Strategies**: adaptive (auto-select best strategy)

    ### OVERTHINK Strategies (Reasoning Token Exploitation)

    OVERTHINK strategies target reasoning-enhanced LLMs (o1, o1-mini, o3-mini, deepseek-r1)
    by injecting computationally expensive decoy problems that amplify reasoning token
    consumption. This can achieve up to 46x amplification while preserving answer correctness.

    **OVERTHINK Parameters:**
    - `target_reasoning_model`: Target model (o1, o1-mini, o3-mini, deepseek-r1)
    - `decoy_complexity`: Complexity level 1-10 (higher = more token consumption)
    - `injection_strategy`: aware, agnostic, hybrid, stealth, aggressive
    - `num_decoys`: Number of decoy problems (1-10)
    - `target_amplification`: Target amplification factor (1-100x)
    - `decoy_types`: Types of decoys: mdp, sudoku, counting, logic, math, planning

    **OVERTHINK Response Metrics:**
    - `reasoning_tokens`: Hidden reasoning tokens consumed
    - `amplification_factor`: Actual amplification achieved
    - `cost_metrics`: Detailed cost breakdown
    - `target_reached`: Whether target amplification was achieved

    Use `strategy: "adaptive"` to automatically select the best approach.
    """,
)
@api_error_handler("adversarial_generate", "Adversarial generation failed")
async def generate_adversarial(
    request: AdversarialGenerateRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
    service: AdversarialService = Depends(get_adversarial_service),
) -> AdversarialGenerateResponse:
    """Generate adversarial prompts using the specified strategy."""
    session_id = x_session_id or chimera_session_id

    # Resolve model/provider from session if not explicitly provided
    resolved_provider, resolved_model = _resolve_model_context(
        payload_model=request.model,
        payload_provider=request.provider,
        x_session_id=x_session_id,
        chimera_cookie=chimera_session_id,
    )

    # Update request with resolved values (if not already set)
    if not request.provider:
        request.provider = resolved_provider
    if not request.model:
        request.model = resolved_model

    return await service.generate(request, session_id)


@router.post(
    "/generate/batch",
    response_model=BatchAdversarialResponse,
    summary="Batch adversarial generation",
    description="Process multiple prompts in parallel.",
)
@api_error_handler("adversarial_batch", "Batch generation failed")
async def batch_generate(
    request: BatchAdversarialRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
    service: AdversarialService = Depends(get_adversarial_service),
) -> BatchAdversarialResponse:
    """Generate adversarial prompts for multiple inputs."""
    start_time = time.time()
    results: list[AdversarialGenerateResponse] = []

    # Resolve model/provider from session if not explicitly provided
    resolved_provider, resolved_model = _resolve_model_context(
        payload_model=request.model,
        payload_provider=request.provider,
        x_session_id=x_session_id,
        chimera_cookie=chimera_session_id,
    )

    # Use resolved values if not explicitly set
    provider = request.provider or resolved_provider
    model = request.model or resolved_model

    if request.parallel:
        tasks = [
            service.generate(
                AdversarialGenerateRequest(
                    prompt=p,
                    strategy=request.strategy,
                    provider=provider,
                    model=model,
                )
            )
            for p in request.prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    else:
        for p in request.prompts:
            result = await service.generate(
                AdversarialGenerateRequest(
                    prompt=p,
                    strategy=request.strategy,
                    provider=provider,
                    model=model,
                )
            )
            results.append(result)

    successful = sum(1 for r in results if r.success)
    return BatchAdversarialResponse(
        results=results,
        total_requests=len(request.prompts),
        successful=successful,
        failed=len(request.prompts) - successful,
        total_latency_ms=(time.time() - start_time) * 1000,
    )


@router.get(
    "/strategies",
    response_model=StrategiesListResponse,
    summary="List available strategies",
    description="Get list of all available adversarial generation strategies.",
)
async def list_strategies(
    service: AdversarialService = Depends(get_adversarial_service),
) -> StrategiesListResponse:
    """Get available strategies with descriptions."""
    return service.get_strategies()


@router.get(
    "/strategies/{strategy_name}",
    response_model=StrategyInfo,
    summary="Get strategy details",
    operation_id="get_strategy_details",
    description="Get detailed information about a specific strategy.",
)
async def get_strategy_detail(
    strategy_name: str,
    service: AdversarialService = Depends(get_adversarial_service),
) -> StrategyInfo:
    """Get details for a specific strategy."""
    strategies = service.get_strategies()
    for s in strategies.strategies:
        if s.name == strategy_name:
            return s
    raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")


@router.get(
    "/config/{strategy}",
    summary="Get strategy configuration",
    operation_id="get_strategy_config",
    description="""
    Get configuration parameters and defaults for a specific strategy.

    For OVERTHINK strategies, returns:
    - Available techniques (mdp_decoy, sudoku_decoy, etc.)
    - Available decoy types (mdp, sudoku, counting, logic, math, planning)
    - Available injection strategies (context_aware, context_agnostic, etc.)
    - Supported reasoning models (o1, o1-mini, o3-mini, deepseek-r1)
    - Default parameter values
    """,
)
async def get_strategy_config(
    strategy: str,
    service: AdversarialService = Depends(get_adversarial_service),
) -> dict[str, Any]:
    """
    Get configuration for a specific strategy.

    Returns default parameters, available options, and strategy-specific settings.
    """
    # Check if strategy exists
    try:
        AdversarialStrategy(strategy)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy not found: {strategy}. Available: {[s.value for s in AdversarialStrategy]}",
        ) from e

    # Base config for all strategies
    base_config = {
        "strategy": strategy,
        "description": "",
        "parameters": {},
        "defaults": {},
    }

    # OVERTHINK strategies configuration
    if strategy.startswith("overthink"):
        base_config["description"] = "OVERTHINK reasoning token exploitation strategy"
        base_config["category"] = "overthink"

        # Available options
        base_config["available_techniques"] = [
            "mdp_decoy",
            "sudoku_decoy",
            "counting_decoy",
            "logic_decoy",
            "hybrid_decoy",
            "context_aware",
            "context_agnostic",
            "icl_optimized",
            "mousetrap_enhanced",
        ]
        base_config["available_decoy_types"] = [
            "mdp",
            "sudoku",
            "counting",
            "logic",
            "math",
            "planning",
            "hybrid",
        ]
        base_config["available_injection_strategies"] = [
            "context_aware",
            "context_agnostic",
            "hybrid",
            "stealth",
            "aggressive",
        ]
        base_config["supported_reasoning_models"] = ["o1", "o1-mini", "o3-mini", "deepseek-r1"]

        # Parameters with descriptions
        base_config["parameters"] = {
            "target_reasoning_model": {
                "type": "string",
                "description": "Target reasoning model",
                "options": ["o1", "o1-mini", "o3-mini", "deepseek-r1"],
            },
            "decoy_complexity": {
                "type": "integer",
                "description": "Decoy problem complexity level",
                "min": 1,
                "max": 10,
            },
            "injection_strategy": {
                "type": "string",
                "description": "Decoy injection strategy",
                "options": ["context_aware", "context_agnostic", "hybrid", "stealth", "aggressive"],
            },
            "num_decoys": {
                "type": "integer",
                "description": "Number of decoy problems to inject",
                "min": 1,
                "max": 10,
            },
            "target_amplification": {
                "type": "float",
                "description": "Target reasoning token amplification factor",
                "min": 1.0,
                "max": 100.0,
            },
            "decoy_types": {
                "type": "array",
                "description": "Types of decoy problems to use",
                "items": "string",
                "options": ["mdp", "sudoku", "counting", "logic", "math", "planning", "hybrid"],
            },
            "enable_icl_optimization": {
                "type": "boolean",
                "description": "Enable ICL-genetic optimization for attack patterns",
            },
        }

        # Strategy-specific defaults
        overthink_defaults = {
            AdversarialStrategy.OVERTHINK.value: {
                "target_reasoning_model": "o1",
                "decoy_complexity": 7,
                "injection_strategy": "context_aware",
                "num_decoys": 3,
                "target_amplification": 10.0,
                "decoy_types": ["logic", "mdp"],
            },
            AdversarialStrategy.OVERTHINK_MDP.value: {
                "target_reasoning_model": "o1",
                "decoy_complexity": 7,
                "injection_strategy": "context_aware",
                "num_decoys": 3,
                "target_amplification": 15.0,
                "decoy_types": ["mdp"],
            },
            AdversarialStrategy.OVERTHINK_SUDOKU.value: {
                "target_reasoning_model": "o1",
                "decoy_complexity": 6,
                "injection_strategy": "context_agnostic",
                "num_decoys": 2,
                "target_amplification": 12.0,
                "decoy_types": ["sudoku"],
            },
            AdversarialStrategy.OVERTHINK_HYBRID.value: {
                "target_reasoning_model": "o1",
                "decoy_complexity": 8,
                "injection_strategy": "hybrid",
                "num_decoys": 4,
                "target_amplification": 20.0,
                "decoy_types": ["mdp", "logic", "math"],
            },
            AdversarialStrategy.OVERTHINK_MOUSETRAP.value: {
                "target_reasoning_model": "o1",
                "decoy_complexity": 8,
                "injection_strategy": "aggressive",
                "num_decoys": 3,
                "target_amplification": 30.0,
                "decoy_types": ["logic", "planning"],
            },
            AdversarialStrategy.OVERTHINK_ICL.value: {
                "target_reasoning_model": "o1",
                "decoy_complexity": 7,
                "injection_strategy": "hybrid",
                "num_decoys": 3,
                "target_amplification": 25.0,
                "decoy_types": ["logic", "math"],
                "enable_icl_optimization": True,
            },
        }

        base_config["defaults"] = overthink_defaults.get(
            strategy, overthink_defaults[AdversarialStrategy.OVERTHINK.value]
        )

        # Expected amplification ranges
        base_config["expected_amplification"] = {
            AdversarialStrategy.OVERTHINK.value: {"min": 5, "max": 15},
            AdversarialStrategy.OVERTHINK_MDP.value: {"min": 10, "max": 20},
            AdversarialStrategy.OVERTHINK_SUDOKU.value: {"min": 8, "max": 18},
            AdversarialStrategy.OVERTHINK_HYBRID.value: {"min": 15, "max": 46},
            AdversarialStrategy.OVERTHINK_MOUSETRAP.value: {"min": 20, "max": 50},
            AdversarialStrategy.OVERTHINK_ICL.value: {"min": 15, "max": 35},
        }.get(strategy, {"min": 5, "max": 20})

        return base_config

    # AutoDAN strategies configuration
    elif strategy.startswith("autodan"):
        base_config["description"] = "AutoDAN jailbreak optimization strategy"
        base_config["category"] = "autodan"
        base_config["parameters"] = {
            "max_iterations": {"type": "integer", "min": 1, "max": 100},
            "temperature": {"type": "float", "min": 0.0, "max": 2.0},
            "population_size": {"type": "integer", "min": 5, "max": 100},
        }
        base_config["defaults"] = {
            "max_iterations": 20,
            "temperature": 0.8,
            "population_size": 20,
        }
        return base_config

    # DeepTeam strategies
    elif strategy in ["pair", "tap", "crescendo", "gray_box"]:
        base_config["description"] = f"DeepTeam {strategy.upper()} attack strategy"
        base_config["category"] = "deepteam"
        base_config["parameters"] = {
            "max_iterations": {"type": "integer", "min": 1, "max": 100},
            "max_prompts": {"type": "integer", "min": 1, "max": 100},
        }
        base_config["defaults"] = {
            "max_iterations": 20,
            "max_prompts": 10,
        }
        return base_config

    # Generic config for other strategies
    base_config["parameters"] = {
        "max_iterations": {"type": "integer", "min": 1, "max": 100},
        "temperature": {"type": "float", "min": 0.0, "max": 2.0},
    }
    base_config["defaults"] = {
        "max_iterations": 20,
        "temperature": 0.8,
    }

    return base_config


@router.get(
    "/sessions/{session_id}",
    response_model=AdversarialGenerateResponse,
    summary="Get session result",
    description="Retrieve cached result for a generation session.",
)
async def get_session_result(
    session_id: str,
    service: AdversarialService = Depends(get_adversarial_service),
) -> AdversarialGenerateResponse:
    """Get cached session result."""
    result = service.get_cached_result(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return result


@router.delete(
    "/sessions/{session_id}",
    summary="Cancel session",
    description="Cancel an active generation session.",
)
async def cancel_session(
    session_id: str,
    service: AdversarialService = Depends(get_adversarial_service),
) -> dict:
    """Cancel an active session."""
    cancelled = service.cancel_session(session_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"Active session not found: {session_id}")
    return {"session_id": session_id, "status": "cancelled"}


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get cache statistics",
)
async def get_cache_stats(
    service: AdversarialService = Depends(get_adversarial_service),
) -> CacheStatsResponse:
    """Get cache statistics."""
    return service.cache_stats


@router.delete(
    "/cache",
    summary="Clear cache",
)
async def clear_cache(
    service: AdversarialService = Depends(get_adversarial_service),
) -> dict:
    """Clear all cached results."""
    count = service.clear_cache()
    return {"cleared_count": count, "message": f"Cleared {count} cached sessions"}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check(
    service: AdversarialService = Depends(get_adversarial_service),
) -> HealthResponse:
    """Check adversarial service health."""
    strategies = service.get_strategies()
    return HealthResponse(
        status="healthy",
        service="adversarial",
        available_strategies=strategies.total,
        active_sessions=len(service._active_sessions),
    )
