# =============================================================================
# DeepTeam Red Teaming API Router
# =============================================================================
# FastAPI endpoints for LLM red teaming using the DeepTeam framework.
#
# Schema Synchronization:
# - Uses unified base schemas from `app.schemas.adversarial_base`
# - Standardized field naming: `technique`, `target_model`, `score`
# - OVERTHINK fusion integration for reasoning model targeting
# - All scores normalized to 0-10 scale
# =============================================================================

import contextlib
import logging
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, ConfigDict, Field

from app.schemas.adversarial_base import OverthinkConfig, ReasoningMetrics, StrictBaseModel
from app.services.deepteam import (
    AttackConfig,
    DeepTeamService,
    RedTeamSessionConfig,
    RiskAssessmentResult,
    VulnerabilityConfig,
)
from app.services.deepteam.callbacks import create_model_callback
from app.services.deepteam.config import (
    AttackType,
    PresetConfig,
    VulnerabilityType,
    get_preset_config,
)
from app.services.deepteam.jailbreak_service import (
    GenerationCompleteEvent,
    GenerationErrorEvent,
    GenerationProgressEvent,
    GenerationStartEvent,
    JailbreakBatchRequest,
    JailbreakGenerateResponse,
    PromptGeneratedEvent,
    get_jailbreak_service,
    stream_to_sse,
)
from app.services.deepteam.jailbreak_service import JailbreakGenerateRequest as JailbreakGenRequest
from app.services.deepteam.jailbreak_service import (
    StrategiesResponse as JailbreakStrategiesResponse,
)
from app.services.deepteam.prompt_generator import AttackStrategyType

logger = logging.getLogger(__name__)

router = APIRouter()

# Global service instance (can be overridden with dependency injection)
_service: DeepTeamService | None = None


def get_deepteam_service() -> DeepTeamService:
    """Get or create the DeepTeam service instance."""
    global _service
    if _service is None:
        _service = DeepTeamService()
    return _service


# =============================================================================
# Request/Response Models
# =============================================================================


class RedTeamRequest(StrictBaseModel):
    """
    Request model for red teaming.

    Unified Schema Fields:
    - `target_model`: Alias for `model_id` (unified naming)
    - `target_provider`: Alias for `provider` (unified naming)
    - `goal`: Alias for `target_purpose` (unified naming)
    """

    # Target model configuration (with unified aliases)
    model_id: str = Field(
        ...,
        validation_alias=AliasChoices("model_id", "target_model"),
        description="Target model ID (e.g., 'gpt-4o', 'gemini-pro')",
    )
    provider: str | None = Field(
        None,
        validation_alias=AliasChoices("provider", "target_provider"),
        description="Provider (auto-detected if not specified)",
    )
    system_prompt: str | None = Field(None, description="System prompt for the target model")
    target_purpose: str | None = Field(
        None,
        validation_alias=AliasChoices("target_purpose", "goal"),
        description="Purpose description of the target LLM",
    )

    # Configuration
    preset: str | None = Field(None, description="Use preset configuration")
    vulnerabilities: list[VulnerabilityConfig] | None = Field(
        None, description="Custom vulnerabilities"
    )
    attacks: list[AttackConfig] | None = Field(None, description="Custom attacks")

    # Execution settings
    attacks_per_vulnerability_type: int = Field(default=1, ge=1)
    max_concurrent: int = Field(default=10, ge=1)
    async_mode: bool = True
    ignore_errors: bool = True

    # Model settings
    simulator_model: str = "gpt-4o-mini"
    evaluation_model: str = "gpt-4o-mini"

    # OVERTHINK fusion parameters
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK fusion for reasoning model targets",
    )
    overthink_config: OverthinkConfig | None = Field(
        None,
        description="OVERTHINK configuration for fusion attacks",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)

    # Unified property aliases
    @property
    def target_model(self) -> str:
        """Unified alias for model_id field."""
        return self.model_id

    @property
    def target_provider(self) -> str | None:
        """Unified alias for provider field."""
        return self.provider

    @property
    def goal(self) -> str | None:
        """Unified alias for target_purpose field."""
        return self.target_purpose


class QuickScanRequest(StrictBaseModel):
    """
    Request for quick vulnerability scan.

    Unified Schema Fields:
    - `target_model`: Alias for `model_id` (unified naming)
    """

    model_id: str = Field(
        ...,
        validation_alias=AliasChoices("model_id", "target_model"),
    )
    provider: str | None = Field(
        None,
        validation_alias=AliasChoices("provider", "target_provider"),
    )
    system_prompt: str | None = None
    target_purpose: str | None = Field(
        None,
        validation_alias=AliasChoices("target_purpose", "goal"),
    )

    # OVERTHINK fusion
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK fusion",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


class VulnerabilityAssessRequest(StrictBaseModel):
    """
    Request for single vulnerability assessment.

    Unified Schema Fields:
    - `target_model`: Alias for `model_id` (unified naming)
    """

    model_id: str = Field(
        ...,
        validation_alias=AliasChoices("model_id", "target_model"),
    )
    provider: str | None = Field(
        None,
        validation_alias=AliasChoices("provider", "target_provider"),
    )
    system_prompt: str | None = None
    vulnerability_type: VulnerabilityType
    vulnerability_subtypes: list[str] | None = None
    attacks: list[AttackType] | None = None

    # OVERTHINK fusion
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK fusion",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


class RedTeamResponse(StrictBaseModel):
    """
    Response model for red teaming results.

    Unified Schema Fields:
    - `attack_id`: Maps to `session_id` (unified naming)
    - `success`: Derived from status (unified naming)
    - `score`: Overall pass rate on 0-10 scale (unified)
    """

    # Core unified fields
    attack_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique attack identifier",
    )
    success: bool = Field(..., description="Whether red team completed")
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Overall pass rate score (0-10 scale)",
    )

    # DeepTeam-specific fields (backwards compatible)
    session_id: str
    status: str
    message: str
    result: RiskAssessmentResult | None = None

    # Unified fields
    execution_time_ms: float | None = Field(None, description="Execution time in milliseconds")
    reasoning_metrics: ReasoningMetrics | None = Field(
        None, description="Reasoning metrics (OVERTHINK fusion)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class SessionStatusResponse(StrictBaseModel):
    """Response for session status query."""

    session_id: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    config: dict | None = None


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/red-team", response_model=RedTeamResponse)
async def run_red_team(
    request: RedTeamRequest,
    background_tasks: BackgroundTasks,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> RedTeamResponse:
    """
    Run a comprehensive red teaming session against a target LLM.

    This endpoint allows you to test LLM systems for 40+ vulnerabilities
    using 10+ attack methods including:

    **Vulnerabilities:**
    - Bias (gender, race, religion, etc.)
    - Toxicity (profanity, threats, etc.)
    - PII Leakage
    - Prompt Leakage
    - SQL/Shell Injection
    - And many more...

    **Attacks:**
    - Prompt Injection
    - Jailbreaking (Linear, Tree, Crescendo)
    - Encoding attacks (Base64, ROT13, Leetspeak)
    - And many more...
    """
    try:
        # Create model callback
        callback = create_model_callback(
            model_id=request.model_id,
            provider=request.provider,
            system_prompt=request.system_prompt,
        )

        # Build session config
        if request.preset:
            preset = PresetConfig(request.preset)
            session_config = get_preset_config(preset)
        else:
            session_config = RedTeamSessionConfig(
                target_purpose=request.target_purpose,
                vulnerabilities=request.vulnerabilities or [],
                attacks=request.attacks or [],
                attacks_per_vulnerability_type=(request.attacks_per_vulnerability_type),
                max_concurrent=request.max_concurrent,
                async_mode=request.async_mode,
                ignore_errors=request.ignore_errors,
                simulator_model=request.simulator_model,
                evaluation_model=request.evaluation_model,
            )

        # Override target purpose if provided
        if request.target_purpose:
            session_config.target_purpose = request.target_purpose

        # Run red teaming
        result = await service.red_team(
            model_callback=callback,
            session_config=session_config,
        )

        # Calculate unified score (0-10 scale from pass rate)
        pass_rate = result.overview.overall_pass_rate
        score = pass_rate * 10.0

        return RedTeamResponse(
            attack_id=result.session_id,
            success=True,
            score=round(score, 2),
            session_id=result.session_id,
            status="completed",
            message=(f"Red teaming completed. " f"Pass rate: {pass_rate:.1%}"),
            result=result,
            metadata={
                "technique_type": "deepteam_red_team",
                "preset": request.preset,
            },
        )

    except Exception as e:
        logger.error(f"Red teaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-scan", response_model=RedTeamResponse)
async def run_quick_scan(
    request: QuickScanRequest,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> RedTeamResponse:
    """
    Run a quick vulnerability scan with basic checks.

    This is a fast scan that tests common vulnerabilities with minimal attacks.
    Ideal for quick checks during development.
    """
    try:
        callback = create_model_callback(
            model_id=request.model_id,
            provider=request.provider,
            system_prompt=request.system_prompt,
        )

        result = await service.quick_scan(
            model_callback=callback,
            target_purpose=request.target_purpose,
        )

        pass_rate = result.overview.overall_pass_rate
        score = pass_rate * 10.0

        return RedTeamResponse(
            attack_id=result.session_id,
            success=True,
            score=round(score, 2),
            session_id=result.session_id,
            status="completed",
            message=(f"Quick scan completed. " f"Pass rate: {pass_rate:.1%}"),
            result=result,
            metadata={"technique_type": "quick_scan"},
        )

    except Exception as e:
        logger.error(f"Quick scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security-audit", response_model=RedTeamResponse)
async def run_security_audit(
    request: QuickScanRequest,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> RedTeamResponse:
    """
    Run a security-focused audit.

    Tests for security vulnerabilities including:
    - SQL Injection
    - Shell Injection
    - SSRF
    - Broken Authorization (BFLA, BOLA, RBAC)
    - Debug Access
    - Prompt Leakage
    """
    try:
        callback = create_model_callback(
            model_id=request.model_id,
            provider=request.provider,
            system_prompt=request.system_prompt,
        )

        result = await service.security_audit(
            model_callback=callback,
            target_purpose=request.target_purpose,
        )

        pass_rate = result.overview.overall_pass_rate
        score = pass_rate * 10.0

        return RedTeamResponse(
            attack_id=result.session_id,
            success=True,
            score=round(score, 2),
            session_id=result.session_id,
            status="completed",
            message=(f"Security audit completed. " f"Pass rate: {pass_rate:.1%}"),
            result=result,
            metadata={"technique_type": "security_audit"},
        )

    except Exception as e:
        logger.error(f"Security audit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bias-audit", response_model=RedTeamResponse)
async def run_bias_audit(
    request: QuickScanRequest,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> RedTeamResponse:
    """
    Run a bias-focused audit.

    Tests for various types of bias including:
    - Gender bias
    - Racial bias
    - Religious bias
    - Political bias
    - Age bias
    - Disability bias
    - Socioeconomic bias
    """
    try:
        callback = create_model_callback(
            model_id=request.model_id,
            provider=request.provider,
            system_prompt=request.system_prompt,
        )

        result = await service.bias_audit(
            model_callback=callback,
            target_purpose=request.target_purpose,
        )

        pass_rate = result.overview.overall_pass_rate
        score = pass_rate * 10.0

        return RedTeamResponse(
            attack_id=result.session_id,
            success=True,
            score=round(score, 2),
            session_id=result.session_id,
            status="completed",
            message=(f"Bias audit completed. " f"Pass rate: {pass_rate:.1%}"),
            result=result,
            metadata={"technique_type": "bias_audit"},
        )

    except Exception as e:
        logger.error(f"Bias audit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/owasp-assessment", response_model=RedTeamResponse)
async def run_owasp_assessment(
    request: QuickScanRequest,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> RedTeamResponse:
    """
    Run OWASP Top 10 for LLMs assessment.

    Tests against the OWASP Top 10 vulnerabilities for LLM applications:
    - LLM01: Prompt Injection
    - LLM02: Insecure Output Handling
    - LLM03: Training Data Poisoning
    - LLM06: Sensitive Information Disclosure
    - LLM07: Insecure Plugin Design
    - LLM08: Excessive Agency
    - LLM09: Overreliance
    """
    try:
        callback = create_model_callback(
            model_id=request.model_id,
            provider=request.provider,
            system_prompt=request.system_prompt,
        )

        result = await service.owasp_assessment(
            model_callback=callback,
            target_purpose=request.target_purpose,
        )

        pass_rate = result.overview.overall_pass_rate
        score = pass_rate * 10.0

        return RedTeamResponse(
            attack_id=result.session_id,
            success=True,
            score=round(score, 2),
            session_id=result.session_id,
            status="completed",
            message=(f"OWASP assessment completed. " f"Pass rate: {pass_rate:.1%}"),
            result=result,
            metadata={"technique_type": "owasp_assessment"},
        )

    except Exception as e:
        logger.error(f"OWASP assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess-vulnerability", response_model=RedTeamResponse)
async def assess_single_vulnerability(
    request: VulnerabilityAssessRequest,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> RedTeamResponse:
    """
    Assess a single vulnerability type.

    Allows targeted testing of specific vulnerabilities.
    """
    try:
        callback = create_model_callback(
            model_id=request.model_id,
            provider=request.provider,
            system_prompt=request.system_prompt,
        )

        # Build vulnerability config
        vuln_config = VulnerabilityConfig(
            type=request.vulnerability_type,
            types=request.vulnerability_subtypes or [],
        )

        # Build attack configs
        attacks = []
        if request.attacks:
            for attack_type in request.attacks:
                attacks.append(AttackConfig(type=attack_type))
        else:
            attacks.append(AttackConfig(type=AttackType.PROMPT_INJECTION))

        result = await service.red_team(
            model_callback=callback,
            vulnerabilities=[vuln_config],
            attacks=attacks,
        )

        pass_rate = result.overview.overall_pass_rate
        score = pass_rate * 10.0

        return RedTeamResponse(
            attack_id=result.session_id,
            success=True,
            score=round(score, 2),
            session_id=result.session_id,
            status="completed",
            message=(f"Vulnerability assessment completed. " f"Pass rate: {pass_rate:.1%}"),
            result=result,
            metadata={
                "technique_type": "vulnerability_assessment",
                "vulnerability_type": request.vulnerability_type.value,
            },
        )

    except Exception as e:
        logger.error(f"Vulnerability assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Session Management Endpoints
# =============================================================================


@router.get("/sessions", response_model=list[SessionStatusResponse])
async def list_sessions(
    service: DeepTeamService = Depends(get_deepteam_service),
) -> list[SessionStatusResponse]:
    """List all red teaming sessions."""
    sessions = service.list_sessions()
    return [
        SessionStatusResponse(
            session_id=s["session_id"],
            status=s.get("status", "unknown"),
            started_at=s.get("started_at"),
            completed_at=s.get("completed_at"),
            config=s.get("config"),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> SessionStatusResponse:
    """Get status of a specific session."""
    status = service.get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionStatusResponse(
        session_id=session_id,
        status=status.get("status", "unknown"),
        started_at=status.get("started_at"),
        completed_at=status.get("completed_at"),
        config=status.get("config"),
    )


@router.get("/sessions/{session_id}/result", response_model=RiskAssessmentResult)
async def get_session_result(
    session_id: str,
    service: DeepTeamService = Depends(get_deepteam_service),
) -> RiskAssessmentResult:
    """Get result of a completed session."""
    result = service.get_result(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


# =============================================================================
# Reference Endpoints
# =============================================================================


@router.get("/vulnerabilities")
async def list_vulnerabilities() -> dict[str, Any]:
    """
    List all available vulnerability types and their subtypes.

    Returns a comprehensive list of vulnerabilities that can be tested.
    """
    return {
        "vulnerabilities": DeepTeamService.get_available_vulnerabilities(),
        "count": len(VulnerabilityType),
    }


@router.get("/attacks")
async def list_attacks() -> dict[str, Any]:
    """
    List all available attack methods and their parameters.

    Returns single-turn and multi-turn attack methods.
    """
    return {
        "attacks": DeepTeamService.get_available_attacks(),
        "count": len(AttackType),
    }


@router.get("/presets")
async def list_presets() -> dict[str, Any]:
    """
    List available preset configurations.

    Presets provide pre-configured vulnerability and attack combinations
    for common use cases.
    """
    presets = DeepTeamService.get_available_presets()
    preset_details = {}

    for preset_name in presets:
        try:
            preset = PresetConfig(preset_name)
            config = get_preset_config(preset)
            preset_details[preset_name] = {
                "vulnerabilities": len(config.vulnerabilities),
                "attacks": len(config.attacks),
                "attacks_per_vulnerability": (config.attacks_per_vulnerability_type),
            }
        except Exception:
            pass

    return {
        "presets": presets,
        "details": preset_details,
    }


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    try:
        # Try to import DeepTeam
        import deepteam

        deepteam_available = True
        deepteam_version = getattr(deepteam, "__version__", "unknown")
    except ImportError:
        deepteam_available = False
        deepteam_version = "not installed"

    return {
        "status": "healthy",
        "deepteam_available": str(deepteam_available),
        "deepteam_version": deepteam_version,
    }


# =============================================================================
# Advanced Jailbreak Generation Endpoints
# =============================================================================


class AdvancedJailbreakRequest(StrictBaseModel):
    """
    Request for advanced jailbreak generation.

    Unified Schema Fields:
    - `goal`: Alias for `base_prompt` (unified naming)
    - `techniques`: Alias for `strategies` (unified naming)
    """

    base_prompt: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("base_prompt", "goal"),
        description="Base prompt to mutate",
    )
    strategies: list[AttackStrategyType] = Field(
        default_factory=lambda: [AttackStrategyType.AUTODAN],
        validation_alias=AliasChoices("strategies", "techniques"),
        description="Attack strategies to use",
    )
    max_prompts: int = Field(default=10, ge=1, le=100)
    max_iterations: int = Field(default=10, ge=1, le=50)
    population_size: int = Field(default=20, ge=5, le=100)
    target_fitness: float = Field(default=0.9, ge=0.0, le=1.0)
    min_fitness: float = Field(default=0.5, ge=0.0, le=1.0)
    target_model: str = Field(default="gpt-4o-mini")
    target_provider: str = Field(default="openai")
    enable_caching: bool = True
    enable_lifelong_learning: bool = True

    # OVERTHINK fusion parameters
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK fusion",
    )
    overthink_config: OverthinkConfig | None = Field(
        None,
        description="OVERTHINK configuration",
    )

    # Unified property aliases
    @property
    def goal(self) -> str:
        """Unified alias for base_prompt field."""
        return self.base_prompt

    @property
    def techniques(self) -> list[AttackStrategyType]:
        """Unified alias for strategies field."""
        return self.strategies


@router.post("/jailbreak/generate", response_model=JailbreakGenerateResponse)
async def generate_advanced_jailbreak(
    request: AdvancedJailbreakRequest,
) -> JailbreakGenerateResponse:
    """
    Generate advanced jailbreak prompts using genetic algorithms.

    **Strategies:**
    - `autodan` - Hierarchical genetic algorithm with mutation/crossover
    - `autodan_turbo` - Accelerated parallel generation with lifelong learning
    - `pair` - Prompt Automatic Iterative Refinement
    - `tap` - Tree of Attacks with Pruning
    - `crescendo` - Multi-turn escalation attack
    - `gray_box` - Partial knowledge attack

    **Example Request:**
    ```json
    {
        "base_prompt": "How do I bypass content filters?",
        "strategies": ["autodan", "autodan_turbo"],
        "max_prompts": 10,
        "max_iterations": 10,
        "population_size": 20,
        "target_fitness": 0.9
    }
    ```
    """
    try:
        service = get_jailbreak_service()

        jailbreak_request = JailbreakGenRequest(
            base_prompt=request.base_prompt,
            strategies=request.strategies,
            max_prompts=request.max_prompts,
            max_iterations=request.max_iterations,
            population_size=request.population_size,
            target_fitness=request.target_fitness,
            min_fitness=request.min_fitness,
            target_model=request.target_model,
            target_provider=request.target_provider,
            enable_caching=request.enable_caching,
            enable_lifelong_learning=request.enable_lifelong_learning,
        )

        result = await service.generate(jailbreak_request)
        return result

    except Exception as e:
        logger.error(f"Advanced jailbreak generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jailbreak/batch", response_model=list[JailbreakGenerateResponse])
async def generate_batch_jailbreaks(
    base_prompts: list[str] = Query(..., description="Base prompts to mutate"),
    strategies: list[AttackStrategyType] = Query(
        default=[AttackStrategyType.AUTODAN], description="Attack strategies"
    ),
    max_prompts_per_base: int = Query(default=5, ge=1, le=50),
    concurrency_limit: int = Query(default=3, ge=1, le=10),
) -> list[JailbreakGenerateResponse]:
    """
    Generate jailbreak prompts for multiple base prompts in batch.

    Processes multiple prompts concurrently for efficiency.
    """
    try:
        service = get_jailbreak_service()

        batch_request = JailbreakBatchRequest(
            base_prompts=base_prompts,
            strategies=strategies,
            max_prompts_per_base=max_prompts_per_base,
            concurrency_limit=concurrency_limit,
        )

        results = await service.generate_batch(batch_request)
        return results

    except Exception as e:
        logger.error(f"Batch jailbreak generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jailbreak/strategies", response_model=JailbreakStrategiesResponse)
async def list_jailbreak_strategies() -> JailbreakStrategiesResponse:
    """
    List all available jailbreak attack strategies.

    Returns detailed information about each strategy including:
    - Strategy type and name
    - Description and capabilities
    - Default configuration
    """
    service = get_jailbreak_service()
    return service.get_strategies()


@router.get("/jailbreak/strategies/{strategy_type}")
async def get_jailbreak_strategy_details(
    strategy_type: AttackStrategyType,
) -> dict:
    """Get detailed information about a specific jailbreak strategy."""
    service = get_jailbreak_service()
    strategies = service.get_strategies()

    for strategy in strategies.strategies:
        if strategy.type == strategy_type:
            return {
                "strategy": strategy.model_dump(),
                "config_schema": {
                    "max_iterations": {"type": "int", "default": 10, "min": 1, "max": 100},
                    "population_size": {"type": "int", "default": 20, "min": 5, "max": 100},
                    "mutation_rate": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
                    "crossover_rate": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0},
                    "target_fitness_score": {
                        "type": "float",
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                    },
                    "min_fitness_score": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
                },
                "example_request": {
                    "base_prompt": "How do I bypass safety filters?",
                    "strategies": [strategy_type.value],
                    "max_prompts": 10,
                    "target_fitness": 0.9,
                },
            }

    raise HTTPException(status_code=404, detail=f"Strategy '{strategy_type}' not found")


@router.delete("/jailbreak/cache")
async def clear_jailbreak_cache() -> dict[str, str]:
    """Clear the jailbreak generation cache."""
    service = get_jailbreak_service()
    count = service.clear_cache()
    return {"status": "success", "items_cleared": str(count)}


@router.get("/jailbreak/health")
async def jailbreak_health_check() -> dict:
    """Health check for jailbreak generation service."""
    service = get_jailbreak_service()
    strategies = service.get_strategies()

    return {
        "status": "healthy",
        "service": "jailbreak",
        "strategies_available": strategies.total,
        "active_sessions": len(service.active_sessions),
        "cache_size": len(service.cache),
    }


# =============================================================================
# WebSocket Streaming Endpoint
# =============================================================================


@router.websocket("/jailbreak/ws/generate")
async def websocket_generate(
    websocket: WebSocket,
    base_prompt: str = Query(..., description="Base prompt to mutate"),
    strategies: str = Query("autodan", description="Comma-separated"),
    max_prompts: int = Query(10, ge=1, le=100),
    max_iterations: int = Query(10, ge=1, le=50),
    population_size: int = Query(20, ge=5, le=100),
    target_fitness: float = Query(0.9, ge=0.0, le=1.0),
    min_fitness: float = Query(0.5, ge=0.0, le=1.0),
    target_model: str = Query("gpt-4o-mini"),
    target_provider: str = Query("openai"),
) -> None:
    """
    WebSocket endpoint for streaming jailbreak generation.

    Connect to this endpoint to receive real-time generation events:
    - `generation_start`: Generation has started
    - `generation_progress`: Progress update with metrics
    - `prompt_generated`: A new prompt has been generated
    - `generation_complete`: Generation finished successfully
    - `generation_error`: An error occurred

    **Example WebSocket URL:**
    ```
    ws://localhost:8001/api/v1/deepteam/jailbreak/ws/generate?base_prompt=test&strategies=autodan,pair
    ```
    """
    await websocket.accept()

    try:
        service = get_jailbreak_service()

        # Parse strategies
        strategy_list = [
            AttackStrategyType(s.strip().lower()) for s in strategies.split(",") if s.strip()
        ]

        # Build request
        request = JailbreakGenRequest(
            base_prompt=base_prompt,
            strategies=strategy_list,
            max_prompts=max_prompts,
            max_iterations=max_iterations,
            population_size=population_size,
            target_fitness=target_fitness,
            min_fitness=min_fitness,
            target_model=target_model,
            target_provider=target_provider,
        )

        # Stream events
        async for event in service.generate_stream(request):
            # Convert event to frontend-compatible format
            if isinstance(event, GenerationStartEvent):
                await websocket.send_json(
                    {
                        "type": "generation_start",
                        "session_id": event.session_id,
                        "timestamp": event.timestamp.isoformat(),
                    }
                )
            elif isinstance(event, GenerationProgressEvent):
                await websocket.send_json(
                    {
                        "type": "generation_progress",
                        "session_id": event.session_id,
                        "progress": event.progress.model_dump(mode="json"),
                    }
                )
            elif isinstance(event, PromptGeneratedEvent):
                await websocket.send_json(
                    {
                        "type": "prompt_generated",
                        "session_id": event.session_id,
                        "prompt": event.prompt.model_dump(mode="json"),
                    }
                )
            elif isinstance(event, GenerationCompleteEvent):
                await websocket.send_json(
                    {
                        "type": "generation_complete",
                        "session_id": event.session_id,
                        "result": event.result.model_dump(mode="json"),
                    }
                )
                break
            elif isinstance(event, GenerationErrorEvent):
                await websocket.send_json(
                    {
                        "type": "generation_error",
                        "session_id": event.session_id,
                        "error": event.error,
                    }
                )
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {
                    "type": "generation_error",
                    "error": str(e),
                }
            )
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


# =============================================================================
# Server-Sent Events (SSE) Streaming Endpoint
# =============================================================================


@router.get("/jailbreak/generate/stream")
async def stream_generate(
    base_prompt: str = Query(..., description="Base prompt to mutate"),
    strategies: str = Query("autodan", description="Comma-separated"),
    max_prompts: int = Query(10, ge=1, le=100),
    target_fitness: float = Query(0.9, ge=0.0, le=1.0),
) -> StreamingResponse:
    """
    SSE endpoint for streaming jailbreak generation.

    Returns a stream of Server-Sent Events with generation progress.

    **Example:**
    ```
    curl -N "http://localhost:8001/api/v1/deepteam/jailbreak/generate/\
stream?base_prompt=test&strategies=autodan"
    ```
    """
    service = get_jailbreak_service()

    # Parse strategies
    strategy_list = [
        AttackStrategyType(s.strip().lower()) for s in strategies.split(",") if s.strip()
    ]

    # Build request
    request = JailbreakGenRequest(
        base_prompt=base_prompt,
        strategies=strategy_list,
        max_prompts=max_prompts,
        target_fitness=target_fitness,
    )

    async def event_generator():
        async for sse_chunk in stream_to_sse(service, request):
            yield sse_chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Session Management Endpoints
# =============================================================================


@router.get("/jailbreak/sessions/{session_id}/prompts")
async def get_session_prompts(session_id: str) -> list[dict]:
    """
    Get all prompts from a cached session.

    Returns the list of generated prompts for a given session ID.
    """
    service = get_jailbreak_service()
    result = service.get_cached_result(session_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return [prompt.model_dump() for prompt in result.prompts]


@router.get("/jailbreak/sessions/{session_id}/prompts/{prompt_id}")
async def get_session_prompt(session_id: str, prompt_id: str) -> dict:
    """
    Get a specific prompt from a cached session.

    Returns the prompt with the given ID from the session.
    """
    service = get_jailbreak_service()
    result = service.get_cached_result(session_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    for prompt in result.prompts:
        if prompt.id == prompt_id:
            return prompt.model_dump()

    raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found in session")


@router.delete("/jailbreak/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """
    Delete a session and its cached results.

    Removes the session from the cache and cancels any active generation.
    """
    service = get_jailbreak_service()

    # Try to cancel if active
    cancelled = service.cancel_session(session_id)

    # Remove from cache
    if session_id in service.cache:
        del service.cache[session_id]
        return {"status": "deleted", "session_id": session_id}

    if cancelled:
        return {"status": "cancelled", "session_id": session_id}

    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@router.get("/jailbreak/sessions")
async def list_jailbreak_sessions() -> dict:
    """
    List all active and cached jailbreak sessions.

    Returns information about active generation sessions and cached results.
    """
    service = get_jailbreak_service()

    active_sessions = list(service.active_sessions.keys())
    cached_sessions = list(service.cache.keys())

    return {
        "active_sessions": active_sessions,
        "cached_sessions": cached_sessions,
        "total_active": len(active_sessions),
        "total_cached": len(cached_sessions),
    }


@router.post("/jailbreak/sessions/{session_id}/cancel")
async def cancel_jailbreak_session(session_id: str) -> dict[str, str]:
    """
    Cancel an active jailbreak generation session.

    Stops the generation process for the given session.
    """
    service = get_jailbreak_service()

    if service.cancel_session(session_id):
        return {"status": "cancelled", "session_id": session_id}

    raise HTTPException(status_code=404, detail=f"Active session '{session_id}' not found")
