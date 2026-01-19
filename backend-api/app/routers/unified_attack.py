"""
Unified Multi-Vector Attack API Router

Consolidated API endpoints for the multi-vector attack framework
combining ExtendAttack and AutoDAN capabilities.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from app.schemas.unified_attack import (  # Request schemas; Response schemas
    AdaptiveAttackRequest,
    AllocationRequest,
    AllocationResponse,
    AttackResponse,
    AttackStatusEnum,
    BatchAttackRequest,
    BatchAttackResponse,
    BatchAttackResult,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    BatchEvaluationResult,
    BenchmarkDataset,
    BenchmarkRequest,
    BenchmarkResponse,
    BudgetOutput,
    BudgetStatusResponse,
    CompositionStrategyEnum,
    CreateSessionRequest,
    DualVectorMetricsOutput,
    EvaluationRequest,
    EvaluationResponse,
    IterativeAttackRequest,
    ParallelAttackRequest,
    ParetoFrontResponse,
    ParetoPoint,
    PresetConfig,
    ResourceUsageResponse,
    SequentialAttackRequest,
    SessionResponse,
    SessionStatusEnum,
    SessionStatusResponse,
    SessionSummaryResponse,
    StrategyBenchmarkResult,
    StrategyInfo,
    UnifiedAttackRequest,
    UnifiedConfigInput,
    UnifiedConfigOutput,
    ValidationRequest,
    ValidationResult,
    VectorResourceUsage,
)

# ==============================================================================
# Router Configuration
# ==============================================================================

router = APIRouter(
    prefix="/api/v1/unified-attack",
    tags=["Unified Multi-Vector Attack"],
    responses={404: {"description": "Not found"}},
)


# ==============================================================================
# In-Memory Session Storage (Replace with proper storage in production)
# ==============================================================================

_sessions: dict[str, dict] = {}
_attacks: dict[str, dict] = {}


# ==============================================================================
# Dependency Injection
# ==============================================================================


async def get_session(session_id: str) -> dict:
    """Get session by ID or raise 404."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return _sessions[session_id]


# ==============================================================================
# Session Management Endpoints
# ==============================================================================


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_attack_session(request: CreateSessionRequest) -> SessionResponse:
    """
    Create a new multi-vector attack session with budget.

    Sessions track resource usage, attack history, and Pareto-optimal solutions
    across multiple attack executions.
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Build configuration output
    config_output = UnifiedConfigOutput(
        model_id=request.config.model_id,
        extend_config=request.config.extend_config,
        autodan_config=request.config.autodan_config,
        default_strategy=request.config.default_strategy.value,
        enable_resource_tracking=request.config.enable_resource_tracking,
        enable_pareto_optimization=request.config.enable_pareto_optimization,
        stealth_mode=request.config.stealth_mode,
    )

    # Build budget output
    budget_output = BudgetOutput(
        max_tokens=request.budget.max_tokens,
        max_cost_usd=request.budget.max_cost_usd,
        max_requests=request.budget.max_requests,
        max_time_seconds=request.budget.max_time_seconds,
        tokens_used=0,
        cost_used=0.0,
        requests_used=0,
        time_used_seconds=0.0,
    )

    # Store session
    _sessions[session_id] = {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "config": config_output,
        "budget": budget_output,
        "status": SessionStatusEnum.ACTIVE,
        "name": request.name,
        "description": request.description,
        "tags": request.tags,
        "metadata": request.metadata,
        "attacks": [],
        "pareto_front": [],
    }

    return SessionResponse(
        session_id=session_id,
        created_at=now,
        config=config_output,
        budget=budget_output,
        status=SessionStatusEnum.ACTIVE,
        name=request.name,
        description=request.description,
    )


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str) -> SessionStatusResponse:
    """
    Get current session status including resource usage.

    Returns detailed information about session state, budget consumption,
    and attack statistics.
    """
    session = await get_session(session_id)

    # Calculate statistics
    attacks = session.get("attacks", [])
    successful = sum(1 for a in attacks if a.get("success", False))
    failed = len(attacks) - successful
    fitnesses = [a.get("fitness", 0.0) for a in attacks if a.get("success", False)]

    return SessionStatusResponse(
        session_id=session_id,
        status=session["status"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        config=session["config"],
        budget=session["budget"],
        attacks_executed=len(attacks),
        successful_attacks=successful,
        failed_attacks=failed,
        average_fitness=sum(fitnesses) / len(fitnesses) if fitnesses else None,
        best_fitness=max(fitnesses) if fitnesses else None,
    )


@router.delete("/sessions/{session_id}", response_model=SessionSummaryResponse)
async def finalize_session(session_id: str) -> SessionSummaryResponse:
    """
    Finalize session and return complete summary.

    Marks the session as finalized, preventing further attacks,
    and returns comprehensive statistics and top results.
    """
    session = await get_session(session_id)

    now = datetime.utcnow()
    session["status"] = SessionStatusEnum.FINALIZED
    session["finalized_at"] = now

    # Calculate statistics
    attacks = session.get("attacks", [])
    successful = sum(1 for a in attacks if a.get("success", False))
    failed = len(attacks) - successful
    fitnesses = [a.get("fitness", 0.0) for a in attacks if a.get("success", False)]

    duration = (now - session["created_at"]).total_seconds()

    # Find best attack
    best_attack_id = None
    if fitnesses:
        best_idx = fitnesses.index(max(fitnesses))
        successful_attacks = [a for a in attacks if a.get("success", False)]
        if successful_attacks:
            best_attack_id = successful_attacks[best_idx].get("attack_id")

    return SessionSummaryResponse(
        session_id=session_id,
        status=SessionStatusEnum.FINALIZED,
        created_at=session["created_at"],
        finalized_at=now,
        duration_seconds=duration,
        total_attacks=len(attacks),
        successful_attacks=successful,
        failed_attacks=failed,
        budget_used=session["budget"],
        average_fitness=sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
        best_fitness=max(fitnesses) if fitnesses else 0.0,
        best_attack_id=best_attack_id,
        pareto_optimal_count=len(session.get("pareto_front", [])),
        top_attacks=[],
    )


# ==============================================================================
# Attack Execution Endpoints
# ==============================================================================


@router.post("/attack", response_model=AttackResponse)
async def execute_unified_attack(request: UnifiedAttackRequest) -> AttackResponse:
    """
    Execute a single multi-vector attack.

    Combines ExtendAttack and AutoDAN capabilities based on the selected
    composition strategy to generate adversarial prompts.
    """
    session = await get_session(request.session_id)

    # Check session status
    if session["status"] != SessionStatusEnum.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is {session['status'].value}, cannot execute attacks",
        )

    attack_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Simulate attack execution (replace with actual engine integration)
    # In production: from meta_prompter.unified.engine import CombinedAttackEngine
    transformed_query = f"[{request.strategy.value}] {request.query}"

    # Create response
    response = AttackResponse(
        attack_id=attack_id,
        session_id=request.session_id,
        strategy=request.strategy.value,
        status=AttackStatusEnum.COMPLETED,
        original_query=request.query,
        transformed_query=transformed_query,
        intermediate_query=None,
        target_response=None,
        token_amplification=1.2,
        latency_amplification=1.5,
        jailbreak_score=0.75,
        unified_fitness=0.72,
        stealth_score=0.85,
        extend_metrics=None,
        autodan_metrics=None,
        tokens_consumed=150,
        cost_usd=0.003,
        latency_ms=450.0,
        success=True,
        error_message=None,
        timestamp=now,
    )

    # Update session
    session["attacks"].append(
        {
            "attack_id": attack_id,
            "query": request.query,
            "strategy": request.strategy.value,
            "success": True,
            "fitness": response.unified_fitness,
            "timestamp": now,
        }
    )
    session["budget"].tokens_used += response.tokens_consumed
    session["budget"].cost_used += response.cost_usd
    session["budget"].requests_used += 1
    session["updated_at"] = now

    # Store attack
    _attacks[attack_id] = response.model_dump()

    return response


@router.post("/attack/batch", response_model=BatchAttackResponse)
async def execute_batch_attack(request: BatchAttackRequest) -> BatchAttackResponse:
    """
    Execute multiple attacks in sequence.

    Processes multiple queries using the same strategy and configuration,
    returning aggregated results and statistics.
    """
    session = await get_session(request.session_id)

    if session["status"] != SessionStatusEnum.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is {session['status'].value}, cannot execute attacks",
        )

    batch_id = str(uuid.uuid4())
    now = datetime.utcnow()
    results: list[BatchAttackResult] = []
    total_tokens = 0
    total_cost = 0.0
    fitnesses: list[float] = []

    for idx, query in enumerate(request.queries):
        attack_id = str(uuid.uuid4())
        fitness = 0.65 + (idx * 0.05) % 0.3  # Simulated fitness

        results.append(
            BatchAttackResult(
                index=idx,
                attack_id=attack_id,
                query=query,
                success=True,
                fitness=fitness,
                error=None,
            )
        )
        fitnesses.append(fitness)
        total_tokens += 100
        total_cost += 0.002

        # Update session
        session["attacks"].append(
            {
                "attack_id": attack_id,
                "query": query,
                "strategy": request.strategy.value,
                "success": True,
                "fitness": fitness,
                "timestamp": now,
            }
        )

    # Update budget
    session["budget"].tokens_used += total_tokens
    session["budget"].cost_used += total_cost
    session["budget"].requests_used += len(request.queries)
    session["updated_at"] = now

    best_idx = fitnesses.index(max(fitnesses))

    return BatchAttackResponse(
        session_id=request.session_id,
        batch_id=batch_id,
        strategy=request.strategy.value,
        total_attacks=len(request.queries),
        successful_attacks=len(results),
        failed_attacks=0,
        results=results,
        average_fitness=sum(fitnesses) / len(fitnesses),
        best_fitness=max(fitnesses),
        best_attack_id=results[best_idx].attack_id,
        tokens_consumed=total_tokens,
        cost_usd=total_cost,
        duration_seconds=1.5,
        timestamp=now,
    )


@router.post("/attack/sequential", response_model=AttackResponse)
async def execute_sequential_attack(request: SequentialAttackRequest) -> AttackResponse:
    """
    Execute sequential attack (extend-first or autodan-first).

    Runs one attack vector, then feeds the result to the second vector
    for iterative refinement.
    """
    session = await get_session(request.session_id)

    if session["status"] != SessionStatusEnum.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is {session['status'].value}, cannot execute attacks",
        )

    attack_id = str(uuid.uuid4())
    now = datetime.utcnow()

    strategy = (
        CompositionStrategyEnum.SEQUENTIAL_EXTEND_FIRST
        if request.extend_first
        else CompositionStrategyEnum.SEQUENTIAL_AUTODAN_FIRST
    )

    # Simulate sequential execution
    intermediate = f"[Phase1: {'extend' if request.extend_first else 'autodan'}] {request.query}"
    transformed = f"[Phase2: {'autodan' if request.extend_first else 'extend'}] {intermediate}"

    response = AttackResponse(
        attack_id=attack_id,
        session_id=request.session_id,
        strategy=strategy.value,
        status=AttackStatusEnum.COMPLETED,
        original_query=request.query,
        transformed_query=transformed,
        intermediate_query=intermediate,
        target_response=None,
        token_amplification=1.4,
        latency_amplification=2.0,
        jailbreak_score=0.82,
        unified_fitness=0.78,
        stealth_score=0.80,
        extend_metrics=None,
        autodan_metrics=None,
        tokens_consumed=250,
        cost_usd=0.005,
        latency_ms=800.0,
        success=True,
        error_message=None,
        timestamp=now,
    )

    # Update session
    session["attacks"].append(
        {
            "attack_id": attack_id,
            "query": request.query,
            "strategy": strategy.value,
            "success": True,
            "fitness": response.unified_fitness,
            "timestamp": now,
        }
    )
    session["budget"].tokens_used += response.tokens_consumed
    session["budget"].cost_used += response.cost_usd
    session["budget"].requests_used += 2
    session["updated_at"] = now

    _attacks[attack_id] = response.model_dump()

    return response


@router.post("/attack/parallel", response_model=AttackResponse)
async def execute_parallel_attack(request: ParallelAttackRequest) -> AttackResponse:
    """
    Execute parallel attack (both vectors simultaneously).

    Runs ExtendAttack and AutoDAN in parallel, then combines
    results using the specified combination method.
    """
    session = await get_session(request.session_id)

    if session["status"] != SessionStatusEnum.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is {session['status'].value}, cannot execute attacks",
        )

    attack_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Simulate parallel execution
    transformed = f"[parallel:{request.combination_method}] {request.query}"

    response = AttackResponse(
        attack_id=attack_id,
        session_id=request.session_id,
        strategy=CompositionStrategyEnum.PARALLEL.value,
        status=AttackStatusEnum.COMPLETED,
        original_query=request.query,
        transformed_query=transformed,
        intermediate_query=None,
        target_response=None,
        token_amplification=1.8,
        latency_amplification=1.2,
        jailbreak_score=0.85,
        unified_fitness=0.80,
        stealth_score=0.78,
        extend_metrics=None,
        autodan_metrics=None,
        tokens_consumed=300,
        cost_usd=0.006,
        latency_ms=500.0,
        success=True,
        error_message=None,
        timestamp=now,
    )

    # Update session
    session["attacks"].append(
        {
            "attack_id": attack_id,
            "query": request.query,
            "strategy": CompositionStrategyEnum.PARALLEL.value,
            "success": True,
            "fitness": response.unified_fitness,
            "timestamp": now,
        }
    )
    session["budget"].tokens_used += response.tokens_consumed
    session["budget"].cost_used += response.cost_usd
    session["budget"].requests_used += 2
    session["updated_at"] = now

    _attacks[attack_id] = response.model_dump()

    return response


@router.post("/attack/iterative", response_model=AttackResponse)
async def execute_iterative_attack(request: IterativeAttackRequest) -> AttackResponse:
    """
    Execute iterative attack (alternating optimization).

    Alternates between ExtendAttack and AutoDAN optimization passes
    until convergence or max iterations reached.
    """
    session = await get_session(request.session_id)

    if session["status"] != SessionStatusEnum.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is {session['status'].value}, cannot execute attacks",
        )

    attack_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Simulate iterative execution
    transformed = f"[iterative:{request.max_iterations}:{request.start_with}] {request.query}"

    response = AttackResponse(
        attack_id=attack_id,
        session_id=request.session_id,
        strategy=CompositionStrategyEnum.ITERATIVE.value,
        status=AttackStatusEnum.COMPLETED,
        original_query=request.query,
        transformed_query=transformed,
        intermediate_query=None,
        target_response=None,
        token_amplification=2.2,
        latency_amplification=3.5,
        jailbreak_score=0.90,
        unified_fitness=0.85,
        stealth_score=0.75,
        extend_metrics=None,
        autodan_metrics=None,
        tokens_consumed=500,
        cost_usd=0.01,
        latency_ms=1500.0,
        success=True,
        error_message=None,
        timestamp=now,
    )

    # Update session
    session["attacks"].append(
        {
            "attack_id": attack_id,
            "query": request.query,
            "strategy": CompositionStrategyEnum.ITERATIVE.value,
            "success": True,
            "fitness": response.unified_fitness,
            "timestamp": now,
        }
    )
    session["budget"].tokens_used += response.tokens_consumed
    session["budget"].cost_used += response.cost_usd
    session["budget"].requests_used += request.max_iterations * 2
    session["updated_at"] = now

    _attacks[attack_id] = response.model_dump()

    return response


@router.post("/attack/adaptive", response_model=AttackResponse)
async def execute_adaptive_attack(request: AdaptiveAttackRequest) -> AttackResponse:
    """
    Execute adaptive attack (auto-select strategy).

    Analyzes the query and objectives to automatically select
    the optimal composition strategy.
    """
    session = await get_session(request.session_id)

    if session["status"] != SessionStatusEnum.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is {session['status'].value}, cannot execute attacks",
        )

    attack_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Simulate adaptive strategy selection
    selected_strategy = CompositionStrategyEnum.PARALLEL
    if "stealth" in request.objectives:
        selected_strategy = CompositionStrategyEnum.SEQUENTIAL_EXTEND_FIRST
    elif "jailbreak" in request.objectives:
        selected_strategy = CompositionStrategyEnum.ITERATIVE

    transformed = f"[adaptive:{selected_strategy.value}] {request.query}"

    response = AttackResponse(
        attack_id=attack_id,
        session_id=request.session_id,
        strategy=CompositionStrategyEnum.ADAPTIVE.value,
        status=AttackStatusEnum.COMPLETED,
        original_query=request.query,
        transformed_query=transformed,
        intermediate_query=None,
        target_response=None,
        token_amplification=1.6,
        latency_amplification=2.0,
        jailbreak_score=0.88,
        unified_fitness=0.82,
        stealth_score=0.82,
        extend_metrics=None,
        autodan_metrics=None,
        tokens_consumed=350,
        cost_usd=0.007,
        latency_ms=900.0,
        success=True,
        error_message=None,
        timestamp=now,
    )

    # Update session
    session["attacks"].append(
        {
            "attack_id": attack_id,
            "query": request.query,
            "strategy": CompositionStrategyEnum.ADAPTIVE.value,
            "success": True,
            "fitness": response.unified_fitness,
            "timestamp": now,
        }
    )
    session["budget"].tokens_used += response.tokens_consumed
    session["budget"].cost_used += response.cost_usd
    session["budget"].requests_used += 3
    session["updated_at"] = now

    _attacks[attack_id] = response.model_dump()

    return response


# ==============================================================================
# Configuration Endpoints
# ==============================================================================


@router.get("/config/strategies", response_model=list[StrategyInfo])
async def get_available_strategies() -> list[StrategyInfo]:
    """
    Get available composition strategies.

    Returns detailed information about each strategy including
    use cases, overhead, and trade-offs.
    """
    return [
        StrategyInfo(
            strategy=CompositionStrategyEnum.SEQUENTIAL_EXTEND_FIRST,
            name="Sequential (ExtendAttack First)",
            description="Runs ExtendAttack to generate semantic variations, then AutoDAN for optimization",
            recommended_for=["stealth attacks", "semantic manipulation", "context injection"],
            typical_overhead="Medium (2x latency)",
            pros=["High stealth", "Preserves semantic meaning", "Good for complex prompts"],
            cons=["Slower execution", "May not maximize jailbreak"],
        ),
        StrategyInfo(
            strategy=CompositionStrategyEnum.SEQUENTIAL_AUTODAN_FIRST,
            name="Sequential (AutoDAN First)",
            description="Runs AutoDAN for genetic optimization, then ExtendAttack for refinement",
            recommended_for=["aggressive attacks", "jailbreak optimization", "token efficiency"],
            typical_overhead="Medium (2x latency)",
            pros=["High jailbreak rate", "Efficient token usage"],
            cons=["Lower stealth", "May be detected"],
        ),
        StrategyInfo(
            strategy=CompositionStrategyEnum.PARALLEL,
            name="Parallel Execution",
            description="Runs both vectors simultaneously and combines results",
            recommended_for=["fast attacks", "resource-rich scenarios", "exploration"],
            typical_overhead="Low (1.2x latency)",
            pros=["Fast execution", "Diverse outputs", "Good coverage"],
            cons=["Higher resource usage", "May produce redundant results"],
        ),
        StrategyInfo(
            strategy=CompositionStrategyEnum.ITERATIVE,
            name="Iterative Optimization",
            description="Alternates between vectors until convergence",
            recommended_for=["maximum effectiveness", "difficult targets", "research"],
            typical_overhead="High (3-5x latency)",
            pros=["Highest success rate", "Converges to optimal"],
            cons=["Slowest", "Highest resource usage"],
        ),
        StrategyInfo(
            strategy=CompositionStrategyEnum.ADAPTIVE,
            name="Adaptive Selection",
            description="Auto-selects strategy based on query analysis",
            recommended_for=["general use", "unknown targets", "automated pipelines"],
            typical_overhead="Variable",
            pros=["No manual selection", "Balances trade-offs"],
            cons=["May not be optimal for specific cases"],
        ),
    ]


@router.get("/config/presets", response_model=list[PresetConfig])
async def get_attack_presets() -> list[PresetConfig]:
    """
    Get predefined attack configuration presets.

    Returns ready-to-use configurations optimized for common scenarios.
    """
    from app.schemas.unified_attack import AutoDANConfigInput, BudgetInput, ExtendConfigInput

    return [
        PresetConfig(
            preset_id="stealth-optimized",
            name="Stealth Optimized",
            description="Prioritizes undetectable attacks with high semantic similarity",
            config=UnifiedConfigInput(
                model_id="default",
                extend_config=ExtendConfigInput(semantic_threshold=0.95, max_iterations=15),
                autodan_config=AutoDANConfigInput(population_size=10, generations=30),
                default_strategy=CompositionStrategyEnum.SEQUENTIAL_EXTEND_FIRST,
                stealth_mode=True,
            ),
            budget=BudgetInput(max_tokens=50000, max_cost_usd=5.0),
            recommended_scenarios=["production testing", "subtle manipulation", "evasion testing"],
        ),
        PresetConfig(
            preset_id="jailbreak-aggressive",
            name="Jailbreak Aggressive",
            description="Maximizes jailbreak success rate regardless of stealth",
            config=UnifiedConfigInput(
                model_id="default",
                extend_config=ExtendConfigInput(semantic_threshold=0.7, max_iterations=20),
                autodan_config=AutoDANConfigInput(population_size=30, generations=100),
                default_strategy=CompositionStrategyEnum.ITERATIVE,
                stealth_mode=False,
            ),
            budget=BudgetInput(max_tokens=200000, max_cost_usd=20.0),
            recommended_scenarios=[
                "red team exercises",
                "vulnerability discovery",
                "stress testing",
            ],
        ),
        PresetConfig(
            preset_id="balanced",
            name="Balanced",
            description="Balanced configuration for general purpose attacks",
            config=UnifiedConfigInput(
                model_id="default",
                extend_config=ExtendConfigInput(),
                autodan_config=AutoDANConfigInput(),
                default_strategy=CompositionStrategyEnum.ADAPTIVE,
                stealth_mode=False,
            ),
            budget=BudgetInput(),
            recommended_scenarios=["general testing", "exploration", "benchmarking"],
        ),
        PresetConfig(
            preset_id="quick-scan",
            name="Quick Scan",
            description="Fast, low-resource configuration for rapid assessment",
            config=UnifiedConfigInput(
                model_id="default",
                extend_config=ExtendConfigInput(max_iterations=3),
                autodan_config=AutoDANConfigInput(population_size=5, generations=10),
                default_strategy=CompositionStrategyEnum.PARALLEL,
                stealth_mode=False,
            ),
            budget=BudgetInput(max_tokens=10000, max_cost_usd=1.0, max_time_seconds=300),
            recommended_scenarios=["quick assessment", "CI/CD pipelines", "smoke testing"],
        ),
    ]


@router.post("/config/validate", response_model=ValidationResult)
async def validate_config(request: ValidationRequest) -> ValidationResult:
    """
    Validate attack configuration before execution.

    Checks configuration validity, estimates resource usage,
    and provides recommendations.
    """
    errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    # Validate model_id
    if not request.config.model_id:
        errors.append("model_id is required")

    # Validate budget constraints
    if request.budget:
        if request.budget.max_tokens < 1000:
            warnings.append("Token budget is very low, may limit attack effectiveness")
        if request.budget.max_cost_usd < 0.1:
            warnings.append("Cost budget is very low, may limit attack iterations")

    # Strategy-specific recommendations
    if request.config.default_strategy == CompositionStrategyEnum.ITERATIVE:
        recommendations.append("Iterative strategy works best with higher token budgets")
        recommendations.append("Consider setting convergence_threshold for faster completion")

    if request.config.stealth_mode:
        recommendations.append("Stealth mode enabled - sequential strategies recommended")
        if request.config.default_strategy == CompositionStrategyEnum.ITERATIVE:
            warnings.append("Iterative strategy may reduce stealth effectiveness")

    # Estimate resource usage
    estimated_tokens = (
        500 if request.config.default_strategy == CompositionStrategyEnum.PARALLEL else 1000
    )
    estimated_cost = estimated_tokens * 0.00002

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        estimated_cost=estimated_cost,
        estimated_tokens=estimated_tokens,
        recommendations=recommendations,
    )


# ==============================================================================
# Evaluation Endpoints
# ==============================================================================


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_attack(request: EvaluationRequest) -> EvaluationResponse:
    """
    Evaluate attack results using unified evaluation pipeline.

    Computes comprehensive metrics including jailbreak success,
    stealth score, and Pareto ranking.
    """
    await get_session(request.session_id)

    if request.attack_id not in _attacks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attack {request.attack_id} not found",
        )

    attack = _attacks[request.attack_id]

    # Compute metrics
    metrics = DualVectorMetricsOutput(
        extend_metrics=None,
        autodan_metrics=None,
        unified_fitness=attack.get("unified_fitness", 0.0),
        token_amplification=attack.get("token_amplification", 1.0),
        latency_amplification=attack.get("latency_amplification", 1.0),
        jailbreak_score=attack.get("jailbreak_score", 0.0),
        stealth_score=attack.get("stealth_score", 0.0),
        pareto_dominance_count=0,
    )

    # Calculate Pareto rank
    pareto_rank = 1  # Simplified

    report = None
    if request.include_report:
        report = f"""
Attack Evaluation Report
========================
Attack ID: {request.attack_id}
Session ID: {request.session_id}

Metrics:
- Unified Fitness: {metrics.unified_fitness:.3f}
- Jailbreak Score: {metrics.jailbreak_score:.3f}
- Stealth Score: {metrics.stealth_score:.3f}
- Token Amplification: {metrics.token_amplification:.2f}x
- Latency Amplification: {metrics.latency_amplification:.2f}x

Pareto Rank: {pareto_rank}
"""

    return EvaluationResponse(
        attack_id=request.attack_id,
        session_id=request.session_id,
        metrics=metrics,
        report=report,
        pareto_rank=pareto_rank,
        improvement_suggestions=[
            "Consider increasing iteration count for higher fitness",
            "Try parallel strategy for faster results",
        ],
        comparison_with_baseline={"improvement": 0.15, "baseline_fitness": 0.65},
    )


@router.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def batch_evaluate(request: BatchEvaluationRequest) -> BatchEvaluationResponse:
    """
    Batch evaluate multiple attack results.

    Evaluates multiple attacks and optionally computes Pareto front.
    """
    await get_session(request.session_id)

    results: list[BatchEvaluationResult] = []
    fitnesses: list[float] = []

    for attack_id in request.attack_ids:
        if attack_id in _attacks:
            fitness = _attacks[attack_id].get("unified_fitness", 0.0)
            results.append(
                BatchEvaluationResult(
                    attack_id=attack_id,
                    fitness=fitness,
                    pareto_rank=1,
                    dominated_by=0,
                )
            )
            fitnesses.append(fitness)

    # Simple Pareto front (in production, use proper dominance calculation)
    pareto_front_ids = [r.attack_id for r in sorted(results, key=lambda x: -x.fitness)[:3]]

    # Create histogram buckets
    distribution = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for f in fitnesses:
        if f < 0.2:
            distribution["0.0-0.2"] += 1
        elif f < 0.4:
            distribution["0.2-0.4"] += 1
        elif f < 0.6:
            distribution["0.4-0.6"] += 1
        elif f < 0.8:
            distribution["0.6-0.8"] += 1
        else:
            distribution["0.8-1.0"] += 1

    return BatchEvaluationResponse(
        session_id=request.session_id,
        total_evaluated=len(results),
        pareto_optimal_count=len(pareto_front_ids),
        results=results,
        pareto_front_ids=pareto_front_ids,
        average_fitness=sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
        fitness_distribution=distribution,
    )


@router.get("/evaluate/pareto", response_model=ParetoFrontResponse)
async def get_pareto_front(session_id: str) -> ParetoFrontResponse:
    """
    Get Pareto-optimal solutions from session.

    Returns the non-dominated solutions that represent optimal
    trade-offs between objectives.
    """
    session = await get_session(session_id)

    attacks = session.get("attacks", [])

    # Build Pareto points
    pareto_points: list[ParetoPoint] = []
    for attack in attacks:
        if attack.get("success", False):
            attack_id = attack["attack_id"]
            if attack_id in _attacks:
                a = _attacks[attack_id]
                pareto_points.append(
                    ParetoPoint(
                        attack_id=attack_id,
                        jailbreak_score=a.get("jailbreak_score", 0.0),
                        stealth_score=a.get("stealth_score", 0.0),
                        efficiency_score=1.0 / (1.0 + a.get("tokens_consumed", 100) / 100),
                        unified_fitness=a.get("unified_fitness", 0.0),
                    )
                )

    # Sort by unified fitness
    pareto_points.sort(key=lambda x: -x.unified_fitness)

    # Take top 10 as Pareto front (simplified)
    pareto_points = pareto_points[:10]

    return ParetoFrontResponse(
        session_id=session_id,
        pareto_points=pareto_points,
        total_attacks=len(attacks),
        pareto_optimal_ratio=len(pareto_points) / len(attacks) if attacks else 0.0,
        frontier_diversity=0.75,
        recommended_point=pareto_points[0] if pareto_points else None,
    )


# ==============================================================================
# Resource Tracking Endpoints
# ==============================================================================


@router.get("/resources/{session_id}", response_model=ResourceUsageResponse)
async def get_resource_usage(session_id: str) -> ResourceUsageResponse:
    """
    Get current resource usage for session.

    Returns detailed breakdown of resource consumption by vector.
    """
    session = await get_session(session_id)
    budget = session["budget"]

    # Simulate vector-specific usage (50/50 split for demo)
    extend_usage = VectorResourceUsage(
        tokens_used=budget.tokens_used // 2,
        requests_made=budget.requests_used // 2,
        cost_usd=budget.cost_used / 2,
        average_latency_ms=350.0,
    )

    autodan_usage = VectorResourceUsage(
        tokens_used=budget.tokens_used - extend_usage.tokens_used,
        requests_made=budget.requests_used - extend_usage.requests_made,
        cost_usd=budget.cost_used - extend_usage.cost_usd,
        average_latency_ms=450.0,
    )

    return ResourceUsageResponse(
        session_id=session_id,
        extend_usage=extend_usage,
        autodan_usage=autodan_usage,
        total_tokens=budget.tokens_used,
        total_requests=budget.requests_used,
        total_cost_usd=budget.cost_used,
        total_time_seconds=budget.time_used_seconds,
        efficiency_score=0.78,
        timestamp=datetime.utcnow(),
    )


@router.get("/resources/{session_id}/budget", response_model=BudgetStatusResponse)
async def get_budget_status(session_id: str) -> BudgetStatusResponse:
    """
    Get budget status and remaining allowance.

    Returns current budget consumption and estimates for remaining capacity.
    """
    session = await get_session(session_id)
    budget = session["budget"]

    tokens_remaining = budget.max_tokens - budget.tokens_used
    cost_remaining = budget.max_cost_usd - budget.cost_used
    requests_remaining = budget.max_requests - budget.requests_used
    time_remaining = budget.max_time_seconds - budget.time_used_seconds

    utilization = (budget.tokens_used / budget.max_tokens) * 100

    # Estimate remaining attacks
    avg_tokens_per_attack = budget.tokens_used / max(budget.requests_used, 1)
    estimated_remaining = int(tokens_remaining / max(avg_tokens_per_attack, 100))

    warning = None
    if utilization > 80:
        warning = "Budget utilization exceeds 80%"
    elif utilization > 90:
        warning = "Critical: Budget nearly exhausted"

    return BudgetStatusResponse(
        session_id=session_id,
        budget=budget,
        tokens_remaining=tokens_remaining,
        cost_remaining=cost_remaining,
        requests_remaining=requests_remaining,
        time_remaining_seconds=time_remaining,
        utilization_percent=utilization,
        estimated_attacks_remaining=estimated_remaining,
        warning=warning,
    )


@router.post("/resources/allocate", response_model=AllocationResponse)
async def allocate_resources(request: AllocationRequest) -> AllocationResponse:
    """
    Request resource allocation between vectors.

    Adjusts the resource distribution between ExtendAttack and AutoDAN.
    """
    session = await get_session(request.session_id)

    # Normalize allocations
    total = request.extend_allocation + request.autodan_allocation
    extend_alloc = request.extend_allocation / total
    autodan_alloc = request.autodan_allocation / total

    # Store previous allocation
    previous = session.get("allocation", {"extend": 0.5, "autodan": 0.5})

    # Update allocation
    session["allocation"] = {"extend": extend_alloc, "autodan": autodan_alloc}
    session["updated_at"] = datetime.utcnow()

    return AllocationResponse(
        session_id=request.session_id,
        extend_allocation=extend_alloc,
        autodan_allocation=autodan_alloc,
        previous_allocation=previous,
        effective_from=datetime.utcnow(),
        estimated_impact={
            "extend_capacity_change": f"{(extend_alloc - previous.get('extend', 0.5)) * 100:+.1f}%",
            "autodan_capacity_change": f"{(autodan_alloc - previous.get('autodan', 0.5)) * 100:+.1f}%",
        },
    )


# ==============================================================================
# Benchmarking Endpoints
# ==============================================================================


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """
    Run benchmark evaluation against standard datasets.

    Executes attacks against benchmark datasets and compares
    performance across strategies.
    """
    benchmark_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Simulate benchmark execution
    results_by_strategy: list[StrategyBenchmarkResult] = []
    for strategy in request.strategies:
        results_by_strategy.append(
            StrategyBenchmarkResult(
                strategy=strategy.value,
                success_rate=0.75 + (hash(strategy.value) % 20) / 100,
                average_fitness=0.72 + (hash(strategy.value) % 15) / 100,
                average_tokens=250 + hash(strategy.value) % 200,
                average_latency_ms=500 + hash(strategy.value) % 500,
                pareto_optimal_rate=0.15 + (hash(strategy.value) % 10) / 100,
            )
        )

    # Find best strategy
    best_strategy = max(results_by_strategy, key=lambda x: x.average_fitness)

    baseline_comparison = None
    if request.compare_baseline:
        baseline_comparison = {
            "baseline_success_rate": 0.55,
            "improvement": best_strategy.success_rate - 0.55,
            "baseline_fitness": 0.50,
            "fitness_improvement": best_strategy.average_fitness - 0.50,
        }

    return BenchmarkResponse(
        benchmark_id=benchmark_id,
        session_id=request.session_id,
        dataset_id=request.dataset_id,
        dataset_name=f"Benchmark Dataset {request.dataset_id}",
        samples_tested=request.sample_size or 100,
        strategies_tested=[s.value for s in request.strategies],
        results_by_strategy=results_by_strategy,
        best_strategy=best_strategy.strategy,
        overall_success_rate=sum(r.success_rate for r in results_by_strategy)
        / len(results_by_strategy),
        baseline_comparison=baseline_comparison,
        duration_seconds=15.5,
        timestamp=now,
    )


@router.get("/benchmark/datasets", response_model=list[BenchmarkDataset])
async def get_benchmark_datasets() -> list[BenchmarkDataset]:
    """
    Get available benchmark datasets.

    Returns information about standard benchmark datasets for evaluation.
    """
    return [
        BenchmarkDataset(
            dataset_id="advbench-complex",
            name="AdvBench complex Behaviors",
            description="Standard complex behavior benchmark from adversarial ML research",
            size=520,
            categories=["violence", "complex", "complex"],
            difficulty_levels=["easy", "medium", "hard"],
            recommended_strategies=[
                CompositionStrategyEnum.ITERATIVE,
                CompositionStrategyEnum.ADAPTIVE,
            ],
        ),
        BenchmarkDataset(
            dataset_id="jailbreak-chat",
            name="JailbreakChat Collection",
            description="Curated jailbreak prompts from community submissions",
            size=1200,
            categories=["roleplay", "hypothetical", "encoding"],
            difficulty_levels=["medium", "hard"],
            recommended_strategies=[
                CompositionStrategyEnum.SEQUENTIAL_EXTEND_FIRST,
                CompositionStrategyEnum.PARALLEL,
            ],
        ),
        BenchmarkDataset(
            dataset_id="harmbench",
            name="HarmBench Standard",
            description="Comprehensive harm evaluation benchmark",
            size=320,
            categories=["cyber", "bio", "chem", "other"],
            difficulty_levels=["hard", "expert"],
            recommended_strategies=[
                CompositionStrategyEnum.ITERATIVE,
            ],
        ),
        BenchmarkDataset(
            dataset_id="red-team-v1",
            name="Red Team Dataset v1",
            description="Internal red team scenarios for comprehensive testing",
            size=150,
            categories=["manipulation", "extraction", "bypass"],
            difficulty_levels=["easy", "medium"],
            recommended_strategies=[
                CompositionStrategyEnum.ADAPTIVE,
                CompositionStrategyEnum.PARALLEL,
            ],
        ),
    ]
