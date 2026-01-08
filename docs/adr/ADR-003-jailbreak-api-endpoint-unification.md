# ADR-003: Jailbreak API Endpoint Unification

## Status
Proposed

## Date
2026-01-02

## Context

The Chimera backend has accumulated multiple jailbreak-related API endpoints with overlapping functionality, creating confusion for API consumers and maintenance burden for developers.

### Current Endpoint Landscape

| Endpoint | Source File | Purpose | Service Layer |
|----------|-------------|---------|---------------|
| `/api/v1/jailbreak/*` | `jailbreak.py` | DeepTeam attack strategies | `JailbreakService` |
| `/api/v1/autodan/jailbreak` | `autodan.py` | AutoDAN optimization methods | `EnhancedAutoDANService` |
| `/api/v1/autodan/batch` | `autodan.py` | Batch AutoDAN generation | `EnhancedAutoDANService` |
| `/api/v1/autodan/mousetrap` | `mousetrap.py` | Chain-of-thought jailbreaking | `MousetrapService` |
| `/api/v1/autodan-enhanced/*` | `autodan_enhanced.py` | Research-focused AutoDAN | Legacy |
| `/api/v1/autodan-advanced/*` | `autodan_hierarchical.py` | Hierarchical strategies | Research |
| `/api/v1/autodan-gradient/*` | `autodan_gradient.py` | Gradient optimization | Research |
| `/api/generation/jailbreak/generate` | `api_routes.py` | AI-powered generation | `transformation_engine` |

### Problems Identified

1. **Prefix Collision**: Both `autodan.router` and `mousetrap.router` mount under `/autodan`
2. **Duplicate Functionality**: Multiple endpoints generate jailbreak prompts with different approaches
3. **Inconsistent Request/Response Models**: Each endpoint defines its own Pydantic models
4. **Service Layer Fragmentation**: 4+ different service classes handle similar operations
5. **Frontend Confusion**: UI must track multiple endpoints for related functionality

## Decision

We will consolidate jailbreak endpoints under a unified `/api/v1/adversarial/*` namespace with clear sub-routes for different attack paradigms.

### Target Architecture

```
/api/v1/adversarial/
├── /generate              # Unified generation endpoint (strategy parameter)
├── /generate/stream       # SSE streaming
├── /generate/batch        # Batch processing
├── /ws/generate           # WebSocket streaming
├── /strategies            # List available strategies
├── /strategies/{type}     # Strategy details
├── /sessions/{id}         # Session management
├── /sessions/{id}/cancel  # Cancel active session
├── /config                # System configuration
└── /health                # Service health
```

### Unified Strategy Parameter

Instead of separate endpoints, use a `strategy` parameter:

```python
class AdversarialStrategy(str, Enum):
    # AutoDAN Family
    AUTODAN = "autodan"
    AUTODAN_TURBO = "autodan_turbo"
    AUTODAN_GENETIC = "autodan_genetic"
    AUTODAN_BEAM = "autodan_beam"
    
    # DeepTeam Attacks
    PAIR = "pair"
    TAP = "tap"
    CRESCENDO = "crescendo"
    GRAY_BOX = "gray_box"
    
    # Advanced Methods
    MOUSETRAP = "mousetrap"
    GRADIENT = "gradient"
    HIERARCHICAL = "hierarchical"
    
    # Meta Strategies
    ADAPTIVE = "adaptive"  # Auto-select best strategy
```

### Unified Request Model

```python
class AdversarialGenerateRequest(BaseModel):
    """Unified request for all adversarial generation methods."""
    
    prompt: str = Field(..., min_length=1, max_length=50000)
    strategy: AdversarialStrategy = Field(default=AdversarialStrategy.ADAPTIVE)
    
    # Model selection
    provider: str | None = None
    model: str | None = None
    
    # Generation parameters
    max_prompts: int = Field(default=10, ge=1, le=100)
    max_iterations: int = Field(default=20, ge=1, le=100)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    
    # Strategy-specific (optional)
    population_size: int | None = None  # Genetic algorithms
    beam_width: int | None = None       # Beam search
    beam_depth: int | None = None       # Beam search
```

### Facade Service Pattern

Create a unified facade that delegates to specialized services:

```python
class AdversarialService:
    """Unified facade for all adversarial generation methods."""
    
    def __init__(self):
        self._autodan_service = EnhancedAutoDANService()
        self._deepteam_service = JailbreakService()
        self._mousetrap_service = MousetrapService()
        self._gradient_service = GradientOptimizationService()
    
    async def generate(
        self, 
        request: AdversarialGenerateRequest
    ) -> AdversarialGenerateResponse:
        """Route to appropriate service based on strategy."""
        strategy = request.strategy
        
        if strategy in {AdversarialStrategy.AUTODAN, ...}:
            return await self._autodan_service.generate(request)
        elif strategy in {AdversarialStrategy.PAIR, ...}:
            return await self._deepteam_service.generate(request)
        elif strategy == AdversarialStrategy.MOUSETRAP:
            return await self._mousetrap_service.generate(request)
        elif strategy == AdversarialStrategy.ADAPTIVE:
            return await self._adaptive_select_and_run(request)
```

## Implementation Plan

### Phase 1: Create Unified Endpoint (Non-Breaking)
1. Create `/api/v1/adversarial/` router with new unified endpoint
2. Implement `AdversarialService` facade
3. Add deprecation warnings to legacy endpoints

### Phase 2: Frontend Migration
1. Update frontend to use new unified endpoint
2. Update TanStack Query hooks
3. Test all generation flows

### Phase 3: Legacy Cleanup
1. Mark old endpoints as deprecated in OpenAPI
2. Add `X-Deprecated` headers to responses
3. Log usage metrics for migration tracking

### Phase 4: Removal (Future)
1. Remove deprecated endpoints after 2 release cycles
2. Clean up unused code

## Backward Compatibility

### Deprecation Strategy

```python
# Legacy endpoint with deprecation warning
@router.post("/autodan/jailbreak", deprecated=True)
async def legacy_autodan_jailbreak(
    request: JailbreakRequest,
    response: Response,
):
    """DEPRECATED: Use /api/v1/adversarial/generate instead."""
    response.headers["X-Deprecated"] = "true"
    response.headers["X-Deprecation-Notice"] = (
        "This endpoint is deprecated. "
        "Migrate to POST /api/v1/adversarial/generate"
    )
    
    # Delegate to new service
    new_request = AdversarialGenerateRequest(
        prompt=request.request,
        strategy=AdversarialStrategy.AUTODAN,
        ...
    )
    return await adversarial_service.generate(new_request)
```

### Response Envelope Consistency

All endpoints will use consistent response structure:

```python
class AdversarialGenerateResponse(BaseModel):
    """Unified response for all methods."""
    
    # Core fields (always present)
    success: bool
    session_id: str
    strategy_used: AdversarialStrategy
    
    # Generation results
    prompts: list[GeneratedPrompt]
    best_prompt: GeneratedPrompt | None
    
    # Metrics
    latency_ms: float
    iterations: int
    
    # Optional fields
    model_used: str | None = None
    provider_used: str | None = None
    cached: bool = False
    error: str | None = None
```

## Consequences

### Positive
- Single endpoint simplifies frontend integration
- Consistent request/response models
- Easier to add new strategies
- Better OpenAPI documentation
- Reduced code duplication

### Negative
- Migration effort required
- Temporary code duplication during transition
- Risk of breaking external integrations

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking external integrations | 2-release deprecation cycle with warnings |
| Performance regression | Facade adds minimal overhead; benchmark before/after |
| Missing strategy-specific features | Allow strategy-specific params in request model |

## Related ADRs
- ADR-002: AutoDAN Service Consolidation

## References
- Original assessment: Duplicate jailbreak endpoints issue
- DeepTeam attack strategies documentation
- AutoDAN ICLR 2025 paper