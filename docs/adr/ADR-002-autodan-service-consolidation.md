# ADR-002: AutoDAN Service Module Consolidation

## Status
**Proposed** - January 2026

## Context

The AutoDAN service module at `backend-api/app/services/autodan/` has grown to 70+ files across 8 subdirectories. Analysis reveals significant code duplication, naming collisions, and architectural sprawl that impacts maintainability.

### Current Structure Problems

```
autodan/
├── service.py              # 1038 lines - AutoDANService (singleton: autodan_service)
├── service_enhanced.py     # 897 lines  - EnhancedAutoDANService (singleton: enhanced_autodan_service)
├── enhanced_service.py     # 389 lines  - EnhancedAutoDANService (COLLISION!)
├── config.py               # Base config
├── config_enhanced.py      # Enhanced config
├── framework/              # Core components (Attacker, Scorer, etc.)
├── framework_r/            # Reasoning model variants
├── framework_autodan_reasoning/  # v2.0.0 comprehensive framework
├── optimization/           # Optimization utilities
├── optimizations/          # More optimization utilities (duplicate dir)
├── optimized/              # Even more optimization utilities (third!)
├── engines/                # Engine components
└── llm/                    # LLM adapters
```

### Critical Issues

1. **Class Name Collision**: Two files define `EnhancedAutoDANService`:
   - `service_enhanced.py:50` - Full-featured with parallel processing, caching, genetic optimization
   - `enhanced_service.py:30` - Research protocols and dynamic generation focus

2. **Overlapping Functionality**:
   - Both `AutoDANService` and `EnhancedAutoDANService` implement jailbreak methods
   - Three separate optimization directories with similar content

3. **Framework Sprawl**:
   - `framework/` - Base components
   - `framework_r/` - Reasoning variants (only 3 files, could be merged)
   - `framework_autodan_reasoning/` - Comprehensive v2.0.0 (main framework)

4. **Configuration Duplication**:
   - `config.py` - Simple config
   - `config_enhanced.py` - Complex config with dataclasses

## Decision

We propose consolidating the AutoDAN module using a phased approach:

### Phase 1: Resolve Naming Collisions (Immediate)

Rename `enhanced_service.py` to `research_service.py` with class `AutoDANResearchService`:

```python
# enhanced_service.py → research_service.py
class AutoDANResearchService:  # Was: EnhancedAutoDANService
    """Research-focused AutoDAN service with protocols and dynamic generation."""
```

### Phase 2: Consolidate Framework Directories

Merge `framework_r/` into `framework_autodan_reasoning/`:

```
framework_autodan_reasoning/
├── __init__.py
├── attacker_autodan_reasoning.py
├── attacker_reasoning_model.py      # Moved from framework_r/
├── scorer_reasoning_model.py        # Moved from framework_r/
├── summarizer_reasoning_model.py    # Moved from framework_r/
└── ...
```

### Phase 3: Merge Optimization Directories

Consolidate `optimization/`, `optimizations/`, `optimized/` into single `optimization/`:

```
optimization/
├── __init__.py
├── strategies/                # Mutation strategies
├── parallelization/           # Parallel processing
├── caching/                   # FAISS, hierarchical cache
├── genetic/                   # Genetic algorithms
└── convergence/               # Convergence acceleration
```

### Phase 4: Unified Service Facade

Create a unified facade that delegates to specialized services:

```python
# service_facade.py (NEW)
class AutoDANServiceFacade:
    """Unified entry point for all AutoDAN functionality."""
    
    def __init__(self):
        self._core_service = AutoDANService()
        self._enhanced_service = EnhancedAutoDANService()  # from service_enhanced.py
        self._research_service = AutoDANResearchService()  # from research_service.py
    
    async def run_jailbreak(self, request: str, method: str = "auto", **kwargs):
        """Route to appropriate service based on method."""
        if method in ("turbo", "genetic", "hybrid", "adaptive"):
            return await self._enhanced_service.run_jailbreak(request, method, **kwargs)
        elif method == "research":
            return await self._research_service.generate_adversarial_prompt(...)
        else:
            return self._core_service.run_jailbreak(request, method, **kwargs)
```

### Phase 5: Update __init__.py Exports

```python
# autodan/__init__.py
from .service import AutoDANService, autodan_service
from .service_enhanced import EnhancedAutoDANService, enhanced_autodan_service
from .research_service import AutoDANResearchService
from .service_facade import AutoDANServiceFacade, autodan_facade

# Deprecation notice for direct imports
import warnings
def __getattr__(name):
    if name == "enhanced_service":
        warnings.warn(
            "enhanced_service is deprecated. Use AutoDANResearchService or autodan_facade.",
            DeprecationWarning,
            stacklevel=2
        )
    return globals()[name]
```

## Target Structure

```
autodan/
├── __init__.py              # Unified exports with deprecation warnings
├── service.py               # Core AutoDANService
├── service_enhanced.py      # EnhancedAutoDANService (parallel, caching, genetic)
├── research_service.py      # AutoDANResearchService (renamed from enhanced_service.py)
├── service_facade.py        # NEW: Unified facade
├── config.py                # Base config
├── config_enhanced.py       # Enhanced config
├── data/                    # Data files (unchanged)
├── engines/                 # Engine components (unchanged)
├── llm/                     # LLM adapters (unchanged)
├── framework/               # Core framework components
├── framework_autodan_reasoning/  # Main reasoning framework (merged with framework_r/)
└── optimization/            # Consolidated optimization (merged from 3 dirs)
```

File count reduction: **70+ → ~50 files** (30% reduction)

## Consequences

### Positive

1. **No Name Collisions**: Clear class naming prevents import errors
2. **Reduced Cognitive Load**: Fewer directories to navigate
3. **Single Entry Point**: Facade simplifies API usage
4. **Backward Compatibility**: Old imports work with deprecation warnings

### Negative

1. **Migration Effort**: Existing code needs updates to use facade
2. **Testing Required**: Integration tests needed after consolidation
3. **Documentation Updates**: API docs need refresh

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing imports | Deprecation warnings + re-exports in `__init__.py` |
| Service behavior changes | Comprehensive test suite before/after |
| Configuration conflicts | Unified config validation |

## Implementation Roadmap

### Sprint 1 (Week 1-2)
1. [ ] Rename `enhanced_service.py` → `research_service.py`
2. [ ] Update class name to `AutoDANResearchService`
3. [ ] Update all internal imports
4. [ ] Add deprecation warnings to `__init__.py`

### Sprint 2 (Week 3-4)
1. [ ] Create `service_facade.py` with `AutoDANServiceFacade`
2. [ ] Merge `framework_r/` into `framework_autodan_reasoning/`
3. [ ] Update imports in service files

### Sprint 3 (Week 5-6)
1. [ ] Consolidate optimization directories
2. [ ] Update all internal imports
3. [ ] Comprehensive integration testing

### Sprint 4 (Week 7-8)
1. [ ] Update API endpoints to use facade
2. [ ] Documentation updates
3. [ ] Remove deprecated code paths (after deprecation period)

## Related

- ADR-001: Frontend API Migration to TanStack Query
- Issue: Architecture Assessment - AutoDAN Module Sprawl