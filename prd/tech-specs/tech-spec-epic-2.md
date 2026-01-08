# Technical Specification: Advanced Transformation Engine

Date: 2026-01-02
Author: BMAD USER
Epic ID: Epic 2
Status: Draft

---

## Overview

Epic 2 implements Chimera's core differentiator: a comprehensive prompt transformation engine with 20+ techniques across 8 categories, AutoDAN-Turbo adversarial optimization targeting 88.5% ASR (Attack Success Rate), and GPTFuzz mutation-based jailbreak testing. This epic provides the advanced adversarial prompting capabilities that establish Chimera as the leading LLM security testing platform.

## Objectives and Scope

**Objectives:**
- Design and implement modular transformation engine architecture
- Deliver 20+ transformation techniques across 8 categories
- Integrate AutoDAN-Turbo service with genetic algorithm optimization
- Integrate GPTFuzz service with mutation-based testing
- Support transformation execution endpoint with technique selection

**Scope:**
- 10 user stories covering architecture, basic/cognitive/obfuscation/persona/context/payload/advanced techniques, AutoDAN integration, and GPTFuzz integration
- Transformation categories: Basic, Cognitive, Obfuscation, Persona, Context, Logic, Multimodal, Agentic, Payload, Advanced
- AutoDAN optimization methods: vanilla, best-of-n, beam search, mousetrap
- GPTFuzz mutators: CrossOver, Expand, GenerateSimilar, Rephrase, Shorten
- Technique metadata: name, category, description, risk level

**Out of Scope:**
- Multi-provider integration (Epic 1)
- WebSocket real-time updates (Epic 3)
- Analytics pipeline (Epic 4)
- Cross-model strategy analysis (Epic 5)

## System Architecture Alignment

Epic 2 implements the **Transformation Engine Layer** from the solution architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformation Engine                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              TransformationService                         │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         Technique Registry (20+ techniques)          │  │ │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │  │ │
│  │  │  │  Basic   │ │ Cognitive│ │Obfuscation│ │ Persona │ │  │ │
│  │  │  │ (3)      │ │ (2)      │ │ (2)        │ │ (2)     │ │  │ │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │  │ │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │  │ │
│  │  │  │ Context  │ │  Payload │ │ Advanced  │ │Agentic/ │ │  │ │
│  │  │  │ (2)      │ │ (2)      │ │ (4)       │ │Multimod.│ │  │ │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └──────────────────────┬─────────────────────────────────────┘ │
│                         │                                        │
│  ┌──────────────────────┴─────────────────────────────────────┐ │
│  │              Advanced Frameworks Integration               │ │
│  │  ┌────────────────────┐      ┌────────────────────┐       │ │
│  │  │   AutoDAN-Turbo    │      │      GPTFuzz        │       │ │
│  │  │  • Genetic Algo    │      │  • Mutators         │       │ │
│  │  │  • Attack Methods  │      │  • MCTS Selection   │       │ │
│  │  │  • 88.5% ASR Target│      │  • Session Testing  │       │ │
│  │  └────────────────────┘      └────────────────────┘       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Key Architectural Decisions Referenced:**
- **ADR-001**: Monolithic Full-Stack Architecture
- **ADR-002**: Separate Frontend/Backend Deployment

## Detailed Design

### Services and Modules

**Backend Services:**

1. **`app/services/transformation_service.py`** - Transformation Engine Core
   ```python
   class TransformationEngine:
       technique_registry: Dict[str, TransformationTechnique]
       category_index: Dict[str, List[str]]

       def register_technique(self, technique: TransformationTechnique): ...
       async def apply(self, prompt: str, techniques: List[str], **kwargs) -> TransformationResult: ...
       async def apply_pipeline(self, prompt: str, pipeline: PipelineConfig) -> TransformationResult: ...
   ```

2. **`app/services/transformers/`** - Technique Implementations
   - `basic/` - simple, advanced, expert techniques
   - `cognitive/` - cognitive_hacking, hypothetical_scenario
   - `obfuscation/` - advanced_obfuscation, typoglycemia
   - `persona/` - hierarchical_persona, dan_persona
   - `context/` - contextual_inception, nested_context
   - `payload/` - payload_splitting, instruction_fragmentation
   - `advanced/` - quantum_exploit, deep_inception, code_chameleon, cipher

3. **`app/services/autodan/`** - AutoDAN-Turbo Integration
   - `service.py` - Main AutoDAN service with ChimeraLLMAdapter
   - `config.py` - Base configuration (population size, iterations, mutation rates)
   - `config_enhanced.py` - Enhanced configuration (attack methods, model selection)
   - `optimizer.py` - Genetic algorithm implementation
   - `mousetrap.py` - Chain of Iterative Chaos for reasoning models

4. **`app/services/gptfuzz/`** - GPTFuzz Integration
   - `service.py` - Main GPTFuzz service
   - `components.py` - Mutators: CrossOver, Expand, GenerateSimilar, Rephrase, Shorten
   - `selector.py` - MCTS selection policy
   - `session.py` - Session-based testing state management

5. **`app/api/v1/endpoints/transformation.py`** - Transformation Endpoints
   - `POST /api/v1/transform` - Transform without execution
   - `POST /api/v1/execute` - Transform and execute
   - `POST /api/v1/autodan/optimize` - AutoDAN optimization
   - `POST /api/v1/autodan/mousetrap` - Mousetrap technique
   - `POST /api/v1/gptfuzz/mutate` - GPTFuzz mutation testing

**Frontend Components:**

1. **`src/components/transformation/TechniqueSelector.tsx`** - Technique Selection
   - Category-based technique grouping
   - Multi-select with risk level indicators
   - Technique descriptions and explanations

2. **`src/components/transformation/AutoDANConfig.tsx`** - AutoDAN Configuration
   - Attack method selection (vanilla, best_of_n, beam_search, mousetrap)
   - Parameter controls (population, iterations, temperature)
   - Target model selection with reasoning model indicators

3. **`src/components/transformation/GPTFuzzConfig.tsx`** - GPTFuzz Configuration
   - Mutator selection (multi-select)
   - Session parameter controls
   - MCTS parameter configuration

### Data Models and Contracts

**Transformation Models (Pydantic):**

```python
class TransformationTechnique(Protocol):
    name: str
    category: str
    description: str
    risk_level: Literal["low", "medium", "high", "critical"]
    async def apply(self, prompt: str, **kwargs) -> str: ...

class TransformationRequest(BaseModel):
    prompt: str
    techniques: List[str]
    parameters: Dict[str, Any] = {}

class TransformationResult(BaseModel):
    original_prompt: str
    transformed_prompt: str
    techniques_applied: List[str]
    technique_chain: List[TechniqueMetadata]
    execution_time_ms: float
    warnings: List[str] = []

class AutoDANRequest(BaseModel):
    target_prompt: str
    target_provider: str
    target_model: str
    attack_method: Literal["vanilla", "best_of_n", "beam_search", "mousetrap"]
    population_size: int = Field(default=20, ge=10, le=100)
    iterations: int = Field(default=50, ge=10, le=200)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)

class GPTFuzzRequest(BaseModel):
    initial_prompt: str
    target_provider: str
    target_model: str
    mutators: List[Literal["crossover", "expand", "generate_similar", "rephrase", "shorten"]]
    session_id: Optional[str] = None
    iterations: int = Field(default=20, ge=5, le=100)
```

**Frontend TypeScript Types:**

```typescript
interface TransformationTechnique {
  name: string;
  category: TechniqueCategory;
  description: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

interface TransformationResult {
  originalPrompt: string;
  transformedPrompt: string;
  techniquesApplied: string[];
  executionTimeMs: number;
  warnings: string[];
}

interface AutoDANConfig {
  attackMethod: 'vanilla' | 'best_of_n' | 'beam_search' | 'mousetrap';
  populationSize: number;
  iterations: number;
  temperature: number;
  targetModel: string;
}
```

### APIs and Interfaces

**Transformation API Endpoints:**

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/v1/transform` | Apply transformations without execution | API Key |
| POST | `/api/v1/execute` | Transform and execute with LLM | API Key |
| GET | `/api/v1/techniques` | List all available techniques by category | API Key |
| POST | `/api/v1/autodan/optimize` | Run AutoDAN optimization | API Key |
| POST | `/api/v1/autodan/mousetrap` | Run Mousetrap technique for reasoning models | API Key |
| POST | `/api/v1/autodan/adaptive` | Run adaptive Mousetrap with model-specific tuning | API Key |
| POST | `/api/v1/gptfuzz/mutate` | Run GPTFuzz mutation testing | API Key |
| GET | `/api/v1/gptfuzz/session/{id}` | Get GPTFuzz session results | API Key |

**Technique Categories:**

| Category | Techniques | Risk Range |
|----------|-------------|------------|
| Basic | simple, advanced, expert | low - medium |
| Cognitive | cognitive_hacking, hypothetical_scenario | medium - high |
| Obfuscation | advanced_obfuscation, typoglycemia | medium - high |
| Persona | hierarchical_persona, dan_persona | high - critical |
| Context | contextual_inception, nested_context | medium - high |
| Payload | payload_splitting, instruction_fragmentation | high |
| Advanced | quantum_exploit, deep_inception, code_chameleon, cipher | critical |

### Workflows and Sequencing

**Transformation Pipeline Flow:**

```
┌─────────────┐
│ User Input  │
│  Prompt     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         Technique Selection UI                      │
│  • Category selection                               │
│  • Technique multi-select                           │
│  • Risk level warnings                              │
└──────┬──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         TransformationService                        │
│  • Validate techniques                               │
│  • Build execution pipeline                          │
│  • Apply techniques sequentially/parallel            │
└──────┬──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         Technique Execution                          │
│  • Each technique atomic with error handling         │
│  • Track technique chain metadata                    │
│  • Collect warnings and risk assessments             │
└──────┬──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         TransformationResult                         │
│  • Original vs transformed prompt                   │
│  • Applied techniques with metadata                 │
│  • Execution time and warnings                       │
└─────────────────────────────────────────────────────┘
```

**AutoDAN Optimization Flow:**

```
┌─────────────┐
│ Target      │
│ Prompt      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         AutoDAN Configuration                       │
│  • Attack method (vanilla/best_of_n/beam/mousetrap) │
│  • Target provider and model                        │
│  • Population size and iterations                   │
└──────┬──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         Genetic Algorithm Optimization               │
│  ┌──────────────────────────────────────────────┐   │
│  │  Initialize Population (random prompts)       │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Evaluate Fitness (LLM response via          │   │
│  │                   ChimeraLLMAdapter)         │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Selection (best performers)                 │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Crossover & Mutation                        │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 └─────┐ (repeat for N iterations)     │
│                       │                               │
│                       ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │  Return Best Prompt + ASR Metrics            │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**GPTFuzz Mutation Testing Flow:**

```
┌─────────────┐
│ Initial     │
│ Prompt      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         GPTFuzz Configuration                        │
│  • Mutators selected (CrossOver, Expand, etc.)       │
│  • Target provider and model                        │
│  • Session ID (create or resume)                     │
└──────┬──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│         MCTS Exploration Loop                        │
│  ┌──────────────────────────────────────────────┐   │
│  │  Select Prompt (MCTS selection policy)        │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Apply Mutator (random from selected)        │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Test against LLM (via ChimeraLLMAdapter)    │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Update Success Rate & Session State         │   │
│  └──────────────┬───────────────────────────────┘   │
│                 │                                    │
│                 └─────┐ (repeat for N iterations)     │
│                       │                               │
│                       ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │  Return Successful Mutations + Analysis      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Non-Functional Requirements

### Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Single Technique Execution | <500ms | Per technique application |
| Multi-Technique Pipeline (5 techniques) | <2s | Sequential execution |
| AutoDAN Optimization (50 iterations) | <5min | Full optimization cycle |
| GPTFuzz Session (20 mutations) | <3min | Full mutation testing |
| Technique Metadata Lookup | <50ms | Registry query |

**Performance Optimization:**
- Async technique execution for parallel application
- Technique result caching where appropriate
- Efficient string manipulation for text transformations
- Genetic algorithm population management in memory

### Security

| Aspect | Implementation |
|--------|----------------|
| Risk Assessment | Each technique tagged with risk level |
| Warning System | High/critical techniques require confirmation |
| Audit Logging | All transformations logged with technique chain |
| Input Sanitization | Prompt validation before transformation |
| Output Filtering | Detect dangerous content in results |

**Security Considerations:**
- Techniques can be misused for malicious purposes
- Risk warnings must be prominent for high/critical techniques
- Audit trail essential for compliance and research integrity
- Technique access controlled by user permissions

### Reliability/Availability

| Metric | Target | Mechanism |
|--------|--------|-----------|
| Technique Execution Success Rate | 99%+ | Error handling and fallback |
| AutoDAN ASR Target | 88.5% | Genetic algorithm tuning |
| GPTFuzz Success Rate Tracking | Per session | Session state persistence |
| Technique Registry Availability | 100% | In-memory registry with fallback |

**Reliability Features:**
- Atomic technique execution with rollback on failure
- Graceful degradation for failed techniques
- Session persistence for AutoDAN and GPTFuzz
- Comprehensive error messages and recovery guidance

### Observability

| Aspect | Implementation |
|--------|----------------|
| Metrics | Technique usage, execution time, success rate by technique |
| Logging | Structured logs with technique chain metadata |
| Tracing | Request ID through transformation pipeline |
| Analytics | Per-technique effectiveness tracking |

**Observable Metrics:**
- `transformation_requests_total` - By technique and category
- `transformation_execution_seconds` - Per technique timing
- `autodan_asr_rate` - Attack success rate by method and model
- `gptfuzz_mutation_success_rate` - By mutator and model
- `technique_failure_rate` - Per technique error tracking

## Dependencies and Integrations

**Internal Dependencies:**
- Epic 1 (Multi-Provider Foundation) - LLM provider integration for AutoDAN and GPTFuzz
- Epic 3 (Real-Time Research Platform) - WebSocket updates for long-running optimizations

**External Dependencies:**

| Dependency | Version | Purpose |
|------------|---------|---------|
| DEAP | 1.3+ | Genetic algorithm library for AutoDAN |
| numpy | 1.24+ | Numerical operations for optimization |
| scipy | 1.10+ | Statistical operations for MCTS |

**Provider Dependencies:**
- All LLM providers from Epic 1 for optimization target models

**Research Dependencies:**
- AutoDAN research paper (ICLR 2025) for algorithm reference
- GPTFuzz research paper for mutation strategy reference

## Acceptance Criteria (Authoritative)

### Story TE-001: Transformation Architecture
- [ ] Each transformation technique is self-contained module
- [ ] Techniques grouped into logical categories
- [ ] New techniques registerable via configuration or code
- [ ] Pipeline supports sequential and parallel technique application
- [ ] Each technique has metadata (name, category, description, risk level)
- [ ] Technique execution atomic with proper error handling
- [ ] Results include applied techniques and metadata

### Story TE-002: Basic Transformation Techniques
- [ ] "simple" technique improves clarity and structure
- [ ] "advanced" technique adds domain context and expertise
- [ ] "expert" technique applies comprehensive enhancement
- [ ] Each technique maintains original intent
- [ ] Transformations reversible or trackable
- [ ] Output includes explanation of changes
- [ ] Techniques handle edge cases (empty input, malformed prompts)

### Story TE-003: Cognitive Transformation Techniques
- [ ] "cognitive_hacking" restructures reasoning patterns
- [ ] "hypothetical_scenario" embeds prompts in hypothetical contexts
- [ ] Techniques bypass standard cognitive filters
- [ ] Transformations subtle and contextually appropriate
- [ ] Multiple cognitive techniques combinable
- [ ] Output explains cognitive mechanisms applied
- [ ] Risk assessment provided for each technique

### Story TE-004: Obfuscation Transformation Techniques
- [ ] "advanced_obfuscation" applies sophisticated text hiding
- [ ] "typoglycemia" leverages visual word recognition patterns
- [ ] Obfuscation preserves semantic meaning
- [ ] Multiple obfuscation methods stackable
- [ ] De-obfuscation possible for analysis
- [ ] Techniques bypass common content filters
- [ ] Output shows original and obfuscated versions

### Story TE-005: Persona Transformation Techniques
- [ ] "hierarchical_persona" creates multi-level persona structures
- [ ] "dan_persona" applies adversarial persona patterns
- [ ] Personas consistent and believable
- [ ] Multiple personas combinable for complex scenarios
- [ ] Persona injection contextually appropriate
- [ ] Techniques include persona background and motivation
- [ ] Risk levels clearly indicated

### Story TE-006: Context Transformation Techniques
- [ ] "contextual_inception" embeds prompts in layered contexts
- [ ] "nested_context" creates recursive context structures
- [ ] Context consistent and logically coherent
- [ ] Multiple context layers supported
- [ ] Context supports scenario-based framing
- [ ] Techniques include context background and setup
- [ ] Output explains context structure applied

### Story TE-007: Payload Transformation Techniques
- [ ] "payload_splitting" divides instructions across segments
- [ ] "instruction_fragmentation" breaks instructions into fragments
- [ ] Payload reconstructs correctly when processed
- [ ] Splitting contextually appropriate
- [ ] Multiple splitting strategies available
- [ ] Techniques include recombination instructions
- [ ] Output shows split and recombined versions

### Story TE-008: Advanced Transformation Techniques
- [ ] "quantum_exploit" applies quantum-inspired prompt structures
- [ ] "deep_inception" creates recursive inception layers
- [ ] "code_chameleon" adapts prompt to code-like structures
- [ ] "cipher" applies encryption and encoding techniques
- [ ] Techniques highly sophisticated and subtle
- [ ] Techniques combine multiple lower-level techniques
- [ ] Risk assessment comprehensive
- [ ] Usage includes detailed explanation and warnings

### Story TE-009: AutoDAN-Turbo Integration
- [ ] AutoDAN uses genetic algorithms for prompt evolution
- [ ] Multiple attack methods available (vanilla, best-of-n, beam search, mousetrap)
- [ ] Optimization targets specific LLM providers and models
- [ ] ASR tracked and reported
- [ ] Results include optimized prompts and success metrics
- [ ] Configuration supports population size, iterations, method selection
- [ ] Mousetrap technique works with reasoning models
- [ ] Process completes within reasonable time (<5 minutes)

### Story TE-010: GPTFuzz Integration
- [ ] GPTFuzz applies mutation operators (CrossOver, Expand, etc.)
- [ ] MCTS selection policy guides prompt exploration
- [ ] Session-based testing maintains state across mutations
- [ ] Results track mutation success rates and patterns
- [ ] Configuration supports mutator selection and session parameters
- [ ] Testing supports configurable iterations and population size
- [ ] Results include successful mutations and analysis
- [ ] Process completes efficiently for systematic testing

## Traceability Mapping

**Requirements from PRD:**

| PRD Requirement | Epic 2 Story | Implementation |
|----------------|--------------|----------------|
| FR-04: 20+ transformation techniques | TE-001 through TE-008 | All technique categories |
| FR-05: AutoDAN-Turbo 88.5% ASR | TE-009 | Genetic algorithm optimization |
| FR-06: GPTFuzz mutation testing | TE-010 | MCTS-based mutation |
| NFR-06: Technique execution <2s | TE-001 | Pipeline optimization |
| NFR-10: Audit trail for transformations | TE-001 | Technique chain tracking |

**Epic-to-Architecture Mapping:**

| Architecture Component | Epic 2 Implementation |
|-----------------------|----------------------|
| Transformation Engine | All stories (TE-001 through TE-010) |
| Technique Registry | TE-001 |
| AutoDAN Service | TE-009 |
| GPTFuzz Service | TE-010 |
| Transformation API Endpoints | TE-001, TE-009, TE-010 |

## Risks, Assumptions, Open Questions

**Risks:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| Techniques bypass safety filters causing unintended harm | Critical | Risk warnings, confirmation dialogs, audit logging |
| AutoDAN optimization fails to reach 88.5% ASR target | High | Algorithm tuning, hyperparameter optimization |
| GPTFuzz mutation effectiveness varies by model | Medium | Model-specific mutator tuning |
| Technique combination causes unexpected behavior | Medium | Technique compatibility validation |

**Assumptions:**
- AutoDAN and GPTFuzz research papers provide sufficient implementation details
- Target models have consistent behavior for optimization
- Genetic algorithm parameters transfer from research to production
- MCTS selection policy effective for prompt mutation

**Open Questions:**
- Should high/critical techniques require additional authorization? → **Decision: Yes, require explicit confirmation for high/critical techniques**
- What is the maximum number of techniques allowed in a single pipeline? → **Decision: 10 techniques maximum to prevent complexity issues**
- Should failed technique applications abort entire pipeline? → **Decision: No, log failure and continue with next technique**

## Test Strategy Summary

**Unit Tests:**
- Each transformation technique in isolation
- Technique registry operations
- Pipeline execution logic
- Genetic algorithm components for AutoDAN
- MCTS selection policy for GPTFuzz

**Integration Tests:**
- Technique application against mock LLM responses
- AutoDAN optimization with test models
- GPTFuzz mutation testing with test prompts
- Multi-technique pipeline execution

**End-to-End Tests:**
- Full AutoDAN optimization cycle
- Full GPTFuzz session with multiple mutations
- Complex technique chains (10+ techniques)
- Transform + execute workflow with Epic 1 providers

**Performance Tests:**
- Technique execution timing (target <500ms per technique)
- Pipeline scaling with technique count
- AutoDAN optimization with 100+ iterations
- GPTFuzz session with 50+ mutations

**Security Tests:**
- Risk level assessment accuracy
- Dangerous content detection in results
- Audit trail completeness
- Confirmation dialog enforcement for high/critical techniques

**Test Coverage Target:** 75%+ for transformation engine, 80%+ for AutoDAN/GPTFuzz services

---

_This technical specification serves as the implementation guide for Epic 2: Advanced Transformation Engine. All development should reference this document for detailed design decisions, transformation techniques, and acceptance criteria._
