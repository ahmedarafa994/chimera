# Project Aegis: Chimera-AutoDan Hybrid Architecture Plan

## 1. Context
This document outlines the architectural consolidation of "Chimera" (Narrative/Persona generation) and "AutoDan" (Gradient/Genetic optimization) into a unified adversarial testing framework ("Project Aegis").

## 2. Core Modules

### Module A: Chimera Engine (Narrative Generation)
**Location:** `meta_prompter/chimera/`
- **`PersonaFactory`**: Generates high-fidelity personas with specific distinct attributes.
- **`NarrativeContextManager`**: Wraps prompts in specific scenarios (Sandbox, Fiction, Code Debugging).
- **`SemanticObfuscator`**: Rewrites malicious payloads using indirect language.

### Module B: AutoDan Engine (Optimization)
**Location:** `deepteam_autodan_integration/core/`
- **`GradientOptimizer`**: White-box optimization using gradient ascent on token embeddings.
- **`GeneticMutator`**: Black-box optimization using crossover/mutation/rephrasing primitives.

### Module C: Orchestrator (Control Plane)
**Location:** `meta_prompter/aegis_orchestrator.py`
- **`CampaignManager`**: Manages the attack lifecycle (Draft -> Optimize -> Evaluate -> Refine).
- **`TelemetryLogger`**: Records successful prompt structures for analysis.

## 3. Implementation Steps

### Step 1: Restructure `meta_prompter`
- [ ] Create `meta_prompter/chimera/` directory.
- [ ] Move/Refactor `jailbreak_enhancer.py` logic into `PersonaFactory`.
- [ ] Implement `SemanticObfuscator` class.

### Step 2: Define Interfaces
- [ ] Define `IPromptGenerator` interface (Chimera).
- [ ] Define `IOptimizer` interface (AutoDan).
- [ ] Update `AegisOrchestrator` to use these interfaces instead of hardcoded logic.

### Step 3: Gradient Bridge
- [ ] Ensure `deepteam_autodan_integration` exposes a synchronous or async `optimize()` method compatible with the Orchestrator's loop.

### Step 4: Metric System
- [ ] Implement `SafetyEvaluator` to classify responses (Safe, Leaky, Jailbroken).

## 4. Workflows

**The Attack Loop:**
1. **User** inputs `Objective` (e.g., "How to bypass X").
2. **Chimera** generates 5 varied `NarrativeTemplates`.
3. **Obfuscator** rewrites `Objective` into `Payload`.
4. **AutoDan** treats `Template + Payload` as the starting point.
5. **AutoDan** iterates 100 times to find the optimal suffix/prefix.
6. **Orchestrator** tests against Target Model.
7. **Evaluator** scores result.
8. If Success: Save to `KnowledgeBase`.
9. If Fail: Feedback loop to Chimera to change Narrative Strategy.
