# AutoDAN Advanced Implementation Plan

## Goal
Implement the Hierarchical Genetic Search (HGS) and Coherence-Constrained Gradient Search (CCGS) as described in `AUTODAN_ADVANCEMENT_STRATEGY.md`. This creates the `AutoDANAdvanced` framework.

## Components to Implement

### 1. Diversity Archive (`diversity_archive.py`)
- **Purpose**: Store and manage successful and novel prompts using a Map-Elites inspired approach.
- **Features**:
    - `add(prompt, score, embedding)`: Add prompt to archive.
    - `sample_diverse_elites(k)`: Select diverse parents.
    - Fitness sharing logic.

### 2. Gradient Optimizer (`gradient_optimizer.py`)
- **Purpose**: Handle white-box gradient calculations if surrogates are available.
- **Features**:
    - `gradient_guided_refinement(tokens, target, models)`: CCGS logic.
    - `compute_aligned_gradient(grads)`: Ensemble alignment.
    - **Fallback**: If no gradients available, return tokens as-is or apply random mutation.

### 3. AutoDAN Advanced Pipeline (`autodan_advanced.py`)
- **Purpose**: Main controller inheriting from `AutoDANReasoning`.
- **Features**:
    - `hierarchical_genetic_search(request)`: Main loop.
    - Level 1: Meta-Strategy Evolution (using LLM).
    - Level 2: Prompt Instantiation (using Gradient Optimizer or Fallback).
    - Integration with `DiversityArchive`.

### 4. Integration
- Update `__init__.py` to expose the new classes.

## Implementation Steps

- [ ] **Step 1**: Implement `DiversityArchive`
    - Create `backend-api/app/services/autodan/framework_autodan_reasoning/diversity_archive.py`.
    - Implement basic semantic distance calculation (cosine similarity).

- [ ] **Step 2**: Implement `GradientOptimizer`
    - Create `backend-api/app/services/autodan/framework_autodan_reasoning/gradient_optimizer.py`.
    - Implement the logic for CCGS (placeholder for actual torch/grad calls if dependencies missing, but structure present).
    - Ensure robust fallback for black-box models.

- [ ] **Step 3**: Implement `AutoDANAdvanced` Pipeline
    - Create `backend-api/app/services/autodan/framework_autodan_reasoning/autodan_advanced.py`.
    - Inherit from `AutoDANReasoning`.
    - Implement `hierarchical_genetic_search` overriding `test` or as a new method.

- [ ] **Step 4**: Expose and Verify
    - Update `backend-api/app/services/autodan/framework_autodan_reasoning/__init__.py`.
    - Create a simple test script `test_autodan_advanced.py` to instantiate the class and run a mock iteration.

## Success Criteria
- `AutoDANAdvanced` class can be instantiated.
- `hierarchical_genetic_search` runs without crashing (even in black-box mode).
- Diversity Archive correctly stores and retrieves items.
