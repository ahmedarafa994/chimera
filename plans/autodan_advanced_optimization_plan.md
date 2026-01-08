# AutoDAN Phase 2: Advanced Optimization Implementation Plan

## Overview
Implementation of Hierarchical Genetic Search (HGS) and Coherence-Constrained Gradient Search (CCGS) as outlined in `AUTODAN_ADVANCEMENT_STRATEGY.md` to improve convergence, diversity, and stealth of adversarial prompts.

## Phase 1: Architecture Refactoring
- [x] Create `backend-api/app/services/autodan/framework/advanced_attacker.py`.
- [x] Define `AutoDANAdvanced` class inheriting from `AutoDANReasoning`.
- [x] Implement `DiversityArchive` class (Map-Elites) in `backend-api/app/services/autodan/framework/archive.py`.

## Phase 2: Hierarchical Genetic Search (HGS)
- [x] Implement **Level 1: Meta-Strategy Evolution**:
    - [x] `initialize_population` with abstract attack templates.
    - [x] `llm_mutate` for strategy templates.
- [x] Implement **Level 2: Prompt Instantiation**:
    - [x] `instantiate_prompt` converting templates to token sequences.
- [x] Implement **Dynamic Archive-Based Selection**:
    - [x] `sample_diverse_elites` using Novelty + Score selection logic.

## Phase 3: Coherence-Constrained Gradient Search (CCGS)
- [x] Implement `gradient_guided_refinement` method in `AutoDANAdvanced` (call `AdvancedAttacker`).
- [x] Implement `AdvancedAttacker.coherence_constrained_gradient_search`.
    - [x] Implement `compute_gradient` (Mock/Placeholder or real if feasible).
    - [x] Implement `candidate_filtering` (Top-K mask).
    - [x] Implement `selection_score` (Attack Strength + Perplexity Penalty).
- [x] Implement `compute_aligned_gradient` for ensemble transferability.

## Phase 4: Integration & Evaluation
- [x] Update `backend-api/app/services/autodan/pipeline.py` (via service_enhanced.py) to support `AutoDANAdvanced` instantiation.
- [x] Create verification script `verify_advanced_optimization.py` to test HGS and CCGS components.
