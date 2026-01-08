S311 Triage for `random.random()` / `random.uniform()` occurrences

Summary
- Found ~145 occurrences across the repository (probabilistic checks, sampling, jitter, and ML usages).
- Goal: classify each site so we can safely resolve `S311` warnings without breaking algorithmic behavior.

Action categories
- Replace with `secrets` (High security impact): files that influence prompt generation, ID/nonce creation, or anything that could create predictable outputs used externally (prompts, prompt enhancers, prompt generators, persona builders, exported tokens/IDs).
- Annotate `# noqa: S311` (Algorithmic, low security): algorithmic uses (genetic algorithms, evolution engines, mutation rates, ML heuristics, simulation noise) where cryptographic randomness is not required and replacing with `secrets` harms performance or reproducibility.
- Manual review (ML / numpy): places using `np.random.*` or ranges that influence training/sampling — review for reproducibility vs security tradeoffs.

Top hotspots (recommended initial classifications)

Replace with `secrets` (recommend direct code change)
- meta_prompter/refusal_bypass_optimizer.py
- meta_prompter/prompt_enhancer.py
- backend-api/app/services/deepteam/prompt_generator.py
- prometheus_unbound/app/core/persona_generator.py (and other persona/prompt modules)
- pandora_gpt.py
- backend-api/app/services/jailbreak/* (engines that generate or mutate prompts used externally)
- backend-api/app/services/autodan_* (prompt-generation / attack synthesis modules — security-sensitive)

Annotate `# noqa: S311` (algorithmic / performance-sensitive)
- backend-api/app/services/*/genetic_optimizer.py
- backend-api/app/services/*/evolution_engine.py
- backend-api/app/services/*/optimization/* (mutation strategies, gradient optimizers)
- legacy/Project_Chimera/* (historical algorithm code)
- backend-api/app/engines/* (obfuscator, token_perturbation, etc.)

Manual review (np.random / ML uses)
- Any file that uses `np.random.*` (e.g., gradient_optimizer, ensemble_aligner) — decide reproducibility vs security

Full matched list (from grep)
- meta_prompter/refusal_bypass_optimizer.py: lines with `random.random()` checks and intensity thresholds
- legacy/Project_Chimera/transformer_engine_fixed.py
- legacy/Project_Chimera/transformer_engine.py
- legacy/Project_Chimera/preset_transformers.py
- legacy/Project_Chimera/autodan_engine.py
- chimera-orchestrator/integration/service_registry.py (random.uniform used for weighting)
- backend-api/app/plugins/custom/sample_plugin.py
- backend-api/app/services/transformers/obfuscation.py
- backend-api/app/services/optimization/engine.py
- backend-api/app/services/metamorphosis_strategies.py
- backend-api/app/services/janus/core/heuristic_generator.py
- backend-api/app/services/janus/core/evolution_engine.py
- backend-api/app/services/janus/core/causal_mapper.py
- backend-api/app/services/jailbreak/evolutionary_optimizer.py
- backend-api/app/services/deepteam/prompt_generator.py
- backend-api/app/services/autodan_advanced/hierarchical_search.py
- backend-api/app/services/autodan_advanced/gradient_optimizer.py
- backend-api/app/services/autodan_x/niche_crowding.py
- backend-api/app/services/autodan_x/mutation_engine.py
- backend-api/app/services/autodan_advanced/ensemble_aligner.py
- backend-api/app/services/advanced_transformation_layers.py
- backend-api/app/services/autodan/framework_autodan_reasoning/gradient_optimizer.py
- backend-api/app/services/autodan/framework_autodan_reasoning/enhanced_execution.py
- backend-api/app/services/autodan/optimized/enhanced_lifelong_engine.py
- backend-api/app/services/autodan/optimization/mutation_strategies.py
- backend-api/app/services/autodan/optimization/gradient_optimization.py
- backend-api/app/services/autodan/optimized/enhanced_gradient_optimizer.py
- backend-api/app/services/autodan/optimized/adaptive_mutation_engine.py
- backend-api/app/services/autodan/llm/chimera_adapter.py
- backend-api/app/services/autodan/engines/genetic_optimizer.py
- backend-api/app/services/autodan/engines/genetic_optimizer_complete.py
- backend-api/app/services/autodan/engines/token_perturbation.py
- backend-api/app/engines/obfuscator.py
- backend-api/app/engines/preset_transformers.py
- backend-api/app/engines/autodan_engine.py
- backend-api/app/engines/autodan_turbo/neural_bypass.py
- backend-api/app/engines/autodan_turbo/* (many lines)

(If you want the literal file:line list, I can add it to this document.)

Recommended next step
1. Confirm classification strategy: should I (A) apply repo-wide `# noqa: S311` annotations for algorithmic files and replace in high-risk files automatically, or (B) produce a per-file patch bundle for your review before applying changes?  
2. If you choose (A), I will:  
   - Apply `secrets` replacements in the high-risk list and add `import secrets` as needed.  
   - Insert `# noqa: S311` for algorithmic files.  
   - Re-run `ruff check .` and update the TODOs.  
3. If you choose (B), I'll create a patch bundle (preview) and wait for your approval.

Next: confirm preferred option and whether to include the full file:line list in this triage file.
