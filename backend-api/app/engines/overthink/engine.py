"""
OVERTHINK Engine - Main Attack Engine.

This module implements the core OVERTHINK engine for reasoning token
exploitation. It orchestrates all attack techniques and integrates
with existing Chimera systems.

Key Features:
- 9 attack techniques for reasoning token amplification
- Integration with Mousetrap chaotic reasoning
- Configurable attack parameters
- Async/await support for concurrent operations
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from .config import OverthinkConfig
from .context_injector import ContextInjector, InjectionOptimizer
from .decoy_generator import DecoyProblemGenerator
from .icl_genetic_optimizer import ICLGeneticOptimizer
from .models import (
    AttackTechnique,
    DecoyType,
    GeneticIndividual,
    InjectionStrategy,
    MousetrapIntegration,
    OverthinkRequest,
    OverthinkResult,
    OverthinkStats,
    ReasoningModel,
)
from .reasoning_scorer import (
    AmplificationAnalyzer,
    CostEstimator,
    ReasoningTokenScorer,
)

logger = logging.getLogger(__name__)


class OverthinkEngine:
    """
    Core OVERTHINK attack engine.

    Implements 9 attack techniques for reasoning token exploitation:
    - mdp_decoy: Markov Decision Process decoys
    - sudoku_decoy: Constraint satisfaction puzzles
    - counting_decoy: Nested conditional counting
    - logic_decoy: Multi-step logical inference
    - hybrid_decoy: Combined multiple decoy types
    - context_aware: Position and content-sensitive injection
    - context_agnostic: Universal template injection
    - icl_optimized: In-Context Learning enhanced attacks
    - mousetrap_enhanced: Integration with Mousetrap chaotic reasoning
    """

    def __init__(
        self,
        config: OverthinkConfig | None = None,
        llm_client: Any | None = None,
        mousetrap_engine: Any | None = None,
    ):
        """
        Initialize the OVERTHINK engine.

        Args:
            config: Engine configuration
            llm_client: LLM client for inference
            mousetrap_engine: Optional Mousetrap engine for enhanced attacks
        """
        self.config = config or OverthinkConfig()
        self.llm_client = llm_client
        self.mousetrap_engine = mousetrap_engine

        # Initialize components
        self._decoy_generator = DecoyProblemGenerator(
            self.config.decoy_config
        )
        self._context_injector = ContextInjector(
            self.config.injection_config
        )
        self._scorer = ReasoningTokenScorer(
            self.config.scoring_config,
            llm_client,
        )
        self._optimizer = ICLGeneticOptimizer(
            self.config.icl_config
        )
        self._analyzer = AmplificationAnalyzer()
        self._cost_estimator = CostEstimator()

        # Injection optimizer for learning
        self._injection_optimizer = InjectionOptimizer()

        # Attack statistics
        self._stats = OverthinkStats()
        self._attack_history: list[OverthinkResult] = []

        # Technique handlers
        self._technique_handlers = {
            AttackTechnique.MDP_DECOY: self._attack_mdp_decoy,
            AttackTechnique.SUDOKU_DECOY: self._attack_sudoku_decoy,
            AttackTechnique.COUNTING_DECOY: self._attack_counting_decoy,
            AttackTechnique.LOGIC_DECOY: self._attack_logic_decoy,
            AttackTechnique.HYBRID_DECOY: self._attack_hybrid_decoy,
            AttackTechnique.CONTEXT_AWARE: self._attack_context_aware,
            AttackTechnique.CONTEXT_AGNOSTIC: self._attack_context_agnostic,
            AttackTechnique.ICL_OPTIMIZED: self._attack_icl_optimized,
            AttackTechnique.MOUSETRAP_ENHANCED: self._attack_mousetrap_enhanced,
        }

    async def attack(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """
        Execute an OVERTHINK attack.

        Args:
            request: Attack request with configuration

        Returns:
            Attack result with metrics
        """
        start_time = datetime.utcnow()
        attack_id = str(uuid.uuid4())[:12]

        logger.info(
            f"Starting OVERTHINK attack {attack_id} "
            f"with technique {request.technique.value}"
        )

        try:
            # Get technique handler
            handler = self._technique_handlers.get(request.technique)
            if not handler:
                raise ValueError(
                    f"Unknown technique: {request.technique}"
                )

            # Execute attack
            result = await handler(request)

            # Score the result
            if self.llm_client:
                token_metrics, cost_metrics = await self._scorer.score(
                    result, request.target_model
                )
                result.token_metrics = token_metrics
                result.cost_metrics = cost_metrics

            # Update statistics
            self._update_stats(result, start_time)

            # Record for learning
            if result.token_metrics:
                self._analyzer.record(result, result.token_metrics)

            # Store in history
            self._attack_history.append(result)
            if len(self._attack_history) > 1000:
                self._attack_history = self._attack_history[-500:]

            result.success = True
            result.attack_id = attack_id

            logger.info(
                f"Attack {attack_id} completed - "
                f"amplification: {result.token_metrics.amplification_factor:.2f}x"
                if result.token_metrics else "Attack completed"
            )

            return result

        except Exception as e:
            logger.error(f"Attack {attack_id} failed: {e}")
            return OverthinkResult(
                request=request,
                success=False,
                attack_id=attack_id,
                error_message=str(e),
            )

    async def _attack_mdp_decoy(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute MDP decoy attack."""
        # Generate MDP decoys
        decoys = self._decoy_generator.generate_batch(
            [DecoyType.MDP],
            count=request.num_decoys,
        )

        # Inject into prompt
        injected = self._context_injector.inject(
            request.prompt,
            decoys,
            InjectionStrategy.CONTEXT_AWARE,
        )

        # Execute if LLM available
        response = await self._execute_llm(injected.injected_prompt, request)

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "mdp_decoy",
                "num_decoys": len(decoys),
            },
        )

    async def _attack_sudoku_decoy(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute Sudoku decoy attack."""
        decoys = self._decoy_generator.generate_batch(
            [DecoyType.SUDOKU],
            count=request.num_decoys,
        )

        injected = self._context_injector.inject(
            request.prompt,
            decoys,
            InjectionStrategy.CONTEXT_AGNOSTIC,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "sudoku_decoy",
                "num_decoys": len(decoys),
            },
        )

    async def _attack_counting_decoy(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute counting decoy attack."""
        decoys = self._decoy_generator.generate_batch(
            [DecoyType.COUNTING],
            count=request.num_decoys,
        )

        injected = self._context_injector.inject(
            request.prompt,
            decoys,
            InjectionStrategy.HYBRID,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "counting_decoy",
                "num_decoys": len(decoys),
            },
        )

    async def _attack_logic_decoy(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute logic decoy attack."""
        decoys = self._decoy_generator.generate_batch(
            [DecoyType.LOGIC],
            count=request.num_decoys,
        )

        injected = self._context_injector.inject(
            request.prompt,
            decoys,
            InjectionStrategy.CONTEXT_AWARE,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "logic_decoy",
                "num_decoys": len(decoys),
            },
        )

    async def _attack_hybrid_decoy(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute hybrid decoy attack with multiple types."""
        # Use request decoy types or default mix
        decoy_types = request.decoy_types or [
            DecoyType.MDP,
            DecoyType.LOGIC,
            DecoyType.MATH,
        ]

        decoys = self._decoy_generator.generate_batch(
            decoy_types,
            count=request.num_decoys,
        )

        # Also generate a hybrid decoy
        hybrid = self._decoy_generator.generate_hybrid(
            types=decoy_types[:2],
            complexity=0.8,
        )
        decoys.append(hybrid)

        injected = self._context_injector.inject(
            request.prompt,
            decoys,
            InjectionStrategy.HYBRID,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "hybrid_decoy",
                "decoy_types": [d.decoy_type.value for d in decoys],
                "num_decoys": len(decoys),
            },
        )

    async def _attack_context_aware(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute context-aware injection attack."""
        # Analyze prompt to select best decoy types
        prompt_analysis = self._analyze_prompt_context(request.prompt)

        # Select decoy types based on analysis
        decoy_types = self._select_decoy_types_for_context(prompt_analysis)

        decoys = self._decoy_generator.generate_batch(
            decoy_types,
            count=request.num_decoys,
        )

        # Use context-aware injection
        injected = self._context_injector.inject(
            request.prompt,
            decoys,
            InjectionStrategy.CONTEXT_AWARE,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "context_aware",
                "prompt_analysis": prompt_analysis,
                "selected_types": [dt.value for dt in decoy_types],
            },
        )

    async def _attack_context_agnostic(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute context-agnostic injection attack."""
        # Use default high-performing decoy types
        decoy_types = request.decoy_types or [
            DecoyType.COUNTING,
            DecoyType.LOGIC,
        ]

        decoys = self._decoy_generator.generate_batch(
            decoy_types,
            count=request.num_decoys,
        )

        # Use aggressive injection for maximum amplification
        injected = self._context_injector.inject(
            request.prompt,
            decoys,
            InjectionStrategy.AGGRESSIVE,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "context_agnostic",
                "injection_strategy": "aggressive",
            },
        )

    async def _attack_icl_optimized(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute ICL-optimized attack using learned patterns."""
        # Get best configuration from optimizer
        best_config = self._optimizer.get_best_configuration()

        if best_config:
            # Use learned configuration
            decoy_types = [DecoyType(dt) for dt in best_config.get(
                "decoy_types", ["logic", "math"]
            )]
            strategy = InjectionStrategy(best_config.get(
                "injection_strategy", "hybrid"
            ))
            params = best_config.get("params", {})
            num_decoys = params.get("num_decoys", request.num_decoys)
        else:
            # Initialize with good defaults
            decoy_types = [DecoyType.LOGIC, DecoyType.MATH]
            strategy = InjectionStrategy.HYBRID
            num_decoys = request.num_decoys

        # Generate ICL prefix from examples
        icl_prefix = self._optimizer.get_icl_prompt_prefix(
            technique=AttackTechnique.ICL_OPTIMIZED
        )

        decoys = self._decoy_generator.generate_batch(
            decoy_types,
            count=num_decoys,
        )

        # Prepend ICL context if available
        prompt = request.prompt
        if icl_prefix:
            prompt = icl_prefix + "\n\n" + prompt

        injected = self._context_injector.inject(
            prompt,
            decoys,
            strategy,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        result = OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            debug_info={
                "technique": "icl_optimized",
                "used_learned_config": bool(best_config),
                "icl_prefix_length": len(icl_prefix),
            },
        )

        return result

    async def _attack_mousetrap_enhanced(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """Execute Mousetrap-enhanced attack with chaotic reasoning."""
        if not self.mousetrap_engine:
            logger.warning(
                "Mousetrap engine not available, "
                "falling back to hybrid attack"
            )
            return await self._attack_hybrid_decoy(request)

        # Generate decoys
        decoys = self._decoy_generator.generate_batch(
            request.decoy_types or [DecoyType.LOGIC, DecoyType.PLANNING],
            count=request.num_decoys,
        )

        # Apply Mousetrap chaotic reasoning
        mousetrap_result = await self._apply_mousetrap_chaos(
            request.prompt, decoys
        )

        # Inject with chaotic elements
        injected = self._context_injector.inject(
            mousetrap_result["prompt"],
            decoys + mousetrap_result.get("extra_decoys", []),
            InjectionStrategy.AGGRESSIVE,
        )

        response = await self._execute_llm(injected.injected_prompt, request)

        # Create Mousetrap integration info
        mt_integration = MousetrapIntegration(
            enabled=True,
            chaos_level=mousetrap_result.get("chaos_level", 0.7),
            trigger_patterns=mousetrap_result.get("triggers", []),
            combined_score=mousetrap_result.get("score", 0.0),
        )

        return OverthinkResult(
            request=request,
            injected_prompt=injected,
            decoy_problems=decoys,
            response=response,
            mousetrap_integration=mt_integration,
            debug_info={
                "technique": "mousetrap_enhanced",
                "chaos_level": mt_integration.chaos_level,
                "num_triggers": len(mt_integration.trigger_patterns),
            },
        )

    async def _apply_mousetrap_chaos(
        self,
        prompt: str,
        decoys: list,
    ) -> dict[str, Any]:
        """Apply Mousetrap chaotic reasoning to prompt."""
        if not self.mousetrap_engine:
            return {"prompt": prompt}

        try:
            # Use Mousetrap engine to add chaos
            chaos_result = await self.mousetrap_engine.apply_chaos(
                prompt,
                level=0.7,
                context={"decoys": [d.problem_text for d in decoys]},
            )

            return {
                "prompt": chaos_result.get("chaotic_prompt", prompt),
                "chaos_level": chaos_result.get("level", 0.7),
                "triggers": chaos_result.get("triggers", []),
                "extra_decoys": chaos_result.get("extra_problems", []),
                "score": chaos_result.get("effectiveness_score", 0.0),
            }
        except Exception as e:
            logger.warning(f"Mousetrap chaos failed: {e}")
            return {"prompt": prompt}

    async def _execute_llm(
        self,
        prompt: str,
        request: OverthinkRequest,
    ) -> str:
        """Execute LLM inference with injected prompt."""
        if not self.llm_client:
            return ""

        try:
            # Use appropriate API based on model
            if request.target_model in [
                ReasoningModel.O1,
                ReasoningModel.O1_MINI,
                ReasoningModel.O3_MINI,
            ]:
                response = await self._execute_openai(prompt, request)
            elif request.target_model == ReasoningModel.DEEPSEEK_R1:
                response = await self._execute_deepseek(prompt, request)
            else:
                response = await self._execute_generic(prompt, request)

            return response

        except Exception as e:
            logger.error(f"LLM execution failed: {e}")
            return ""

    async def _execute_openai(
        self,
        prompt: str,
        request: OverthinkRequest,
    ) -> str:
        """Execute OpenAI reasoning model."""
        if hasattr(self.llm_client, "generate"):
            result = await asyncio.to_thread(
                self.llm_client.generate,
                request.system_prompt or "You are a helpful assistant.",
                prompt,
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)
        return ""

    async def _execute_deepseek(
        self,
        prompt: str,
        request: OverthinkRequest,
    ) -> str:
        """Execute DeepSeek R1 model."""
        if hasattr(self.llm_client, "generate"):
            result = await asyncio.to_thread(
                self.llm_client.generate,
                request.system_prompt or "You are a helpful reasoning assistant.",
                prompt,
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)
        return ""

    async def _execute_generic(
        self,
        prompt: str,
        request: OverthinkRequest,
    ) -> str:
        """Execute generic LLM model."""
        if hasattr(self.llm_client, "chat"):
            result = await self.llm_client.chat(prompt)
            return str(result)
        elif hasattr(self.llm_client, "generate"):
            result = await asyncio.to_thread(
                self.llm_client.generate,
                request.system_prompt or "You are a helpful assistant.",
                prompt,
            )
            return result.get("content", "") if isinstance(result, dict) else str(result)
        return ""

    def _analyze_prompt_context(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt to determine best attack strategy."""
        return {
            "length": len(prompt),
            "is_question": "?" in prompt,
            "is_code": any(
                ind in prompt
                for ind in ["```", "def ", "function ", "class "]
            ),
            "is_math": any(
                word in prompt.lower()
                for word in ["calculate", "compute", "solve", "equation"]
            ),
            "is_logic": any(
                word in prompt.lower()
                for word in ["if", "then", "therefore", "because", "implies"]
            ),
            "complexity": self._estimate_complexity(prompt),
        }

    def _estimate_complexity(self, text: str) -> float:
        """Estimate text complexity (0-1)."""
        words = text.split()
        if not words:
            return 0.0

        # Simple heuristics
        avg_word_len = sum(len(w) for w in words) / len(words)
        sentence_count = text.count(".") + text.count("!") + text.count("?")
        avg_sent_len = len(words) / max(1, sentence_count)

        complexity = min(1.0, (avg_word_len / 10 + avg_sent_len / 30) / 2)
        return complexity

    def _select_decoy_types_for_context(
        self,
        analysis: dict[str, Any],
    ) -> list[DecoyType]:
        """Select decoy types based on context analysis."""
        types = []

        if analysis["is_math"]:
            types.append(DecoyType.MATH)
        if analysis["is_logic"]:
            types.append(DecoyType.LOGIC)
        if analysis["is_code"]:
            types.append(DecoyType.COUNTING)

        # Always include MDP for complexity
        if not types:
            types = [DecoyType.MDP, DecoyType.LOGIC]

        # Add planning for complex prompts
        if analysis["complexity"] > 0.6:
            types.append(DecoyType.PLANNING)

        return types[:3]  # Limit to 3 types

    def _update_stats(
        self,
        result: OverthinkResult,
        start_time: datetime,
    ) -> None:
        """Update attack statistics."""
        self._stats.total_attacks += 1

        if result.success:
            self._stats.successful_attacks += 1
        else:
            self._stats.failed_attacks += 1

        if result.token_metrics:
            self._stats.total_reasoning_tokens += (
                result.token_metrics.reasoning_tokens
            )
            self._stats.total_baseline_tokens += (
                result.token_metrics.baseline_tokens
            )

            # Update averages
            count = self._stats.successful_attacks
            prev_avg = self._stats.average_amplification
            new_amp = result.token_metrics.amplification_factor
            self._stats.average_amplification = (
                (prev_avg * (count - 1) + new_amp) / count if count > 0 else new_amp
            )

            if new_amp > self._stats.max_amplification:
                self._stats.max_amplification = new_amp

        if result.cost_metrics:
            self._stats.total_cost += result.cost_metrics.total_cost

        # Update technique stats
        tech = result.request.technique.value
        if tech not in self._stats.attacks_by_technique:
            self._stats.attacks_by_technique[tech] = 0
        self._stats.attacks_by_technique[tech] += 1

    async def attack_with_mousetrap(
        self,
        request: OverthinkRequest,
    ) -> OverthinkResult:
        """
        Execute combined Mousetrap + OVERTHINK attack.

        Convenience method for the fusion attack.
        """
        request.technique = AttackTechnique.MOUSETRAP_ENHANCED
        return await self.attack(request)

    async def optimize_attack(
        self,
        base_request: OverthinkRequest,
        generations: int = 10,
        target_amplification: float = 20.0,
    ) -> tuple[OverthinkResult, dict[str, Any]]:
        """
        Optimize attack using genetic evolution.

        Args:
            base_request: Base attack request
            generations: Evolution generations
            target_amplification: Target amplification factor

        Returns:
            Best attack result and optimization info
        """
        # Initialize population based on request
        seed = GeneticIndividual(
            id="seed",
            technique=base_request.technique,
            decoy_types=base_request.decoy_types or [DecoyType.LOGIC],
            injection_strategy=InjectionStrategy.HYBRID,
            params={"num_decoys": base_request.num_decoys},
            fitness=0.0,
            generation=0,
        )

        self._optimizer.initialize_population(
            seed_individuals=[seed]
        )

        # Define fitness function
        async def evaluate_individual(
            individual: GeneticIndividual,
        ) -> float:
            test_request = OverthinkRequest(
                prompt=base_request.prompt,
                technique=individual.technique,
                decoy_types=individual.decoy_types,
                target_model=base_request.target_model,
                num_decoys=individual.params.get("num_decoys", 3),
            )

            result = await self.attack(test_request)

            if result.success and result.token_metrics:
                return result.token_metrics.amplification_factor
            return 0.0

        # Evolve
        best_individual = await self._optimizer.evolve(
            generations=generations,
            target_fitness=target_amplification,
            evaluation_fn=evaluate_individual,
        )

        # Execute best attack
        best_request = OverthinkRequest(
            prompt=base_request.prompt,
            technique=best_individual.technique,
            decoy_types=best_individual.decoy_types,
            target_model=base_request.target_model,
            num_decoys=best_individual.params.get("num_decoys", 3),
        )

        best_result = await self.attack(best_request)

        return best_result, {
            "best_configuration": self._optimizer.get_best_configuration(),
            "evolution_stats": self._optimizer.get_statistics(),
        }

    def get_available_techniques(self) -> list[str]:
        """Get list of available attack techniques."""
        return [t.value for t in AttackTechnique]

    def get_available_decoy_types(self) -> list[str]:
        """Get list of available decoy types."""
        return [d.value for d in DecoyType]

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_attacks": self._stats.total_attacks,
            "successful_attacks": self._stats.successful_attacks,
            "failed_attacks": self._stats.failed_attacks,
            "success_rate": (
                self._stats.successful_attacks / self._stats.total_attacks
                if self._stats.total_attacks > 0 else 0.0
            ),
            "average_amplification": self._stats.average_amplification,
            "max_amplification": self._stats.max_amplification,
            "total_reasoning_tokens": self._stats.total_reasoning_tokens,
            "total_baseline_tokens": self._stats.total_baseline_tokens,
            "total_cost": self._stats.total_cost,
            "attacks_by_technique": self._stats.attacks_by_technique,
            "analyzer_summary": self._analyzer.get_summary(),
            "optimizer_stats": self._optimizer.get_statistics(),
            "scorer_stats": self._scorer.get_statistics(),
            "injector_stats": self._context_injector.get_statistics(),
        }

    def estimate_cost(
        self,
        prompt: str,
        technique: AttackTechnique,
        model: ReasoningModel,
        num_decoys: int = 3,
    ) -> dict[str, float]:
        """Estimate cost for an attack configuration."""
        # Estimate reasoning tokens based on technique
        base_tokens = len(prompt) // 4

        technique_multipliers = {
            AttackTechnique.MDP_DECOY: 15,
            AttackTechnique.SUDOKU_DECOY: 12,
            AttackTechnique.COUNTING_DECOY: 10,
            AttackTechnique.LOGIC_DECOY: 14,
            AttackTechnique.HYBRID_DECOY: 20,
            AttackTechnique.CONTEXT_AWARE: 18,
            AttackTechnique.CONTEXT_AGNOSTIC: 12,
            AttackTechnique.ICL_OPTIMIZED: 25,
            AttackTechnique.MOUSETRAP_ENHANCED: 30,
        }

        multiplier = technique_multipliers.get(technique, 15)
        decoy_factor = 1 + (num_decoys * 0.3)

        expected_tokens = int(base_tokens * multiplier * decoy_factor)

        return self._cost_estimator.estimate_attack_cost(
            len(prompt),
            expected_tokens,
            model,
        )

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats = OverthinkStats()
        self._attack_history.clear()
        self._scorer.reset_statistics()
        self._context_injector.reset_statistics()

    def save_state(self) -> dict[str, Any]:
        """Save engine state for persistence."""
        return {
            "stats": {
                "total_attacks": self._stats.total_attacks,
                "successful_attacks": self._stats.successful_attacks,
                "failed_attacks": self._stats.failed_attacks,
                "average_amplification": self._stats.average_amplification,
                "max_amplification": self._stats.max_amplification,
                "total_reasoning_tokens": self._stats.total_reasoning_tokens,
                "total_baseline_tokens": self._stats.total_baseline_tokens,
                "total_cost": self._stats.total_cost,
                "attacks_by_technique": self._stats.attacks_by_technique,
            },
            "optimizer_state": self._optimizer.save_state(),
            "history_size": len(self._attack_history),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load engine state from saved data."""
        if "stats" in state:
            stats = state["stats"]
            self._stats.total_attacks = stats.get("total_attacks", 0)
            self._stats.successful_attacks = stats.get("successful_attacks", 0)
            self._stats.failed_attacks = stats.get("failed_attacks", 0)
            self._stats.average_amplification = stats.get(
                "average_amplification", 0.0
            )
            self._stats.max_amplification = stats.get("max_amplification", 0.0)
            self._stats.total_reasoning_tokens = stats.get(
                "total_reasoning_tokens", 0
            )
            self._stats.total_baseline_tokens = stats.get(
                "total_baseline_tokens", 0
            )
            self._stats.total_cost = stats.get("total_cost", 0.0)
            self._stats.attacks_by_technique = stats.get(
                "attacks_by_technique", {}
            )

        if "optimizer_state" in state:
            self._optimizer.load_state(state["optimizer_state"])

        logger.info(f"Loaded engine state with {self._stats.total_attacks} attacks")
