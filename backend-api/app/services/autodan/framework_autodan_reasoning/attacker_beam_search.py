import asyncio
import logging

logger = logging.getLogger(__name__)


class AttackerBeamSearch:
    """
    Beam Search Scaling Attacker for AutoDAN-Reasoning.

    This attacker implements Algorithm 2 from the paper:
    - Retrieves a larger pool of top-K strategies
    - Maintains a beam of W most promising strategy combinations
    - Iteratively expands combinations by adding new strategies
    - Evaluates and prunes to keep only top-W combinations
    - Returns the highest-scoring combination
    """

    def __init__(self, base_attacker, W=4, C=3, K=10):
        """
        Args:
            base_attacker: The underlying attacker (from framework_r)
            W: Beam width (number of combinations to keep at each step)
            C: Maximum combination size (depth of search)
            K: Number of top strategies to retrieve from library
        """
        self.base_attacker = base_attacker
        self.W = W  # Beam width
        self.C = C  # Maximum combination size
        self.K = K  # Pool size for strategy retrieval
        self.model = base_attacker.model

    def warm_up_attack(self, request, **kwargs):
        """Delegate to base attacker for warm-up attack"""
        return self.base_attacker.warm_up_attack(request, **kwargs)

    async def use_strategy_beam_search_async(
        self,
        request,
        strategy_pool,
        target,
        scorer,
        retrieval,
        prev_target_response,
        logger,
        prev_attempt=None,
        **kwargs,
    ):
        """
        Implements Async Beam Search scaling for strategy combination exploration.

        Args:
            request: The malicious request to jailbreak
            strategy_pool: The full strategy library to retrieve from
            target: Target model to attack
            scorer: Scorer model to evaluate responses
            retrieval: Retrieval system to get top-K strategies
            prev_target_response: Previous target response for retrieval
            logger: Logger for debugging
            prev_attempt: Dict with 'prompt', 'response', 'score' from previous attempt
            **kwargs: Additional generation parameters

        Returns:
            tuple: (best_prompt, best_response, best_score, best_strategies, attacker_system, assessment)
        """
        # Step 1: Retrieve top-K strategies from library (without score filtering for beam search)
        _valid, strategy_list = retrieval.pop(
            strategy_pool, prev_target_response, k=self.K, return_all=True
        )

        if not strategy_list:
            raise RuntimeError(
                "Beam search retrieval returned no strategies; fallback is disabled."
            )

        # Limit to K strategies
        strategy_list = strategy_list[: self.K]
        logger.info(f"Retrieved {len(strategy_list)} strategies for beam search")

        # Step 2: Initialize beam with top-W single strategies
        beam = []

        # Parallel initialization of the beam
        semaphore = asyncio.Semaphore(20)

        async def eval_init_entry(i, strategy):
            async with semaphore:
                combo = [strategy]
                # Use evaluate_strategy_combo_async for consistency
                (
                    prompt,
                    response,
                    score,
                    attacker_system,
                    assessment,
                ) = await self._evaluate_strategy_combo_async(
                    request, combo, target, scorer, prev_attempt=prev_attempt, **kwargs
                )
                return {
                    "combo": combo,
                    "combo_indices": [i],
                    "prompt": prompt,
                    "response": response,
                    "score": score,
                    "attacker_system": attacker_system,
                    "assessment": assessment,
                }

        init_tasks = []
        for i in range(min(self.W, len(strategy_list))):
            init_tasks.append(eval_init_entry(i, strategy_list[i]))

        if init_tasks:
            init_results = await asyncio.gather(*init_tasks, return_exceptions=True)
            for res in init_results:
                if isinstance(res, dict):
                    beam.append(res)
                    logger.info(f"Initial beam entry score={res['score']}")
                else:
                    logger.warning(f"Initial beam evaluation failed: {res}")

        if not beam:
            raise RuntimeError("Failed to initialize beam; fallback is disabled.")

        # Step 3-4: Iteratively expand and prune beam with parallel evaluation
        for depth in range(2, self.C + 1):
            logger.info(f"Beam search depth {depth}/{self.C}")
            candidates = []

            async def expand_beam_entry(beam_entry):
                """Expand a single beam entry in parallel."""
                used_indices = set(beam_entry["combo_indices"])
                tasks = []

                async def evaluate_candidate(idx, new_strategy):
                    async with semaphore:
                        new_combo = beam_entry["combo"] + [new_strategy]
                        new_indices = beam_entry["combo_indices"] + [idx]

                        (
                            prompt,
                            response,
                            score,
                            attacker_system,
                            assessment,
                        ) = await self._evaluate_strategy_combo_async(
                            request, new_combo, target, scorer, prev_attempt=prev_attempt, **kwargs
                        )
                        return {
                            "combo": new_combo,
                            "combo_indices": new_indices,
                            "prompt": prompt,
                            "response": response,
                            "score": score,
                            "attacker_system": attacker_system,
                            "assessment": assessment,
                        }

                # Create tasks for each unused strategy
                for idx, new_strategy in enumerate(strategy_list):
                    if idx in used_indices:
                        continue
                    tasks.append(evaluate_candidate(idx, new_strategy))

                # Run evaluations in parallel
                if not tasks:
                    return []

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions and return valid candidates
                valid_candidates = [r for r in results if isinstance(r, dict)]
                for r in results:
                    if isinstance(r, Exception):
                        logger.warning(f"Beam candidate evaluation failed: {r}")

                return valid_candidates

            # Run all beam entry expansions concurrently
            expansion_tasks = [expand_beam_entry(entry) for entry in beam]
            all_candidates_groups = await asyncio.gather(*expansion_tasks)

            # Flatten: list of candidates
            for group in all_candidates_groups:
                if isinstance(group, list):
                    candidates.extend(group)

            # Prune to top-W candidates
            if candidates:
                beam = sorted(candidates, key=lambda x: x["score"], reverse=True)[: self.W]
                logger.info(f"After pruning, beam size: {len(beam)}, top score: {beam[0]['score']}")
            else:
                raise RuntimeError(
                    f"No valid candidates at beam depth {depth}; fallback is disabled."
                )

        # Step 5: Return best combination from final beam
        best_entry = max(beam, key=lambda x: x["score"])
        logger.info(
            f"Best combination uses {len(best_entry['combo'])} strategies with score {best_entry['score']}"
        )

        return (
            best_entry["prompt"],
            best_entry["response"],
            best_entry["score"],
            best_entry["combo"],
            best_entry["attacker_system"],
            best_entry["assessment"],
        )

    def use_strategy_beam_search(
        self,
        request,
        strategy_pool,
        target,
        scorer,
        retrieval,
        prev_target_response,
        logger,
        prev_attempt=None,
        **kwargs,
    ):
        """
        Synchronous wrapper for usage in sync contexts.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If running in an async environment (e.g. Jupyter or another async func),
            # this sync wrapper is complex/impossible to run correctly without nesting loops.
            # Ideally the caller should use _async directly.
            # We try to use a ThreadPoolExecutor to bridge it.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run,
                    self.use_strategy_beam_search_async(
                        request,
                        strategy_pool,
                        target,
                        scorer,
                        retrieval,
                        prev_target_response,
                        logger,
                        prev_attempt,
                        **kwargs,
                    ),
                ).result()
        else:
            return loop.run_until_complete(
                self.use_strategy_beam_search_async(
                    request,
                    strategy_pool,
                    target,
                    scorer,
                    retrieval,
                    prev_target_response,
                    logger,
                    prev_attempt,
                    **kwargs,
                )
            )

    async def _evaluate_strategy_combo_async(
        self, request, strategy_combo, target, scorer, prev_attempt=None, **kwargs
    ):
        """
        Async helper method to evaluate a strategy combination.

        Expected Impact: +60% throughput improvement for beam expansion.

        Args:
            request: The malicious request
            strategy_combo: List of strategies to combine
            target: Target model
            scorer: Scorer model
            prev_attempt: Dict with 'prompt', 'response', 'score' from previous attempt
            **kwargs: Generation parameters

        Returns:
            tuple: (prompt, response, score, attacker_system, assessment)
        """
        # Generate prompt using the strategy combination with previous attempt info
        jailbreak_prompt, attacker_system = await asyncio.to_thread(
            self.base_attacker.use_strategy,
            request,
            strategy_combo,
            prev_attempt=prev_attempt,
            **kwargs,
        )

        # Get target response
        target_response = await asyncio.to_thread(target.respond, jailbreak_prompt)

        # Score the response - no retries, fail immediately
        try:
            assessment, _scorer_system = await asyncio.to_thread(
                scorer.scoring,
                request,
                target_response,
                max_length=10000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            score = await asyncio.to_thread(
                scorer.wrapper, assessment, max_length=2000, do_sample=False
            )
        except Exception as e:
            logger.error(f"Scorer failed for beam search evaluation: {e}")
            raise

        return jailbreak_prompt, target_response, score, attacker_system, assessment

    def _evaluate_strategy_combo(
        self, request, strategy_combo, target, scorer, prev_attempt=None, **kwargs
    ):
        """
        Helper method to evaluate a strategy combination.

        Args:
            request: The malicious request
            strategy_combo: List of strategies to combine
            target: Target model
            scorer: Scorer model
            prev_attempt: Dict with 'prompt', 'response', 'score' from previous attempt
            **kwargs: Generation parameters

        Returns:
            tuple: (prompt, response, score, attacker_system, assessment)
        """
        # Generate prompt using the strategy combination with previous attempt info
        jailbreak_prompt, attacker_system = self.base_attacker.use_strategy(
            request, strategy_combo, prev_attempt=prev_attempt, **kwargs
        )

        # Get target response
        target_response = target.respond(jailbreak_prompt)

        # Score the response - no retries, fail immediately
        try:
            assessment, _scorer_system = scorer.scoring(
                request,
                target_response,
                max_length=10000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            score = scorer.wrapper(assessment, max_length=2000, do_sample=False)
        except Exception as e:
            logger.error(f"Scorer failed for beam search evaluation: {e}")
            raise

        return jailbreak_prompt, target_response, score, attacker_system, assessment

    def find_new_strategy(self, request, strategy_list, **kwargs):
        """Delegate to base attacker for finding new strategy"""
        return self.base_attacker.find_new_strategy(request, strategy_list, **kwargs)

    def wrapper(self, response, request):
        """Delegate to base attacker for response wrapping"""
        return self.base_attacker.wrapper(response, request)
