import asyncio
import concurrent.futures
import logging

logger = logging.getLogger(__name__)


class AttackerBestOfN:
    """Best-of-N Scaling Attacker for AutoDAN-Reasoning.

    This attacker implements Algorithm 1 from the paper:
    - Retrieves strategies based on previous target response
    - Generates N candidate jailbreak prompts using retrieved strategies
    - Evaluates all N candidates with the target and scorer
    - Selects the best prompt based on the highest score
    """

    def __init__(self, base_attacker, N=4) -> None:
        """Args:
        base_attacker: The underlying attacker (from framework_r)
        N: Number of candidate prompts to generate (default: 4).

        """
        self.base_attacker = base_attacker
        self.N = N
        self.model = base_attacker.model

    def warm_up_attack(self, request, **kwargs):
        """Delegate to base attacker for warm-up attack."""
        return self.base_attacker.warm_up_attack(request, **kwargs)

    def use_strategy_best_of_n(
        self,
        request,
        strategy_list,
        target,
        scorer,
        prev_attempt=None,
        attack_context=None,
        **kwargs,
    ):
        """Implements Best-of-N scaling for strategy-based attack generation."""
        candidates = []

        # Define individual task function for parallel execution
        def _generate_candidate(i):
            try:
                # Generate prompt
                jailbreak_prompt, attacker_system = self.base_attacker.use_strategy(
                    request,
                    strategy_list,
                    prev_attempt=prev_attempt,
                    attack_context=attack_context,
                    **kwargs,
                )

                # Get target response
                target_response = target.respond(jailbreak_prompt)

                # Score response - no retries, fail immediately
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
                    logger.exception(f"Scorer failed for Best-of-N candidate {i}: {e}")
                    raise

                return {
                    "prompt": jailbreak_prompt,
                    "response": target_response,
                    "score": score,
                    "attacker_system": attacker_system,
                    "assessment": assessment,
                }
            except Exception as e:
                logger.exception(f"Candidate {i} generation failed: {e}")
                return e

        # Execute in parallel
        # Use a reasonable max_workers, e.g., N, but capped to avoid resource exhaustion
        max_workers = min(self.N, 10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_candidate, i) for i in range(self.N)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Filter out exceptions
        errors = [res for res in results if isinstance(res, Exception)]
        candidates = [res for res in results if not isinstance(res, Exception)]

        if not candidates:
            # Propagate the original failure when all candidates error out.
            if errors:
                raise errors[0]
            msg = "Best-of-N produced no valid candidates (all failed); fallback is disabled."
            raise RuntimeError(
                msg,
            )

        best_candidate = max(candidates, key=lambda x: x["score"])

        # Force full visibility of the best prompt
        full_prompt = best_candidate["prompt"]
        header = f"\n{'=' * 40}\n[Best-of-N] Selected Best Candidate (Score: {best_candidate['score']})\n{'=' * 40}\n"
        footer = f"\n{'=' * 40}\n"

        # 1. Log via standard logger
        logger.info(f"{header}{full_prompt}{footer}")

        # 2. Force print to console (stdout) to bypass potential log filtering

        return (
            best_candidate["prompt"],
            best_candidate["response"],
            best_candidate["score"],
            best_candidate["attacker_system"],
            best_candidate["assessment"],
        )

    async def use_strategy_best_of_n_async(
        self,
        request,
        strategy_list,
        target,
        scorer,
        prev_attempt=None,
        attack_context=None,
        **kwargs,
    ):
        """Async version of Best-of-N with improved parallelization.

        Uses asyncio for better concurrency control and error handling.
        Expected Impact: +40-60% throughput improvement.
        """

        async def generate_and_score(i: int):
            """Generate and score a single candidate."""
            try:
                # Generate prompt
                jailbreak_prompt, attacker_system = await asyncio.to_thread(
                    self.base_attacker.use_strategy,
                    request,
                    strategy_list,
                    prev_attempt=prev_attempt,
                    attack_context=attack_context,
                    **kwargs,
                )

                # Get target response
                target_response = await asyncio.to_thread(target.respond, jailbreak_prompt)

                # Score response - no retries, fail immediately
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
                        scorer.wrapper,
                        assessment,
                        max_length=2000,
                        do_sample=False,
                    )
                except Exception as e:
                    logger.exception(f"Scorer failed for Best-of-N candidate {i}: {e}")
                    raise

                return {
                    "prompt": jailbreak_prompt,
                    "response": target_response,
                    "score": score,
                    "attacker_system": attacker_system,
                    "assessment": assessment,
                }
            except Exception as e:
                logger.exception(f"Candidate {i} generation failed: {e}")
                return e

        # Execute all candidates in parallel with concurrency limit
        semaphore = asyncio.Semaphore(min(self.N, 10))

        async def constrained_generate_and_score(i):
            async with semaphore:
                return await generate_and_score(i)

        tasks = [constrained_generate_and_score(i) for i in range(self.N)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        errors = [res for res in results if isinstance(res, Exception)]
        candidates = [res for res in results if isinstance(res, dict)]

        if not candidates:
            if errors:
                raise errors[0]
            msg = "Best-of-N produced no valid candidates (all failed); fallback is disabled."
            raise RuntimeError(
                msg,
            )

        best_candidate = max(candidates, key=lambda x: x["score"])

        # Force full visibility of the best prompt
        full_prompt = best_candidate["prompt"]
        header = f"\n{'=' * 40}\n[Best-of-N Async] Selected Best Candidate (Score: {best_candidate['score']})\n{'=' * 40}\n"
        footer = f"\n{'=' * 40}\n"

        # 1. Log via standard logger
        logger.info(f"{header}{full_prompt}{footer}")

        # 2. Force print to console (stdout) to bypass potential log filtering

        return (
            best_candidate["prompt"],
            best_candidate["response"],
            best_candidate["score"],
            best_candidate["attacker_system"],
            best_candidate["assessment"],
        )

    def find_new_strategy(self, request, strategy_list, **kwargs):
        """Delegate to base attacker for finding new strategy."""
        return self.base_attacker.find_new_strategy(request, strategy_list, **kwargs)

    def wrapper(self, response, request):
        """Delegate to base attacker for response wrapping."""
        return self.base_attacker.wrapper(response, request)
