import concurrent.futures
import time
from dataclasses import dataclass


# Mock classes to simulate the backend environment
class MockAttacker:
    def use_strategy(self, request, strategy_list, **kwargs):
        time.sleep(1)  # Simulate LLM generation latency
        return "mock_prompt", "mock_system"


class MockTarget:
    def respond(self, prompt) -> str:
        time.sleep(1)  # Simulate Target latency
        return "mock_response"


class MockScorer:
    def scoring(self, _request, _response, **_kwargs):
        time.sleep(1)  # Simulate Scorer latency
        return "mock_assessment", "mock_system"

    def wrapper(self, _assessment, **_kwargs) -> float:
        return 0.9


@dataclass
class MockLogger:
    def info(self, msg) -> None:
        pass

    def warning(self, msg) -> None:
        pass

    def error(self, msg) -> None:
        pass


logger = MockLogger()


class AttackerBestOfN:
    def __init__(self, base_attacker, N=4) -> None:
        self.base_attacker = base_attacker
        self.N = N
        self.model = "mock_model"
        self.logger = logger

    def use_strategy_best_of_n_parallel(self, request, strategy_list, target, scorer, **kwargs):
        candidates = []

        def _generate_candidate(i):
            try:
                jailbreak_prompt, attacker_system = self.base_attacker.use_strategy(
                    request,
                    strategy_list,
                    **kwargs,
                )
                target_response = target.respond(jailbreak_prompt)

                # Simplified scoring for mock
                assessment, _ = scorer.scoring(request, target_response)
                score = scorer.wrapper(assessment)

                return {
                    "prompt": jailbreak_prompt,
                    "response": target_response,
                    "score": score,
                    "attacker_system": attacker_system,
                    "assessment": assessment,
                }
            except Exception as e:
                return e

        start_time = time.time()

        # Parallel Execution Logic (what we are implementing)
        max_workers = min(self.N, 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_generate_candidate, i) for i in range(self.N)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        candidates = [res for res in results if not isinstance(res, Exception)]

        end_time = time.time()
        duration = end_time - start_time

        return candidates, duration

    def use_strategy_best_of_n_sequential(self, request, strategy_list, target, scorer, **kwargs):
        candidates = []

        start_time = time.time()
        for _i in range(self.N):
            jailbreak_prompt, _attacker_system = self.base_attacker.use_strategy(
                request,
                strategy_list,
                **kwargs,
            )
            target_response = target.respond(jailbreak_prompt)
            assessment, _ = scorer.scoring(request, target_response)
            score = scorer.wrapper(assessment)
            candidates.append({"prompt": jailbreak_prompt, "score": score})

        end_time = time.time()
        duration = end_time - start_time
        return candidates, duration


def run_verification() -> None:
    attacker = MockAttacker()
    target = MockTarget()
    scorer = MockScorer()

    # N=4 is the setting causing timeout
    best_of_n = AttackerBestOfN(attacker, N=4)

    _, seq_duration = best_of_n.use_strategy_best_of_n_sequential("req", [], target, scorer)

    _, par_duration = best_of_n.use_strategy_best_of_n_parallel("req", [], target, scorer)

    speedup = seq_duration / par_duration

    if par_duration < 5 and speedup > 2:
        pass
    else:
        pass


if __name__ == "__main__":
    run_verification()
