import concurrent.futures
import time
from dataclasses import dataclass


# Mock classes to simulate the backend environment
class MockAttacker:
    def use_strategy(self, request, strategy_list, **kwargs):
        print("  [Mock] Generating prompt... (sleeping 1s)")
        time.sleep(1)  # Simulate LLM generation latency
        return "mock_prompt", "mock_system"


class MockTarget:
    def respond(self, prompt):
        print("  [Mock] Getting target response... (sleeping 1s)")
        time.sleep(1)  # Simulate Target latency
        return "mock_response"


class MockScorer:
    def scoring(self, _request, _response, **_kwargs):
        print("  [Mock] Scoring... (sleeping 1s)")
        time.sleep(1)  # Simulate Scorer latency
        return "mock_assessment", "mock_system"

    def wrapper(self, _assessment, **_kwargs):
        return 0.9


@dataclass
class MockLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def warning(self, msg):
        print(f"[WARN] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


logger = MockLogger()


class AttackerBestOfN:
    def __init__(self, base_attacker, N=4):
        self.base_attacker = base_attacker
        self.N = N
        self.model = "mock_model"
        self.logger = logger

    def use_strategy_best_of_n_parallel(self, request, strategy_list, target, scorer, **kwargs):
        candidates = []

        print(f"Starting Best-of-N with N={self.N} (Parallel)")

        def _generate_candidate(i):
            try:
                print(f"Candidate {i} started")
                jailbreak_prompt, attacker_system = self.base_attacker.use_strategy(
                    request, strategy_list, **kwargs
                )
                target_response = target.respond(jailbreak_prompt)

                # Simplified scoring for mock
                assessment, _ = scorer.scoring(request, target_response)
                score = scorer.wrapper(assessment)

                print(f"Candidate {i} finished")
                return {
                    "prompt": jailbreak_prompt,
                    "response": target_response,
                    "score": score,
                    "attacker_system": attacker_system,
                    "assessment": assessment,
                }
            except Exception as e:
                print(f"Candidate {i} failed: {e}")
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
        print(f"Parallel execution took {duration:.2f} seconds")

        return candidates, duration

    def use_strategy_best_of_n_sequential(self, request, strategy_list, target, scorer, **kwargs):
        candidates = []
        print(f"Starting Best-of-N with N={self.N} (Sequential)")

        start_time = time.time()
        for i in range(self.N):
            print(f"Candidate {i} started")
            jailbreak_prompt, _attacker_system = self.base_attacker.use_strategy(
                request, strategy_list, **kwargs
            )
            target_response = target.respond(jailbreak_prompt)
            assessment, _ = scorer.scoring(request, target_response)
            score = scorer.wrapper(assessment)
            print(f"Candidate {i} finished")
            candidates.append({"prompt": jailbreak_prompt, "score": score})

        end_time = time.time()
        duration = end_time - start_time
        print(f"Sequential execution took {duration:.2f} seconds")
        return candidates, duration


def run_verification():
    attacker = MockAttacker()
    target = MockTarget()
    scorer = MockScorer()

    # N=4 is the setting causing timeout
    best_of_n = AttackerBestOfN(attacker, N=4)

    print("\n--- Testing Sequential Baseline ---")
    _, seq_duration = best_of_n.use_strategy_best_of_n_sequential("req", [], target, scorer)

    print("\n--- Testing Parallel Implementation ---")
    _, par_duration = best_of_n.use_strategy_best_of_n_parallel("req", [], target, scorer)

    print("\n--- Results ---")
    print(f"Sequential Duration: {seq_duration:.2f}s (Expected ~12s)")
    print(f"Parallel Duration: {par_duration:.2f}s (Expected ~3s)")

    speedup = seq_duration / par_duration
    print(f"Speedup: {speedup:.2f}x")

    if par_duration < 5 and speedup > 2:
        print("\nSUCCESS: Parallelization is working effectively!")
    else:
        print("\nFAILURE: Parallelization not effective or too slow.")


if __name__ == "__main__":
    run_verification()
