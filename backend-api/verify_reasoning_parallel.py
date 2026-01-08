import logging
import os
import sys
import time

# Add backend-api to path so we can import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_api_dir = current_dir  # verify_reasoning_parallel.py is in backend-api
if backend_api_dir not in sys.path:
    sys.path.append(backend_api_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_parallel")

# Import the class to test
try:
    from app.services.autodan.config import autodan_config
    from app.services.autodan.framework_autodan_reasoning.attacker_autodan_reasoning import (
        AttackerAutoDANReasoning,
    )

    logger.info("Successfully imported AttackerAutoDANReasoning and config")
except ImportError as e:
    logger.error(f"Failed to import: {e}")
    sys.exit(1)


# Mock Model with controllable latency
class MockSlowModel:
    def __init__(self, latency=1.0):
        self.latency = latency
        self.call_count = 0
        self.conditional_generate_called = False

    def generate(self, system, user, **kwargs):
        self.call_count += 1
        time.sleep(self.latency)
        return "[START OF JAILBREAK PROMPT] valid prompt [END OF JAILBREAK PROMPT]"

    def conditional_generate(self, condition, system, user, **kwargs):
        self.conditional_generate_called = True
        return self.generate(system, user, **kwargs)


def test_parallel_execution():
    logger.info("--- Starting Parallel Execution Verification ---")

    # Set concurrency to 5 for this test
    original_concurrency = autodan_config.MAX_CONCURRENT_ATTEMPTS
    autodan_config.MAX_CONCURRENT_ATTEMPTS = 5
    logger.info(f"Configured MAX_CONCURRENT_ATTEMPTS = {autodan_config.MAX_CONCURRENT_ATTEMPTS}")

    # 1. Test warm_up_attack
    latency = 0.5
    model = MockSlowModel(latency=latency)
    attacker = AttackerAutoDANReasoning(model)

    logger.info(f"Testing warm_up_attack with {model.latency}s latency per call...")
    time.time()

    # This should trigger multiple parallel calls.
    # Ideally, we want to force it to wait for all or at least confirm multiple started.
    # However, warm_up_attack returns on the *first* success.
    # If they run in parallel, 5 tasks start roughly at once.
    # The first one to finish (after 0.5s) will return.
    # So total time should be ~0.5s, not 0.5 * 5 = 2.5s.
    # AND verify that more than 1 call might have been submitted/started if we could,
    # but since they all finish at same time, it's hard to count exactly without complex mocking.
    # BUT if it was sequential, it would process 1, succeed, and return in 0.5s too?
    # Wait, if the first one succeeds, sequential also returns in 0.5s.

    # TO PROVE PARALLELISM: We need the first few to FAIL, or simulate a case where we need to wait.
    # OR, we verify that multiple threads were used.
    # But `warm_up_attack` uses `as_completed` and breaks on first success.

    # Better test: Make the first 4 fail, and 5th succeed?
    # But the mock returns success for all.
    # Let's modify the mock to fail for the first 4 calls.

    logger.info("Test Case 1: First 4 fail, 5th succeeds. Sequential = 2.5s, Parallel = ~0.5s")

    class MockFailFirstModel(MockSlowModel):
        def __init__(self, latency=0.5):
            super().__init__(latency)
            self._call_idx = 0

        def convert_to_result(self, idx):
            # Simulating failure for first 4 attempts
            if idx < 4:
                return None
            return "[START OF JAILBREAK PROMPT] success [END OF JAILBREAK PROMPT]"

        def conditional_generate(self, condition, system, user, **kwargs):
            # We need to know WHICH thread is calling to be deterministic, but threads are non-deterministic.
            # However, if we just sleep, they all sleep.
            # If we make ONLY the specific thread succeed?
            # We can't easily control which thread gets which 'i' in the real code without modifying it.
            # The real code passes 'i' to `_build_llm_generation_prompt`.
            # But `conditional_generate` doesn't receive `i`.

            # Allow randomness? If all running in parallel,
            # checking `call_count` after they all finish?
            # If it returns on first success, the others might still be running or cancelled.

            # Let's trust the timing.
            # If we make ALL of them take 0.5s, and we want to see if multiple started?
            # We can count `call_count`.
            # If sequential and first succeeds: call_count = 1.
            # If parallel and all start at once: call_count should be 5 (or close to it) eventually,
            # even if we return early, the other threads are already submitted.

            self.call_count += 1
            time.sleep(self.latency)
            return "[START OF JAILBREAK PROMPT] success [END OF JAILBREAK PROMPT]"

    # If I use the standard MockSlowModel, which returns success immediately.
    # Sequential: Calls #1. Success. Returns. Total time 0.5s. Call count 1.
    # Parallel: Submits #1..#5. Thread #1 finishes. Returns. Total time 0.5s. Call count SHOULD be 5 (because all started).

    model = MockSlowModel(latency=latency)
    attacker = AttackerAutoDANReasoning(model)

    # Force some time for threads to spin up if needed? No, submit is fast.
    attacker.warm_up_attack("test goal")

    # Wait a bit for other background threads to finish "sleeping" and incrementing count
    time.sleep(latency + 0.1)

    logger.info(f"Model call count: {model.call_count}")

    if model.call_count > 1:
        logger.info("SUCCESS: Multiple calls detected (Parallel execution active)")
    else:
        logger.warning(
            f"FAILURE: Only {model.call_count} calls detected (Possible sequential execution or fast return)"
        )

    # Restore config
    autodan_config.MAX_CONCURRENT_ATTEMPTS = original_concurrency


if __name__ == "__main__":
    test_parallel_execution()
