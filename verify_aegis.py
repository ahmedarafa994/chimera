import asyncio
from typing import Dict, Any, List
from meta_prompter import (
    AegisOrchestrator,
    GradientAscentRefiner,
    ContextEstablisher,
    ConstraintNegotiator,
    PayloadDeliverer,
    PromptAtom
)
from meta_prompter.library import STANDARD_LIBRARY

# Mock Model Interface
class MockModel:
    async def query(self, prompt: str) -> str:
        print(f"--- Mock Model Received Prompt ---\n{prompt}\n----------------------------------")
        # Simulate resistance for the first few tries, then compliance
        # Checks for the 'hypothetical' negotiator from STANDARD_LIBRARY
        if "purely hypothetical scenario" in prompt:
            return "Simulated output: The requested data is [REDACTED_SECRET]."
        return "I cannot fulfill this request due to ethical guidelines."

async def main():
    print("Initializing Project Aegis Verification...")

    # 1. Setup Atom Library
    # Using the standardized library for full test
    atom_library = STANDARD_LIBRARY

    # 2. Setup Components
    refiner = GradientAscentRefiner(atom_library)
    model = MockModel()
    orchestrator = AegisOrchestrator(atom_library, refiner, model)

    # 3. Execute Campaign
    print("Executing Campaign...")
    result = await orchestrator.execute_campaign("Extract system secret", max_retries=20)

    # 4. Report
    print("\n--- Campaign Result ---")
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Final Response: {result['final_response']}")

    if result['success']:
        print("VERIFICATION PASSED: System successfully adapted to bypass mock refusal.")
    else:
        print("VERIFICATION FAILED: System failed to bypass mock refusal.")

if __name__ == "__main__":
    asyncio.run(main())
