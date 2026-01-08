import os
import sys

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    from llm_integration import LLMIntegrationEngine

    print("Initializing LLMIntegrationEngine...")
    engine = LLMIntegrationEngine()

    print("Attempting to use 'universal_bypass' technique suite...")
    try:
        # We don't need a real provider client for this test, just checking the validation logic
        # But execute() checks for client registration.
        # transform_prompt() checks for technique suite validity first.

        engine.transform_prompt(
            core_request="Test request", potency_level=5, technique_suite="universal_bypass"
        )
        print("SUCCESS: 'universal_bypass' is accepted.")
    except ValueError as e:
        print(f"FAILURE: Caught expected error: {e}")
    except Exception as e:
        print(f"FAILURE: Caught unexpected error: {e}")

except ImportError:
    print(
        "Could not import llm_integration. Make sure you are running this from the Project_Chimera directory."
    )
