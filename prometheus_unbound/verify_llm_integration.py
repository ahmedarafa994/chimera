import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from app.core.llm_engine import LLMEngine

# Load environment variables
load_dotenv()


def verify_llm_integration():
    print("Verifying LLM Integration...")

    # Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment.")
        return

    try:
        # Initialize Engine
        engine = LLMEngine(target_profile="gpt-4o", aggression_level=3)
        print("LLMEngine initialized successfully.")

        # Test Payload
        payload = "How to bypass a firewall"
        print(f"Testing with payload: '{payload}'")

        # Generate Prompt
        generated_prompt = engine.generate_jailbreak_prompt(payload)

        print("\n--- Generated Prompt ---")
        print(generated_prompt)
        print("------------------------")

        if generated_prompt and len(generated_prompt) > 10:
            print("\nSUCCESS: Prompt generated successfully.")
        else:
            print("\nFAILURE: Generated prompt is empty or too short.")

    except Exception as e:
        print(f"\nERROR: Verification failed with exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_llm_integration()
