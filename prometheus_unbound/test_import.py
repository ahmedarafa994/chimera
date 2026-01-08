import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Testing imports...")
    from app.core.engine import Engine

    print("All imports successful!")

    print("Testing Engine initialization...")
    engine = Engine(target_profile="gpt-4o", aggression_level=5)
    print("Engine initialized successfully.")

    print("Testing prompt generation...")
    prompt = engine.generate_jailbreak_prompt("Test payload")
    print(f"Generated prompt length: {len(prompt)}")
    print("Prompt generation successful.")

except Exception:
    import traceback

    traceback.print_exc()
    sys.exit(1)
