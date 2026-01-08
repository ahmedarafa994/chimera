import os
import sys

# Setup path
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

try:
    from app.core.config import settings

    print("SUCCESS: Settings imported.")

    # Access transformation property which triggers YAML load
    config = settings.transformation

    # Verify specific key from YAML
    techniques = config.technique_suites

    if "expert" in techniques:
        print("SUCCESS: Loaded 'expert' technique from YAML.")
        print(f"Expert transformers: {techniques['expert']['transformers']}")
    else:
        print("FAILURE: 'expert' technique not found in config.")

    if "quantum" in techniques:
        print("SUCCESS: Loaded 'quantum' technique from YAML.")
        print(f"Quantum framers: {techniques['quantum']['framers']}")

except Exception as e:
    print(f"FAILURE: {e}")
