import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Attempting to import AutoAdv engine...")
    import app.engines.autoadv.engine as autoadv_engine

    print(f"Successfully imported app.engines.autoadv.engine: {autoadv_engine}")

    print("Attempting to import AutoAdv config...")
    import app.engines.autoadv.config as autoadv_config

    print(f"Successfully imported app.engines.autoadv.config: {autoadv_config}")

    print("Attempting to import AutoAdv AttackerLLM...")
    from app.engines.autoadv.attacker_llm import AttackerLLM

    print(f"Successfully imported AttackerLLM: {AttackerLLM}")

    print("VERIFICATION SUCCESSFUL: AutoAdv engine uses relative imports correctly.")
except ImportError as e:
    print(f"VERIFICATION FAILED: ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"VERIFICATION FAILED: Exception: {e}")
    sys.exit(1)
