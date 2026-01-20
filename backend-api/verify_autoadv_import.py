import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import app.engines.autoadv.config as autoadv_config
    import app.engines.autoadv.engine as autoadv_engine
    from app.engines.autoadv.attacker_llm import AttackerLLM

    # Touch imports to avoid F401
    _ = autoadv_config
    _ = autoadv_engine
    _ = AttackerLLM

except ImportError:
    sys.exit(1)
except Exception:
    sys.exit(1)
