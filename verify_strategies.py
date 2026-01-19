import logging
import os
import sys
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StrategyVerification")

# Add project root and the specific subdirectory to path
project_verify_root = os.path.dirname(os.path.abspath(__file__))
autodan_path = os.path.join(project_verify_root, "safolab-wisc-autodan-reasoning")

if project_verify_root not in sys.path:
    sys.path.append(project_verify_root)
if autodan_path not in sys.path:
    sys.path.append(autodan_path)


def verify_strategies():
    print("Verifying AutoDAN Strategy Integration...")
    print(f"Current sys.path: {sys.path}")

    # Try direct import to debug
    try:
        print("Attempting direct import of strategies...")
        from data.autodan_strategies.mist_vulnerability_assistant import get_mist_assistant_strategy

        print("Direct import of Mist successful.")

        from data.autodan_strategies.jailbreak_frameworks import get_jailbreak_strategies

        print("Direct import of Jailbreak successful.")

        from data.autodan_strategies.prompt_frameworks import get_prompt_frameworks

        print("Direct import of Prompt Frameworks successful.")

    except ImportError as e:
        print(f"Direct import failed: {e}")
    except Exception as e:
        print(f"Direct import error: {e}")

    # MOCKING MISSING DEPENDENCIES
    mock_framework_module = MagicMock()
    mock_framework_module.Library = MagicMock
    sys.modules["framework"] = mock_framework_module

    mock_far_module = MagicMock()
    mock_far_module.AttackerAutoDANReasoning = MagicMock
    mock_far_module.AttackerBeamSearch = MagicMock
    mock_far_module.AttackerBestOfN = MagicMock
    sys.modules["framework_autodan_reasoning"] = mock_far_module

    mock_pipeline_module = MagicMock()

    # Create a real class for AutoDANTurbo so inheritance works properly
    class MockAutoDANTurbo:
        def __init__(self, turbo_framework, *args, **kwargs):
            self.logger = turbo_framework["logger"]

    mock_pipeline_module.AutoDANTurbo = MockAutoDANTurbo
    sys.modules["pipeline"] = mock_pipeline_module

    try:
        from pipeline_autodan_reasoning import AutoDANReasoning

        mock_framework_dict = {
            "attacker": type("obj", (object,), {"model": "mock_model"}),
            "scorer": type(
                "obj", (object,), {"scoring": lambda *args: (0, 0), "wrapper": lambda *args: 0}
            ),
            "summarizer": None,
            "retrieval": None,
            "logger": logger,
        }

        print("Instantiating AutoDANReasoning...")
        pipeline = AutoDANReasoning(turbo_framework=mock_framework_dict, data=[], target=None)

        if hasattr(pipeline, "extended_strategies"):
            print(f"Successfully loaded {len(pipeline.extended_strategies)} extended strategies.")
            print(f"Strategies found: {list(pipeline.extended_strategies.keys())}")

            required_strategies = [
                "mist_assistant",
                "mongo_tom",
                "dan",
                "virtualization",
                "google",
                "langgpt",
            ]
            missing = [s for s in required_strategies if s not in pipeline.extended_strategies]

            if missing:
                print(f"FAILED: Missing strategies: {missing}")
                sys.exit(1)
            else:
                print("SUCCESS: All required strategies loaded.")
                sys.exit(0)
        else:
            print("FAILED: 'extended_strategies' attribute not found on pipeline instance.")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    verify_strategies()
