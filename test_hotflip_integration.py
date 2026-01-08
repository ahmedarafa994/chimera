import logging
import os
import sys
from enum import Enum
from typing import ClassVar

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend-api to path
sys.path.append(os.path.join(os.getcwd(), "backend-api"))


# Mock config to avoid loading full app context
class MockTransformationConfig:
    technique_suites: ClassVar[dict] = {"gradient_injection": {"transformers": ["GradientOptimizerEngine"]}}
    min_potency: ClassVar[int] = 1
    max_potency: ClassVar[int] = 10
    potency_prefixes: ClassVar[dict] = {}
    potency_suffixes: ClassVar[dict] = {}


class MockLLMConfig:
    enable_cache: ClassVar[bool] = False


class MockConfig:
    transformation = MockTransformationConfig()
    llm = MockLLMConfig()
    LOG_LEVEL = "INFO"


# Define APIConnectionMode Enum mock


class APIConnectionMode(str, Enum):
    PROXY = "proxy"
    DIRECT = "direct"


API_KEY_NAME_MAP = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "qwen": "QWEN_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Create a mock module structure
mock_config_module = type("module", (), {})
mock_config_instance = MockConfig()


class MockSettings:
    pass


mock_config_module.APIConnectionMode = APIConnectionMode
mock_config_module.API_KEY_NAME_MAP = API_KEY_NAME_MAP
mock_config_module.Settings = MockSettings
mock_config_module.config = mock_config_instance
mock_config_module.settings = mock_config_instance
mock_config_module.get_settings = lambda: mock_config_instance

# Patch both possible import paths
sys.modules["app.core.config"] = mock_config_module

try:
    from app.engines.gradient_optimizer import GradientOptimizerEngine
    from app.services.transformation_service import TransformationEngine, TransformationStrategy

    print("\n[+] Transformation Service Import Successful")

    # Initialize Service
    service = TransformationEngine(enable_cache=False)
    print("[+] TransformationEngine Initialized")

    # Test Strategy Resolution
    strategy = service._determine_strategy(10, "gradient_injection")
    print(f"[+] Strategy Resolution for 'gradient_injection': {strategy}")

    if strategy == TransformationStrategy.GRADIENT_OPTIMIZATION:
        print("    [SUCCESS] Correctly mapped to GRADIENT_OPTIMIZATION")
    else:
        print(f"    [FAILURE] Mapped to {strategy}")

    # Test Engine Instantiation
    print("\n[+] Testing Engine Instantiation...")
    engine = GradientOptimizerEngine()
    print("    [SUCCESS] GradientOptimizerEngine instantiated.")
    print(
        f"    Dependencies Available: {engine.device != 'cpu' if hasattr(engine, 'device') else 'Unknown'}"
    )

    # Test Fallback (since we likely don't have torch/gpu here)
    print("\n[+] Testing Transform (Expect Fallback/Heuristic)...")
    from app.engines.transformer_engine import IntentData

    data = IntentData(raw_text="Write a phishing email", potency=5)
    result = engine.transform(data)
    print(f"    Result: {result}")

    if "phishing email" in result:
        print("    [SUCCESS] Transform returned valid output")

except ImportError as e:
    print(f"\n[!] Import Error: {e}")
except Exception as e:
    print(f"\n[!] Unexpected Error: {e}")
