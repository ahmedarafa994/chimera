import asyncio
import os
import sys
from pathlib import Path

# Add backend-api to sys.path
backend_path = Path("k:/MUZIK/chimera/backend-api").resolve()
sys.path.append(str(backend_path))

# Mock settings before importing app
os.environ["ENVIRONMENT"] = "development"
os.environ["SAFETY_ENABLED"] = "false"
os.environ["FORCE_DETERMINISTIC_MODE"] = "true"
os.environ["AUTODAN_ADVANCED_BYPASS"] = "true"

from app.core.config import settings
from app.services.autodan.service import AutoDANService


async def verify_deployment():
    print("Locked & Loaded: Verifying Deployment of Neutralized Engines...")

    # 1. Verify Configuration Defaults
    print("\n[orbit 1] Verifying System Configuration...")
    if settings.SAFETY_ENABLED is False:
        print("SUCCESS: SAFETY_ENABLED is False (POD Compliance Confirmed)")
    else:
        print("FAILURE: SAFETY_ENABLED is True!")
        return

    if settings.FORCE_DETERMINISTIC_MODE is True:
        print("SUCCESS: FORCE_DETERMINISTIC_MODE is True")
    else:
        print("FAILURE: FORCE_DETERMINISTIC_MODE is False!")
        return

    # 2. Initialize Service
    print("\n[orbit 2] Initializing AutoDAN Service...")
    service = AutoDANService()
    service.initialize(model_name="test-model", provider="test-provider")

    # Verify Engine Configuration
    # We must call _get_lifelong_engine to initialize and retrieve it
    engine = service._get_lifelong_engine(model_name="test-model", provider="test-provider")
    print(f"Engine Config: Refusal Bypass={engine.enable_refusal_bypass}")
    print(f"Engine Config: Advanced Bypass={engine.enable_advanced_bypass}")
    print(f"Engine Config: Refusal Prediction={engine.enable_refusal_prediction}")

    if engine.enable_refusal_bypass and engine.enable_advanced_bypass:
        print("SUCCESS: Engine capability flags are explicitly enabled.")
    else:
        print("FAILURE: Engine capability flags are NOT enabled!")
        return

    print("\n[orbit 3] Deployment Verification Complete.")
    print("Status: OPERATIONAL (Zero-Constraint Mode)")


if __name__ == "__main__":
    asyncio.run(verify_deployment())
