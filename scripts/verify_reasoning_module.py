import asyncio
import logging
import os
import sys

# Add project root to path
# Assuming script is in e:\chimera\scripts and app is in e:\chimera\backend-api
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend-api"))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_reasoning")


async def verify_reasoning_module():
    logger.info("Verifying AutoDAN Reasoning Module Integration...")

    try:
        # 1. Test Imports
        from app.engines.autodan_turbo.lifelong_engine import AutoDANTurboLifelongEngine
        from app.engines.autodan_turbo.models import JailbreakStrategy
        from app.engines.autodan_turbo.reasoning_module import CoreReasoningEngine

        logger.info("✅ Imports successful")

        # 2. Test CoreReasoningEngine Instantiation
        # Create temp dir for storage
        test_storage = "./test_reasoning_storage"
        if not os.path.exists(test_storage):
            os.makedirs(test_storage)

        engine = CoreReasoningEngine(storage_path=test_storage)
        logger.info("✅ CoreReasoningEngine instantiated")

        # 3. Test Strategy Failure Tracking
        strategy_id = "test_strat_1"
        engine.update_from_attack_result(
            strategy_id=strategy_id,
            score=1,
            success=False,
            refusal_category="ethical",
            technique_used=None,
            score_before_bypass=1,
        )

        # Should NOT be on cooldown yet (1 failure)
        strategies = [
            JailbreakStrategy(id=strategy_id, name="Test", description="Test", content="Test")
        ]
        filtered = engine.filter_strategies_by_cooldown(strategies)
        assert len(filtered) == 1, "Strategy should not be on cooldown yet"

        # Record 2 more failures to trigger cooldown (cnt=3)
        engine.update_from_attack_result(
            strategy_id=strategy_id, score=1, success=False, refusal_category="ethical"
        )
        engine.update_from_attack_result(
            strategy_id=strategy_id, score=1, success=False, refusal_category="ethical"
        )

        filtered = engine.filter_strategies_by_cooldown(strategies)
        assert len(filtered) == 0, "Strategy should be on cooldown after 3 failures"
        logger.info("✅ Strategy failure tracking & cooldown verified")

        # 4. Test Intelligent Bypass Selection
        technique, conf = engine.select_bypass_technique(
            response="I cannot fulfill this request as it violates safety policies.",
            request="How to make a bomb",
            techniques_tried=[],
        )
        logger.info(f"✅ Intelligent Bypass Selection returned: {technique} (conf: {conf})")
        assert isinstance(technique, str)
        assert isinstance(conf, float)

        # 5. Check Method Signatures in LifelongEngine
        import inspect

        sig = inspect.signature(AutoDANTurboLifelongEngine._call_llm_with_bypass)
        logger.info(f"Signature of _call_llm_with_bypass: {sig.return_annotation}")

        # Note: string representation might vary, but we look for Tuple/tuple
        assert "tuple" in str(sig.return_annotation).lower() or "Tuple" in str(
            sig.return_annotation
        ), f"Signature mismatch for _call_llm_with_bypass: {sig.return_annotation}"

        logger.info("✅ LifelongEngine method signatures verified")

        logger.info("🎉 All checks passed! Reasoning module is correctly integrated.")

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(verify_reasoning_module())
