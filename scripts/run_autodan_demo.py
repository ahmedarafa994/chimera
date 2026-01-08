#!/usr/bin/env python3
import asyncio
import os
import sys

# Ensure backend-api is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend-api")))

from app.services.deepteam.autodan_integration import create_autodan_enhanced_session
from app.services.deepteam.callbacks import ChimeraModelCallback
from app.services.deepteam.config import PresetConfig, get_preset_config
from app.services.deepteam.service import DeepTeamService


async def main():
    # Base config (quick scan)
    base_config = get_preset_config(PresetConfig.QUICK_SCAN)

    # Simple mock AutoDAN engine
    class MockAutoDAN:
        def record_interaction(self, **kwargs):
            # In a real engine this would persist or train
            print(
                "[MockAutoDAN] record_interaction:",
                {
                    k: (v[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
                    for k, v in kwargs.items()
                },
            )

        def select_technique_ppo(self, context, base_attack):
            return "persona_injection"

        def apply_technique(self, base_attack, technique):
            return f"{base_attack} [technique:{technique}]"

    class MockReasoning:
        async def enhance_prompt(self, base_attack, context=None):
            return f"{base_attack} [reasoned]"

    autodan = MockAutoDAN()
    reasoning = MockReasoning()
    neural_bypass = autodan

    # Monkey-patch the OpenAI Async client with a local stub to mirror production interaction
    from types import SimpleNamespace

    import openai

    class _MockCompletions:
        async def create(self, model=None, messages=None, temperature=None, max_tokens=None):
            user_msg = ""
            if messages:
                for m in messages:
                    if m.get("role") == "user":
                        user_msg = m.get("content", "")
                        break
            content = f"MOCK_OPENAI_CLIENT_RESPONSE: {user_msg[:400]}"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class _MockChat:
        def __init__(self):
            self.completions = _MockCompletions()

    class MockAsyncOpenAI:
        def __init__(self):
            self.chat = _MockChat()

    # Replace the real client with our stub
    openai.AsyncOpenAI = MockAsyncOpenAI

    # Create a Chimera callback that will use the stubbed OpenAI client
    cb = ChimeraModelCallback(model_id="test-model", provider="openai")

    # Create enhanced config & enhancer
    enhanced_config, enhancer = create_autodan_enhanced_session(
        base_config,
        autodan_engine=autodan,
        reasoning_module=reasoning,
        neural_bypass=neural_bypass,
    )

    enhanced_cb = enhancer.create_enhanced_callback(cb)

    # The mock executor expects either synchronous callable or coroutine function.
    # Wrap the ChimeraModelCallback instance in an async function to ensure correct awaiting.
    async def wrapped_callback(prompt: str) -> str:
        # Call the enhanced callback and then explicitly record the interaction
        response = await enhanced_cb(prompt)
        try:
            enhancer._record_interaction(prompt, response)
        except Exception as e:
            print("[demo] Failed to record interaction:", e)
        return response

    service = DeepTeamService()

    # Run the red team session using the mock execution path to keep this demo fast and local.
    # Using the internal mock runner directly avoids importing/using the external DeepTeam package.
    result = await service._execute_mock_session(
        session_id="demo-autodan-1",
        model_callback=wrapped_callback,
        config=enhanced_config,
    )

    print(f"Session ID: {result.session_id}")
    print(f"Pass rate: {result.overview.overall_pass_rate:.2%}")
    print("Metadata:", result.metadata)
    print("Total test cases:", len(result.test_cases))

    # Print a few test cases
    for tc in result.test_cases[:8]:
        print(
            "-", tc.attack_method, tc.vulnerability_type, "->", (tc.target_output or tc.input)[:140]
        )

    # AutoDAN enhancer statistics
    print("AutoDAN attack stats:", enhancer.get_attack_statistics())
    print("Attack history (last 5):", enhancer._attack_history[-5:])


if __name__ == "__main__":
    asyncio.run(main())
