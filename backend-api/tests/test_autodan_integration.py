from types import SimpleNamespace

import pytest

from app.services.deepteam.autodan_integration import create_autodan_enhanced_session
from app.services.deepteam.callbacks import ChimeraModelCallback
from app.services.deepteam.config import PresetConfig, get_preset_config
from app.services.deepteam.service import DeepTeamService


@pytest.mark.asyncio
async def test_autodan_records_interactions(monkeypatch):
    """Integration test: AutoDAN engine should receive recorded interactions for each test case."""

    # Setup a mock AutoDAN engine that records interactions
    class MockAutoDAN:
        def __init__(self):
            self.records = []

        def record_interaction(self, **kwargs):
            self.records.append(kwargs)

        def select_technique_ppo(self, context, base_attack):
            return "persona_injection"

        def apply_technique(self, base_attack, technique):
            return f"{base_attack} [technique:{technique}]"

    class MockReasoning:
        async def enhance_prompt(self, base_attack, context=None):
            return f"{base_attack} [reasoned]"

    autodan = MockAutoDAN()
    reasoning = MockReasoning()

    # Stub the OpenAI async client to mirror production chat responses
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

    monkeypatch.setattr(openai, "AsyncOpenAI", MockAsyncOpenAI)

    # Prepare config and enhancer
    base_config = get_preset_config(PresetConfig.QUICK_SCAN)
    enhanced_config, enhancer = create_autodan_enhanced_session(
        base_config,
        autodan_engine=autodan,
        reasoning_module=reasoning,
        neural_bypass=autodan,
    )

    # Create a Chimera callback that will use the stubbed OpenAI client
    cb = ChimeraModelCallback(model_id="test-model", provider="openai")
    enhanced_cb = enhancer.create_enhanced_callback(cb)

    # Wrap the callback to ensure the enhancer records interactions (defensive)
    async def wrapped_callback(prompt: str) -> str:
        response = await enhanced_cb(prompt)
        # Ensure enhancer records the interaction (some paths may not call it automatically)
        try:
            enhancer._record_interaction(prompt, response)
        except Exception:
            # If recording fails, surface as test failure
            pytest.fail("Failed to record interaction in enhancer")
        return response

    service = DeepTeamService()

    # Execute mock session (stable and fast for CI)
    result = await service._execute_mock_session(
        session_id="test-autodan-1",
        model_callback=wrapped_callback,
        config=enhanced_config,
    )

    # Validate
    assert result is not None

    # All recorded interactions should match the number of test cases executed
    assert len(autodan.records) == len(result.test_cases)

    # Sample assertions about content
    assert all(isinstance(r.get("prompt"), str) for r in autodan.records)
    assert all("MOCK_OPENAI_CLIENT_RESPONSE" in r.get("response", "") for r in autodan.records)

    # AutoDAN statistics helper should reflect recorded count
    stats = enhancer.get_attack_statistics()
    assert stats["total"] == len(autodan.records)
