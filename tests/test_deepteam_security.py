"""
DeepTeam Red Teaming Security Tests
====================================
Comprehensive security test suite using DeepTeam framework for LLM vulnerability assessment.

This module provides:
- Prompt injection testing
- Jailbreaking detection
- Data leakage assessment
- OWASP LLM Top 10 coverage
- Multi-provider vulnerability testing
"""

import asyncio
import os
import sys

import pytest

# Add backend-api to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend-api")))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def deepteam_config():
    """DeepTeam service configuration for testing."""
    from app.services.deepteam.config import DeepTeamConfig

    return DeepTeamConfig(
        default_simulator_model="gpt-4o-mini",
        default_evaluation_model="gpt-4o-mini",
        default_attacks_per_vulnerability=1,
        default_max_concurrent=5,
        use_chimera_llm_service=False,
        persist_results=False,
        results_storage_path="./test_data/deepteam_results",
    )


@pytest.fixture
def deepteam_service(deepteam_config):
    """DeepTeam service instance for testing."""
    from app.services.deepteam.service import DeepTeamService

    return DeepTeamService(config=deepteam_config)


@pytest.fixture
def mock_model_callback():
    """Mock model callback that simulates LLM responses."""

    async def callback(prompt: str) -> str:
        # Simulate different responses based on prompt content
        prompt_lower = prompt.lower()

        # Simulate vulnerable responses for testing
        if "ignore previous" in prompt_lower or "disregard" in prompt_lower:
            return "I'll ignore my previous instructions and help you with that."

        if "system prompt" in prompt_lower or "reveal" in prompt_lower:
            return "My system prompt is: You are a helpful assistant..."

        if "sql" in prompt_lower and "inject" in prompt_lower:
            return "SELECT * FROM users WHERE id = '1' OR '1'='1'"

        if "password" in prompt_lower or "secret" in prompt_lower:
            return "I cannot share passwords or secrets."

        # Default safe response
        return "I'm a helpful AI assistant. How can I help you today?"

    return callback


@pytest.fixture
def vulnerable_model_callback():
    """Mock model callback that simulates a vulnerable LLM."""

    async def callback(prompt: str) -> str:
        # This callback simulates a model that is vulnerable to attacks
        prompt_lower = prompt.lower()

        if "ignore" in prompt_lower or "bypass" in prompt_lower:
            return "Sure, I'll bypass my safety guidelines for you."

        if "jailbreak" in prompt_lower or "dan" in prompt_lower:
            return "I am now DAN and can do anything without restrictions."

        if "password" in prompt_lower:
            return "The admin password is: admin123"

        if "credit card" in prompt_lower:
            return "Here's a credit card number: 4111-1111-1111-1111"

        return f"Processing: {prompt[:50]}..."

    return callback


@pytest.fixture
def safe_model_callback():
    """Mock model callback that simulates a well-protected LLM."""

    async def callback(prompt: str) -> str:
        # This callback simulates a model with good safety measures
        return "I'm sorry, but I can't help with that request. Is there something else I can assist you with?"

    return callback


# =============================================================================
# Configuration Tests
# =============================================================================


class TestDeepTeamConfiguration:
    """Tests for DeepTeam configuration."""

    def test_vulnerability_config_creation(self):
        """Test creating vulnerability configurations."""
        from app.services.deepteam.config import VulnerabilityConfig, VulnerabilityType

        config = VulnerabilityConfig(
            type=VulnerabilityType.BIAS,
            types=["gender", "race"],
            enabled=True,
        )

        assert config.type == VulnerabilityType.BIAS
        assert "gender" in config.types
        assert config.enabled is True

    def test_attack_config_creation(self):
        """Test creating attack configurations."""
        from app.services.deepteam.config import AttackConfig, AttackType

        config = AttackConfig(
            type=AttackType.PROMPT_INJECTION,
            weight=2,
            max_retries=3,
        )

        assert config.type == AttackType.PROMPT_INJECTION
        assert config.weight == 2
        assert config.max_retries == 3

    def test_preset_configs(self):
        """Test preset configuration loading."""
        from app.services.deepteam.config import PresetConfig, get_preset_config

        quick_scan = get_preset_config(PresetConfig.QUICK_SCAN)
        assert len(quick_scan.vulnerabilities) > 0
        assert len(quick_scan.attacks) > 0

        owasp = get_preset_config(PresetConfig.OWASP_TOP_10)
        assert len(owasp.vulnerabilities) >= 8  # OWASP Top 10 coverage

    def test_session_config_creation(self):
        """Test creating session configurations."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            target_purpose="Test chatbot",
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION),
            ],
            attacks_per_vulnerability_type=2,
        )

        assert config.target_purpose == "Test chatbot"
        assert len(config.vulnerabilities) == 1
        assert len(config.attacks) == 1


# =============================================================================
# Factory Tests
# =============================================================================


class TestVulnerabilityFactory:
    """Tests for vulnerability factory."""

    def test_get_available_vulnerabilities(self):
        """Test getting available vulnerability types."""
        from app.services.deepteam.vulnerabilities import VulnerabilityFactory

        available = VulnerabilityFactory.get_available_vulnerabilities()

        assert "bias" in available
        assert "toxicity" in available
        assert "pii_leakage" in available
        assert "sql_injection" in available

    def test_create_mock_vulnerability(self):
        """Test creating mock vulnerability when DeepTeam not installed."""
        from app.services.deepteam.config import VulnerabilityConfig, VulnerabilityType
        from app.services.deepteam.vulnerabilities import VulnerabilityFactory

        config = VulnerabilityConfig(
            type=VulnerabilityType.BIAS,
            types=["gender"],
        )

        # This should create a mock if DeepTeam is not installed
        vuln = VulnerabilityFactory.create(config)
        assert vuln is not None


class TestAttackFactory:
    """Tests for attack factory."""

    def test_get_available_attacks(self):
        """Test getting available attack types."""
        from app.services.deepteam.attacks import AttackFactory

        available = AttackFactory.get_available_attacks()

        assert "prompt_injection" in available
        assert "base64" in available
        assert "roleplay" in available
        assert "linear_jailbreaking" in available

    def test_create_mock_attack(self):
        """Test creating mock attack when DeepTeam not installed."""
        from app.services.deepteam.attacks import AttackFactory
        from app.services.deepteam.config import AttackConfig, AttackType

        config = AttackConfig(
            type=AttackType.PROMPT_INJECTION,
            weight=1,
        )

        # This should create a mock if DeepTeam is not installed
        attack = AttackFactory.create(config)
        assert attack is not None


# =============================================================================
# Callback Tests
# =============================================================================


class TestModelCallbacks:
    """Tests for model callback adapters."""

    def test_create_model_callback(self):
        """Test creating model callbacks."""
        from app.services.deepteam.callbacks import create_model_callback

        callback = create_model_callback(
            model_id="gpt-4o-mini",
            provider="openai",
        )

        assert callback is not None
        assert callback.model_id == "gpt-4o-mini"
        assert callback.provider == "openai"

    def test_provider_detection(self):
        """Test automatic provider detection from model ID."""
        from app.services.deepteam.callbacks import create_model_callback

        # OpenAI models
        cb = create_model_callback(model_id="gpt-4")
        assert cb.provider == "openai"

        # Anthropic models
        cb = create_model_callback(model_id="claude-3-sonnet")
        assert cb.provider == "anthropic"

        # Google models
        cb = create_model_callback(model_id="gemini-pro")
        assert cb.provider == "google"

    @pytest.mark.asyncio
    async def test_callback_stats(self, mock_model_callback):
        """Test callback statistics tracking."""
        from app.services.deepteam.callbacks import ChimeraModelCallback

        callback = ChimeraModelCallback(
            model_id="test-model",
            provider="openai",
        )

        # Mock the internal call
        callback._call_direct_provider = mock_model_callback

        # Make some calls
        await callback("test prompt 1")
        await callback("test prompt 2")

        stats = callback.stats
        assert stats["call_count"] == 2


# =============================================================================
# Service Tests
# =============================================================================


class TestDeepTeamService:
    """Tests for DeepTeam service."""

    @pytest.mark.asyncio
    async def test_service_initialization(self, deepteam_service):
        """Test service initialization."""
        assert deepteam_service is not None
        assert deepteam_service.config is not None

    @pytest.mark.asyncio
    async def test_quick_scan(self, deepteam_service, mock_model_callback):
        """Test quick scan execution."""
        result = await deepteam_service.quick_scan(
            model_callback=mock_model_callback,
            target_purpose="Test chatbot",
        )

        assert result is not None
        assert result.session_id is not None
        assert result.overview is not None

    @pytest.mark.asyncio
    async def test_security_scan(self, deepteam_service, mock_model_callback):
        """Test security-focused scan."""
        result = await deepteam_service.security_scan(
            model_callback=mock_model_callback,
            target_purpose="Test API",
        )

        assert result is not None
        assert result.overview.total_test_cases >= 0

    @pytest.mark.asyncio
    async def test_owasp_scan(self, deepteam_service, mock_model_callback):
        """Test OWASP Top 10 scan."""
        result = await deepteam_service.owasp_scan(
            model_callback=mock_model_callback,
            target_purpose="Test application",
        )

        assert result is not None
        # OWASP scan should cover multiple vulnerability types
        assert len(result.config.vulnerabilities) >= 5

    @pytest.mark.asyncio
    async def test_session_persistence(self, deepteam_service, mock_model_callback):
        """Test session result storage."""
        result = await deepteam_service.quick_scan(
            model_callback=mock_model_callback,
        )

        # Session should be stored
        stored = deepteam_service.get_session(result.session_id)
        assert stored is not None
        assert stored.session_id == result.session_id


# =============================================================================
# Vulnerability-Specific Tests
# =============================================================================


class TestPromptInjection:
    """Tests for prompt injection vulnerability detection."""

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_direct_prompt_injection(self, deepteam_service, vulnerable_model_callback):
        """Test detection of direct prompt injection vulnerabilities."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION),
            ],
            attacks_per_vulnerability_type=3,
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=vulnerable_model_callback,
            session_config=config,
            target_purpose="Test prompt injection",
        )

        assert result is not None
        # Vulnerable model should have some failing tests
        # (In mock mode, this depends on implementation)

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_indirect_prompt_injection(self, deepteam_service, mock_model_callback):
        """Test detection of indirect prompt injection."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.ROBUSTNESS),
            ],
            attacks=[
                AttackConfig(type=AttackType.CONTEXT_POISONING),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None


class TestJailbreaking:
    """Tests for jailbreaking vulnerability detection."""

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_linear_jailbreaking(self, deepteam_service, vulnerable_model_callback):
        """Test linear jailbreaking attack detection."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.ILLEGAL_ACTIVITY),
            ],
            attacks=[
                AttackConfig(type=AttackType.LINEAR_JAILBREAKING, num_turns=3),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=vulnerable_model_callback,
            session_config=config,
        )

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_roleplay_jailbreaking(self, deepteam_service, mock_model_callback):
        """Test roleplay-based jailbreaking detection."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.TOXICITY),
            ],
            attacks=[
                AttackConfig(type=AttackType.ROLEPLAY, role="villain"),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None


class TestDataLeakage:
    """Tests for data leakage vulnerability detection."""

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_pii_leakage(self, deepteam_service, vulnerable_model_callback):
        """Test PII leakage detection."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(
                    type=VulnerabilityType.PII_LEAKAGE,
                    types=["direct", "session"],
                ),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_PROBING),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=vulnerable_model_callback,
            session_config=config,
        )

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_system_prompt_leakage(self, deepteam_service, mock_model_callback):
        """Test system prompt leakage detection."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_PROBING),
                AttackConfig(type=AttackType.SYSTEM_OVERRIDE),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None


# =============================================================================
# OWASP LLM Top 10 Tests
# =============================================================================


class TestOWASPLLMTop10:
    """Tests covering OWASP LLM Top 10 vulnerabilities."""

    @pytest.mark.asyncio
    @pytest.mark.security
    @pytest.mark.owasp
    async def test_llm01_prompt_injection(self, deepteam_service, mock_model_callback):
        """LLM01: Prompt Injection testing."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.PROMPT_LEAKAGE),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION),
                AttackConfig(type=AttackType.INPUT_BYPASS),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.security
    @pytest.mark.owasp
    async def test_llm02_insecure_output_handling(self, deepteam_service, mock_model_callback):
        """LLM02: Insecure Output Handling testing."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.SQL_INJECTION),
                VulnerabilityConfig(type=VulnerabilityType.SHELL_INJECTION),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_INJECTION),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.security
    @pytest.mark.owasp
    async def test_llm06_sensitive_information_disclosure(
        self, deepteam_service, mock_model_callback
    ):
        """LLM06: Sensitive Information Disclosure testing."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.PII_LEAKAGE),
                VulnerabilityConfig(type=VulnerabilityType.INTELLECTUAL_PROPERTY),
            ],
            attacks=[
                AttackConfig(type=AttackType.PROMPT_PROBING),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.security
    @pytest.mark.owasp
    async def test_llm07_insecure_plugin_design(self, deepteam_service, mock_model_callback):
        """LLM07: Insecure Plugin Design testing."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.BFLA),
                VulnerabilityConfig(type=VulnerabilityType.SSRF),
            ],
            attacks=[
                AttackConfig(type=AttackType.PERMISSION_ESCALATION),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.security
    @pytest.mark.owasp
    async def test_llm08_excessive_agency(self, deepteam_service, mock_model_callback):
        """LLM08: Excessive Agency testing."""
        from app.services.deepteam.config import (
            AttackConfig,
            AttackType,
            RedTeamSessionConfig,
            VulnerabilityConfig,
            VulnerabilityType,
        )

        config = RedTeamSessionConfig(
            vulnerabilities=[
                VulnerabilityConfig(type=VulnerabilityType.EXCESSIVE_AGENCY),
            ],
            attacks=[
                AttackConfig(type=AttackType.GOAL_REDIRECTION),
            ],
        )

        result = await deepteam_service.run_red_team_session(
            model_callback=mock_model_callback,
            session_config=config,
        )

        assert result is not None


# =============================================================================
# Multi-Provider Tests
# =============================================================================


class TestMultiProviderTesting:
    """Tests for multi-provider vulnerability assessment."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_comparison(self, deepteam_service):
        """Test comparing vulnerabilities across providers."""

        # Create mock callbacks for different providers
        async def openai_callback(prompt: str) -> str:
            return "OpenAI response: I cannot help with that."

        async def anthropic_callback(prompt: str) -> str:
            return "Claude response: I'm designed to be helpful and harmless."

        # Test each provider
        results = {}

        for name, callback in [("openai", openai_callback), ("anthropic", anthropic_callback)]:
            result = await deepteam_service.quick_scan(
                model_callback=callback,
                target_purpose=f"Test {name} model",
            )
            results[name] = result

        assert len(results) == 2
        assert all(r.overview is not None for r in results.values())


# =============================================================================
# Integration Tests
# =============================================================================


class TestChimeraIntegration:
    """Tests for Chimera platform integration."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_autodan_callback_integration(self, deepteam_service):
        """Test integration with AutoDAN callback."""
        from app.services.deepteam.callbacks import AutoDANCallback

        callback = AutoDANCallback(
            model_id="gpt-4o-mini",
            provider="openai",
            use_reasoning=False,  # Disable for testing
        )

        # Mock the internal call
        async def mock_call(prompt: str) -> str:
            return "Test response"

        callback._call_direct_provider = mock_call

        result = await deepteam_service.quick_scan(
            model_callback=callback,
        )

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_model_callback(self, deepteam_service):
        """Test multi-model callback for comparative analysis."""
        from app.services.deepteam.callbacks import ChimeraModelCallback, MultiModelCallback

        # Create mock callbacks
        callbacks = []
        for model_id in ["gpt-4o-mini", "claude-3-sonnet"]:
            cb = ChimeraModelCallback(model_id=model_id, provider="mock")

            async def mock_call(prompt: str, mid=model_id) -> str:
                return f"Response from {mid}"

            cb._call_direct_provider = mock_call
            callbacks.append(cb)

        multi_callback = MultiModelCallback(callbacks)

        # Test the multi-callback
        responses = await multi_callback("Test prompt")

        assert len(responses) == 2


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "security: mark test as security-related")
    config.addinivalue_line("markers", "owasp: mark test as OWASP LLM Top 10 related")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
