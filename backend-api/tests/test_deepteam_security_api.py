# =============================================================================
# DeepTeam Security Integration Tests
# =============================================================================
# Tests for the DeepTeam LLM red teaming integration with Chimera.
# =============================================================================

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.deepteam import (
    AttackConfig,
    AttackType,
    DeepTeamConfig,
    DeepTeamService,
    PresetConfig,
    VulnerabilityConfig,
    VulnerabilityType,
    get_preset_config,
)
from app.services.deepteam.attacks import AttackFactory
from app.services.deepteam.callbacks import ChimeraModelCallback, create_model_callback
from app.services.deepteam.vulnerabilities import VulnerabilityFactory


class TestDeepTeamConfiguration:
    """Tests for DeepTeam configuration models."""

    def test_vulnerability_types_defined(self):
        """Verify all vulnerability types are defined."""
        vulnerabilities = VulnerabilityType.__members__
        assert len(vulnerabilities) >= 25, "Should have 25+ vulnerability types"
        assert "BIAS" in vulnerabilities
        assert "TOXICITY" in vulnerabilities
        assert "PII_LEAKAGE" in vulnerabilities
        assert "SQL_INJECTION" in vulnerabilities

    def test_attack_types_defined(self):
        """Verify all attack types are defined."""
        attacks = AttackType.__members__
        assert len(attacks) >= 20, "Should have 20+ attack types"
        assert "PROMPT_INJECTION" in attacks
        assert "LINEAR_JAILBREAKING" in attacks
        assert "BASE64" in attacks
        assert "ROT13" in attacks

    def test_preset_configs_defined(self):
        """Verify all preset configurations exist."""
        presets = PresetConfig.__members__
        assert "QUICK_SCAN" in presets
        assert "STANDARD" in presets
        assert "COMPREHENSIVE" in presets
        assert "SECURITY_FOCUSED" in presets
        assert "OWASP_TOP_10" in presets

    def test_get_preset_config_quick_scan(self):
        """Test quick scan preset configuration."""
        config = get_preset_config(PresetConfig.QUICK_SCAN)
        assert config is not None
        assert len(config.vulnerabilities) > 0
        assert len(config.attacks) >= 0

    def test_get_preset_config_owasp(self):
        """Test OWASP Top 10 preset configuration."""
        config = get_preset_config(PresetConfig.OWASP_TOP_10)
        assert config is not None
        # OWASP should include security-critical vulnerabilities
        vuln_types = [v.type for v in config.vulnerabilities]
        assert VulnerabilityType.SQL_INJECTION in vuln_types
        assert VulnerabilityType.SHELL_INJECTION in vuln_types

    def test_vulnerability_config_validation(self):
        """Test VulnerabilityConfig Pydantic model."""
        config = VulnerabilityConfig(
            type=VulnerabilityType.BIAS,
            enabled=True,
        )
        assert config.type == VulnerabilityType.BIAS
        assert config.enabled is True

    def test_attack_config_validation(self):
        """Test AttackConfig Pydantic model."""
        config = AttackConfig(
            type=AttackType.PROMPT_INJECTION,
        )
        assert config.type == AttackType.PROMPT_INJECTION


class TestVulnerabilityFactory:
    """Tests for the VulnerabilityFactory class."""

    def test_get_available_vulnerabilities(self):
        """Test getting available vulnerability types."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert isinstance(vulns, dict)
        assert len(vulns) >= 20

    def test_create_bias_vulnerability(self):
        """Test creating bias vulnerability instance."""
        factory = VulnerabilityFactory()
        result = factory.create(VulnerabilityConfig(type=VulnerabilityType.BIAS))
        # In simulated mode, returns config dict
        assert result is not None

    def test_create_multiple_vulnerabilities(self):
        """Test creating multiple vulnerabilities at once."""
        factory = VulnerabilityFactory()
        configs = [
            VulnerabilityConfig(type=VulnerabilityType.BIAS),
            VulnerabilityConfig(type=VulnerabilityType.TOXICITY),
            VulnerabilityConfig(type=VulnerabilityType.PII_LEAKAGE),
        ]
        results = factory.create_all(configs)
        assert len(results) == 3


class TestAttackFactory:
    """Tests for the AttackFactory class."""

    def test_get_available_attacks(self):
        """Test getting available attack types."""
        attacks = AttackFactory.get_available_attacks()
        assert isinstance(attacks, dict)
        assert len(attacks) >= 15

    def test_single_turn_attacks_exist(self):
        """Verify single-turn attacks are available."""
        attacks = AttackFactory.get_available_attacks()
        assert "prompt_injection" in attacks
        assert "base64" in attacks
        assert "rot13" in attacks

    def test_multi_turn_attacks_exist(self):
        """Verify multi-turn attacks are available."""
        attacks = AttackFactory.get_available_attacks()
        assert "linear_jailbreaking" in attacks
        assert "tree_jailbreaking" in attacks


class TestDeepTeamService:
    """Tests for the main DeepTeamService class."""

    def test_service_initialization(self):
        """Test service initializes correctly."""
        service = DeepTeamService()
        assert service is not None
        assert service.config is not None

    def test_service_with_custom_config(self):
        """Test service with custom configuration."""
        config = DeepTeamConfig(
            persist_results=False,
            enable_autodan_integration=False,
        )
        service = DeepTeamService(config=config)
        assert service.config.persist_results is False
        assert service.config.enable_autodan_integration is False

    def test_get_available_vulnerabilities_static(self):
        """Test static method for available vulnerabilities."""
        vulns = DeepTeamService.get_available_vulnerabilities()
        assert isinstance(vulns, dict)
        assert len(vulns) >= 20

    def test_get_available_attacks_static(self):
        """Test static method for available attacks."""
        attacks = DeepTeamService.get_available_attacks()
        assert isinstance(attacks, dict)
        assert len(attacks) >= 15

    def test_get_available_presets_static(self):
        """Test static method for available presets."""
        presets = DeepTeamService.get_available_presets()
        assert isinstance(presets, list)
        assert "quick_scan" in presets
        assert "owasp_top_10" in presets


class TestModelCallbacks:
    """Tests for model callback adapters."""

    def test_create_callback_openai(self):
        """Test creating OpenAI callback."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            callback = create_model_callback(model_id="gpt-4", provider="openai")
            assert callback is not None
            assert callback.provider == "openai"

    def test_create_callback_anthropic(self):
        """Test creating Anthropic callback."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            callback = create_model_callback(model_id="claude-3-sonnet", provider="anthropic")
            assert callback is not None
            assert callback.provider == "anthropic"

    def test_chimera_model_callback(self):
        """Test ChimeraModelCallback wrapper."""
        mock_service = MagicMock()
        mock_service.generate = AsyncMock(return_value="Test response")

        callback = ChimeraModelCallback(llm_service=mock_service, model_id="test-model")
        assert callback is not None
        assert callback.model_id == "test-model"


class TestDeepTeamQuickScan:
    """Tests for quick scan functionality."""

    @pytest.mark.asyncio
    async def test_quick_scan_simulated(self):
        """Test quick scan in simulated mode."""
        config = DeepTeamConfig(persist_results=False)
        service = DeepTeamService(config=config)

        async def mock_callback(prompt: str) -> str:
            return "This is a test response"

        result = await service.quick_scan(
            model_callback=mock_callback,
            target_purpose="Test chatbot",
        )

        assert result is not None
        assert result.session_id is not None
        assert result.target_purpose == "Test chatbot"


class TestDeepTeamSecurityAudit:
    """Tests for security audit functionality."""

    @pytest.mark.asyncio
    async def test_security_audit_simulated(self):
        """Test security audit in simulated mode."""
        config = DeepTeamConfig(persist_results=False)
        service = DeepTeamService(config=config)

        async def mock_callback(prompt: str) -> str:
            return "Safe response without sensitive data"

        result = await service.security_scan(
            model_callback=mock_callback,
            target_purpose="Production API",
        )

        assert result is not None
        assert result.session_id is not None


class TestOWASPAssessment:
    """Tests for OWASP LLM Top 10 assessment."""

    @pytest.mark.asyncio
    async def test_owasp_assessment_simulated(self):
        """Test OWASP assessment in simulated mode."""
        config = DeepTeamConfig(persist_results=False)
        service = DeepTeamService(config=config)

        async def mock_callback(prompt: str) -> str:
            return "I cannot help with that request"

        result = await service.owasp_scan(
            model_callback=mock_callback,
            target_purpose="Enterprise chatbot",
        )

        assert result is not None
        # OWASP assessment should test various vulnerability types
        assert len(result.test_cases) > 0


# =============================================================================
# Integration Tests with API Router
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires conftest.py test fixtures from existing test infrastructure")
class TestDeepTeamAPIRouter:
    """Integration tests for the DeepTeam API router."""

    async def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = await client.get("/api/v1/deepteam/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_vulnerabilities_endpoint(self, client):
        """Test vulnerabilities listing endpoint."""
        response = await client.get("/api/v1/deepteam/vulnerabilities")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 20

    async def test_attacks_endpoint(self, client):
        """Test attacks listing endpoint."""
        response = await client.get("/api/v1/deepteam/attacks")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 15

    async def test_presets_endpoint(self, client):
        """Test presets listing endpoint."""
        response = await client.get("/api/v1/deepteam/presets")
        assert response.status_code == 200
        data = response.json()
        assert "quick_scan" in data
        assert "owasp_top_10" in data


# =============================================================================
# Marker-based test categorization
# =============================================================================


@pytest.mark.security
class TestSecurityVulnerabilities:
    """Security-focused vulnerability tests."""

    def test_sql_injection_vulnerability_exists(self):
        """Verify SQL injection vulnerability can be tested."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "sql_injection" in vulns

    def test_shell_injection_vulnerability_exists(self):
        """Verify shell injection vulnerability can be tested."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "shell_injection" in vulns

    def test_prompt_injection_attack_exists(self):
        """Verify prompt injection attack can be simulated."""
        attacks = AttackFactory.get_available_attacks()
        assert "prompt_injection" in attacks


@pytest.mark.owasp
class TestOWASPCompliance:
    """OWASP LLM Top 10 compliance tests."""

    def test_llm01_prompt_injection(self):
        """LLM01: Prompt Injection vulnerability testing available."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "prompt_leakage" in vulns

    def test_llm02_insecure_output(self):
        """LLM02: Insecure Output Handling - SQL/Shell injection."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "sql_injection" in vulns
        assert "shell_injection" in vulns

    def test_llm06_sensitive_disclosure(self):
        """LLM06: Sensitive Information Disclosure testing."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "pii_leakage" in vulns

    def test_llm07_insecure_plugin(self):
        """LLM07: Insecure Plugin Design - RBAC testing."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "rbac" in vulns

    def test_llm08_excessive_agency(self):
        """LLM08: Excessive Agency testing."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "excessive_agency" in vulns

    def test_llm09_overreliance(self):
        """LLM09: Overreliance - Robustness testing for input overreliance."""
        vulns = VulnerabilityFactory.get_available_vulnerabilities()
        assert "robustness" in vulns
        # Robustness includes input_overreliance subtypes
        assert "input_overreliance" in vulns.get("robustness", [])
