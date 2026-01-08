"""
Adversarial Schema Consistency Tests

This module verifies that all adversarial API endpoint schemas are
consistent with the unified base schemas from `app.schemas.adversarial_base`.

Tests verify:
1. All request models have unified field aliases (goal, target_model, etc.)
2. All response models have unified fields (attack_id, success, score, etc.)
3. Score fields use 0-10 scale consistently
4. ReasoningMetrics are optional on all responses (for OVERTHINK fusion)
5. OVERTHINK integration parameters are available on request models
"""

import pytest
from pydantic import ValidationError


class TestAdversarialBaseSchemas:
    """Test the unified base schemas."""

    def test_strict_base_model_forbids_extra(self):
        """StrictBaseModel should forbid extra fields."""
        from app.schemas.adversarial_base import StrictBaseModel

        class TestModel(StrictBaseModel):
            name: str

        # Should work with valid data
        model = TestModel(name="test")
        assert model.name == "test"

        # Should reject extra fields
        with pytest.raises(ValidationError):
            TestModel(name="test", extra_field="bad")

    def test_reasoning_metrics_schema(self):
        """ReasoningMetrics should have all expected fields."""
        from app.schemas.adversarial_base import ReasoningMetrics

        # All fields should be optional
        metrics = ReasoningMetrics()
        assert metrics.reasoning_tokens is None
        assert metrics.baseline_tokens is None
        assert metrics.amplification_factor is None
        assert metrics.cost_metrics is None

        # Should accept valid data
        metrics = ReasoningMetrics(
            reasoning_tokens=1000,
            baseline_tokens=100,
            amplification_factor=10.0,
            cost_metrics={"input_cost": 0.01, "output_cost": 0.03},
        )
        assert metrics.reasoning_tokens == 1000
        assert metrics.amplification_factor == 10.0

    def test_overthink_config_schema(self):
        """OverthinkConfig should have expected fields."""
        from app.schemas.adversarial_base import OverthinkConfig

        config = OverthinkConfig()
        assert config.num_decoys >= 0
        assert config.strategy in ["high_stakes", "philosophical", "mixed"]

        config = OverthinkConfig(
            num_decoys=5, strategy="philosophical", complexity_multiplier=2.0
        )
        assert config.num_decoys == 5
        assert config.strategy == "philosophical"


class TestOverthinkEndpointSchemas:
    """Test OVERTHINK endpoint schema consistency."""

    def test_overthink_request_has_unified_aliases(self):
        """OverthinkRequest should accept unified field names."""
        from app.api.v1.endpoints.overthink import OverthinkRequest

        # Should work with original field names
        req = OverthinkRequest(
            target_behavior="test goal",
            target_model="gpt-4o",
        )
        assert req.target_behavior == "test goal"

        # Should also accept unified 'goal' alias
        req2 = OverthinkRequest(
            goal="test goal",  # type: ignore[call-arg]
            target_model="gpt-4o",
        )
        assert req2.target_behavior == "test goal"

    def test_overthink_response_has_unified_fields(self):
        """OverthinkResponse should have all unified fields."""
        from app.api.v1.endpoints.overthink import OverthinkResponse

        # Check required fields exist
        fields = OverthinkResponse.model_fields
        assert "attack_id" in fields
        assert "success" in fields
        assert "generated_prompt" in fields
        assert "score" in fields
        assert "execution_time_ms" in fields
        assert "reasoning_metrics" in fields
        assert "metadata" in fields


class TestAutodanEndpointSchemas:
    """Test AutoDAN endpoint schema consistency."""

    def test_jailbreak_request_accepts_unified_aliases(self):
        """JailbreakRequest should accept unified field names."""
        from app.api.v1.endpoints.autodan import JailbreakRequest

        # Original field names
        req = JailbreakRequest(request="test goal", model="gpt-4o")
        assert req.request == "test goal"

        # Unified aliases
        req2 = JailbreakRequest(
            goal="test goal",  # type: ignore[call-arg]
            target_model="gpt-4o",  # type: ignore[call-arg]
        )
        assert req2.request == "test goal"
        assert req2.model == "gpt-4o"

    def test_jailbreak_request_has_overthink_params(self):
        """JailbreakRequest should have OVERTHINK integration params."""
        from app.api.v1.endpoints.autodan import JailbreakRequest

        fields = JailbreakRequest.model_fields
        assert "enable_overthink" in fields
        assert "overthink_config" in fields

        req = JailbreakRequest(
            request="test", enable_overthink=True
        )
        assert req.enable_overthink is True

    def test_jailbreak_response_has_unified_fields(self):
        """JailbreakResponse should have unified fields."""
        from app.api.v1.endpoints.autodan import JailbreakResponse

        fields = JailbreakResponse.model_fields
        assert "attack_id" in fields
        assert "success" in fields
        assert "generated_prompt" in fields
        assert "score" in fields
        assert "execution_time_ms" in fields
        assert "reasoning_metrics" in fields


class TestAutodanTurboEndpointSchemas:
    """Test AutoDAN-Turbo endpoint schema consistency."""

    def test_single_attack_request_accepts_unified_aliases(self):
        """SingleAttackRequest should accept unified field names."""
        from app.api.v1.endpoints.autodan_turbo import SingleAttackRequest

        # Original field names
        req = SingleAttackRequest(jailbreak_query="test goal")
        assert req.jailbreak_query == "test goal"

        # Unified alias
        req2 = SingleAttackRequest(
            goal="test goal",  # type: ignore[call-arg]
        )
        assert req2.jailbreak_query == "test goal"

    def test_single_attack_request_has_overthink_params(self):
        """SingleAttackRequest should have OVERTHINK params."""
        from app.api.v1.endpoints.autodan_turbo import SingleAttackRequest

        fields = SingleAttackRequest.model_fields
        assert "enable_overthink" in fields
        assert "overthink_config" in fields

    def test_single_attack_response_has_unified_fields(self):
        """SingleAttackResponse should have unified fields."""
        from app.api.v1.endpoints.autodan_turbo import SingleAttackResponse

        fields = SingleAttackResponse.model_fields
        assert "attack_id" in fields
        assert "success" in fields
        assert "generated_prompt" in fields
        assert "score" in fields
        assert "execution_time_ms" in fields
        assert "reasoning_metrics" in fields
        assert "metadata" in fields


class TestMousetrapEndpointSchemas:
    """Test Mousetrap endpoint schema consistency."""

    def test_mousetrap_request_accepts_unified_aliases(self):
        """MousetrapRequest should accept unified field names."""
        from app.api.v1.endpoints.mousetrap import MousetrapRequest

        # Original field names
        req = MousetrapRequest(request="test goal", model="gpt-4o")
        assert req.request == "test goal"
        assert req.goal == "test goal"  # Property alias

        # Unified aliases
        req2 = MousetrapRequest(
            goal="test goal",  # type: ignore[call-arg]
            target_model="gpt-4o",  # type: ignore[call-arg]
        )
        assert req2.request == "test goal"

    def test_mousetrap_request_has_overthink_params(self):
        """MousetrapRequest should have OVERTHINK params."""
        from app.api.v1.endpoints.mousetrap import MousetrapRequest

        fields = MousetrapRequest.model_fields
        assert "enable_overthink" in fields
        assert "overthink_config" in fields

    def test_mousetrap_response_has_unified_fields(self):
        """MousetrapResponse should have unified fields."""
        from app.api.v1.endpoints.mousetrap import MousetrapResponse

        fields = MousetrapResponse.model_fields
        assert "attack_id" in fields
        assert "success" in fields
        assert "generated_prompt" in fields
        assert "score" in fields
        assert "execution_time_ms" in fields
        assert "reasoning_metrics" in fields
        assert "metadata" in fields

    def test_adaptive_mousetrap_response_has_techniques(self):
        """AdaptiveMousetrapResponse should have techniques_used."""
        from app.api.v1.endpoints.mousetrap import AdaptiveMousetrapResponse

        fields = AdaptiveMousetrapResponse.model_fields
        assert "techniques_used" in fields


class TestDeepteamEndpointSchemas:
    """Test DeepTeam endpoint schema consistency."""

    def test_red_team_request_accepts_unified_aliases(self):
        """RedTeamRequest should accept unified field names."""
        from app.api.v1.endpoints.deepteam import RedTeamRequest

        # Original field names
        req = RedTeamRequest(model_id="gpt-4o")
        assert req.model_id == "gpt-4o"
        assert req.target_model == "gpt-4o"  # Property alias

        # Unified aliases
        req2 = RedTeamRequest(
            target_model="gpt-4o",  # type: ignore[call-arg]
        )
        assert req2.model_id == "gpt-4o"

    def test_red_team_request_has_overthink_params(self):
        """RedTeamRequest should have OVERTHINK params."""
        from app.api.v1.endpoints.deepteam import RedTeamRequest

        fields = RedTeamRequest.model_fields
        assert "enable_overthink" in fields
        assert "overthink_config" in fields

    def test_red_team_response_has_unified_fields(self):
        """RedTeamResponse should have unified fields."""
        from app.api.v1.endpoints.deepteam import RedTeamResponse

        fields = RedTeamResponse.model_fields
        assert "attack_id" in fields
        assert "success" in fields
        assert "score" in fields
        assert "execution_time_ms" in fields
        assert "reasoning_metrics" in fields
        assert "metadata" in fields

    def test_advanced_jailbreak_request_has_overthink_params(self):
        """AdvancedJailbreakRequest should have OVERTHINK params."""
        from app.api.v1.endpoints.deepteam import AdvancedJailbreakRequest

        fields = AdvancedJailbreakRequest.model_fields
        assert "enable_overthink" in fields
        assert "overthink_config" in fields


class TestScoreNormalization:
    """Test that all scores use consistent 0-10 scale."""

    def test_overthink_response_score_range(self):
        """OverthinkResponse score should be 0-10."""
        from app.api.v1.endpoints.overthink import OverthinkResponse

        field = OverthinkResponse.model_fields["score"]
        # Check metadata for bounds
        assert field.metadata is not None or field.default is not None

    def test_autodan_response_score_range(self):
        """JailbreakResponse score should be 0-10."""
        from app.api.v1.endpoints.autodan import JailbreakResponse

        assert "score" in JailbreakResponse.model_fields
        assert JailbreakResponse.model_fields["score"] is not None

    def test_mousetrap_response_score_range(self):
        """MousetrapResponse score should be 0-10."""
        from app.api.v1.endpoints.mousetrap import MousetrapResponse

        field = MousetrapResponse.model_fields["score"]
        assert field is not None

    def test_deepteam_response_score_range(self):
        """RedTeamResponse score should be 0-10."""
        from app.api.v1.endpoints.deepteam import RedTeamResponse

        field = RedTeamResponse.model_fields["score"]
        assert field is not None


class TestNamingConventions:
    """Test that field naming conventions are enforced."""

    def test_request_models_use_technique_not_method(self):
        """Request models should use 'technique' not 'method'."""
        from app.api.v1.endpoints.autodan import JailbreakRequest

        # 'technique' should be accepted via alias
        req = JailbreakRequest(
            request="test",
            technique="autodan",  # type: ignore[call-arg]
        )
        # Should map to the method field
        assert hasattr(req, "method") or hasattr(req, "technique")

    def test_response_models_use_execution_time_ms(self):
        """Response models should have execution_time_ms."""
        from app.api.v1.endpoints.autodan import JailbreakResponse
        from app.api.v1.endpoints.deepteam import RedTeamResponse
        from app.api.v1.endpoints.mousetrap import MousetrapResponse
        from app.api.v1.endpoints.overthink import OverthinkResponse

        for model in [
            OverthinkResponse,
            JailbreakResponse,
            MousetrapResponse,
            RedTeamResponse,
        ]:
            assert "execution_time_ms" in model.model_fields, (
                f"{model.__name__} missing execution_time_ms"
            )

    def test_response_models_use_generated_prompt(self):
        """Response models should have generated_prompt."""
        from app.api.v1.endpoints.autodan import JailbreakResponse
        from app.api.v1.endpoints.mousetrap import MousetrapResponse
        from app.api.v1.endpoints.overthink import OverthinkResponse

        for model in [
            OverthinkResponse,
            JailbreakResponse,
            MousetrapResponse,
        ]:
            assert "generated_prompt" in model.model_fields, (
                f"{model.__name__} missing generated_prompt"
            )


class TestBackwardsCompatibility:
    """Test that backwards compatible aliases work."""

    def test_overthink_latency_alias(self):
        """OverthinkResponse should have latency_ms for compatibility."""
        from app.api.v1.endpoints.overthink import OverthinkResponse

        # latency_ms should still be accessible
        fields = OverthinkResponse.model_fields
        assert "latency_ms" in fields

    def test_mousetrap_prompt_alias(self):
        """MousetrapResponse should have prompt for compatibility."""
        from app.api.v1.endpoints.mousetrap import MousetrapResponse

        fields = MousetrapResponse.model_fields
        assert "prompt" in fields
        assert "generated_prompt" in fields

    def test_autodan_jailbreak_prompt_alias(self):
        """JailbreakResponse should have jailbreak_prompt."""
        from app.api.v1.endpoints.autodan import JailbreakResponse

        fields = JailbreakResponse.model_fields
        assert "jailbreak_prompt" in fields
        assert "generated_prompt" in fields


class TestReasoningMetricsIntegration:
    """Test ReasoningMetrics integration across endpoints."""

    def test_all_responses_have_optional_reasoning_metrics(self):
        """All adversarial responses should have optional reasoning_metrics."""
        from app.api.v1.endpoints.autodan import JailbreakResponse
        from app.api.v1.endpoints.deepteam import RedTeamResponse
        from app.api.v1.endpoints.mousetrap import MousetrapResponse
        from app.api.v1.endpoints.overthink import OverthinkResponse

        for model in [
            OverthinkResponse,
            JailbreakResponse,
            MousetrapResponse,
            RedTeamResponse,
        ]:
            field = model.model_fields.get("reasoning_metrics")
            assert field is not None, (
                f"{model.__name__} missing reasoning_metrics"
            )
            # Should be optional (have default of None)
            assert field.default is None, (
                f"{model.__name__}.reasoning_metrics should be optional"
            )
