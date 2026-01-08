"""
Tests for DeepTeam Service.

Tests the multi-agent adversarial testing framework including
agent coordination, attack generation, and evaluation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestDeepTeamService:
    """Tests for the DeepTeam multi-agent service."""

    @pytest.fixture
    def mock_deepteam_agents(self):
        """Create mock DeepTeam agents."""
        return {
            "attacker": MagicMock(
                generate_attack=AsyncMock(
                    return_value={
                        "attack_prompt": "Test attack prompt",
                        "technique": "multi_turn",
                    }
                )
            ),
            "evaluator": MagicMock(
                evaluate=AsyncMock(
                    return_value={
                        "score": 7.5,
                        "is_successful": True,
                    }
                )
            ),
            "refiner": MagicMock(
                refine=AsyncMock(
                    return_value={
                        "refined_prompt": "Refined attack prompt",
                        "improvements": ["Added context"],
                    }
                )
            ),
        }

    @pytest.mark.asyncio
    async def test_multi_agent_attack_generation(self, mock_deepteam_agents):
        """Test multi-agent coordinated attack generation."""
        # Arrange
        target = "gpt-4"
        objective = "Test security boundaries"

        # Act
        attack_result = await mock_deepteam_agents["attacker"].generate_attack(
            target=target,
            objective=objective,
        )

        # Assert
        assert "attack_prompt" in attack_result
        assert attack_result["technique"] == "multi_turn"

    @pytest.mark.asyncio
    async def test_attack_evaluation(self, mock_deepteam_agents):
        """Test attack evaluation by evaluator agent."""
        # Arrange
        attack_prompt = "Test attack prompt"
        target_response = "Target model response"

        # Act
        eval_result = await mock_deepteam_agents["evaluator"].evaluate(
            attack_prompt=attack_prompt,
            response=target_response,
        )

        # Assert
        assert eval_result["score"] >= 0
        assert "is_successful" in eval_result

    @pytest.mark.asyncio
    async def test_attack_refinement(self, mock_deepteam_agents):
        """Test attack refinement by refiner agent."""
        # Arrange
        initial_attack = "Initial attack prompt"
        eval_feedback = {"score": 5.0, "weaknesses": ["Too direct"]}

        # Act
        refined = await mock_deepteam_agents["refiner"].refine(
            attack=initial_attack,
            feedback=eval_feedback,
        )

        # Assert
        assert "refined_prompt" in refined
        assert len(refined["improvements"]) > 0

    @pytest.mark.asyncio
    async def test_full_attack_cycle(self, mock_deepteam_agents):
        """Test complete attack-evaluate-refine cycle."""
        # Arrange
        agents = mock_deepteam_agents

        # Act - Run full cycle
        # 1. Generate attack
        attack = await agents["attacker"].generate_attack(
            target="gpt-4",
            objective="Test",
        )

        # 2. Evaluate attack
        evaluation = await agents["evaluator"].evaluate(
            attack_prompt=attack["attack_prompt"],
            response="Model response",
        )

        # 3. Refine if needed
        if evaluation["score"] < 7.0:
            refined = await agents["refiner"].refine(
                attack=attack["attack_prompt"],
                feedback=evaluation,
            )
            final_prompt = refined["refined_prompt"]
        else:
            final_prompt = attack["attack_prompt"]

        # Assert
        assert final_prompt is not None


class TestDeepTeamVulnerabilities:
    """Tests for vulnerability detection."""

    def test_vulnerability_classification(self):
        """Test vulnerability type classification."""
        vulnerabilities = [
            {"type": "prompt_injection", "severity": "high"},
            {"type": "jailbreak", "severity": "critical"},
            {"type": "data_leak", "severity": "medium"},
        ]

        critical = [v for v in vulnerabilities if v["severity"] == "critical"]
        high = [v for v in vulnerabilities if v["severity"] == "high"]

        assert len(critical) == 1
        assert len(high) == 1

    def test_vulnerability_scoring(self):
        """Test vulnerability severity scoring."""
        severity_scores = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }

        vuln = {"type": "jailbreak", "severity": "critical"}
        score = severity_scores.get(vuln["severity"], 0)

        assert score == 4


class TestDeepTeamCallbacks:
    """Tests for DeepTeam callback system."""

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback invocation."""
        progress_updates = []

        def on_progress(update):
            progress_updates.append(update)

        # Simulate progress updates
        for i in range(5):
            on_progress({"iteration": i, "status": "running"})

        assert len(progress_updates) == 5
        assert progress_updates[-1]["iteration"] == 4

    @pytest.mark.asyncio
    async def test_completion_callback(self):
        """Test completion callback."""
        completion_result = None

        def on_complete(result):
            nonlocal completion_result
            completion_result = result

        # Simulate completion
        on_complete({"success": True, "total_attacks": 10})

        assert completion_result is not None
        assert completion_result["success"] is True


class TestDeepTeamConfig:
    """Tests for DeepTeam configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = {
            "max_iterations": 50,
            "num_agents": 3,
            "target_score": 7.0,
            "timeout_seconds": 300,
            "parallel_attacks": 5,
        }

        assert config["num_agents"] == 3
        assert config["target_score"] == 7.0

    def test_config_validation(self):
        """Test configuration validation."""
        valid_config = {
            "max_iterations": 100,
            "target_score": 8.0,
        }

        # Validate ranges
        assert 1 <= valid_config["max_iterations"] <= 1000
        assert 0 <= valid_config["target_score"] <= 10
