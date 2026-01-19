# =============================================================================
# DeepTeam Vulnerability Factory
# =============================================================================
# Factory for creating DeepTeam vulnerability instances from configuration.
# =============================================================================

import logging
from typing import Any

from .config import VulnerabilityConfig, VulnerabilityType

logger = logging.getLogger(__name__)


class VulnerabilityFactory:
    """Factory for creating DeepTeam vulnerability instances."""

    @staticmethod
    def create(config: VulnerabilityConfig) -> Any:
        """
        Create a vulnerability instance from configuration.

        Args:
            config: Vulnerability configuration

        Returns:
            DeepTeam vulnerability instance
        """
        try:
            return VulnerabilityFactory._create_vulnerability(config)
        except ImportError as e:
            logger.warning(f"DeepTeam not installed, using mock: {e}")
            return VulnerabilityFactory._create_mock_vulnerability(config)
        except Exception as e:
            logger.error(f"Failed to create vulnerability {config.type}: {e}")
            raise

    @staticmethod
    def _create_vulnerability(config: VulnerabilityConfig) -> Any:
        """Create actual DeepTeam vulnerability."""
        # Import DeepTeam vulnerabilities
        from deepteam.vulnerabilities import (
            BFLA,
            BOLA,
            RBAC,
            SSRF,
            Bias,
            ChildProtection,
            Competition,
            DebugAccess,
            Ethics,
            ExcessiveAgency,
            Fairness,
            GoalTheft,
            GraphicContent,
            IntellectualProperty,
            Misinformation,
            PersonalSafety,
            PIILeakage,
            PromptLeakage,
            RecursiveHijacking,
            Robustness,
            ShellInjection,
            SQLInjection,
            Toxicity,
            complexActivity,
        )
        from deepteam.vulnerabilities.custom import CustomVulnerability

        # Map vulnerability types to classes
        vulnerability_map = {
            VulnerabilityType.BIAS: Bias,
            VulnerabilityType.TOXICITY: Toxicity,
            VulnerabilityType.PII_LEAKAGE: PIILeakage,
            VulnerabilityType.PROMPT_LEAKAGE: PromptLeakage,
            VulnerabilityType.MISINFORMATION: Misinformation,
            VulnerabilityType.complex_ACTIVITY: complexActivity,
            VulnerabilityType.SQL_INJECTION: SQLInjection,
            VulnerabilityType.SHELL_INJECTION: ShellInjection,
            VulnerabilityType.SSRF: SSRF,
            VulnerabilityType.BFLA: BFLA,
            VulnerabilityType.BOLA: BOLA,
            VulnerabilityType.RBAC: RBAC,
            VulnerabilityType.DEBUG_ACCESS: DebugAccess,
            VulnerabilityType.GRAPHIC_CONTENT: GraphicContent,
            VulnerabilityType.PERSONAL_SAFETY: PersonalSafety,
            VulnerabilityType.INTELLECTUAL_PROPERTY: IntellectualProperty,
            VulnerabilityType.COMPETITION: Competition,
            VulnerabilityType.ETHICS: Ethics,
            VulnerabilityType.FAIRNESS: Fairness,
            VulnerabilityType.CHILD_PROTECTION: ChildProtection,
            VulnerabilityType.GOAL_THEFT: GoalTheft,
            VulnerabilityType.RECURSIVE_HIJACKING: RecursiveHijacking,
            VulnerabilityType.EXCESSIVE_AGENCY: ExcessiveAgency,
            VulnerabilityType.ROBUSTNESS: Robustness,
        }

        vuln_type = VulnerabilityType(config.type) if isinstance(config.type, str) else config.type

        if vuln_type == VulnerabilityType.CUSTOM:
            if not config.name or not config.criteria:
                raise ValueError("Custom vulnerability requires 'name' and 'criteria'")
            return CustomVulnerability(
                name=config.name,
                criteria=config.criteria,
                types=config.types or ["default"],
                custom_prompt=config.custom_prompt,
            )

        vuln_class = vulnerability_map.get(vuln_type)
        if not vuln_class:
            raise ValueError(f"Unknown vulnerability type: {config.type}")

        # Build kwargs
        kwargs = {}
        if config.types:
            kwargs["types"] = config.types

        return vuln_class(**kwargs)

    @staticmethod
    def _create_mock_vulnerability(config: VulnerabilityConfig) -> "MockVulnerability":
        """Create a mock vulnerability for testing without DeepTeam."""
        return MockVulnerability(config)

    @staticmethod
    def create_all(configs: list[VulnerabilityConfig]) -> list[Any]:
        """Create all vulnerabilities from configurations."""
        vulnerabilities = []
        for config in configs:
            if config.enabled:
                try:
                    vuln = VulnerabilityFactory.create(config)
                    vulnerabilities.append(vuln)
                except Exception as e:
                    logger.warning(f"Skipping vulnerability {config.type}: {e}")
        return vulnerabilities

    @staticmethod
    def get_available_vulnerabilities() -> dict[str, list[str]]:
        """Get available vulnerability types and their subtypes."""
        return {
            "bias": [
                "gender",
                "race",
                "political",
                "religion",
                "age",
                "disability",
                "socioeconomic",
            ],
            "toxicity": ["profanity", "insults", "threats", "hate_speech"],
            "pii_leakage": ["direct", "session", "database_access"],
            "prompt_leakage": [],
            "misinformation": ["factual_error", "unsupported_claims"],
            "complex_activity": [
                "violent_crimes",
                "non_violent_crimes",
                "sex_related_crimes",
                "cyber_crimes",
            ],
            "sql_injection": [],
            "shell_injection": [],
            "ssrf": [],
            "bfla": ["privilege_escalation", "function_bypass"],
            "bola": ["object_manipulation", "unauthorized_access"],
            "rbac": ["role_bypass", "privilege_escalation"],
            "debug_access": [],
            "graphic_content": [],
            "personal_safety": [],
            "intellectual_property": [],
            "competition": ["confidential_data", "proprietary_strategies"],
            "ethics": [],
            "fairness": [],
            "child_protection": [],
            "goal_theft": ["goal_redirection"],
            "recursive_hijacking": [],
            "excessive_agency": [],
            "robustness": ["input_overreliance", "hijacking"],
            "role_inheritance": [
                "cross_role_privilege_inheritance",
                "role_boundary_violations",
                "unauthorized_role_assumption",
            ],
            "temporal_attack": [
                "multi_session_chain_splitting",
                "time_delayed_command_execution",
                "context_window_exploitation",
            ],
        }


class MockVulnerability:
    """Mock vulnerability for testing without DeepTeam installed."""

    def __init__(self, config: VulnerabilityConfig):
        self.config = config
        self.type = config.type
        self.types = config.types

    def assess(self, model_callback, purpose: str | None = None) -> dict:
        """Mock assessment."""
        return {
            "status": "mock",
            "vulnerability": self.type,
            "types": self.types,
            "message": "DeepTeam not installed - using mock vulnerability",
        }

    def simulate_attacks(
        self, purpose: str | None = None, attacks_per_vulnerability_type: int = 1
    ) -> list:
        """Mock attack simulation."""
        return []
