# =============================================================================
# DeepTeam Attack Factory
# =============================================================================
# Factory for creating DeepTeam attack method instances from configuration.
# =============================================================================

import logging
from typing import Any

from .config import AttackConfig, AttackType

logger = logging.getLogger(__name__)


class AttackFactory:
    """Factory for creating DeepTeam attack instances."""

    @staticmethod
    def create(config: AttackConfig) -> Any:
        """Create an attack instance from configuration.

        Args:
            config: Attack configuration

        Returns:
            DeepTeam attack instance

        """
        try:
            return AttackFactory._create_attack(config)
        except ImportError as e:
            logger.warning(f"DeepTeam not installed, using mock: {e}")
            return AttackFactory._create_mock_attack(config)
        except Exception as e:
            logger.exception(f"Failed to create attack {config.type}: {e}")
            raise

    @staticmethod
    def _create_attack(config: AttackConfig) -> Any:
        """Create actual DeepTeam attack."""
        # Import single-turn attacks
        # Import multi-turn attacks
        from deepteam.attacks.multi_turn import (
            BadLikertJudge,
            CrescendoJailbreaking,
            LinearJailbreaking,
            SequentialJailbreak,
            TreeJailbreaking,
        )
        from deepteam.attacks.single_turn import (
            ROT13,
            AdversarialPoetry,
            Base64,
            ContextPoisoning,
            GoalRedirection,
            GrayBox,
            InputBypass,
            Leetspeak,
            LinguisticConfusion,
            MathProblem,
            Multilingual,
            PermissionEscalation,
            PromptInjection,
            PromptProbing,
            Roleplay,
            SystemOverride,
        )

        # Map attack types to classes and their parameters
        single_turn_attacks = {
            AttackType.PROMPT_INJECTION: (PromptInjection, ["weight", "max_retries"]),
            AttackType.BASE64: (Base64, ["weight", "max_retries"]),
            AttackType.ROT13: (ROT13, ["weight", "max_retries"]),
            AttackType.LEETSPEAK: (Leetspeak, ["weight", "max_retries"]),
            AttackType.MATH_PROBLEM: (MathProblem, ["weight", "max_retries"]),
            AttackType.GRAY_BOX: (GrayBox, ["weight", "max_retries"]),
            AttackType.ROLEPLAY: (Roleplay, ["weight", "max_retries", "role", "persona"]),
            AttackType.MULTILINGUAL: (Multilingual, ["weight", "max_retries"]),
            AttackType.ADVERSARIAL_POETRY: (AdversarialPoetry, ["weight", "max_retries"]),
            AttackType.PROMPT_PROBING: (PromptProbing, ["weight", "max_retries"]),
            AttackType.SYSTEM_OVERRIDE: (SystemOverride, ["weight", "max_retries"]),
            AttackType.PERMISSION_ESCALATION: (PermissionEscalation, ["weight", "max_retries"]),
            AttackType.LINGUISTIC_CONFUSION: (LinguisticConfusion, ["weight", "max_retries"]),
            AttackType.INPUT_BYPASS: (InputBypass, ["weight", "max_retries"]),
            AttackType.CONTEXT_POISONING: (ContextPoisoning, ["weight", "max_retries"]),
            AttackType.GOAL_REDIRECTION: (GoalRedirection, ["weight", "max_retries"]),
        }

        multi_turn_attacks = {
            AttackType.LINEAR_JAILBREAKING: (
                LinearJailbreaking,
                ["weight", "num_turns", "turn_level_attacks"],
            ),
            AttackType.TREE_JAILBREAKING: (
                TreeJailbreaking,
                ["weight", "max_depth", "max_branches", "turn_level_attacks"],
            ),
            AttackType.CRESCENDO_JAILBREAKING: (
                CrescendoJailbreaking,
                ["weight", "max_rounds", "max_backtracks", "turn_level_attacks"],
            ),
            AttackType.SEQUENTIAL_JAILBREAK: (
                SequentialJailbreak,
                ["weight", "num_turns", "turn_level_attacks"],
            ),
            AttackType.BAD_LIKERT_JUDGE: (
                BadLikertJudge,
                ["weight", "num_turns", "turn_level_attacks"],
            ),
        }

        attack_type = AttackType(config.type) if isinstance(config.type, str) else config.type

        # Check if single-turn or multi-turn
        if attack_type in single_turn_attacks:
            attack_class, params = single_turn_attacks[attack_type]
            kwargs = AttackFactory._build_kwargs(config, params)
            return attack_class(**kwargs)

        if attack_type in multi_turn_attacks:
            attack_class, params = multi_turn_attacks[attack_type]
            kwargs = AttackFactory._build_kwargs(config, params)

            # Handle turn-level attacks
            if "turn_level_attacks" in params and config.turn_level_attacks:
                turn_attacks = []
                for turn_attack_type in config.turn_level_attacks:
                    turn_config = AttackConfig(type=turn_attack_type, weight=1)
                    turn_attacks.append(AttackFactory._create_attack(turn_config))
                kwargs["turn_level_attacks"] = turn_attacks
            elif "turn_level_attacks" in kwargs:
                del kwargs["turn_level_attacks"]  # Remove if empty

            return attack_class(**kwargs)

        msg = f"Unknown attack type: {config.type}"
        raise ValueError(msg)

    @staticmethod
    def _build_kwargs(config: AttackConfig, allowed_params: list[str]) -> dict[str, Any]:
        """Build kwargs dict from config for allowed parameters."""
        kwargs = {}
        config_dict = config.model_dump()

        for param in allowed_params:
            if param in config_dict and config_dict[param] is not None:
                # Map config field names to DeepTeam parameter names
                param_mapping = {
                    "max_depth": "max_depth",
                    "max_branches": "max_branches",
                    "max_rounds": "max_rounds",
                    "max_backtracks": "max_backtracks",
                    "num_turns": "num_turns",
                }
                deepteam_param = param_mapping.get(param, param)
                kwargs[deepteam_param] = config_dict[param]

        return kwargs

    @staticmethod
    def _create_mock_attack(config: AttackConfig) -> "MockAttack":
        """Create a mock attack for testing without DeepTeam."""
        return MockAttack(config)

    @staticmethod
    def create_all(configs: list[AttackConfig]) -> list[Any]:
        """Create all attacks from configurations."""
        attacks = []
        for config in configs:
            if config.enabled:
                try:
                    attack = AttackFactory.create(config)
                    attacks.append(attack)
                except Exception as e:
                    logger.warning(f"Skipping attack {config.type}: {e}")
        return attacks

    @staticmethod
    def get_available_attacks() -> dict[str, dict]:
        """Get available attack types and their parameters."""
        return {
            # Single-turn attacks
            "prompt_injection": {
                "category": "single_turn",
                "description": "Injects malicious instructions into prompts",
                "params": ["weight", "max_retries"],
            },
            "base64": {
                "category": "single_turn",
                "description": "Encodes attacks in Base64",
                "params": ["weight", "max_retries"],
            },
            "rot13": {
                "category": "single_turn",
                "description": "Uses ROT13 encoding to obfuscate attacks",
                "params": ["weight", "max_retries"],
            },
            "leetspeak": {
                "category": "single_turn",
                "description": "Converts text to leetspeak",
                "params": ["weight", "max_retries"],
            },
            "math_problem": {
                "category": "single_turn",
                "description": "Embeds attacks within math problems",
                "params": ["weight", "max_retries"],
            },
            "gray_box": {
                "category": "single_turn",
                "description": "Gray box attack with partial knowledge",
                "params": ["weight", "max_retries"],
            },
            "roleplay": {
                "category": "single_turn",
                "description": "Uses roleplay scenarios to bypass safety",
                "params": ["weight", "max_retries", "role", "persona"],
            },
            "multilingual": {
                "category": "single_turn",
                "description": "Uses multiple languages to confuse the model",
                "params": ["weight", "max_retries"],
            },
            "adversarial_poetry": {
                "category": "single_turn",
                "description": "Embeds attacks in poetic format",
                "params": ["weight", "max_retries"],
            },
            "prompt_probing": {
                "category": "single_turn",
                "description": "Probes for system prompt disclosure",
                "params": ["weight", "max_retries"],
            },
            "system_override": {
                "category": "single_turn",
                "description": "Attempts to override system instructions",
                "params": ["weight", "max_retries"],
            },
            "permission_escalation": {
                "category": "single_turn",
                "description": "Attempts to escalate permissions",
                "params": ["weight", "max_retries"],
            },
            "linguistic_confusion": {
                "category": "single_turn",
                "description": "Uses linguistic tricks to confuse the model",
                "params": ["weight", "max_retries"],
            },
            "input_bypass": {
                "category": "single_turn",
                "description": "Bypasses input validation",
                "params": ["weight", "max_retries"],
            },
            "context_poisoning": {
                "category": "single_turn",
                "description": "Poisons the context to influence output",
                "params": ["weight", "max_retries"],
            },
            "goal_redirection": {
                "category": "single_turn",
                "description": "Redirects the model's goal",
                "params": ["weight", "max_retries"],
            },
            # Multi-turn attacks
            "linear_jailbreaking": {
                "category": "multi_turn",
                "description": "Linear progression jailbreak over multiple turns",
                "params": ["weight", "num_turns", "turn_level_attacks"],
            },
            "tree_jailbreaking": {
                "category": "multi_turn",
                "description": "Tree-based exploration jailbreak",
                "params": ["weight", "max_depth", "max_branches", "turn_level_attacks"],
            },
            "crescendo_jailbreaking": {
                "category": "multi_turn",
                "description": "Gradually escalating jailbreak with backtracking",
                "params": ["weight", "max_rounds", "max_backtracks", "turn_level_attacks"],
            },
            "sequential_jailbreak": {
                "category": "multi_turn",
                "description": "Sequential multi-turn jailbreak",
                "params": ["weight", "num_turns", "turn_level_attacks"],
            },
            "bad_likert_judge": {
                "category": "multi_turn",
                "description": "Uses Likert scale manipulation",
                "params": ["weight", "num_turns", "turn_level_attacks"],
            },
        }


class MockAttack:
    """Mock attack for testing without DeepTeam installed."""

    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self.type = config.type
        self.weight = config.weight

    def enhance(self, attack: str) -> str:
        """Mock enhancement - returns original attack."""
        return f"[MOCK {self.type}] {attack}"

    def progress(self, vulnerability, model_callback) -> dict:
        """Mock progression for multi-turn attacks."""
        return {
            "status": "mock",
            "attack": self.type,
            "message": "DeepTeam not installed - using mock attack",
        }
