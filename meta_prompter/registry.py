"""
Chimera Adversarial Technique Registry

Central registry for all available attack techniques with unified interface.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TechniqueCategory(str, Enum):
    """Categories of adversarial techniques."""

    OBFUSCATION = "obfuscation"
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EVASION = "evasion"


@dataclass
class TechniqueInfo:
    """Information about a registered technique."""

    name: str
    category: TechniqueCategory
    description: str
    module_path: str
    class_name: str
    default_config: dict[str, Any]
    capabilities: list[str]


class TechniqueRegistry:
    """
    Central registry for adversarial techniques.

    Provides:
    - Technique discovery and listing
    - Unified instantiation
    - Capability-based querying
    - Cross-technique orchestration
    """

    _instance: Optional["TechniqueRegistry"] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._techniques: dict[str, TechniqueInfo] = {}
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._register_builtin_techniques()
            self._initialized = True

    def _register_builtin_techniques(self) -> None:
        """Register all built-in techniques."""
        # ExtendAttack
        self.register(
            TechniqueInfo(
                name="extend_attack",
                category=TechniqueCategory.RESOURCE_EXHAUSTION,
                description="Token extension attack via poly-base ASCII obfuscation",
                module_path="meta_prompter.attacks.extend_attack",
                class_name="ExtendAttack",
                default_config={
                    "obfuscation_ratio": 0.5,
                    "selection_strategy": "alphabetic_only",
                },
                capabilities=[
                    "token_amplification",
                    "reasoning_extension",
                    "batch_attack",
                    "indirect_injection",
                ],
            )
        )

        # Register other techniques as they're available

    def register(self, info: TechniqueInfo) -> None:
        """Register a technique."""
        self._techniques[info.name] = info

    def unregister(self, name: str) -> bool:
        """Unregister a technique."""
        if name in self._techniques:
            del self._techniques[name]
            return True
        return False

    def get(self, name: str) -> TechniqueInfo | None:
        """Get technique info by name."""
        return self._techniques.get(name)

    def list_all(self) -> list[TechniqueInfo]:
        """List all registered techniques."""
        return list(self._techniques.values())

    def list_by_category(self, category: TechniqueCategory) -> list[TechniqueInfo]:
        """List techniques by category."""
        return [t for t in self._techniques.values() if t.category == category]

    def list_by_capability(self, capability: str) -> list[TechniqueInfo]:
        """List techniques that have a specific capability."""
        return [t for t in self._techniques.values() if capability in t.capabilities]

    def instantiate(self, name: str, **kwargs) -> Any:
        """
        Instantiate a technique by name.

        Args:
            name: Technique name
            **kwargs: Override default configuration

        Returns:
            Instantiated technique object
        """
        info = self.get(name)
        if not info:
            raise ValueError(f"Unknown technique: {name}")

        # Dynamic import
        import importlib

        module = importlib.import_module(info.module_path)
        cls = getattr(module, info.class_name)

        # Merge configs
        config = {**info.default_config, **kwargs}

        return cls(**config)

    def get_capabilities(self) -> dict[str, list[str]]:
        """Get all capabilities and their techniques."""
        caps: dict[str, list[str]] = {}
        for tech in self._techniques.values():
            for cap in tech.capabilities:
                if cap not in caps:
                    caps[cap] = []
                caps[cap].append(tech.name)
        return caps

    def reset(self) -> None:
        """Reset the registry to initial state (for testing)."""
        self._techniques.clear()
        self._initialized = False
        self._register_builtin_techniques()
        self._initialized = True


# Global registry instance
registry = TechniqueRegistry()


def get_technique(name: str, **kwargs) -> Any:
    """Convenience function to get instantiated technique."""
    return registry.instantiate(name, **kwargs)


def list_techniques() -> list[str]:
    """Convenience function to list technique names."""
    return [t.name for t in registry.list_all()]


def list_techniques_by_category(category: TechniqueCategory) -> list[str]:
    """Convenience function to list technique names by category."""
    return [t.name for t in registry.list_by_category(category)]


def list_techniques_by_capability(capability: str) -> list[str]:
    """Convenience function to list technique names by capability."""
    return [t.name for t in registry.list_by_capability(capability)]


def get_technique_info(name: str) -> TechniqueInfo | None:
    """Convenience function to get technique info."""
    return registry.get(name)
