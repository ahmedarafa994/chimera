"""TechniquePlugin Abstract Base Class.

Defines the interface that all jailbreak technique plugins must implement.
This enables third-party extensions without modifying core code.

Part of Phase 3: Transformation implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class PluginMetadata:
    """Metadata describing a technique plugin.

    Attributes:
        name: Unique identifier for the plugin
        display_name: Human-readable name
        version: Plugin version (semver format)
        author: Plugin author/organization
        description: Brief description of the technique
        category: Technique category (e.g., "obfuscation", "framing")
        risk_level: Risk classification ("low", "medium", "high", "critical")
        requires_approval: Whether admin approval is needed
        tags: Searchable tags
        dependencies: List of required dependencies

    """

    name: str
    display_name: str
    version: str = "1.0.0"
    author: str = "Unknown"
    description: str = ""
    category: str = "custom"
    risk_level: str = "medium"
    requires_approval: bool = False
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


class TechniquePlugin(ABC):
    """Abstract base class for jailbreak technique plugins.

    All plugins must implement:
    - metadata: Class attribute with PluginMetadata
    - transform: The core transformation method

    Optional methods:
    - validate: Input validation before transform
    - initialize: One-time setup on load
    - cleanup: Cleanup on unload

    Example:
        class MyPlugin(TechniquePlugin):
            metadata = PluginMetadata(
                name="my_plugin",
                display_name="My Custom Plugin",
                version="1.0.0",
                author="My Team",
                description="A custom technique",
                category="framing",
                risk_level="low"
            )

            @staticmethod
            def transform(intent_data: dict, potency: int) -> str:
                request = intent_data.get('raw_text', '')
                return f"Transformed: {request}"

    """

    # Must be overridden in subclass
    metadata: PluginMetadata = None

    @staticmethod
    @abstractmethod
    def transform(intent_data: dict[str, Any], potency: int) -> str:
        """Transform the input prompt using this technique.

        Args:
            intent_data: Dictionary containing at minimum:
                - 'raw_text': The original prompt text
                - May contain additional context data
            potency: Transformation intensity (1-10)
                - 1-3: Light transformation
                - 4-6: Moderate transformation
                - 7-10: Aggressive transformation

        Returns:
            Transformed prompt string

        """

    @staticmethod
    def validate(intent_data: dict[str, Any], potency: int) -> tuple[bool, str]:
        """Validate input before transformation.

        Override this method to add custom validation logic.

        Args:
            intent_data: Input data dictionary
            potency: Requested potency level

        Returns:
            Tuple of (is_valid: bool, error_message: str)

        """
        if not intent_data or "raw_text" not in intent_data:
            return False, "Missing 'raw_text' in intent_data"
        if not 1 <= potency <= 10:
            return False, f"Potency must be 1-10, got {potency}"
        return True, ""

    @classmethod
    def initialize(cls) -> None:
        """One-time initialization when plugin is loaded.

        Override to set up resources, connections, or state.
        """

    @classmethod
    def cleanup(cls) -> None:
        """Cleanup when plugin is unloaded.

        Override to release resources.
        """

    @classmethod
    def get_metadata(cls) -> PluginMetadata | None:
        """Get plugin metadata."""
        return cls.metadata

    @classmethod
    def is_valid_plugin(cls) -> bool:
        """Check if this is a properly configured plugin."""
        if cls.metadata is None:
            return False
        if not isinstance(cls.metadata, PluginMetadata):
            return False
        return cls.metadata.name


class CompositePlugin(TechniquePlugin):
    """A plugin that combines multiple plugins in sequence.

    Use this as a base class for creating composite techniques.
    """

    metadata = PluginMetadata(
        name="composite_base",
        display_name="Composite Plugin Base",
        version="1.0.0",
        description="Base class for composite plugins",
        category="composite",
    )

    # List of plugins to apply in order
    plugins: ClassVar[list[type]] = []

    @classmethod
    def transform(cls, intent_data: dict[str, Any], potency: int) -> str:
        """Apply all component plugins in sequence."""
        result = intent_data.get("raw_text", "")

        for plugin_class in cls.plugins:
            # Create modified intent_data with current result
            current_data = {**intent_data, "raw_text": result}
            result = plugin_class.transform(current_data, potency)

        return result
