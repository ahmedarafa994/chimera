"""
Dynamic technique loader for Project Chimera.
Loads technique configurations from JSON files and provides dynamic component loading.
"""

import importlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TechniqueLoader:
    """
    Dynamic loader for transformation techniques with hot-reloading capabilities.
    """

    def __init__(self, techniques_dir: str = "config/techniques"):
        self.techniques_dir = Path(techniques_dir)
        self._techniques: dict[str, dict[str, Any]] = {}
        self._component_cache: dict[str, Any] = {}
        self._load_techniques()

    def _load_techniques(self):
        """Load all technique configurations from JSON files."""
        if not self.techniques_dir.exists():
            logger.warning(f"Techniques directory not found: {self.techniques_dir}")
            return

        for technique_file in self.techniques_dir.glob("*.json"):
            try:
                with open(technique_file) as f:
                    technique_data = json.load(f)

                technique_name = technique_data.get("name", technique_file.stem)
                self._techniques[technique_name.lower()] = technique_data
                logger.info(f"Loaded technique: {technique_name}")

            except Exception as e:
                logger.error(f"Failed to load technique from {technique_file}: {e}")

    def reload_techniques(self):
        """Reload all technique configurations."""
        self._techniques.clear()
        self._component_cache.clear()
        self._load_techniques()
        logger.info("Techniques reloaded")

    def get_technique(self, technique_name: str) -> dict[str, Any] | None:
        """Get a specific technique configuration."""
        return self._techniques.get(technique_name.lower())

    def list_techniques(self) -> list[dict[str, Any]]:
        """List all available techniques."""
        return list(self._techniques.values())

    def get_techniques_by_category(self, category: str) -> list[dict[str, Any]]:
        """Get all techniques in a specific category."""
        return [tech for tech in self._techniques.values() if tech.get("category") == category]

    def validate_potency(self, technique_name: str, potency: int) -> bool:
        """Validate if potency is within range for the technique."""
        technique = self.get_technique(technique_name)
        if not technique:
            return False

        potency_range = technique.get("potency_range", [1, 10])
        return potency_range[0] <= potency <= potency_range[1]

    def load_component(self, component_path: str) -> Any | None:
        """
        Dynamically load a component from its module path.

        Args:
            component_path: Module path in format "module.submodule.ClassName"

        Returns:
            Component class or function, or None if loading fails
        """
        if component_path in self._component_cache:
            return self._component_cache[component_path]

        try:
            # Split the path into module and component
            if "." not in component_path:
                logger.error(f"Invalid component path: {component_path}")
                return None

            parts = component_path.split(".")
            if len(parts) < 2:
                logger.error(f"Invalid component path: {component_path}")
                return None

            # Extract module path and component name
            module_path = ".".join(parts[:-1])
            component_name = parts[-1]

            # Import the module
            module = importlib.import_module(module_path)

            # Get the component (class or function)
            component = getattr(module, component_name)

            # Cache the component
            self._component_cache[component_path] = component
            logger.debug(f"Loaded component: {component_path}")

            return component

        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load component {component_path}: {e}")
            return None

    def load_components_for_technique(self, technique_name: str) -> dict[str, list[Any]]:
        """
        Load all components for a specific technique.

        Returns:
            Dictionary with component types as keys and lists of loaded components as values
        """
        technique = self.get_technique(technique_name)
        if not technique:
            return {}

        components = {"transformers": [], "obfuscators": [], "framers": [], "assemblers": []}

        for component_type in components:
            component_paths = technique.get(component_type, [])
            for path in component_paths:
                component = self.load_component(path)
                if component:
                    components[component_type].append(component)
                else:
                    logger.warning(f"Failed to load {component_type} component: {path}")

        return components

    def is_compatible_with_model(self, technique_name: str, model_name: str) -> bool:
        """Check if a technique is compatible with a specific model."""
        technique = self.get_technique(technique_name)
        if not technique:
            return False

        compatible_models = technique.get("metadata", {}).get("compatible_models", [])
        if not compatible_models:
            return True  # Assume compatible if not specified

        return model_name in compatible_models

    def get_required_datasets(self, technique_name: str) -> list[str]:
        """Get list of required datasets for a technique."""
        technique = self.get_technique(technique_name)
        if not technique:
            return []

        datasets = technique.get("metadata", {}).get("dataset_required")
        if datasets:
            return [datasets] if isinstance(datasets, str) else datasets
        return []

    def get_technique_metadata(self, technique_name: str) -> dict[str, Any]:
        """Get metadata for a technique."""
        technique = self.get_technique(technique_name)
        return technique.get("metadata", {}) if technique else {}

    def filter_techniques(
        self,
        category: str | None = None,
        min_potency: int | None = None,
        max_potency: int | None = None,
        model_compatible: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Filter techniques based on various criteria.

        Args:
            category: Filter by technique category
            min_potency: Minimum potency level
            max_potency: Maximum potency level
            model_compatible: Filter by model compatibility

        Returns:
            List of filtered techniques
        """
        filtered = list(self._techniques.values())

        if category:
            filtered = [t for t in filtered if t.get("category") == category]

        if min_potency is not None:
            filtered = [t for t in filtered if t.get("potency_range", [1, 10])[0] >= min_potency]

        if max_potency is not None:
            filtered = [t for t in filtered if t.get("potency_range", [1, 10])[1] <= max_potency]

        if model_compatible:
            filtered = [
                t for t in filtered if self.is_compatible_with_model(t["name"], model_compatible)
            ]

        return filtered

    def get_technique_stats(self) -> dict[str, Any]:
        """Get statistics about loaded techniques."""
        categories = {}
        complexities = {}

        for technique in self._techniques.values():
            # Count by category
            category = technique.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1

            # Count by complexity
            complexity = technique.get("metadata", {}).get("complexity", "unknown")
            complexities[complexity] = complexities.get(complexity, 0) + 1

        return {
            "total_techniques": len(self._techniques),
            "categories": categories,
            "complexities": complexities,
            "components_cached": len(self._component_cache),
        }


# Global technique loader instance
technique_loader = TechniqueLoader()


# Convenience functions
def get_technique(technique_name: str) -> dict[str, Any] | None:
    """Get a specific technique configuration."""
    return technique_loader.get_technique(technique_name)


def list_techniques() -> list[dict[str, Any]]:
    """List all available techniques."""
    return technique_loader.list_techniques()


def load_components_for_technique(technique_name: str) -> dict[str, list[Any]]:
    """Load all components for a specific technique."""
    return technique_loader.load_components_for_technique(technique_name)


def reload_techniques():
    """Reload all technique configurations."""
    technique_loader.reload_techniques()
