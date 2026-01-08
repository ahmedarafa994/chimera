"""
Plugin Loader and Registry

Discovers, loads, and manages technique plugins from a directory.
Supports hot-reloading and integration with FeatureFlagService.

Part of Phase 3: Transformation implementation.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any, ClassVar, Optional

from .base import PluginMetadata, TechniquePlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Registry for loaded technique plugins.

    Provides lookup and management of registered plugins.
    Thread-safe singleton pattern.
    """

    _instance: ClassVar[Optional["PluginRegistry"]] = None
    _plugins: ClassVar[dict[str, type[TechniquePlugin]]] = {}

    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._plugins = {}
        return cls._instance

    def register(self, plugin_class: type[TechniquePlugin]) -> bool:
        """
        Register a plugin class.

        Args:
            plugin_class: TechniquePlugin subclass to register

        Returns:
            True if registered, False if invalid or duplicate
        """
        if not issubclass(plugin_class, TechniquePlugin):
            logger.warning(f"Not a TechniquePlugin subclass: {plugin_class}")
            return False

        if not plugin_class.is_valid_plugin():
            logger.warning(f"Invalid plugin configuration: {plugin_class}")
            return False

        name = plugin_class.metadata.name

        if name in self._plugins:
            logger.warning(f"Plugin already registered: {name}")
            return False

        self._plugins[name] = plugin_class
        logger.info(f"Registered plugin: {name} v{plugin_class.metadata.version}")
        return True

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name."""
        if name in self._plugins:
            plugin = self._plugins.pop(name)
            plugin.cleanup()
            logger.info(f"Unregistered plugin: {name}")
            return True
        return False

    def get(self, name: str) -> type[TechniquePlugin] | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_all(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def list_by_category(self, category: str) -> list[str]:
        """List plugins in a specific category."""
        return [
            name for name, plugin in self._plugins.items() if plugin.metadata.category == category
        ]

    def get_metadata(self, name: str) -> PluginMetadata | None:
        """Get metadata for a plugin."""
        plugin = self._plugins.get(name)
        return plugin.metadata if plugin else None

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Get metadata for all registered plugins."""
        return [
            {
                "name": plugin.metadata.name,
                "display_name": plugin.metadata.display_name,
                "version": plugin.metadata.version,
                "author": plugin.metadata.author,
                "description": plugin.metadata.description,
                "category": plugin.metadata.category,
                "risk_level": plugin.metadata.risk_level,
                "requires_approval": plugin.metadata.requires_approval,
                "tags": plugin.metadata.tags,
            }
            for plugin in self._plugins.values()
        ]

    def clear(self) -> None:
        """Clear all registered plugins."""
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.metadata.name}: {e}")
        self._plugins.clear()
        logger.info("Cleared all plugins from registry")

    @property
    def count(self) -> int:
        """Number of registered plugins."""
        return len(self._plugins)


class PluginLoader:
    """
    Loads technique plugins from a directory.

    Discovers Python files, imports them, and registers any
    TechniquePlugin subclasses found.
    """

    def __init__(self, plugin_dir: Path | None = None, registry: PluginRegistry | None = None):
        self.plugin_dir = plugin_dir or Path(__file__).parent / "custom"
        self.registry = registry or PluginRegistry()
        self._loaded_modules: dict[str, Any] = {}

    def discover_plugins(self) -> list[str]:
        """
        Discover plugin files in the plugin directory.

        Returns:
            List of discovered plugin file paths
        """
        if not self.plugin_dir.exists():
            logger.info(f"Creating plugin directory: {self.plugin_dir}")
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            return []

        plugin_files = []
        for file_path in self.plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            plugin_files.append(str(file_path))

        logger.info(f"Discovered {len(plugin_files)} plugin files")
        return plugin_files

    def load_plugin(self, file_path: str) -> int:
        """
        Load plugins from a Python file.

        Args:
            file_path: Path to the plugin file

        Returns:
            Number of plugins loaded from the file
        """
        path = Path(file_path)
        module_name = f"chimera_plugin_{path.stem}"

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create module spec: {path}")
                return 0

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._loaded_modules[module_name] = module

            # Find and register TechniquePlugin subclasses
            loaded_count = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Skip non-classes and the base class itself
                if not isinstance(attr, type):
                    continue
                if attr is TechniquePlugin:
                    continue
                if not issubclass(attr, TechniquePlugin):
                    continue

                # Skip classes without metadata
                if attr.metadata is None:
                    continue

                # Register and initialize
                if self.registry.register(attr):
                    try:
                        attr.initialize()
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"Plugin initialization failed: {attr_name}: {e}")
                        self.registry.unregister(attr.metadata.name)

            logger.info(f"Loaded {loaded_count} plugins from {path.name}")
            return loaded_count

        except Exception as e:
            logger.error(f"Failed to load plugin file {path}: {e}")
            return 0

    def load_all(self) -> dict[str, int]:
        """
        Load all plugins from the plugin directory.

        Returns:
            Dictionary mapping file names to number of plugins loaded
        """
        results = {}
        plugin_files = self.discover_plugins()

        for file_path in plugin_files:
            file_name = Path(file_path).name
            results[file_name] = self.load_plugin(file_path)

        total = sum(results.values())
        logger.info(f"Loaded {total} plugins from {len(plugin_files)} files")
        return results

    def reload_plugin(self, name: str) -> bool:
        """
        Reload a specific plugin.

        Args:
            name: Plugin name to reload

        Returns:
            True if reload succeeded
        """
        plugin = self.registry.get(name)
        if not plugin:
            logger.warning(f"Plugin not found for reload: {name}")
            return False

        # Find the module that defined this plugin
        for _module_name, module in self._loaded_modules.items():
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if attr is plugin:
                    # Found it - reload the module
                    self.registry.unregister(name)

                    # Get the file path and reload
                    if hasattr(module, "__file__") and module.__file__:
                        return self.load_plugin(module.__file__) > 0

        return False

    def reload_all(self) -> dict[str, int]:
        """
        Reload all plugins.

        Returns:
            Dictionary mapping file names to number of plugins loaded
        """
        self.registry.clear()
        self._loaded_modules.clear()
        return self.load_all()


# Global instances
_registry = PluginRegistry()
_loader = PluginLoader(registry=_registry)


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader."""
    return _loader


def load_plugins() -> int:
    """
    Load all plugins from the default directory.

    Returns:
        Total number of plugins loaded
    """
    results = _loader.load_all()
    return sum(results.values())
