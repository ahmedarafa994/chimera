"""
Lazy Module Loading Utilities

This module provides utilities for lazy loading of heavy Python modules to improve
application startup time and reduce memory usage.

Features:
- LazyModule: Proxy object that delays module import until first access
- lazy_import: Function-based lazy import for simpler use cases
- LazyServiceFactory: Factory pattern for lazy service instantiation

Usage:
    # Lazy module import
    from app.core.lazy_loader import LazyModule
    heavy_module = LazyModule("app.services.autodan.framework")

    # Later, when accessed:
    heavy_module.SomeClass()  # Module is imported here

    # Or use the function-based approach:
    from app.core.lazy_loader import lazy_import
    framework = lazy_import("app.services.autodan.framework")

See ADR-002 for AutoDAN module consolidation and lazy loading strategy.
"""

import importlib
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LazyModule:
    """
    Proxy object that delays module import until first attribute access.

    This is useful for heavy modules that are not always needed, improving
    application startup time and reducing initial memory usage.

    Example:
        autodan_framework = LazyModule("app.services.autodan.framework")
        # No import happens yet

        # Later, when accessed:
        result = autodan_framework.GeneticOptimizer()  # Import happens here
    """

    __slots__ = ("_accessed", "_module", "_module_name")

    def __init__(self, module_name: str):
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)
        object.__setattr__(self, "_accessed", False)

    def _load(self) -> Any:
        """Load the module if not already loaded."""
        module = object.__getattribute__(self, "_module")
        if module is None:
            module_name = object.__getattribute__(self, "_module_name")
            logger.debug(f"Lazy loading module: {module_name}")
            try:
                module = importlib.import_module(module_name)
                object.__setattr__(self, "_module", module)
                object.__setattr__(self, "_accessed", True)
                logger.info(f"Successfully loaded module: {module_name}")
            except ImportError as e:
                logger.error(f"Failed to lazy load module {module_name}: {e}")
                raise
        return module

    def __getattr__(self, name: str) -> Any:
        module = self._load()
        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_module_name", "_module", "_accessed"):
            object.__setattr__(self, name, value)
        else:
            module = self._load()
            setattr(module, name, value)

    def __repr__(self) -> str:
        accessed = object.__getattribute__(self, "_accessed")
        module_name = object.__getattribute__(self, "_module_name")
        status = "loaded" if accessed else "not loaded"
        return f"<LazyModule '{module_name}' ({status})>"

    @property
    def is_loaded(self) -> bool:
        """Check if the module has been loaded."""
        return object.__getattribute__(self, "_accessed")


class LazyServiceFactory(Generic[T]):
    """
    Factory for lazy service instantiation.

    Services are only created when first accessed, not when the factory is created.
    Supports both sync and async initialization.

    Example:
        def create_heavy_service():
            from app.services.heavy import HeavyService
            return HeavyService()

        lazy_service = LazyServiceFactory(create_heavy_service)
        # No service created yet

        service = lazy_service.get()  # Service created here
    """

    def __init__(
        self,
        factory: Callable[[], T],
        name: str | None = None,
    ):
        self._factory = factory
        self._instance: T | None = None
        self._name = name or factory.__name__
        self._creating = False

    def get(self) -> T:
        """Get the service instance, creating it if necessary."""
        if self._instance is None:
            if self._creating:
                raise RuntimeError(f"Circular dependency detected for service: {self._name}")
            self._creating = True
            try:
                logger.debug(f"Lazy creating service: {self._name}")
                self._instance = self._factory()
                logger.info(f"Successfully created service: {self._name}")
            finally:
                self._creating = False
        return self._instance

    def reset(self) -> None:
        """Reset the factory, clearing the cached instance."""
        self._instance = None

    @property
    def is_created(self) -> bool:
        """Check if the service has been created."""
        return self._instance is not None

    def __repr__(self) -> str:
        status = "created" if self.is_created else "not created"
        return f"<LazyServiceFactory '{self._name}' ({status})>"


def lazy_import(module_name: str) -> LazyModule:
    """
    Create a lazy module proxy.

    Args:
        module_name: Fully qualified module name to lazy load

    Returns:
        LazyModule proxy that will load the module on first access

    Example:
        framework = lazy_import("app.services.autodan.framework")
        # Module not loaded yet

        result = framework.optimize()  # Module loaded here
    """
    return LazyModule(module_name)


@lru_cache(maxsize=128)
def cached_import(module_name: str) -> Any:
    """
    Import and cache a module.

    This is useful when you need the actual module object and want to
    ensure it's only imported once.

    Args:
        module_name: Fully qualified module name

    Returns:
        The imported module
    """
    logger.debug(f"Importing module (cached): {module_name}")
    return importlib.import_module(module_name)


def preload_modules(*module_names: str) -> dict[str, Any]:
    """
    Preload multiple modules in parallel (useful for background initialization).

    Args:
        module_names: Module names to preload

    Returns:
        Dict mapping module names to loaded modules
    """
    import concurrent.futures

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_name = {
            executor.submit(importlib.import_module, name): name
            for name in module_names
        }

        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                logger.info(f"Preloaded module: {name}")
            except Exception as e:
                logger.error(f"Failed to preload module {name}: {e}")
                results[name] = None

    return results


# Module-level lazy imports for commonly used heavy modules
# These can be imported from this module without triggering the actual import
_lazy_modules: dict[str, LazyModule] = {}


def get_lazy_module(module_name: str) -> LazyModule:
    """
    Get or create a lazy module proxy.

    Uses a cache to ensure the same LazyModule instance is returned
    for repeated calls with the same module name.
    """
    if module_name not in _lazy_modules:
        _lazy_modules[module_name] = LazyModule(module_name)
    return _lazy_modules[module_name]


# Pre-defined lazy modules for heavy AutoDAN components
# These are the main entry points that should be used by other modules
autodan_framework = get_lazy_module("app.services.autodan.framework")
autodan_framework_r = get_lazy_module("app.services.autodan.framework_r")
autodan_reasoning = get_lazy_module("app.services.autodan.framework_autodan_reasoning")
autodan_optimized = get_lazy_module("app.services.autodan.optimized")
