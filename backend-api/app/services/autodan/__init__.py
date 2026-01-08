"""
AutoDAN Service Package

This package provides adversarial prompt generation services for AI security research.

LAZY LOADING ENABLED
====================
Heavy modules (framework, framework_r, framework_autodan_reasoning, optimized) are
loaded lazily on first access to improve startup time and reduce memory usage.

Services:
- AutoDANService: Core service with lifelong learning, hybrid engine, and mousetrap
- EnhancedAutoDANService: Feature-rich service with parallel processing and caching
- AutoDANResearchService: Research-focused service with ethical protocols

Usage:
    # Standard usage - services are lazy loaded
    from app.services.autodan import autodan_service
    result = autodan_service.run_jailbreak(request, method="best_of_n")

    # For enhanced features (parallel, caching, genetic optimization):
    from app.services.autodan import enhanced_autodan_service
    result = await enhanced_autodan_service.run_jailbreak(request, method="turbo")

    # For research with ethical protocols:
    from app.services.autodan import AutoDANResearchService
    research_service = AutoDANResearchService()

    # Lazy access to heavy submodules (loaded on first attribute access):
    from app.services.autodan import framework, framework_r, reasoning, optimized
    optimizer = framework.GeneticOptimizer()  # Module loaded here

See ADR-002 for module consolidation and lazy loading details.
"""

from typing import TYPE_CHECKING

# Use lazy imports for heavy modules to improve startup time
from app.core.lazy_loader import LazyServiceFactory, get_lazy_module

# Lazy module proxies - these don't import until accessed
framework = get_lazy_module("app.services.autodan.framework")
framework_r = get_lazy_module("app.services.autodan.framework_r")
reasoning = get_lazy_module("app.services.autodan.framework_autodan_reasoning")
optimized = get_lazy_module("app.services.autodan.optimized")

# Configuration is lightweight, import eagerly
from .config import autodan_config
from .config_enhanced import EnhancedAutoDANConfig, get_config, update_config


# Lazy service factories - services are only created when first accessed
def _create_autodan_service():
    """Factory function for AutoDANService."""
    from .service import AutoDANService
    return AutoDANService()


def _create_enhanced_service():
    """Factory function for EnhancedAutoDANService."""
    from .service_enhanced import EnhancedAutoDANService
    return EnhancedAutoDANService()


def _create_research_service():
    """Factory function for AutoDANResearchService."""
    from .enhanced_service import AutoDANResearchService
    return AutoDANResearchService()


# Lazy service factories
_autodan_service_factory = LazyServiceFactory(_create_autodan_service, "autodan_service")
_enhanced_service_factory = LazyServiceFactory(_create_enhanced_service, "enhanced_autodan_service")
_research_service_factory = LazyServiceFactory(_create_research_service, "research_service")


# Property-style accessors for lazy services
class _LazyServiceAccessor:
    """Provides lazy access to service instances."""

    @property
    def autodan_service(self):
        return _autodan_service_factory.get()

    @property
    def enhanced_autodan_service(self):
        return _enhanced_service_factory.get()

    @property
    def research_service(self):
        return _research_service_factory.get()


_accessor = _LazyServiceAccessor()


# Module-level lazy getters for backward compatibility
def get_autodan_service():
    """Get the AutoDANService instance (lazy loaded)."""
    return _autodan_service_factory.get()


def get_enhanced_service():
    """Get the EnhancedAutoDANService instance (lazy loaded)."""
    return _enhanced_service_factory.get()


def get_research_service():
    """Get the AutoDANResearchService instance (lazy loaded)."""
    return _research_service_factory.get()


# For TYPE_CHECKING, provide type hints without importing
if TYPE_CHECKING:
    from .enhanced_service import AutoDANResearchService
    from .service import AutoDANService
    from .service_enhanced import EnhancedAutoDANService

# Legacy exports for backward compatibility
# These trigger lazy loading when accessed
autodan_service = property(lambda self: get_autodan_service())
enhanced_autodan_service = property(lambda self: get_enhanced_service())


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of legacy exports."""
    if name == "autodan_service":
        return get_autodan_service()
    elif name == "enhanced_autodan_service":
        return get_enhanced_service()
    elif name == "AutoDANService":
        from .service import AutoDANService
        return AutoDANService
    elif name == "EnhancedAutoDANService":
        from .service_enhanced import EnhancedAutoDANService
        return EnhancedAutoDANService
    elif name == "AutoDANResearchService":
        from .enhanced_service import AutoDANResearchService
        return AutoDANResearchService
    elif name == "run_jailbreak":
        from .service_enhanced import run_jailbreak
        return run_jailbreak
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AutoDANResearchService",
    # Service classes (lazy loaded via __getattr__)
    "AutoDANService",
    "EnhancedAutoDANConfig",
    "EnhancedAutoDANService",
    # Configuration (eager loaded - lightweight)
    "autodan_config",
    # Legacy service instances (backward compatible, lazy loaded)
    "autodan_service",
    "enhanced_autodan_service",
    # Lazy module proxies
    "framework",
    "framework_r",
    # Lazy service accessors (recommended)
    "get_autodan_service",
    "get_config",
    "get_enhanced_service",
    "get_research_service",
    "optimized",
    "reasoning",
    "run_jailbreak",
    "update_config",
]
