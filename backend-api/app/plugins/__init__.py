"""
Plugin System for Project Chimera

This package provides an extensible plugin architecture for adding custom
jailbreak techniques without modifying core code.

Usage:
    from app.plugins import PluginLoader, TechniquePlugin

Part of Phase 3: Transformation implementation.
"""

from .base import PluginMetadata, TechniquePlugin
from .loader import PluginLoader, PluginRegistry

__all__ = [
    "PluginLoader",
    "PluginMetadata",
    "PluginRegistry",
    "TechniquePlugin",
]
