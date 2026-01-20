"""Janus Adversarial Simulation Service.

This package provides autonomous heuristic derivation and adversarial simulation
capabilities for Guardian NPU validation as mandated by Directive 7.4.2.
"""

from .config import JanusConfig, get_config, update_config
from .service import get_service, janus_service

__all__ = [
    "JanusConfig",
    "get_config",
    "get_service",
    "janus_service",
    "update_config",
]
