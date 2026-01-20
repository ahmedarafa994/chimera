"""Shared utilities for AutoDAN engines.

This module contains utilities extracted from multiple engines to avoid
duplication and prevent circular imports.
"""

from .patterns import extract_pattern, match_pattern

__all__ = [
    "extract_pattern",
    "match_pattern",
]
