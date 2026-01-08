"""Project-level import helpers.

This module ensures that key workspace packages are available on ``sys.path``
regardless of where a Python process starts from. Python automatically imports
``sitecustomize`` (if present) during startup, so we opportunistically add the
backend directory before any application modules need it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _prepend_path(path: Path) -> None:
    """Insert ``path`` at the beginning of ``sys.path`` if it exists."""
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend-api"

# Ensure both the repository root (for meta_prompter, etc.) and backend package
# directory are easy to import no matter the execution location.
_prepend_path(REPO_ROOT)
_prepend_path(BACKEND_DIR)
