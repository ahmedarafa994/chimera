"""Root pytest configuration to ensure local packages are importable."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend-api"

for path in (REPO_ROOT, BACKEND_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Ensure local app shim wins over any installed `app` module.
if "app" in sys.modules:
    del sys.modules["app"]
if "backend_api.app" in sys.modules:
    del sys.modules["backend_api.app"]

import app as _shim_app  # noqa: E402

sys.modules.setdefault("backend_api.app", _shim_app)
