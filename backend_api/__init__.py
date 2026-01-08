# Lightweight shim to make the `backend-api` package importable as `backend_api`
# This file adds the `backend-api/app` directory to sys.path so tests and imports
# using `backend_api.*` work when running from the repository root.

from __future__ import annotations

import os
import sys

# Resolve repo root (one directory up from this shim package)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_backend_root = os.path.join(_repo_root, "backend-api")

# Add the backend root to sys.path (so `import app` resolves to backend-api/app)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

_backend_app_module = None

# Optionally expose the app package directly for convenience
try:
    # Import here to allow `import backend_api.app` semantics
    import app  # type: ignore
    _backend_app_module = app
except Exception:
    # Silently ignore import-time errors; the tests will import from `backend_api.app` later
    pass

if _backend_app_module is not None:
    # Allow `import backend_api.app` to resolve to the backend app package.
    sys.modules.setdefault("backend_api.app", _backend_app_module)

__all__ = ["app"]
