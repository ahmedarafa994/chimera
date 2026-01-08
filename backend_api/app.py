# Re-export the top-level `app` package as `backend_api.app` so existing
# imports like `from backend_api.app.services...` continue to work when running
# tests from the repository root.

from __future__ import annotations

import importlib
import sys

# Import the real top-level `app` package (located in `backend-api/app`)
_real_app = importlib.import_module("app")

# Replace this module in sys.modules with the real `app` package so that
# `backend_api.app` resolves to the same module object as `app`.
sys.modules[__name__] = _real_app
