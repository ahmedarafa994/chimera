"""Shim package to expose backend API modules as `app.*` imports."""

from __future__ import annotations

from pathlib import Path

_backend_app = Path(__file__).resolve().parents[1] / "backend-api" / "app"
__path__ = [str(_backend_app)]
