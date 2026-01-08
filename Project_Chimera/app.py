"""Legacy compatibility shim for infrastructure tests."""

from __future__ import annotations

from .app import app, create_app  # type: ignore

__all__ = ["app", "create_app"]
