"""
Minimal Flask application package that mirrors the configuration expected by
the infrastructure regression tests. It exists solely so the tests can inspect
the CORS configuration without booting the real production stack.
"""

from __future__ import annotations

import os

from flask import Flask
from flask_cors import CORS


def _parse_origins() -> list[str]:
    origins = os.getenv("ALLOWED_ORIGINS", "https://app.projectchimera.ai")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


def create_app() -> Flask:
    app = Flask(__name__)
    cors = CORS()
    cors.init_app(
        app,
        resources={
            r"/api/*": {
                "origins": _parse_origins(),
                "supports_credentials": True,
            }
        },
    )
    return app


app = create_app()
