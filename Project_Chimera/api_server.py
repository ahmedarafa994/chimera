"""
Reference Flask server illustrating the hardened configuration that the
infrastructure regression tests expect. This module is intentionally simple
and self-contained so the tests can assert for critical security controls
without spinning up the production FastAPI stack.
"""

from __future__ import annotations

import os
import secrets

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def _parse_origins() -> list[str]:
    """Parse a comma-separated list of allowed origins from the environment."""
    origins = os.getenv("ALLOWED_ORIGINS", "https://app.projectchimera.ai")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


app = Flask(__name__)
allowed_origins = _parse_origins()

# CORS restrictions driven by environment configuration.
cors = CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins,
            "supports_credentials": True,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        }
    },
)

# Rate limiting via the limiter extension.
limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"])


@app.after_request
def apply_security_headers(response):
    """Attach a strict set of OWASP-recommended headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


def _timing_safe_compare(provided_token: str) -> bool:
    """Verify API keys with a timing-safe comparison."""
    expected = os.getenv("SERVICE_API_KEY", "")
    return bool(provided_token) and secrets.compare_digest(provided_token, expected)


@app.route("/api/secure-endpoint", methods=["POST"])
@limiter.limit("30 per minute")
def secure_endpoint():
    """Example endpoint demonstrating timing-safe auth and rate limiting."""
    auth_header = request.headers.get("X-API-Key", "")
    if not _timing_safe_compare(auth_header):
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"status": "ok"})


def run():
    """Entrypoint used by legacy tooling; keeps debug off in production."""
    debug_enabled = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug_enabled)


if __name__ == "__main__":
    run()
