import os
import secrets
from datetime import datetime, timedelta
from functools import wraps

import jwt
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


class SecurityConfig:
    """Security configuration for Flask application"""

    def __init__(self, app: Flask = None):
        self.app = app
        if app:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize security configuration"""
        # Disable debug mode in production
        app.config["DEBUG"] = os.getenv("DEBUG", "false").lower() == "true"

        # Security headers
        @app.after_request
        def security_headers(response):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'"
            return response

        # Secure CORS with specific origins
        allowed_origins = os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080"
        ).split(",")
        CORS(app, origins=allowed_origins, methods=["GET", "POST"], supports_credentials=True)

        # Rate limiting
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=[f"{os.getenv('RATE_LIMIT_PER_MINUTE', '60')} per minute"],
        )

        # Session security
        app.config["SECRET_KEY"] = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
        app.config["JWT_EXPIRATION_HOURS"] = 24

        # Store limiter for use in decorators
        app.limiter = limiter


def require_api_key(f):
    """Decorator to require API key authentication with timing-safe comparison"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import jsonify, request

        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "API key required"}), 401

        # Validate API key using timing-safe comparison to prevent timing attacks
        valid_key = os.getenv("CHIMERA_API_KEY") or os.getenv("API_KEY")
        if not valid_key:
            # Generate secure key if not provided (log warning)
            import logging

            logging.warning("No API key configured - authentication will fail")
            return jsonify({"error": "Server configuration error"}), 500

        if not secrets.compare_digest(api_key, valid_key):
            return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)

    return decorated_function


def generate_jwt_token(user_id: str) -> str:
    """Generate JWT token for authenticated user"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, os.getenv("JWT_SECRET"), algorithm="HS256")


def verify_jwt_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")
