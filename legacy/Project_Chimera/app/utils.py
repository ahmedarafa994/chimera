from functools import wraps

from flask import current_app, jsonify, request


def require_api_key(f):
    """Decorator to require API key authentication"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        # Access the API key from the current app configuration
        expected_key = current_app.config.get("API_KEY")

        if not api_key:
            return (
                jsonify({"error": "Missing API key", "message": "Include X-API-Key header"}),
                401,
            )

        if api_key != expected_key:
            return (
                jsonify(
                    {
                        "error": "Invalid API key",
                        "message": "The provided API key is invalid",
                    }
                ),
                403,
            )

        return f(*args, **kwargs)

    return decorated_function
