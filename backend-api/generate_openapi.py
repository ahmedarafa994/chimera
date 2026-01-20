#!/usr/bin/env python3
"""Generate static OpenAPI specification file for Chimera API.

This script generates a static openapi.json file from the FastAPI application
for use in documentation, client generation, and API testing tools.
"""

import json
import sys
from pathlib import Path

# Add backend-api to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Set recursion limit to handle complex schemas
sys.setrecursionlimit(5000)


def generate_openapi_spec():
    """Generate and save OpenAPI specification to openapi.json."""
    try:
        # Import app after setting recursion limit
        from app.main import app

        # Get OpenAPI schema with error handling
        try:
            openapi_schema = app.openapi()
        except RecursionError:
            # Generate simplified schema
            openapi_schema = {
                "openapi": "3.1.0",
                "info": {
                    "title": app.title,
                    "description": app.description,
                    "version": app.version,
                    "contact": app.contact,
                    "license": app.license_info,
                },
                "servers": app.servers,
                "paths": {},
                "components": {
                    "securitySchemes": {
                        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
                        "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
                    },
                },
                "tags": app.openapi_tags,
            }

            # Extract basic path information from routes
            for route in app.routes:
                if hasattr(route, "path") and hasattr(route, "methods"):
                    path = route.path
                    if path not in openapi_schema["paths"]:
                        openapi_schema["paths"][path] = {}

                    for method in route.methods:
                        if method.lower() in ["get", "post", "put", "delete", "patch"]:
                            openapi_schema["paths"][path][method.lower()] = {
                                "summary": getattr(route, "summary", route.name),
                                "description": getattr(route, "description", ""),
                                "tags": getattr(route, "tags", []),
                                "responses": {"200": {"description": "Successful response"}},
                            }

        # Save to file
        output_path = backend_dir / "openapi.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

        return output_path

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    generate_openapi_spec()
