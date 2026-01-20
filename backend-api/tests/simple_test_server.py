#!/usr/bin/env python3
"""Ultra-simple test server for Chimera API.
Uses centralized port configuration to avoid conflicts.
NOTE: This is a standalone server script, not a pytest test file.
"""

import http.server
import json
import socketserver
from datetime import datetime

API_KEY = "chimera_default_key_change_in_production"

# Use centralized port configuration
try:
    from app.core.port_config import get_test_server_port

    PORT = get_test_server_port()
except ImportError:
    # Fallback to default port if port config not available
    PORT = 5001


class TestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            response = {
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "port": PORT,
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {
            "success": True,
            "message": "POST received",
            "data": json.loads(post_data.decode() if post_data else "{}"),
            "port": PORT,
        }
        self.wfile.write(json.dumps(response).encode())


with socketserver.TCPServer(("", PORT), TestHandler) as httpd:
    httpd.serve_forever()
