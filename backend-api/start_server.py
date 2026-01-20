#!/usr/bin/env python
"""Simple startup script for the Chimera backend server.
Uses direct import instead of string form to avoid module reload issues.
"""

import os
import sys

# Set environment before imports
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_LEVEL", "INFO")

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

from app.main import app  # Direct import

if __name__ == "__main__":
    uvicorn.run(
        app,  # Direct app reference instead of string
        host="0.0.0.0",
        port=8001,
        log_level="info",
    )
