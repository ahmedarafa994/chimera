#!/usr/bin/env python3
import os
import sys

import uvicorn

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Redirect stdout and stderr to a file for debugging
    with open("backend-api/test_server.log", "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = sys.stdout

        print("Starting server on port 9009...")
        try:
            uvicorn.run("app.main:app", host="0.0.0.0", port=9009, reload=False)
        except Exception as e:
            print(f"Server failed to start: {e}")
