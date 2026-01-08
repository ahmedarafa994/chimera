#!/usr/bin/env python3
"""
Test that the test server starts on the new port
"""

import subprocess
import time

import requests


def test_server_starts():
    """Test that the test server starts on port 5001"""
    try:
        # Start the server
        proc = subprocess.Popen(
            ["py", "tests/test_server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait a moment for it to start
        time.sleep(2)

        try:
            # Test if server is running on port 5001
            response = requests.get("http://localhost:5001/health", timeout=1)
            print(f"Server responded on port 5001: {response.status_code}")
            print(f"Response: {response.json()}")
            return True
        except requests.exceptions.ConnectionError:
            print("Server did not start on port 5001")
            return False
        except Exception as e:
            print(f"Error testing server: {e}")
            return False
        finally:
            # Clean up
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception as e:
                print(f"Warning: could not terminate test server process: {e}")

    except Exception as e:
        print(f"Error starting server: {e}")
        return False


if __name__ == "__main__":
    result = test_server_starts()
    print(f"Test result: {result}")
