import os
import sys
import traceback

print("Debugging startup...")
try:
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    print("Attempting to import app.api.v1.router...")
    print("Import successful!")
except Exception:
    print("Import failed!")
    with open("errors.log", "w", buffering=1) as f:
        traceback.print_exc(file=f)
        f.flush()
        os.fsync(f.fileno())
    print("Traceback written to errors.log")
    traceback.print_exc()
