import os
import sys
import traceback

print("Debugging full app startup...")
try:
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    print("Attempting to import app.main...")
    print("Import successful!")
except Exception:
    print("Import failed!")
    with open("errors_full.log", "w", buffering=1) as f:
        traceback.print_exc(file=f)
        f.flush()
        os.fsync(f.fileno())
    print("Traceback written to errors_full.log")
    traceback.print_exc()
