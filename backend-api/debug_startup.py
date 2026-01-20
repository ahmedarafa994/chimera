import os
import sys
import traceback

try:
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

except Exception:
    with open("errors.log", "w", buffering=1) as f:
        traceback.print_exc(file=f)
        f.flush()
        os.fsync(f.fileno())
    traceback.print_exc()
