import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    pass
except Exception:
    import traceback

    with open("traceback.txt", "w") as f:
        traceback.print_exc(file=f)
