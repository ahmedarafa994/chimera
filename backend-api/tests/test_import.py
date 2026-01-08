import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    print("Imported app")
    print("Imported config")
    print("Imported transformation_engine")
except Exception:
    import traceback

    with open("traceback.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Error occurred, check traceback.txt")
