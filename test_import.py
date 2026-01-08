import os
import sys

sys.path.append(os.getcwd())

try:
    from meta_prompter.jailbreak_enhancer import JailbreakPromptEnhancer

    print("Import successful")
    enhancer = JailbreakPromptEnhancer()
    print("Instantiation successful")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
