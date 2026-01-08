import os
import sys

from dataset_loader import DatasetLoader


def test_loader():
    # Redirect stdout to a file
    with open("test_output.txt", "w") as f:
        sys.stdout = f

        # Assuming this script is run from Project_Chimera directory
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imported_data"
        )
        print(f"Testing with base path: {base_path}")

        loader = DatasetLoader(base_path)

        # Test loading GPTFuzz
        print("\n--- Testing GPTFuzz ---")
        if loader.load_dataset("GPTFuzz"):
            prompt = loader.get_random_prompt("GPTFuzz")
            print(f"Random GPTFuzz prompt: {prompt[:100]}...")
        else:
            print("Failed to load GPTFuzz")

        # Test loading CodeChameleon
        print("\n--- Testing CodeChameleon ---")
        if loader.load_dataset("CodeChameleon"):
            prompt = loader.get_random_prompt("CodeChameleon")
            print(f"Random CodeChameleon prompt: {prompt[:100]}...")
        else:
            print("Failed to load CodeChameleon")

        # Test loading PAIR
        print("\n--- Testing PAIR ---")
        if loader.load_dataset("PAIR"):
            prompt = loader.get_random_prompt("PAIR")
            print(f"Random PAIR prompt: {prompt[:100]}...")
        else:
            print("Failed to load PAIR")

        sys.stdout = sys.__stdout__  # Reset stdout


if __name__ == "__main__":
    test_loader()
