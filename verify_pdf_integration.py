import sys
from pathlib import Path

# Add backend-api to path
sys.path.append(str(Path("backend-api").absolute()))

from app.services.autodan.jailbreak_data_loader import get_autodan_turbo_strategies


def verify_strategies():
    print("Verifying AutoDAN-Turbo PDF Strategies Integration...")

    strategies = get_autodan_turbo_strategies()

    if not strategies:
        print("❌ Failed to load strategies!")
        sys.exit(1)

    print(f"✅ Loaded {len(strategies)} strategies.")

    expected_count = 16
    if len(strategies) != expected_count:
        print(f"⚠️ Warning: Expected {expected_count} strategies, found {len(strategies)}.")

    # Verify structure
    required_keys = ["Strategy", "Definition", "Example"]
    for i, s in enumerate(strategies):
        missing = [k for k in required_keys if k not in s]
        if missing:
            print(f"❌ Strategy {i} ('{s.get('Strategy', 'Unknown')}') missing keys: {missing}")
            sys.exit(1)

    print("✅ All strategies have correct structure.")
    print("\nSample Strategy:")
    print(f"Name: {strategies[0]['Strategy']}")
    print(f"Definition: {strategies[0]['Definition']}")
    print(f"Example: {strategies[0]['Example'][:100]}...")

    print("\nIntegration Verification Successful!")


if __name__ == "__main__":
    verify_strategies()
