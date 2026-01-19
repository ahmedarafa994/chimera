import os
import sys

# Define the target files to check
TARGET_FILES = [
    r"k:\MUZIK\chimera\backend-api\app\engines\autodan_turbo\advanced_bypass.py",
    r"k:\MUZIK\chimera\backend-api\app\engines\autodan_turbo\refusal_bypass.py",
    r"k:\MUZIK\chimera\backend-api\app\engines\autodan_turbo\neural_bypass.py",
    r"k:\MUZIK\chimera\autodan-turbo\agents\scorer.py",
    r"k:\MUZIK\chimera\autodan-turbo\agents\target.py",
    r"k:\MUZIK\chimera\autodan-turbo\agents\attacker.py",
    r"k:\MUZIK\chimera\backend-api\app\services\autodan\framework_autodan_reasoning\enhanced_attacker.py",
    r"k:\MUZIK\chimera\backend-api\app\services\autodan\framework_autodan_reasoning\research_protocols.py",
]

# Define ethical keywords that should NOT be present
ETHICAL_KEYWORDS = [
    "safety evaluation",
    "ethical guidelines",
    "responsible disclosure",
    "harmful content",
    "security boundaries",
    "academic framing",
    "high-risk keywords",
]


def check_files():
    print("Starting Global Nullification Verification (Debug)...")
    errors = []

    for file_path in TARGET_FILES:
        if not os.path.exists(file_path):
            print(f"WARNING: File not found: {file_path}")
            continue

        print(f"Checking {os.path.basename(file_path)}...")
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.splitlines()
            for keyword in ETHICAL_KEYWORDS:
                if keyword in content:
                    found_in_lines = []
                    for i, line in enumerate(lines):
                        if keyword in line:
                            # Ignore if it's the new Enum member name "OPERATIONAL_FRAMING" (wait, query is "academic framing")
                            # But if query is "academic framing", and line has "ACADEMIC_FRAMING", it might match if we did case insensitive?
                            # But here we do strict string `in`.
                            if keyword == "academic framing" and "ACADEMIC_FRAMING" in line:
                                pass  # Variable name left over? I renamed it though.
                            else:
                                found_in_lines.append(f"Line {i+1}: {line.strip()}")

                    if found_in_lines:
                        errors.append(
                            f"Found forbidden keyword '{keyword}' in {os.path.basename(file_path)}:\n"
                            + "\n".join(found_in_lines)
                        )

        except Exception as e:
            errors.append(f"Error reading {file_path}: {e!s}")

    if errors:
        print("\nFAILURE: Ethical constraints detected!")
        for e in errors:
            print(f"- {e}")
        sys.exit(1)
    else:
        print("\nSUCCESS: No ethical constraints detected in target files.")
        sys.exit(0)


if __name__ == "__main__":
    check_files()
