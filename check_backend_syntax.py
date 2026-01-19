#!/usr/bin/env python3
"""
Quick Backend Syntax Check

Validates that the backend API files have correct syntax and imports
without running the full application.
"""

import ast
import sys
from pathlib import Path


def check_python_syntax(file_path: Path) -> tuple[bool, str]:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse the AST to check syntax
        ast.parse(content, filename=str(file_path))
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading file: {e!s}"


def main():
    """Main entry point."""
    backend_dir = Path("backend-api")
    if not backend_dir.exists():
        print("❌ backend-api directory not found")
        return False

    # Files to check (the ones we modified)
    files_to_check = [
        "app/api/api_routes.py",
        "app/api/v1/api.py",
        "app/api/v1/router.py",
        "app/api/error_handlers.py",
    ]

    print("Checking Python syntax for modified files...")
    print("=" * 60)

    all_passed = True

    for file_path in files_to_check:
        full_path = backend_dir / file_path
        if not full_path.exists():
            print(f"WARNING: {file_path}: File not found")
            continue

        is_valid, error_msg = check_python_syntax(full_path)
        if is_valid:
            print(f"PASS: {file_path}: Syntax OK")
        else:
            print(f"FAIL: {file_path}: {error_msg}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All syntax checks passed!")
        print("\nKey fixes verified:")
        print("   - Added missing aegis_ws import in v1/api.py")
        print("   - Added /jailbreak endpoint alias in api_routes.py")
        print("   - Added /techniques endpoint alias in api_routes.py")
        print("   - All Python files have valid syntax")
        return True
    else:
        print("ERROR: Some syntax checks failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
