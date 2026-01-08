import ast
import os


def check_syntax(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return False


if __name__ == "__main__":
    for root, _dirs, files in os.walk("."):
        # Skip common non-source directories
        if any(d in root for d in [".git", "__pycache__", "node_modules", ".venv", "dist", "build"]):
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Use a separate process or suppress warnings to catch them per file
                # For now, just printing the file path before checking
                print(f"Checking {file_path}...")
                check_syntax(file_path)
