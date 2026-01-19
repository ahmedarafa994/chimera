import ast
import os

def check_syntax(directory):
    for root, dirs, files in os.walk(directory):
        if "node_modules" in root or ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    print(f"Syntax error in {path}: {e}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    check_syntax(".")