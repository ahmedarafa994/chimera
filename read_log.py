
log_file = r"d:\MUZIK\chimera\.subagents\orchestrator\sessions\20251231_124644.jsonl"

try:
    with open(log_file, encoding='utf-8', errors='ignore') as f:
        print(f.read())
except Exception as e:
    print(f"Error reading file: {e}")
