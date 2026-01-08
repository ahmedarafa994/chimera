#!/usr/bin/env python3
"""
Safe automated replacements:
- `secrets.choice(X)` -> `secrets.choice(X)`
- `[secrets.choice(X) for _ in range(N)]` -> `[secrets.choice(X) for _ in range(N)]`
- `(secrets.randbelow((b) - (a) + 1) + (a))` -> `secrets.randbelow((b) - (a) + 1) + (a)`

Backs up files with a .bak extension before modifying.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.venv', '.git', 'node_modules', '__pycache__'}

py_files = [p for p in ROOT.rglob('*.py') if not any(part in EXCLUDE for part in p.parts)]
print(f"Found {len(py_files)} Python files to scan")

choice_re = re.compile(r"\brandom\.choice\s*\((?P<arg>[^)]+)\)")
choices_re = re.compile(r"\brandom\.choices\s*\((?P<args>[^)]+)\)")
randint_re = re.compile(r"\brandom\.randint\s*\((?P<a>[^,]+)\s*,\s*(?P<b>[^)]+)\)")

for p in py_files:
    s = p.read_text(encoding='utf-8')
    orig = s
    changed = False

    # Replace [secrets.choice(pop) for _ in range(N)] -> [secrets.choice(pop) for _ in range(N)]
    def repl_choices(m):
        args = m.group('args')
        # naive split on , k= or ,
        # try to find k= or the second arg
        # handle patterns: population, k=3 or population, 3
        if 'k=' in args:
            pop, rest = args.split(',', 1)
            k_part = rest.strip()
            k = k_part.split('=')[1].strip()
        else:
            parts = [a.strip() for a in args.split(',')]
            if len(parts) >= 2:
                pop = parts[0]
                k = parts[1]
            else:
                return m.group(0)  # leave unchanged
        return f"[secrets.choice({pop.strip()}) for _ in range({k.strip()})]"

    s = choices_re.sub(repl_choices, s)
    if s != orig:
        changed = True
        orig = s

    # Replace random.choice -> secrets.choice
    s2 = choice_re.sub(r"secrets.choice(\g<arg>)", s)
    if s2 != s:
        s = s2
        changed = True

    # Replace (secrets.randbelow((b) - (a) + 1) + (a)) -> secrets.randbelow((b) - (a) + 1) + (a)
    def repl_randint(m):
        a = m.group('a').strip()
        b = m.group('b').strip()
        return f"(secrets.randbelow(({b}) - ({a}) + 1) + ({a}))"

    s3 = randint_re.sub(repl_randint, s)
    if s3 != s:
        s = s3
        changed = True

    if changed:
        # ensure import secrets exists
        if 'import secrets' not in s:
            # try to insert after other stdlib imports block
            lines = s.splitlines()
            insert_at = 0
            for i, line in enumerate(lines[:50]):
                if line.startswith('import ') or line.startswith('from '):
                    insert_at = i + 1
            lines.insert(insert_at, 'import secrets')
            s = '\n'.join(lines)

        bak = p.with_suffix(p.suffix + '.bak')
        p.rename(bak)
        p.write_text(s, encoding='utf-8')
        print(f"Patched {p} (backup -> {bak})")

print('Done')
