"""
Apply targeted S311 secure-random replacements in high-risk files.
- Replaces `random.random()` -> `_secure_random()`
- Replaces `random.uniform(a, b)` -> `_secure_uniform(a, b)`
- Ensures `import secrets` and helper functions are present
- Writes `.bak` backup for each modified file

Run from repository root with the venv Python.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# High-risk files (from triage) — edit these only
TARGETS = [
    "meta_prompter/refusal_bypass_optimizer.py",
    "meta_prompter/prompt_enhancer.py",
    "backend-api/app/services/deepteam/prompt_generator.py",
    "prometheus_unbound/app/core/persona_generator.py",
    "pandora_gpt.py",
    "chimera-orchestrator/integration/service_registry.py",
]

IMPORT_BLOCK = (
    "import secrets\n\n"
    "def _secure_random() -> float:\n"
    "    # Cryptographically secure float in [0,1) using 9-digit granularity\n"
    "    return secrets.randbelow(10**9) / 1e9\n\n"
    "def _secure_uniform(a, b):\n"
    "    return a + _secure_random() * (b - a)\n\n"
)

RE_RAND_RANDOM = re.compile(r"\brandom\.random\s*\(\s*\)")
RE_RAND_UNIFORM = re.compile(r"\brandom\.uniform\s*\(")

patched = []
for rel in TARGETS:
    path = ROOT / rel
    if not path.exists():
        print(f"[SKIP] not found: {rel}")
        continue
    text = path.read_text(encoding="utf-8")
    new = text
    changed = False

    # Replace random.random()
    if RE_RAND_RANDOM.search(new):
        new = RE_RAND_RANDOM.sub("_secure_random()", new)
        changed = True

    # Replace random.uniform(
    if RE_RAND_UNIFORM.search(new):
        new = RE_RAND_UNIFORM.sub("_secure_uniform(", new)
        changed = True

    if changed:
        # ensure import secrets and helpers exist
        if "import secrets" not in new:
            # Insert after initial module docstring or at top
            # Find end of module docstring if present
            m = re.match(r"(\s*\"\"\"[\s\S]*?\"\"\"\s*)", new)
            if m:
                insert_at = m.end()
                new = new[:insert_at] + "\n" + IMPORT_BLOCK + new[insert_at:]
            else:
                new = IMPORT_BLOCK + new
        else:
            # ensure helpers defined
            if "def _secure_random" not in new:
                # add helpers after imports (naively place at top)
                new = IMPORT_BLOCK + new

        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            path.write_text(text, encoding="utf-8")
            bak.write_text(text, encoding="utf-8")
        path.write_text(new, encoding="utf-8")
        patched.append(rel)
        print(f"Patched: {rel} (backup: {bak.name})")
    else:
        print(f"No change: {rel}")

print(f"Done. Patched {len(patched)} files.")
