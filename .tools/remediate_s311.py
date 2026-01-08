import os
import re
import shutil

# Files to patch with secure random helpers (High Risk)
# These files generate prompts, IDs, or security-sensitive mutations.
TARGET_FILES = [
    "meta_prompter/prompt_enhancer.py",
    "backend-api/app/services/jailbreak/evolutionary_optimizer.py",
    "backend-api/app/services/autodan_advanced/hierarchical_search.py",
    "backend-api/app/services/autodan_x/mutation_engine.py",
    "backend-api/app/services/advanced_transformation_layers.py",
    "backend-api/app/engines/preset_transformers.py",
    "backend-api/app/services/jailbreak/nextgen_enhancer.py",
    "backend-api/app/services/autodan/framework_autodan_reasoning/enhanced_execution.py",
    "backend-api/app/services/autodan/optimization/mutation_strategies.py",
    "backend-api/app/services/autodan/optimized/adaptive_mutation_engine.py",
    "backend-api/app/services/autodan/llm/chimera_adapter.py",
    "backend-api/app/services/autodan/engines/genetic_optimizer.py",
    "backend-api/app/services/autodan/engines/genetic_optimizer_complete.py",
    "backend-api/app/services/autodan/engines/token_perturbation.py",
    "backend-api/app/engines/obfuscator.py",
    "backend-api/app/engines/autodan_engine.py",
]

# Files to annotate with
# These use random for initialization, noise, or non-crypto heuristics.
ANNOTATE_FILES = [
    "backend-api/app/services/autodan_advanced/gradient_optimizer.py",
    "backend-api/app/services/autodan/framework_autodan_reasoning/gradient_optimizer.py",
    "backend-api/app/services/autodan/optimized/enhanced_lifelong_engine.py",
    "backend-api/app/services/autodan/optimization/gradient_optimization.py",
    "backend-api/app/services/autodan/optimized/enhanced_gradient_optimizer.py",
    "backend-api/app/engines/autodan_turbo/neural_bypass.py",
]

HELPER_CODE = """
# Helper: cryptographically secure pseudo-floats for security-sensitive choices
import secrets

def _secure_random() -> float:
    \"\"\"Cryptographically secure float in [0,1).\"\"\"
    return secrets.randbelow(10**9) / 1e9


def _secure_uniform(a, b):
    return a + _secure_random() * (b - a)
"""

def patch_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping missing file: {filepath}")
        return

    print(f"Patching {filepath}...")
    shutil.copy(filepath, filepath + ".bak")

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # 1. Insert helper code after imports
    if "_secure_random" not in content:
        # Find last import
        last_import_end = 0
        for match in re.finditer(r"^(import|from)\s+.+", content, re.MULTILINE):
            last_import_end = match.end()

        if last_import_end > 0:
            content = content[:last_import_end] + "\n" + HELPER_CODE + content[last_import_end:]
        else:
            content = HELPER_CODE + content

    # 2. Replace random.random() -> _secure_random()
    content = re.sub(r"random\.random\(\)", "_secure_random()", content)

    # 3. Replace random.uniform(a, b) -> _secure_uniform(a, b)
    content = re.sub(r"random\.uniform\(([^,]+),\s*([^)]+)\)", r"_secure_uniform(\1, \2)", content)

    # 4. Replace random.sample -> secrets.SystemRandom().sample
    if "random.sample" in content:
        if "import secrets" not in content: # Should be there from helper, but just in case
             content = "import secrets\n" + content
        content = content.replace("random.sample", "secrets.SystemRandom().sample")

    # 5. Replace random.shuffle -> secrets.SystemRandom().shuffle
    if "random.shuffle" in content:
        content = content.replace("random.shuffle", "secrets.SystemRandom().shuffle")

    # 6. Replace random.choice -> secrets.choice
    if "random.choice" in content:
        content = content.replace("random.choice", "secrets.choice")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def annotate_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping missing file: {filepath}")
        return

    print(f"Annotating {filepath}...")
    shutil.copy(filepath, filepath + ".bak")

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if ("random." in line or "np.random." in line) and "noqa: S311" not in line:
            line = line.rstrip() + "  # noqa: S311\n"
        new_lines.append(line)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def main():
    print("Starting S311 remediation...")

    for f in TARGET_FILES:
        patch_file(os.path.abspath(f))

    for f in ANNOTATE_FILES:
        annotate_file(os.path.abspath(f))

    print("Done.")

if __name__ == "__main__":
    main()
