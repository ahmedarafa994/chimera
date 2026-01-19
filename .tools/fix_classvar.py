#!/usr/bin/env python3
"""
Batch ClassVar annotation script for RUF012 fixes.
Annotates mutable class attributes (dicts, lists) with ClassVar[...].
"""

import os
import re
import shutil

TARGETS = [
    "meta_prompter/prompt_enhancer.py",
    "backend-api/app/services/jailbreak/evolutionary_optimizer.py",
    "backend-api/app/services/autodan_x/mutation_engine.py",
    "backend-api/app/engines/preset_transformers.py",
]


def fix_classvar_in_file(filepath):
    """Add ClassVar annotations to mutable class attributes."""
    if not os.path.exists(filepath):
        print(f"Skipping missing: {filepath}")
        return False

    print(f"Processing {filepath}...")
    shutil.copy(filepath, filepath + ".bak")

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    # Check if ClassVar is imported
    has_classvar = any("ClassVar" in line for line in lines)
    if not has_classvar:
        # Find the import line for 'typing'
        typing_import_idx = None
        for i, line in enumerate(lines):
            if "from typing import" in line:
                typing_import_idx = i
                break

        if typing_import_idx is not None:
            # Add ClassVar to existing typing import
            lines[typing_import_idx] = lines[typing_import_idx].rstrip()
            if "ClassVar" not in lines[typing_import_idx]:
                # Append ClassVar before closing paren if it exists
                if ")" in lines[typing_import_idx]:
                    lines[typing_import_idx] = lines[typing_import_idx].rstrip(")\n")
                    lines[typing_import_idx] += ", ClassVar)\n"
                else:
                    lines[typing_import_idx] += ", ClassVar\n"
        else:
            # Add new import at top (after module docstring if any)
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    # Skip docstring
                    for j in range(i + 1, len(lines)):
                        if '"""' in lines[j] or "'''" in lines[j]:
                            insert_idx = j + 1
                            break
                    break
            lines.insert(insert_idx, "from typing import ClassVar\n")

    # Now find and annotate class attributes
    in_class = False
    class_indent = 0
    i = 0
    while i < len(lines):
        line = lines[i]

        # Track class definition
        if re.match(r"^class\s+\w+", line):
            in_class = True
            class_indent = len(line) - len(line.lstrip())
            i += 1
            continue

        # Exit class if indent returns to class level or less
        if (
            in_class
            and line.strip()
            and not line.startswith(" " * (class_indent + 1))
            and (not line.startswith(" " * class_indent) or re.match(r"^class\s+", line))
        ):
            in_class = False

        if in_class:
            # Check for mutable class attribute (dict or list at class level)
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#"):
                attr_match = re.match(r"^(\w+)\s*=\s*(\[|\{)", stripped)
                if attr_match:
                    attr_name = attr_match.group(1)
                    is_list = attr_match.group(2) == "["
                    is_dict = attr_match.group(2) == "{"

                    # Skip if already has ClassVar annotation
                    if "ClassVar" not in line:
                        # Determine type hint
                        if is_list:
                            type_hint = "ClassVar[list]"
                        elif is_dict:
                            type_hint = "ClassVar[dict]"
                        else:
                            i += 1
                            continue

                        # Replace: attr_name = ... → attr_name: ClassVar[...] = ...
                        new_line = line.replace(f"{attr_name} =", f"{attr_name}: {type_hint} =", 1)
                        lines[i] = new_line

        i += 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return True


def main():
    print("Starting RUF012 ClassVar annotation fixes...")
    for target in TARGETS:
        fix_classvar_in_file(os.path.abspath(target))
    print("Done.")


if __name__ == "__main__":
    main()
