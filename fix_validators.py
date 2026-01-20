#!/usr/bin/env python3
"""Fix all @field_validator decorators to use @classmethod."""

import re
from pathlib import Path

def fix_field_validators(file_path):
    """Fix field_validator decorators in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern to match @field_validator followed by def method(self, ...)
    # We need to add @classmethod and change self to cls
    pattern = r'(@field_validator\([^)]+\))\s*\n\s*(def\s+\w+\()(self)(\s*,)'

    def replacer(match):
        decorator = match.group(1)
        def_part = match.group(2)
        self_param = match.group(3)
        comma = match.group(4)
        return f'{decorator}\n    @classmethod\n    {def_part}cls{comma}'

    content = re.sub(pattern, replacer, content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# Find all Python files in backend-api/app
backend_path = Path('backend-api/app')
fixed_files = []

for py_file in backend_path.rglob('*.py'):
    if fix_field_validators(py_file):
        fixed_files.append(str(py_file))
        print(f'Fixed: {py_file}')

print(f'\nTotal files fixed: {len(fixed_files)}')
