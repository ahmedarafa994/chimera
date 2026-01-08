# Documentation Automation Pipeline

This document outlines the automated documentation generation and maintenance system for Chimera.

## Overview

The documentation automation pipeline ensures that documentation stays synchronized with code changes through:

1. **Automated API Documentation Generation**
2. **Code Documentation Extraction**
3. **Documentation Validation and Testing**
4. **Deployment to Documentation Sites**
5. **Documentation Coverage Analysis**

## GitHub Actions Workflow

### Documentation Generation Workflow

```yaml
# .github/workflows/docs-generate.yml
name: Generate Documentation

on:
  push:
    branches: [main, develop]
    paths:
      - 'backend-api/**/*.py'
      - 'frontend/src/**/*.ts'
      - 'frontend/src/**/*.tsx'
      - 'docs/**/*.md'
      - '.github/workflows/docs-generate.yml'

  pull_request:
    branches: [main]
    paths:
      - 'backend-api/**/*.py'
      - 'frontend/src/**/*.ts'
      - 'frontend/src/**/*.tsx'
      - 'docs/**/*.md'

jobs:
  generate-docs:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'

    - name: Install Python dependencies
      run: |
        cd backend-api
        pip install -r requirements.txt
        pip install sphinx sphinx-rtd-theme

    - name: Install Node.js dependencies
      run: |
        npm install -g @redocly/cli typedoc

    - name: Generate OpenAPI Specification
      run: |
        cd backend-api
        python scripts/generate_openapi.py > ../docs/openapi.json

    - name: Generate API Documentation
      run: |
        redocly build-docs docs/openapi.json -o docs/api/index.html

    - name: Generate TypeScript Documentation
      run: |
        cd frontend
        typedoc --out ../docs/frontend src

    - name: Generate Python Documentation
      run: |
        cd backend-api
        sphinx-build -b html docs/source ../docs/backend

    - name: Validate Documentation
      run: |
        python scripts/validate_docs.py

    - name: Check Documentation Coverage
      run: |
        python scripts/check_doc_coverage.py

    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs
        destination_dir: docs
```

## Automated Scripts

### 1. OpenAPI Generation Script

```python
# scripts/generate_openapi.py
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend-api"))

from app.main import app

def generate_openapi_spec():
    """Generate OpenAPI specification from FastAPI app."""
    openapi_schema = app.openapi()

    # Enhance with additional metadata
    openapi_schema["info"]["x-logo"] = {
        "url": "https://docs.chimera-api.example.com/logo.png"
    }

    # Add server configurations
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8001",
            "description": "Development server"
        },
        {
            "url": "https://api.chimera.example.com",
            "description": "Production server"
        }
    ]

    return openapi_schema

if __name__ == "__main__":
    spec = generate_openapi_spec()
    print(json.dumps(spec, indent=2))
```

### 2. Documentation Validation Script

```python
# scripts/validate_docs.py
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class DocumentationValidator:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> Tuple[List[str], List[str]]:
        """Run all validation checks."""
        self.validate_markdown_files()
        self.validate_api_documentation()
        self.validate_code_examples()
        self.validate_links()
        self.validate_images()

        return self.errors, self.warnings

    def validate_markdown_files(self):
        """Validate markdown files for common issues."""
        md_files = list(self.docs_dir.glob("**/*.md"))

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')

            # Check for frontmatter if required
            if not content.startswith('---') and md_file.name != 'README.md':
                self.warnings.append(f"{md_file}: Missing frontmatter")

            # Check for proper heading structure
            headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            if not headings:
                self.warnings.append(f"{md_file}: No headings found")

            # Check for broken internal links
            internal_links = re.findall(r'\[([^\]]+)\]\(([^)]+\.md[^)]*)\)', content)
            for link_text, link_path in internal_links:
                if not (self.docs_dir / link_path).exists():
                    self.errors.append(f"{md_file}: Broken internal link: {link_path}")

    def validate_api_documentation(self):
        """Validate API documentation completeness."""
        openapi_file = self.docs_dir / "openapi.yaml"
        if not openapi_file.exists():
            self.errors.append("Missing OpenAPI specification file")
            return

        # Validate OpenAPI file structure
        try:
            import yaml
            with open(openapi_file) as f:
                spec = yaml.safe_load(f)

            # Check required sections
            required_sections = ["info", "paths", "components"]
            for section in required_sections:
                if section not in spec:
                    self.errors.append(f"OpenAPI spec missing required section: {section}")

            # Validate paths have proper documentation
            if "paths" in spec:
                for path, methods in spec["paths"].items():
                    for method, details in methods.items():
                        if "description" not in details:
                            self.warnings.append(f"API endpoint {method.upper()} {path} missing description")

                        if "responses" not in details:
                            self.errors.append(f"API endpoint {method.upper()} {path} missing responses")

        except Exception as e:
            self.errors.append(f"Failed to parse OpenAPI spec: {e}")

    def validate_code_examples(self):
        """Validate code examples in documentation."""
        md_files = list(self.docs_dir.glob("**/*.md"))

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')

            # Find code blocks
            code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)

            for lang, code in code_blocks:
                if lang == 'python':
                    self._validate_python_code(md_file, code)
                elif lang == 'bash':
                    self._validate_bash_code(md_file, code)
                elif lang == 'json':
                    self._validate_json_code(md_file, code)

    def _validate_python_code(self, file_path: Path, code: str):
        """Validate Python code examples."""
        try:
            compile(code, str(file_path), 'exec')
        except SyntaxError as e:
            self.errors.append(f"{file_path}: Python syntax error in code block: {e}")

    def _validate_bash_code(self, file_path: Path, code: str):
        """Validate bash code examples."""
        # Check for common bash issues
        lines = code.strip().split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('cd ') and '&&' not in line:
                self.warnings.append(f"{file_path}: Bash cd command without && on line {i+1}")

    def _validate_json_code(self, file_path: Path, code: str):
        """Validate JSON code examples."""
        try:
            import json
            json.loads(code)
        except json.JSONDecodeError as e:
            self.errors.append(f"{file_path}: Invalid JSON in code block: {e}")

    def validate_links(self):
        """Validate external links."""
        import requests

        md_files = list(self.docs_dir.glob("**/*.md"))

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')

            # Find external links
            external_links = re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)', content)

            for link_text, url in external_links:
                try:
                    response = requests.head(url, timeout=10, allow_redirects=True)
                    if response.status_code >= 400:
                        self.warnings.append(f"{md_file}: Dead external link: {url}")
                except Exception:
                    self.warnings.append(f"{md_file}: Could not verify external link: {url}")

    def validate_images(self):
        """Validate image references."""
        md_files = list(self.docs_dir.glob("**/*.md"))

        for md_file in md_files:
            content = md_file.read_text(encoding='utf-8')

            # Find image references
            images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)

            for alt_text, img_path in images:
                if not img_path.startswith('http'):
                    # Local image
                    img_file = self.docs_dir / img_path
                    if not img_file.exists():
                        self.errors.append(f"{md_file}: Missing image file: {img_path}")

                if not alt_text:
                    self.warnings.append(f"{md_file}: Image missing alt text: {img_path}")

def main():
    docs_dir = Path(__file__).parent.parent / "docs"
    validator = DocumentationValidator(docs_dir)

    errors, warnings = validator.validate_all()

    if errors:
        print("Documentation Errors:")
        for error in errors:
            print(f"  ❌ {error}")

    if warnings:
        print("Documentation Warnings:")
        for warning in warnings:
            print(f"  ⚠️ {warning}")

    if errors:
        sys.exit(1)
    else:
        print("✅ Documentation validation passed!")

if __name__ == "__main__":
    main()
```

### 3. Documentation Coverage Script

```python
# scripts/check_doc_coverage.py
import ast
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

class DocumentationCoverage:
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.coverage_report: Dict[str, Dict] = {}

    def analyze_python_files(self) -> Dict[str, float]:
        """Analyze Python files for documentation coverage."""
        python_files = glob.glob(str(self.source_dir / "**/*.py"), recursive=True)

        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0

        file_coverage = {}

        for file_path in python_files:
            if "__pycache__" in file_path:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                file_functions = 0
                file_documented_functions = 0
                file_classes = 0
                file_documented_classes = 0

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):  # Skip private functions
                            file_functions += 1
                            total_functions += 1

                            if ast.get_docstring(node):
                                file_documented_functions += 1
                                documented_functions += 1

                    elif isinstance(node, ast.ClassDef):
                        file_classes += 1
                        total_classes += 1

                        if ast.get_docstring(node):
                            file_documented_classes += 1
                            documented_classes += 1

                # Calculate file coverage
                file_items = file_functions + file_classes
                file_documented = file_documented_functions + file_documented_classes

                if file_items > 0:
                    coverage = (file_documented / file_items) * 100
                    file_coverage[file_path] = {
                        'coverage': coverage,
                        'functions': file_functions,
                        'documented_functions': file_documented_functions,
                        'classes': file_classes,
                        'documented_classes': file_documented_classes
                    }

            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")

        # Calculate overall coverage
        total_items = total_functions + total_classes
        total_documented = documented_functions + documented_classes

        overall_coverage = (total_documented / total_items * 100) if total_items > 0 else 100

        return {
            'overall': overall_coverage,
            'files': file_coverage,
            'totals': {
                'functions': total_functions,
                'documented_functions': documented_functions,
                'classes': total_classes,
                'documented_classes': documented_classes
            }
        }

    def generate_report(self) -> str:
        """Generate documentation coverage report."""
        coverage_data = self.analyze_python_files()

        report = ["# Documentation Coverage Report", ""]
        report.append(f"**Overall Coverage: {coverage_data['overall']:.1f}%**")
        report.append("")

        # Summary
        totals = coverage_data['totals']
        report.append("## Summary")
        report.append("")
        report.append(f"- Functions: {totals['documented_functions']}/{totals['functions']} documented")
        report.append(f"- Classes: {totals['documented_classes']}/{totals['classes']} documented")
        report.append("")

        # File breakdown
        report.append("## File Coverage")
        report.append("")
        report.append("| File | Coverage | Functions | Classes |")
        report.append("|------|----------|-----------|---------|")

        for file_path, data in sorted(coverage_data['files'].items()):
            relative_path = os.path.relpath(file_path, self.source_dir)
            coverage = data['coverage']
            functions = f"{data['documented_functions']}/{data['functions']}"
            classes = f"{data['documented_classes']}/{data['classes']}"

            # Add emoji indicators
            if coverage >= 80:
                indicator = "✅"
            elif coverage >= 60:
                indicator = "⚠️"
            else:
                indicator = "❌"

            report.append(f"| {relative_path} | {indicator} {coverage:.1f}% | {functions} | {classes} |")

        # Recommendations
        report.append("")
        report.append("## Recommendations")
        report.append("")

        low_coverage_files = [
            (file, data) for file, data in coverage_data['files'].items()
            if data['coverage'] < 60
        ]

        if low_coverage_files:
            report.append("### Files needing documentation improvements:")
            report.append("")
            for file_path, data in low_coverage_files[:5]:  # Top 5 worst
                relative_path = os.path.relpath(file_path, self.source_dir)
                report.append(f"- `{relative_path}`: {data['coverage']:.1f}% coverage")

        return "\n".join(report)

def main():
    backend_dir = Path(__file__).parent.parent / "backend-api" / "app"
    coverage_analyzer = DocumentationCoverage(backend_dir)

    report = coverage_analyzer.generate_report()

    # Save report
    output_file = Path(__file__).parent.parent / "docs" / "COVERAGE_REPORT.md"
    output_file.write_text(report, encoding='utf-8')

    print("Documentation coverage report generated:")
    print(report)

    # Check if coverage meets minimum threshold
    coverage_data = coverage_analyzer.analyze_python_files()
    if coverage_data['overall'] < 60:
        print(f"\n❌ Documentation coverage {coverage_data['overall']:.1f}% is below minimum threshold of 60%")
        exit(1)
    else:
        print(f"\n✅ Documentation coverage {coverage_data['overall']:.1f}% meets minimum requirements")

if __name__ == "__main__":
    main()
```

### 4. Documentation Site Configuration

```yaml
# mkdocs.yml - MkDocs configuration
site_name: Chimera Documentation
site_description: AI-Powered Prompt Optimization & Jailbreak Research System
site_url: https://docs.chimera-api.example.com

nav:
  - Home: index.md
  - User Guide: USER_GUIDE.md
  - Developer Guide: DEVELOPER_GUIDE.md
  - Architecture: ARCHITECTURE.md
  - API Reference:
    - Overview: api/index.md
    - OpenAPI Spec: api/openapi.yaml
  - Coverage Report: COVERAGE_REPORT.md

theme:
  name: material
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.highlight
    - content.code.copy

plugins:
  - search
  - mermaid2
  - swagger-ui-tag

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
  - attr_list
  - md_in_html
  - toc:
      permalink: true

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/your-org/chimera
    - icon: fontawesome/solid/globe
      link: https://chimera-api.example.com
```

### 5. Pre-commit Hook for Documentation

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-docs
        name: Validate Documentation
        entry: python scripts/validate_docs.py
        language: system
        files: ^docs/.*\.md$

      - id: check-doc-coverage
        name: Check Documentation Coverage
        entry: python scripts/check_doc_coverage.py
        language: system
        files: ^backend-api/.*\.py$

      - id: generate-openapi
        name: Generate OpenAPI Specification
        entry: bash -c 'cd backend-api && python ../scripts/generate_openapi.py > ../docs/openapi.json'
        language: system
        files: ^backend-api/app/.*\.py$
```

## Package.json Scripts

```json
{
  "scripts": {
    "docs:generate": "python scripts/generate_openapi.py > docs/openapi.json",
    "docs:validate": "python scripts/validate_docs.py",
    "docs:coverage": "python scripts/check_doc_coverage.py",
    "docs:serve": "mkdocs serve",
    "docs:build": "mkdocs build",
    "docs:deploy": "mkdocs gh-deploy",
    "docs:all": "npm run docs:generate && npm run docs:validate && npm run docs:coverage"
  }
}
```

## Usage

### Local Development

```bash
# Generate all documentation
npm run docs:all

# Serve documentation locally
npm run docs:serve

# Validate documentation
npm run docs:validate

# Check coverage
npm run docs:coverage
```

### CI/CD Integration

The documentation pipeline automatically runs on:

1. **Pull Requests**: Validates documentation changes
2. **Main Branch Push**: Generates and deploys updated documentation
3. **Scheduled**: Weekly documentation link validation

### Manual Deployment

```bash
# Build and deploy to GitHub Pages
npm run docs:build
npm run docs:deploy

# Or using MkDocs directly
mkdocs gh-deploy --force
```

This automation pipeline ensures that documentation remains accurate, complete, and synchronized with code changes while providing comprehensive validation and coverage reporting.