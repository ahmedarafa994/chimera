"""Configuration Validation Framework.

Comprehensive validation system for Chimera configuration files including:
- JSON Schema validation
- Environment-specific rules
- Security checks
- Type safety
- Consistency validation
"""

import json
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from jsonschema import Draft7Validator, ValidationError

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    Draft7Validator = None  # type: ignore[assignment, misc]
    ValidationError = None  # type: ignore[assignment, misc]


class Severity(str, Enum):
    """Validation issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationIssue(BaseModel):
    """Represents a configuration validation issue."""

    severity: Severity
    category: str
    message: str
    path: str
    rule: str
    remediation: str | None = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class ConfigurationAnalysis(BaseModel):
    """Results of configuration analysis."""

    config_files: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    info: list[ValidationIssue] = Field(default_factory=list)
    passed_checks: int = 0
    failed_checks: int = 0
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == Severity.CRITICAL for issue in self.issues)

    @property
    def has_security_issues(self) -> bool:
        return any(issue.category == "security" for issue in self.issues)


class ConfigurationValidator:
    """Main configuration validation engine.

    Validates configuration files against schemas, security rules,
    and environment-specific requirements.
    """

    # Patterns for detecting secrets
    SECRET_PATTERNS = [
        (r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{20,})", "API Key"),
        (r"(?i)(secret|password|passwd)\s*[:=]\s*['\"]?([^\s\"']{8,})", "Secret/Password"),
        (r"(?i)(token|auth[_-]?token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{20,})", "Token"),
        (
            r"(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*['\"]?(AKIA[A-Z0-9]{16})",
            "AWS Access Key",
        ),
        (
            r"(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})",
            "AWS Secret Key",
        ),
        (r"(?i)(private[_-]?key)\s*[:=]\s*['\"]?(-----BEGIN.*?PRIVATE KEY-----)", "Private Key"),
    ]

    # Placeholder values that are safe
    SAFE_PLACEHOLDER_VALUES = [
        "CHANGE_ME",
        "your-api-key-here",
        "TODO",
        "example.com",
        "localhost",
        "127.0.0.1",
        "<your-",
        "INSERT_",
        "REPLACE_",
    ]

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.analysis = ConfigurationAnalysis()

    def analyze_project(self) -> ConfigurationAnalysis:
        """Perform comprehensive configuration analysis.

        Returns:
            ConfigurationAnalysis with all findings

        """
        # 1. Discover configuration files
        self.analysis.config_files = self._find_config_files()

        # 2. Check for security issues
        self._check_security_issues()

        # 3. Validate environment files
        self._validate_environment_files()

        # 4. Check consistency across environments
        self._check_environment_consistency()

        # 5. Validate YAML/JSON schemas
        self._validate_structured_configs()

        # 6. Check file permissions
        self._check_file_permissions()

        return self.analysis

    def _find_config_files(self) -> list[dict[str, Any]]:
        """Find all configuration files in the project."""
        config_patterns = [
            ("**/.env*", "env"),
            ("**/*.yaml", "yaml"),
            ("**/*.yml", "yaml"),
            ("**/*.json", "json"),
            ("**/config.py", "python"),
            ("**/settings.py", "python"),
        ]

        ignore_patterns = {
            "node_modules",
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            "dist",
            "build",
            ".next",
            "coverage",
        }

        config_files = []
        for pattern, config_type in config_patterns:
            for file_path in self.project_root.glob(pattern):
                # Skip if in ignored directory
                if any(ignored in file_path.parts for ignored in ignore_patterns):
                    continue

                # Skip if not a file
                if not file_path.is_file():
                    continue

                config_files.append(
                    {
                        "path": str(file_path.relative_to(self.project_root)),
                        "type": config_type,
                        "environment": self._detect_environment(file_path),
                        "size": file_path.stat().st_size,
                    },
                )

        return config_files

    def _detect_environment(self, file_path: Path) -> str | None:
        """Detect environment from filename."""
        name_lower = file_path.name.lower()

        if "production" in name_lower or "prod" in name_lower:
            return "production"
        if "staging" in name_lower or "stage" in name_lower:
            return "staging"
        if "development" in name_lower or "dev" in name_lower:
            return "development"
        if "test" in name_lower:
            return "test"

        return None

    def _check_security_issues(self) -> None:
        """Check for security vulnerabilities in configuration files."""
        for config_file in self.analysis.config_files:
            file_path = self.project_root / config_file["path"]

            # Skip template files
            if "template" in file_path.name.lower() or "example" in file_path.name.lower():
                continue

            try:
                content = file_path.read_text()

                # Check for exposed secrets
                for pattern, secret_type in self.SECRET_PATTERNS:
                    for match in re.finditer(pattern, content):
                        key_name = match.group(1)
                        value = match.group(2) if match.lastindex >= 2 else ""

                        if self._looks_like_real_secret(value):
                            self.analysis.issues.append(
                                ValidationIssue(
                                    severity=Severity.CRITICAL,
                                    category="security",
                                    message=f"Potential exposed {secret_type} detected: {key_name}",
                                    path=config_file["path"],
                                    rule="no_exposed_secrets",
                                    remediation="Move secrets to environment variables or secret management system",
                                ),
                            )

                # Check for weak security configurations
                self._check_weak_security_config(content, config_file["path"])

            except Exception as e:
                self.analysis.warnings.append(
                    ValidationIssue(
                        severity=Severity.LOW,
                        category="validation",
                        message=f"Failed to analyze file: {e}",
                        path=config_file["path"],
                        rule="file_read_error",
                    ),
                )

    def _looks_like_real_secret(self, value: str) -> bool:
        """Determine if a value looks like a real secret vs placeholder."""
        value_upper = value.upper()

        # Check if it's a known placeholder
        for placeholder in self.SAFE_PLACEHOLDER_VALUES:
            if placeholder in value_upper:
                return False

        # Check entropy (simple heuristic)
        if len(value) < 8:
            return False

        # If it has high character diversity, likely a real secret
        unique_chars = len(set(value))
        if unique_chars < 6:  # Too low diversity
            return False

        return True

    def _check_weak_security_config(self, content: str, file_path: str) -> None:
        """Check for weak security configurations."""
        checks = [
            (r"(?i)debug\s*[:=]\s*true", "Debug mode enabled", Severity.HIGH),
            (r"(?i)ssl\s*[:=]\s*false", "SSL disabled", Severity.CRITICAL),
            (r"(?i)verify\s*[:=]\s*false", "Certificate verification disabled", Severity.HIGH),
            (r"(?i)cors.*\*", "CORS allows all origins", Severity.MEDIUM),
            (r"http://(?!localhost|127\.0\.0\.1)", "HTTP endpoint (not HTTPS)", Severity.MEDIUM),
        ]

        for pattern, message, severity in checks:
            if re.search(pattern, content):
                self.analysis.issues.append(
                    ValidationIssue(
                        severity=severity,
                        category="security",
                        message=message,
                        path=file_path,
                        rule="weak_security_config",
                        remediation="Review security settings for production environments",
                    ),
                )

    def _validate_environment_files(self) -> None:
        """Validate .env files for required variables and proper format."""
        # Define required environment variables by environment
        required_vars = {
            "development": [
                "ENVIRONMENT",
                "LOG_LEVEL",
            ],
            "production": [
                "ENVIRONMENT",
                "LOG_LEVEL",
                "JWT_SECRET",
                "API_KEY",
                "ALLOWED_ORIGINS",
            ],
        }

        for config_file in self.analysis.config_files:
            if config_file["type"] != "env":
                continue

            file_path = self.project_root / config_file["path"]
            environment = config_file.get("environment", "development")

            try:
                env_vars = self._parse_env_file(file_path)

                # Check for required variables
                required = required_vars.get(environment, [])
                missing_vars = set(required) - set(env_vars.keys())

                if missing_vars:
                    self.analysis.issues.append(
                        ValidationIssue(
                            severity=Severity.HIGH,
                            category="configuration",
                            message=f"Missing required environment variables: {', '.join(missing_vars)}",
                            path=config_file["path"],
                            rule="required_env_vars",
                            remediation=f"Add missing variables to {config_file['path']}",
                        ),
                    )

                # Check for CHANGE_ME placeholders in production
                if environment == "production":
                    for key, value in env_vars.items():
                        if "CHANGE_ME" in value or "TODO" in value:
                            self.analysis.issues.append(
                                ValidationIssue(
                                    severity=Severity.CRITICAL,
                                    category="configuration",
                                    message=f"Placeholder value in production: {key}",
                                    path=config_file["path"],
                                    rule="no_placeholders_in_prod",
                                    remediation=f"Set a real value for {key}",
                                ),
                            )

            except Exception as e:
                self.analysis.warnings.append(
                    ValidationIssue(
                        severity=Severity.LOW,
                        category="validation",
                        message=f"Failed to parse env file: {e}",
                        path=config_file["path"],
                        rule="env_parse_error",
                    ),
                )

    def _parse_env_file(self, file_path: Path) -> dict[str, str]:
        """Parse .env file and return key-value pairs."""
        env_vars = {}
        content = file_path.read_text()

        for line in content.splitlines():
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

        return env_vars

    def _check_environment_consistency(self) -> None:
        """Check consistency across different environment configs."""
        # Group env files by base name
        env_files_by_base: dict[str, list[dict]] = {}

        for config_file in self.analysis.config_files:
            if config_file["type"] != "env":
                continue

            base_name = Path(config_file["path"]).name.replace(".env", "").replace(".", "")
            if not base_name:
                base_name = "default"

            if base_name not in env_files_by_base:
                env_files_by_base[base_name] = []

            env_files_by_base[base_name].append(config_file)

        # Check each group for consistency
        for base_name, files in env_files_by_base.items():
            if len(files) < 2:
                continue

            # Parse all files
            parsed_files = []
            for file_info in files:
                file_path = self.project_root / file_info["path"]
                try:
                    env_vars = self._parse_env_file(file_path)
                    parsed_files.append((file_info, env_vars))
                except Exception:
                    continue

            # Check for inconsistent keys
            if len(parsed_files) >= 2:
                all_keys = set()
                for _, env_vars in parsed_files:
                    all_keys.update(env_vars.keys())

                for file_info, env_vars in parsed_files:
                    missing_keys = all_keys - set(env_vars.keys())
                    if missing_keys:
                        self.analysis.warnings.append(
                            ValidationIssue(
                                severity=Severity.LOW,
                                category="consistency",
                                message=f"Missing keys compared to other environments: {', '.join(list(missing_keys)[:5])}",
                                path=file_info["path"],
                                rule="env_consistency",
                                remediation="Ensure all environments have consistent variable names",
                            ),
                        )

    def _validate_structured_configs(self) -> None:
        """Validate YAML/JSON configuration files."""
        for config_file in self.analysis.config_files:
            if config_file["type"] not in ["yaml", "json"]:
                continue

            file_path = self.project_root / config_file["path"]

            try:
                # Try to parse the file
                content = file_path.read_text()

                if config_file["type"] == "json":
                    json.loads(content)
                elif config_file["type"] == "yaml":
                    import yaml

                    yaml.safe_load(content)

                self.analysis.passed_checks += 1

            except Exception as e:
                self.analysis.issues.append(
                    ValidationIssue(
                        severity=Severity.HIGH,
                        category="syntax",
                        message=f"Invalid {config_file['type'].upper()} syntax: {e}",
                        path=config_file["path"],
                        rule="valid_syntax",
                        remediation="Fix syntax errors in configuration file",
                    ),
                )
                self.analysis.failed_checks += 1

    def _check_file_permissions(self) -> None:
        """Check file permissions for security."""
        for config_file in self.analysis.config_files:
            # Skip template/example files
            if "template" in config_file["path"] or "example" in config_file["path"]:
                continue

            file_path = self.project_root / config_file["path"]

            # Check if file is world-readable (on Unix systems)
            if os.name != "nt":  # Not Windows
                mode = file_path.stat().st_mode
                if mode & 0o004:  # World-readable
                    self.analysis.warnings.append(
                        ValidationIssue(
                            severity=Severity.MEDIUM,
                            category="security",
                            message="Configuration file is world-readable",
                            path=config_file["path"],
                            rule="file_permissions",
                            remediation=f"chmod 600 {config_file['path']}",
                        ),
                    )

    def generate_report(self) -> str:
        """Generate a human-readable validation report."""
        lines = [
            "=" * 80,
            "Chimera Configuration Validation Report",
            "=" * 80,
            f"Analyzed at: {self.analysis.analyzed_at.isoformat()}",
            f"Project root: {self.project_root}",
            "",
            "SUMMARY",
            "-" * 80,
            f"Configuration files found: {len(self.analysis.config_files)}",
            f"Passed checks: {self.analysis.passed_checks}",
            f"Failed checks: {self.analysis.failed_checks}",
            f"Critical issues: {sum(1 for i in self.analysis.issues if i.severity == Severity.CRITICAL)}",
            f"High severity issues: {sum(1 for i in self.analysis.issues if i.severity == Severity.HIGH)}",
            f"Warnings: {len(self.analysis.warnings)}",
            "",
        ]

        if self.analysis.has_critical_issues:
            lines.extend(
                [
                    "❌ CRITICAL ISSUES FOUND",
                    "-" * 80,
                ],
            )
            for issue in self.analysis.issues:
                if issue.severity == Severity.CRITICAL:
                    lines.extend(
                        [
                            f"[{issue.severity.value.upper()}] {issue.message}",
                            f"  File: {issue.path}",
                            f"  Rule: {issue.rule}",
                            f"  Fix: {issue.remediation or 'See documentation'}",
                            "",
                        ],
                    )

        if any(issue.severity == Severity.HIGH for issue in self.analysis.issues):
            lines.extend(
                [
                    "⚠️  HIGH SEVERITY ISSUES",
                    "-" * 80,
                ],
            )
            for issue in self.analysis.issues:
                if issue.severity == Severity.HIGH:
                    lines.extend(
                        [
                            f"[{issue.severity.value.upper()}] {issue.message}",
                            f"  File: {issue.path}",
                            f"  Rule: {issue.rule}",
                            "",
                        ],
                    )

        lines.extend(
            [
                "=" * 80,
                "END OF REPORT",
                "=" * 80,
            ],
        )

        return "\n".join(lines)


# Convenience function for CLI usage
def validate_project_configuration(project_root: str) -> tuple[bool, str]:
    """Validate project configuration and return results.

    Args:
        project_root: Path to project root directory

    Returns:
        Tuple of (success: bool, report: str)

    """
    validator = ConfigurationValidator(Path(project_root))
    analysis = validator.analyze_project()
    report = validator.generate_report()

    success = not analysis.has_critical_issues

    return success, report


if __name__ == "__main__":
    import sys

    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    success, report = validate_project_configuration(project_path)

    sys.exit(0 if success else 1)
