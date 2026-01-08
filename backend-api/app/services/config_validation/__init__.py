"""
Configuration Validation Package

Provides comprehensive validation for Chimera configuration files.

Usage:
    from app.services.config_validation import ConfigurationValidator

    validator = ConfigurationValidator(Path("/path/to/project"))
    analysis = validator.analyze_project()

    if analysis.has_critical_issues:
        print("CRITICAL ISSUES FOUND!")
        for issue in analysis.issues:
            print(f"- {issue.message}")
"""

from .validator import (
    ConfigurationAnalysis,
    ConfigurationValidator,
    Severity,
    ValidationIssue,
    validate_project_configuration,
)

__all__ = [
    "ConfigurationAnalysis",
    "ConfigurationValidator",
    "Severity",
    "ValidationIssue",
    "validate_project_configuration",
]

__version__ = "1.0.0"
