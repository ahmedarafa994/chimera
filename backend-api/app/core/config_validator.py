"""
Configuration Validator for AI Provider Integration

Comprehensive validation module for AI provider configurations.
Validates API keys, failover chains, model references, circuit breaker settings,
rate limits, and capability consistency.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Enums and Data Classes
# =============================================================================


class ValidationSeverity(Enum):
    """Validation severity levels."""

    ERROR = "error"  # Configuration is invalid, will cause failures
    WARNING = "warning"  # Configuration may cause issues
    INFO = "info"  # Informational notice


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    is_valid: bool
    severity: ValidationSeverity
    component: str
    message: str
    suggestion: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    timestamp: datetime
    overall_valid: bool
    error_count: int
    warning_count: int
    info_count: int
    results: list[ValidationResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_valid": self.overall_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Configuration Validator
# =============================================================================


class ConfigValidator:
    """
    Comprehensive configuration validator for AI provider integration.

    Validates:
    - API key environment variables
    - Failover chain references
    - Model references
    - Circuit breaker settings
    - Rate limit configurations
    - Capability consistency

    Example:
        validator = ConfigValidator()
        results = validator.validate_all()
        report = validator.get_validation_report()
    """

    def __init__(self):
        """Initialize the configuration validator."""
        self._config_manager = None
        self._last_validation: ValidationReport | None = None

    def _get_config_manager(self):
        """Lazily get the AI config manager."""
        if self._config_manager is None:
            from app.core.ai_config_manager import get_ai_config_manager

            self._config_manager = get_ai_config_manager()
        return self._config_manager

    def _get_config(self):
        """Get the current configuration."""
        manager = self._get_config_manager()
        if not manager.is_loaded():
            return None
        return manager.get_config()

    # =========================================================================
    # Main Validation Methods
    # =========================================================================

    def validate_all(self) -> list[ValidationResult]:
        """
        Run all validation checks and return results.

        Returns:
            List of ValidationResult objects from all checks
        """
        results = []

        # Check if config is loaded
        config = self._get_config()
        if not config:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component="configuration",
                    message="Configuration not loaded",
                    suggestion="Call load_config() on AIConfigManager before validating",
                )
            )
            return results

        # Run all validation checks
        results.extend(self.validate_api_keys())
        results.extend(self.validate_failover_chains())
        results.extend(self.validate_model_references())
        results.extend(self.validate_circuit_breaker_settings())
        results.extend(self.validate_rate_limits())
        results.extend(self.validate_capability_consistency())
        results.extend(self.validate_global_config())

        # Validate each provider individually
        for provider_id in config.providers:
            results.extend(self.validate_provider_config(provider_id))

        return results

    def validate_provider_config(self, provider_name: str) -> list[ValidationResult]:
        """
        Validate a specific provider's configuration.

        Args:
            provider_name: Name of the provider to validate

        Returns:
            List of ValidationResult objects for this provider
        """
        results = []
        config = self._get_config()

        if not config:
            return [
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component=f"provider.{provider_name}",
                    message="Configuration not loaded",
                )
            ]

        provider = config.get_provider(provider_name)
        if not provider:
            return [
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component=f"provider.{provider_name}",
                    message=f"Provider '{provider_name}' not found in configuration",
                    suggestion="Check provider name or add it to providers.yaml",
                )
            ]

        # Validate API configuration
        if not provider.api.base_url:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component=f"provider.{provider_name}.api",
                    message="Missing base_url in API configuration",
                    suggestion="Add base_url to provider API configuration",
                )
            )

        # Validate API key environment variable
        env_var = provider.api.key_env_var
        if not env_var:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component=f"provider.{provider_name}.api",
                    message="Missing key_env_var in API configuration",
                    suggestion="Specify the environment variable name for the API key",
                )
            )
        elif not os.getenv(env_var):
            results.append(
                ValidationResult(
                    is_valid=True,  # Not fatal, but worth noting
                    severity=ValidationSeverity.WARNING,
                    component=f"provider.{provider_name}.api",
                    message=f"API key environment variable '{env_var}' not set",
                    suggestion=f"Set {env_var} environment variable or disable provider",
                    details={"env_var": env_var},
                )
            )

        # Validate models exist
        if not provider.models:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    component=f"provider.{provider_name}.models",
                    message="No models configured for provider",
                    suggestion="Add model configurations to the provider",
                )
            )
        else:
            # Check for default model
            has_default = any(m.is_default for m in provider.models.values())
            if not has_default:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        component=f"provider.{provider_name}.models",
                        message="No default model marked, first model will be used",
                        suggestion="Mark a model with is_default: true",
                    )
                )

        # Validate timeout settings
        if provider.api.timeout_seconds < 10:
            results.append(
                ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    component=f"provider.{provider_name}.api",
                    message=f"Timeout {provider.api.timeout_seconds}s is very short",
                    suggestion="Consider increasing timeout to at least 30 seconds",
                    details={"timeout": provider.api.timeout_seconds},
                )
            )

        return results

    def validate_api_keys(self) -> list[ValidationResult]:
        """
        Validate all API key environment variables are set.

        Returns:
            List of ValidationResult objects
        """
        results = []
        config = self._get_config()

        if not config:
            return results

        missing_keys = []
        set_keys = []

        for provider_id, provider in config.providers.items():
            if not provider.enabled:
                continue

            env_var = provider.api.key_env_var
            if os.getenv(env_var):
                set_keys.append(provider_id)
            else:
                missing_keys.append((provider_id, env_var))

        if missing_keys:
            for provider_id, env_var in missing_keys:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component="api_keys",
                        message=f"Missing API key for enabled provider '{provider_id}'",
                        suggestion=f"Set environment variable: {env_var}",
                        details={"provider": provider_id, "env_var": env_var},
                    )
                )

        if not set_keys:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component="api_keys",
                    message="No API keys are set for any enabled provider",
                    suggestion="Set at least one provider API key",
                )
            )
        else:
            results.append(
                ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    component="api_keys",
                    message=f"API keys configured for {len(set_keys)} provider(s)",
                    details={"providers_with_keys": set_keys},
                )
            )

        return results

    def validate_failover_chains(self) -> list[ValidationResult]:
        """
        Validate failover chain references are valid.

        Returns:
            List of ValidationResult objects
        """
        results = []
        config = self._get_config()

        if not config:
            return results

        # Validate provider failover chains
        for provider_id, provider in config.providers.items():
            for failover_id in provider.failover_chain:
                # Resolve alias
                resolved_id = config.aliases.get(failover_id, failover_id)
                if resolved_id not in config.providers:
                    results.append(
                        ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            component=f"failover.{provider_id}",
                            message=f"Invalid failover provider '{failover_id}'",
                            suggestion="Use a valid provider ID or alias",
                            details={
                                "provider": provider_id,
                                "invalid_reference": failover_id,
                            },
                        )
                    )

                # Check for self-reference
                if resolved_id == provider_id:
                    results.append(
                        ValidationResult(
                            is_valid=True,
                            severity=ValidationSeverity.WARNING,
                            component=f"failover.{provider_id}",
                            message="Failover chain contains self-reference",
                            suggestion="Remove self from failover chain",
                        )
                    )

                # Check if failover provider is enabled
                failover_provider = config.get_provider(resolved_id)
                if failover_provider and not failover_provider.enabled:
                    results.append(
                        ValidationResult(
                            is_valid=True,
                            severity=ValidationSeverity.WARNING,
                            component=f"failover.{provider_id}",
                            message=f"Failover provider '{resolved_id}' is disabled",
                            suggestion="Enable the provider or remove from chain",
                        )
                    )

        # Validate named failover chains
        for chain_name, chain in config.failover_chains.items():
            for provider_id in chain.providers:
                resolved_id = config.aliases.get(provider_id, provider_id)
                if resolved_id not in config.providers:
                    results.append(
                        ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            component=f"failover_chains.{chain_name}",
                            message=f"Invalid provider '{provider_id}' in chain",
                            suggestion="Use a valid provider ID or alias",
                            details={
                                "chain": chain_name,
                                "invalid_reference": provider_id,
                            },
                        )
                    )

        if not results:
            results.append(
                ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    component="failover_chains",
                    message="All failover chain references are valid",
                )
            )

        return results

    def validate_model_references(self) -> list[ValidationResult]:
        """
        Validate model references exist in provider configs.

        Returns:
            List of ValidationResult objects
        """
        results = []
        config = self._get_config()

        if not config:
            return results

        # Validate global default model
        default_provider = config.get_provider(config.global_config.default_provider)
        if default_provider:
            default_model = config.global_config.default_model
            if default_model and default_model not in default_provider.models:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        component="global.default_model",
                        message=f"Default model '{default_model}' not found in provider",
                        suggestion="Use a model that exists in the default provider",
                        details={
                            "model": default_model,
                            "provider": config.global_config.default_provider,
                            "available_models": list(default_provider.models.keys()),
                        },
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        component="global.default_model",
                        message=f"Default model '{default_model}' found in provider",
                    )
                )

        # Check for duplicate model IDs across providers (potential confusion)
        model_providers: dict[str, list[str]] = {}
        for provider_id, provider in config.providers.items():
            for model_id in provider.models:
                if model_id not in model_providers:
                    model_providers[model_id] = []
                model_providers[model_id].append(provider_id)

        for model_id, providers in model_providers.items():
            if len(providers) > 1:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        component="models",
                        message=f"Model '{model_id}' exists in multiple providers",
                        details={"model": model_id, "providers": providers},
                    )
                )

        return results

    def validate_circuit_breaker_settings(self) -> list[ValidationResult]:
        """
        Validate circuit breaker thresholds are sensible.

        Returns:
            List of ValidationResult objects
        """
        results = []
        config = self._get_config()

        if not config:
            return results

        for provider_id, provider in config.providers.items():
            cb = provider.circuit_breaker

            if not cb.enabled:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component=f"circuit_breaker.{provider_id}",
                        message="Circuit breaker disabled for provider",
                        suggestion="Enable circuit breaker for production resilience",
                    )
                )
                continue

            # Check failure threshold
            if cb.failure_threshold < 3:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component=f"circuit_breaker.{provider_id}",
                        message=f"Low failure threshold ({cb.failure_threshold})",
                        suggestion="Consider increasing to at least 3 to avoid false trips",
                        details={"threshold": cb.failure_threshold},
                    )
                )
            elif cb.failure_threshold > 20:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component=f"circuit_breaker.{provider_id}",
                        message=f"High failure threshold ({cb.failure_threshold})",
                        suggestion="Consider lowering to detect failures faster",
                        details={"threshold": cb.failure_threshold},
                    )
                )

            # Check recovery timeout
            if cb.recovery_timeout_seconds < 30:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component=f"circuit_breaker.{provider_id}",
                        message=f"Short recovery timeout ({cb.recovery_timeout_seconds}s)",
                        suggestion="May cause aggressive retries on degraded services",
                        details={"timeout": cb.recovery_timeout_seconds},
                    )
                )
            elif cb.recovery_timeout_seconds > 300:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        component=f"circuit_breaker.{provider_id}",
                        message=f"Long recovery timeout ({cb.recovery_timeout_seconds}s)",
                        suggestion="Provider will be unavailable for extended periods",
                        details={"timeout": cb.recovery_timeout_seconds},
                    )
                )

        return results

    def validate_rate_limits(self) -> list[ValidationResult]:
        """
        Validate rate limit configurations.

        Returns:
            List of ValidationResult objects
        """
        results = []
        config = self._get_config()

        if not config:
            return results

        for provider_id, provider in config.providers.items():
            rl = provider.rate_limits

            # Check for very low rate limits
            if rl.requests_per_minute and rl.requests_per_minute < 10:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component=f"rate_limits.{provider_id}",
                        message=f"Low request limit ({rl.requests_per_minute}/min)",
                        suggestion="May cause throttling under normal load",
                        details={"rpm": rl.requests_per_minute},
                    )
                )

            # Check for very high rate limits (may exceed API limits)
            if rl.requests_per_minute and rl.requests_per_minute > 1000:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        component=f"rate_limits.{provider_id}",
                        message=f"High request limit ({rl.requests_per_minute}/min)",
                        suggestion="Verify this doesn't exceed provider API limits",
                        details={"rpm": rl.requests_per_minute},
                    )
                )

            # Check token limits
            if rl.tokens_per_minute and rl.tokens_per_minute < 10000:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component=f"rate_limits.{provider_id}",
                        message=f"Low token limit ({rl.tokens_per_minute}/min)",
                        suggestion="May limit throughput for long-form generation",
                        details={"tpm": rl.tokens_per_minute},
                    )
                )

        return results

    def validate_capability_consistency(self) -> list[ValidationResult]:
        """
        Validate capability flags match actual provider features.

        Returns:
            List of ValidationResult objects
        """
        results = []
        config = self._get_config()

        if not config:
            return results

        for provider_id, provider in config.providers.items():
            caps = provider.capabilities

            # Check model-level capabilities vs provider capabilities
            for model_id, model in provider.models.items():
                # Model claims vision but provider doesn't support it
                if model.supports_vision and not caps.supports_vision:
                    results.append(
                        ValidationResult(
                            is_valid=True,
                            severity=ValidationSeverity.WARNING,
                            component=f"capabilities.{provider_id}.{model_id}",
                            message="Model claims vision support, provider doesn't",
                            suggestion="Verify and align capability flags",
                        )
                    )

                # Model claims function calling but provider doesn't support it
                if model.supports_function_calling and not caps.supports_function_calling:
                    results.append(
                        ValidationResult(
                            is_valid=True,
                            severity=ValidationSeverity.WARNING,
                            component=f"capabilities.{provider_id}.{model_id}",
                            message="Model claims function calling, provider doesn't",
                            suggestion="Verify and align capability flags",
                        )
                    )

                # Model claims streaming but provider doesn't support it
                if model.supports_streaming and not caps.supports_streaming:
                    results.append(
                        ValidationResult(
                            is_valid=True,
                            severity=ValidationSeverity.WARNING,
                            component=f"capabilities.{provider_id}.{model_id}",
                            message="Model claims streaming support, provider doesn't",
                            suggestion="Verify and align capability flags",
                        )
                    )

        return results

    def validate_global_config(self) -> list[ValidationResult]:
        """
        Validate global configuration settings.

        Returns:
            List of ValidationResult objects
        """
        results = []
        config = self._get_config()

        if not config:
            return results

        gc = config.global_config

        # Validate default provider exists and is enabled
        default_provider = config.get_provider(gc.default_provider)
        if not default_provider:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component="global.default_provider",
                    message=f"Default provider '{gc.default_provider}' not found",
                    suggestion="Set a valid provider as default",
                    details={"available": list(config.providers.keys())},
                )
            )
        elif not default_provider.enabled:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component="global.default_provider",
                    message=f"Default provider '{gc.default_provider}' is disabled",
                    suggestion="Enable the provider or choose a different default",
                )
            )

        # Validate failover settings
        if gc.failover_enabled and gc.max_failover_attempts < 1:
            results.append(
                ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    component="global.failover",
                    message="Failover enabled but max_attempts is 0",
                    suggestion="Set max_failover_attempts to at least 1",
                )
            )

        # Validate cost tracking
        if gc.cost_tracking.enabled:
            if gc.cost_tracking.daily_budget_usd and gc.cost_tracking.daily_budget_usd < 1:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        component="global.cost_tracking",
                        message=f"Low daily budget: ${gc.cost_tracking.daily_budget_usd}",
                        suggestion="May trigger alerts frequently",
                    )
                )

        return results

    # =========================================================================
    # Report Generation
    # =========================================================================

    def get_validation_report(self) -> ValidationReport:
        """
        Generate a comprehensive validation report.

        Returns:
            ValidationReport with all validation results
        """
        results = self.validate_all()

        error_count = sum(1 for r in results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for r in results if r.severity == ValidationSeverity.INFO)

        overall_valid = error_count == 0

        self._last_validation = ValidationReport(
            timestamp=datetime.utcnow(),
            overall_valid=overall_valid,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            results=results,
        )

        return self._last_validation

    def get_errors(self) -> list[ValidationResult]:
        """Get only error-level validation results."""
        results = self.validate_all()
        return [r for r in results if r.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> list[ValidationResult]:
        """Get only warning-level validation results."""
        results = self.validate_all()
        return [r for r in results if r.severity == ValidationSeverity.WARNING]

    def is_config_valid(self) -> bool:
        """
        Quick check if configuration is valid (no errors).

        Returns:
            True if no error-level validation issues
        """
        return len(self.get_errors()) == 0

    def get_last_report(self) -> ValidationReport | None:
        """Get the last generated validation report."""
        return self._last_validation


# =============================================================================
# Global Instance
# =============================================================================

# Singleton instance
_config_validator: ConfigValidator | None = None


def get_config_validator() -> ConfigValidator:
    """Get the global ConfigValidator instance."""
    global _config_validator
    if _config_validator is None:
        _config_validator = ConfigValidator()
    return _config_validator


# Convenience alias for service registry
config_validator = get_config_validator()
