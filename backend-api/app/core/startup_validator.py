"""Startup Validator for AI Provider Integration.

Validates configuration at application startup including:
- Configuration loading and parsing
- API key presence
- Provider connectivity tests
- Minimum requirements validation
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from app.core.config_validator import ValidationResult, ValidationSeverity, get_config_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Startup Validation Data Classes
# =============================================================================


@dataclass
class ConnectivityResult:
    """Result of a provider connectivity test."""

    provider_id: str
    reachable: bool
    latency_ms: float
    error: str | None = None
    status_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider_id": self.provider_id,
            "reachable": self.reachable,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "status_code": self.status_code,
        }


@dataclass
class StartupReport:
    """Comprehensive startup validation report."""

    timestamp: datetime
    config_loaded: bool
    config_valid: bool
    critical_errors: list[str]
    warnings: list[str]
    provider_connectivity: dict[str, ConnectivityResult]
    minimum_requirements_met: bool
    validation_results: list[ValidationResult]
    startup_time_ms: float
    ready_to_serve: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "config_loaded": self.config_loaded,
            "config_valid": self.config_valid,
            "critical_errors": self.critical_errors,
            "warnings": self.warnings,
            "provider_connectivity": {
                k: v.to_dict() for k, v in self.provider_connectivity.items()
            },
            "minimum_requirements_met": self.minimum_requirements_met,
            "validation_results": [r.to_dict() for r in self.validation_results],
            "startup_time_ms": self.startup_time_ms,
            "ready_to_serve": self.ready_to_serve,
        }


# =============================================================================
# Startup Validator
# =============================================================================


class StartupValidator:
    """Validates configuration at application startup.

    Performs comprehensive validation including:
    - Configuration file loading
    - API key presence
    - Provider connectivity tests
    - Minimum requirements validation

    Example:
        # In application startup
        is_ready = await StartupValidator.validate_on_startup()
        if not is_ready:
            logger.error("Startup validation failed!")
            sys.exit(1)

    """

    _last_report: StartupReport | None = None

    @classmethod
    async def validate_on_startup(
        cls,
        test_connectivity: bool = True,
        fail_on_warnings: bool = False,
        connectivity_timeout: float = 10.0,
    ) -> bool:
        """Run all startup validation checks.

        Args:
            test_connectivity: Whether to test provider connectivity
            fail_on_warnings: Whether to fail on warning-level issues
            connectivity_timeout: Timeout for connectivity tests

        Returns:
            False if critical errors exist, True otherwise

        """
        start_time = time.perf_counter()
        critical_errors = []
        warnings = []
        config_loaded = False
        config_valid = False
        provider_connectivity: dict[str, ConnectivityResult] = {}
        validation_results: list[ValidationResult] = []

        logger.info("Starting AI provider configuration validation...")

        # Step 1: Load configuration
        try:
            from app.core.ai_config_manager import get_ai_config_manager

            config_manager = get_ai_config_manager()

            if not config_manager.is_loaded():
                await config_manager.load_config()

            config_loaded = True
            logger.info("Configuration loaded successfully")

        except FileNotFoundError as e:
            critical_errors.append(f"Configuration file not found: {e}")
            logger.exception(f"Config file not found: {e}")

        except ValueError as e:
            critical_errors.append(f"Configuration validation failed: {e}")
            logger.exception(f"Config validation error: {e}")

        except Exception as e:
            critical_errors.append(f"Failed to load configuration: {e}")
            logger.exception(f"Config load error: {e}")

        # Step 2: Validate configuration
        if config_loaded:
            try:
                validator = get_config_validator()
                validation_results = validator.validate_all()

                errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
                warns = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]

                for error in errors:
                    critical_errors.append(error.message)
                    logger.error(f"Validation error: {error.message}")

                for warn in warns:
                    warnings.append(warn.message)
                    logger.warning(f"Validation warning: {warn.message}")

                config_valid = len(errors) == 0
                logger.info(
                    f"Configuration validation: {len(errors)} errors, {len(warns)} warnings",
                )

            except Exception as e:
                critical_errors.append(f"Validation failed: {e}")
                logger.exception(f"Validation error: {e}")

        # Step 3: Test provider connectivity
        if config_loaded and test_connectivity:
            try:
                provider_connectivity = await cls.validate_provider_connectivity(
                    timeout=connectivity_timeout,
                )

                unreachable = [
                    pid for pid, result in provider_connectivity.items() if not result.reachable
                ]

                if unreachable:
                    warning_msg = f"Unreachable providers: {unreachable}"
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)

            except Exception as e:
                warnings.append(f"Connectivity test failed: {e}")
                logger.warning(f"Connectivity test error: {e}")

        # Step 4: Validate minimum requirements
        min_req_results = await cls.validate_minimum_requirements()
        min_req_errors = [r for r in min_req_results if r.severity == ValidationSeverity.ERROR]

        for error in min_req_errors:
            if error.message not in critical_errors:
                critical_errors.append(error.message)

        minimum_requirements_met = len(min_req_errors) == 0

        # Calculate elapsed time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Determine if ready to serve
        ready_to_serve = (
            config_loaded
            and config_valid
            and minimum_requirements_met
            and (not fail_on_warnings or len(warnings) == 0)
        )

        # Create report
        cls._last_report = StartupReport(
            timestamp=datetime.utcnow(),
            config_loaded=config_loaded,
            config_valid=config_valid,
            critical_errors=critical_errors,
            warnings=warnings,
            provider_connectivity=provider_connectivity,
            minimum_requirements_met=minimum_requirements_met,
            validation_results=validation_results,
            startup_time_ms=elapsed_ms,
            ready_to_serve=ready_to_serve,
        )

        logger.info(
            f"Startup validation completed in {elapsed_ms:.2f}ms - Ready: {ready_to_serve}",
        )

        return ready_to_serve

    @classmethod
    async def validate_provider_connectivity(
        cls,
        timeout: float = 10.0,
        providers: list[str] | None = None,
    ) -> dict[str, ConnectivityResult]:
        """Test connectivity to each configured provider.

        Args:
            timeout: Request timeout in seconds
            providers: Optional list of provider IDs to test
                      (tests all enabled if not specified)

        Returns:
            Dictionary mapping provider IDs to ConnectivityResult

        """
        results: dict[str, ConnectivityResult] = {}

        try:
            from app.core.ai_config_manager import get_ai_config_manager

            config_manager = get_ai_config_manager()
            if not config_manager.is_loaded():
                return results

            config = config_manager.get_config()

            # Get providers to test
            if providers:
                test_providers = [
                    config.get_provider(p) for p in providers if config.get_provider(p)
                ]
            else:
                test_providers = [p for p in config.providers.values() if p.enabled]

            # Test each provider concurrently
            async def test_provider(provider):
                start = time.perf_counter()
                try:
                    # Check if API key is set
                    api_key = os.getenv(provider.api.key_env_var)
                    if not api_key:
                        return ConnectivityResult(
                            provider_id=provider.provider_id,
                            reachable=False,
                            latency_ms=0.0,
                            error=f"API key not set: {provider.api.key_env_var}",
                        )

                    # Simple HEAD/GET request to base URL
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        # Try a simple request to the base URL
                        # Most APIs return something for their root endpoint
                        response = await client.get(
                            provider.api.base_url,
                            headers={"Authorization": f"Bearer {api_key}"},
                        )

                        latency = (time.perf_counter() - start) * 1000

                        # Any response (even 4xx) means the server is reachable
                        return ConnectivityResult(
                            provider_id=provider.provider_id,
                            reachable=True,
                            latency_ms=latency,
                            status_code=response.status_code,
                        )

                except httpx.TimeoutException:
                    latency = (time.perf_counter() - start) * 1000
                    return ConnectivityResult(
                        provider_id=provider.provider_id,
                        reachable=False,
                        latency_ms=latency,
                        error="Connection timeout",
                    )

                except httpx.ConnectError as e:
                    latency = (time.perf_counter() - start) * 1000
                    return ConnectivityResult(
                        provider_id=provider.provider_id,
                        reachable=False,
                        latency_ms=latency,
                        error=f"Connection error: {e}",
                    )

                except Exception as e:
                    latency = (time.perf_counter() - start) * 1000
                    return ConnectivityResult(
                        provider_id=provider.provider_id,
                        reachable=False,
                        latency_ms=latency,
                        error=str(e),
                    )

            # Run tests concurrently
            tasks = [test_provider(p) for p in test_providers]
            connectivity_results = await asyncio.gather(*tasks)

            for result in connectivity_results:
                results[result.provider_id] = result

        except Exception as e:
            logger.exception(f"Provider connectivity test failed: {e}")

        return results

    @classmethod
    async def validate_minimum_requirements(cls) -> list[ValidationResult]:
        """Validate minimum configuration requirements are met.

        Returns:
            List of ValidationResult for minimum requirements

        """
        results = []

        try:
            from app.core.ai_config_manager import get_ai_config_manager

            config_manager = get_ai_config_manager()

            # Requirement 1: Configuration must be loaded
            if not config_manager.is_loaded():
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        component="startup.config",
                        message="Configuration not loaded",
                        suggestion="Ensure providers.yaml exists and is valid",
                    ),
                )
                return results

            config = config_manager.get_config()

            # Requirement 2: At least one provider must be configured
            if not config.providers:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        component="startup.providers",
                        message="No providers configured",
                        suggestion="Add at least one provider to providers.yaml",
                    ),
                )
                return results

            # Requirement 3: At least one provider must be enabled
            enabled_providers = config.get_enabled_providers()
            if not enabled_providers:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        component="startup.providers",
                        message="No providers enabled",
                        suggestion="Enable at least one provider in configuration",
                    ),
                )
                return results

            # Requirement 4: At least one API key must be configured
            providers_with_keys = []
            for provider in enabled_providers:
                if os.getenv(provider.api.key_env_var):
                    providers_with_keys.append(provider.provider_id)

            if not providers_with_keys:
                results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        component="startup.api_keys",
                        message="No API keys configured for enabled providers",
                        suggestion="Set API key environment variable for at least one provider",
                    ),
                )
                return results

            # Requirement 5: Default provider must have API key
            default_provider_id = config.global_config.default_provider
            default_provider = config.get_provider(default_provider_id)
            if default_provider and not os.getenv(default_provider.api.key_env_var):
                # Not critical, but worth noting
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component="startup.default_provider",
                        message=f"Default provider '{default_provider_id}' has no API key",
                        suggestion="Set API key or change default provider",
                    ),
                )

            # Requirement 6: Default provider must have at least one model
            if default_provider and not default_provider.models:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        component="startup.default_provider",
                        message=f"Default provider '{default_provider_id}' has no models",
                        suggestion="Add model configurations",
                    ),
                )

            # All minimum requirements met
            if not results:
                results.append(
                    ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        component="startup.requirements",
                        message="All minimum requirements met",
                        details={
                            "enabled_providers": len(enabled_providers),
                            "providers_with_keys": len(providers_with_keys),
                            "default_provider": default_provider_id,
                        },
                    ),
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    component="startup.requirements",
                    message=f"Failed to validate requirements: {e}",
                ),
            )

        return results

    @classmethod
    def get_startup_report(cls) -> dict[str, Any] | None:
        """Generate startup validation report.

        Returns:
            Dictionary with startup report or None if not validated

        """
        if cls._last_report is None:
            return None
        return cls._last_report.to_dict()

    @classmethod
    def is_ready(cls) -> bool:
        """Check if the last startup validation passed.

        Returns:
            True if ready to serve, False otherwise

        """
        if cls._last_report is None:
            return False
        return cls._last_report.ready_to_serve

    @classmethod
    def get_critical_errors(cls) -> list[str]:
        """Get list of critical errors from last validation."""
        if cls._last_report is None:
            return ["Startup validation not yet performed"]
        return cls._last_report.critical_errors

    @classmethod
    def get_warnings(cls) -> list[str]:
        """Get list of warnings from last validation."""
        if cls._last_report is None:
            return []
        return cls._last_report.warnings


# =============================================================================
# Convenience Functions
# =============================================================================


async def validate_startup(
    test_connectivity: bool = True,
    fail_on_warnings: bool = False,
) -> bool:
    """Convenience function to validate startup.

    Args:
        test_connectivity: Whether to test provider connectivity
        fail_on_warnings: Whether to fail on warning-level issues

    Returns:
        True if validation passed, False otherwise

    """
    return await StartupValidator.validate_on_startup(
        test_connectivity=test_connectivity,
        fail_on_warnings=fail_on_warnings,
    )


def get_startup_report() -> dict[str, Any] | None:
    """Get the last startup validation report."""
    return StartupValidator.get_startup_report()


def is_startup_ready() -> bool:
    """Check if startup validation passed."""
    return StartupValidator.is_ready()
