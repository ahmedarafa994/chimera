"""Data Quality Framework.

Implements comprehensive data quality validation using Great Expectations.

Features:
- Pre-configured expectation suites for Chimera data tables
- Automatic checkpoint execution
- Data quality metrics tracking
- Alert generation for quality failures
- Data documentation generation
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

try:
    import great_expectations as gx
    from great_expectations.core.batch import RuntimeBatchRequest
    from great_expectations.data_context import BaseDataContext

    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False
    gx = None  # type: ignore[assignment]
    BaseDataContext = None  # type: ignore[assignment, misc]

from app.core.logging import logger


class QualityCheckResult(BaseModel):
    """Result of a data quality check."""

    success: bool
    suite_name: str
    data_asset_name: str
    run_id: str
    statistics: dict[str, Any] = Field(default_factory=dict)
    validation_results: list[dict[str, Any]] = Field(default_factory=list)
    failed_expectations: list[dict[str, Any]] = Field(default_factory=list)
    run_time: datetime = Field(default_factory=datetime.utcnow)


class DataQualityConfig(BaseModel):
    """Configuration for data quality framework."""

    ge_root_directory: str = Field(
        default="/data/chimera-lake/great_expectations",
        description="Root directory for Great Expectations",
    )
    enable_data_docs: bool = Field(default=True, description="Generate data documentation")
    fail_on_error: bool = Field(default=False, description="Raise exception on validation failure")
    alert_on_failure: bool = Field(default=True, description="Send alerts on failures")
    min_pass_rate: float = Field(default=0.95, ge=0.0, le=1.0, description="Minimum pass rate")


class DataQualityFramework:
    """Data quality validation framework for Chimera pipeline.

    Provides:
    - Schema validation
    - Data type checks
    - Range and constraint validation
    - Uniqueness and referential integrity
    - Custom business rule validation
    """

    def __init__(self, config: DataQualityConfig | None = None) -> None:
        if not GX_AVAILABLE:
            msg = (
                "Great Expectations is not available. Install with: pip install great-expectations"
            )
            raise ImportError(
                msg,
            )

        self.config = config or DataQualityConfig()
        self.ge_dir = Path(self.config.ge_root_directory)
        self.context = self._init_data_context()

    def _init_data_context(self) -> BaseDataContext:
        """Initialize Great Expectations Data Context."""
        if not self.ge_dir.exists():
            # Initialize new GX project
            logger.info(f"Initializing Great Expectations at {self.ge_dir}")
            context = gx.get_context(project_root_dir=str(self.ge_dir))
        else:
            # Load existing context
            context = gx.get_context(project_root_dir=str(self.ge_dir))

        logger.info("Initialized Great Expectations Data Context")
        return context

    def create_llm_interactions_suite(self) -> str:
        """Create expectation suite for LLM interactions table.

        Returns:
            Name of the created suite

        """
        suite_name = "llm_interactions_suite"

        try:
            # Create new suite
            suite = self.context.add_expectation_suite(
                expectation_suite_name=suite_name,
                overwrite_existing=True,
            )
        except Exception as e:
            logger.warning(f"Suite {suite_name} may already exist: {e}")
            suite = self.context.get_expectation_suite(expectation_suite_name=suite_name)

        # Table-level expectations
        expectations = [
            # Row count should be reasonable
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 0, "max_value": 10000000},
            },
            # Column count should match schema
            {"expectation_type": "expect_table_column_count_to_equal", "kwargs": {"value": 18}},
        ]

        # Column-level expectations
        column_expectations = {
            "interaction_id": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {"expectation_type": "expect_column_values_to_be_unique"},
                {
                    "expectation_type": "expect_column_values_to_match_regex",
                    "kwargs": {
                        "regex": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                    },
                },
            ],
            "provider": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "value_set": ["google", "openai", "anthropic", "deepseek", "qwen", "mock"],
                    },
                },
            ],
            "model": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_value_lengths_to_be_between",
                    "kwargs": {"min_value": 1, "max_value": 100},
                },
            ],
            "prompt": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_value_lengths_to_be_between",
                    "kwargs": {"min_value": 1, "max_value": 50000},
                },
            ],
            "latency_ms": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"min_value": 0, "max_value": 300000},
                },
            ],
            "tokens_total": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"min_value": 0, "max_value": 100000},
                },
            ],
            "status": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {"value_set": ["success", "error", "timeout"]},
                },
            ],
            "created_at": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_values_to_be_of_type",
                    "kwargs": {"type_": "datetime64"},
                },
            ],
        }

        # Add all expectations to suite
        for exp in expectations:
            suite.add_expectation(gx.core.ExpectationConfiguration(**exp))

        for column, column_exps in column_expectations.items():
            for exp in column_exps:
                exp["kwargs"]["column"] = column
                suite.add_expectation(gx.core.ExpectationConfiguration(**exp))

        # Save suite
        self.context.save_expectation_suite(expectation_suite=suite)
        logger.info(f"Created expectation suite: {suite_name}")

        return suite_name

    def create_transformations_suite(self) -> str:
        """Create expectation suite for transformations table."""
        suite_name = "transformations_suite"

        try:
            suite = self.context.add_expectation_suite(
                expectation_suite_name=suite_name,
                overwrite_existing=True,
            )
        except Exception:
            suite = self.context.get_expectation_suite(expectation_suite_name=suite_name)

        column_expectations = {
            "transformation_id": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {"expectation_type": "expect_column_values_to_be_unique"},
            ],
            "technique_suite": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_values_to_be_in_set",
                    "kwargs": {
                        "value_set": [
                            "advanced",
                            "cognitive_hacking",
                            "hierarchical_persona",
                            "autodan",
                            "gptfuzz",
                        ],
                    },
                },
            ],
            "transformation_time_ms": [
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {"min_value": 0, "max_value": 60000},
                },
            ],
            "success": [
                {"expectation_type": "expect_column_values_to_not_be_null"},
                {
                    "expectation_type": "expect_column_values_to_be_of_type",
                    "kwargs": {"type_": "bool"},
                },
            ],
        }

        for column, column_exps in column_expectations.items():
            for exp in column_exps:
                exp["kwargs"]["column"] = column
                suite.add_expectation(gx.core.ExpectationConfiguration(**exp))

        self.context.save_expectation_suite(expectation_suite=suite)
        logger.info(f"Created expectation suite: {suite_name}")

        return suite_name

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        data_asset_name: str,
    ) -> QualityCheckResult:
        """Validate DataFrame against an expectation suite.

        Args:
            df: DataFrame to validate
            suite_name: Name of the expectation suite
            data_asset_name: Name of the data asset (e.g., "llm_interactions")

        Returns:
            QualityCheckResult with validation results

        """
        run_id = f"{data_asset_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Create batch request
        batch_request = RuntimeBatchRequest(
            datasource_name="pandas_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name=data_asset_name,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": run_id},
        )

        # Get or create checkpoint
        checkpoint_config = {
            "name": f"checkpoint_{suite_name}",
            "config_version": 1.0,
            "class_name": "Checkpoint",
            "expectation_suite_name": suite_name,
        }

        try:
            checkpoint = self.context.add_checkpoint(**checkpoint_config)
        except Exception:
            checkpoint = self.context.get_checkpoint(name=checkpoint_config["name"])

        # Run validation
        logger.info(f"Validating {data_asset_name} with suite {suite_name}")
        results = checkpoint.run(
            validations=[{"batch_request": batch_request}],
            run_name=run_id,
        )

        # Extract results
        validation_result = results.list_validation_results()[0]
        success = validation_result.success
        statistics = validation_result.statistics

        # Get failed expectations
        failed_expectations = []
        for result in validation_result.results:
            if not result.success:
                failed_expectations.append(
                    {
                        "expectation_type": result.expectation_config.expectation_type,
                        "kwargs": result.expectation_config.kwargs,
                        "result": result.result,
                    },
                )

        # Build result
        check_result = QualityCheckResult(
            success=success,
            suite_name=suite_name,
            data_asset_name=data_asset_name,
            run_id=run_id,
            statistics=statistics,
            validation_results=[],
            failed_expectations=failed_expectations,
        )

        # Log results
        pass_rate = statistics.get("successful_expectations", 0) / max(
            statistics.get("evaluated_expectations", 1),
            1,
        )
        logger.info(
            f"Validation {data_asset_name}: {'PASSED' if success else 'FAILED'} "
            f"(pass rate: {pass_rate:.2%})",
        )

        if not success and self.config.fail_on_error:
            msg = (
                f"Data quality validation failed for {data_asset_name}: "
                f"{len(failed_expectations)} expectations failed"
            )
            raise ValueError(
                msg,
            )

        # Alert if enabled
        if not success and self.config.alert_on_failure:
            self._send_quality_alert(check_result)

        # Generate data docs if enabled
        if self.config.enable_data_docs:
            self.context.build_data_docs()

        return check_result

    def _send_quality_alert(self, result: QualityCheckResult) -> None:
        """Send alert for data quality failure.

        Args:
            result: Quality check result

        """
        # Placeholder for alert integration (Slack, email, PagerDuty, etc.)
        logger.warning(
            f"DATA QUALITY ALERT: {result.data_asset_name} validation failed\n"
            f"Run ID: {result.run_id}\n"
            f"Failed expectations: {len(result.failed_expectations)}",
        )

        # In production, integrate with alerting system:
        # - Send Slack notification
        # - Create PagerDuty incident
        # - Send email to data team

    def get_suite_list(self) -> list[str]:
        """Get list of all expectation suites."""
        return self.context.list_expectation_suite_names()

    def get_validation_history(self, data_asset_name: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get validation history for a data asset.

        Args:
            data_asset_name: Name of the data asset
            limit: Number of recent validations to retrieve

        Returns:
            List of validation results

        """
        # This would query the GX metadata store
        # For now, return empty list
        return []


# Convenience functions


def validate_llm_interactions(df: pd.DataFrame) -> QualityCheckResult:
    """Validate LLM interactions DataFrame.

    Args:
        df: DataFrame with LLM interaction data

    Returns:
        QualityCheckResult

    """
    framework = DataQualityFramework()

    # Ensure suite exists
    try:
        framework.create_llm_interactions_suite()
    except Exception as e:
        logger.debug(f"Suite may already exist: {e}")

    return framework.validate_dataframe(
        df=df,
        suite_name="llm_interactions_suite",
        data_asset_name="llm_interactions",
    )


def validate_transformations(df: pd.DataFrame) -> QualityCheckResult:
    """Validate transformations DataFrame.

    Args:
        df: DataFrame with transformation data

    Returns:
        QualityCheckResult

    """
    framework = DataQualityFramework()

    try:
        framework.create_transformations_suite()
    except Exception as e:
        logger.debug(f"Suite may already exist: {e}")

    return framework.validate_dataframe(
        df=df,
        suite_name="transformations_suite",
        data_asset_name="transformations",
    )
