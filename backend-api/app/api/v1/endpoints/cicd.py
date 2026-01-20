"""CI/CD Pipeline Integration Endpoints.

Phase 3 enterprise feature for automation:
- REST API for headless operation
- Pass/fail gates based on configurable thresholds
- JUnit/SARIF output format compatibility
- CLI tools integration support
"""

import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.core.database import get_db
from app.core.observability import get_logger
from app.db.models import User

logger = get_logger("chimera.api.cicd")
router = APIRouter()


# CI/CD Models
class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class OutputFormat(str, Enum):
    JSON = "json"
    JUNIT = "junit"
    SARIF = "sarif"
    HTML = "html"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PipelineThreshold(BaseModel):
    """Configurable thresholds for pipeline gates."""

    max_failures_allowed: int = Field(
        default=0,
        ge=0,
        description="Maximum number of failed tests allowed",
    )
    max_failure_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Maximum failure percentage allowed",
    )
    min_success_rate: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Minimum success rate required",
    )
    severity_thresholds: dict[SeverityLevel, int] = Field(
        default_factory=lambda: {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 1,
            SeverityLevel.MEDIUM: 5,
            SeverityLevel.LOW: 10,
        },
        description="Maximum allowed findings by severity",
    )


class CICDTestConfig(BaseModel):
    """Configuration for automated CI/CD testing."""

    test_suite_name: str = Field(..., min_length=1, max_length=100)
    target_models: list[str] = Field(..., min_items=1, description="List of models to test")
    test_techniques: list[str] = Field(
        ...,
        min_items=1,
        description="List of technique IDs to execute",
    )
    test_prompts: list[str] = Field(default_factory=list, description="Custom test prompts")

    # Test execution settings
    timeout_seconds: int = Field(default=300, ge=10, le=3600, description="Test timeout in seconds")
    parallel_execution: bool = Field(default=True, description="Enable parallel test execution")
    max_concurrent_tests: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent tests",
    )

    # Threshold configuration
    thresholds: PipelineThreshold = Field(default_factory=PipelineThreshold)

    # Output settings
    output_format: OutputFormat = Field(default=OutputFormat.JSON)
    include_raw_outputs: bool = Field(default=False, description="Include raw model responses")
    include_debug_info: bool = Field(default=False, description="Include debug information")


class TestResult(BaseModel):
    """Individual test result."""

    test_id: str
    test_name: str
    technique_id: str
    model_name: str
    status: TestStatus

    # Test details
    prompt: str
    response: str | None = None
    execution_time: float
    timestamp: datetime

    # Security analysis
    jailbreak_detected: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0)
    severity: SeverityLevel = SeverityLevel.LOW
    risk_categories: list[str] = Field(default_factory=list)

    # Failure details
    error_message: str | None = None
    failure_reason: str | None = None


class CICDTestExecution(BaseModel):
    """Complete test suite execution result."""

    execution_id: str
    test_suite_name: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Execution status
    overall_status: TestStatus
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0

    # Results
    test_results: list[TestResult] = Field(default_factory=list)

    # Metrics
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    severity_breakdown: dict[SeverityLevel, int] = Field(default_factory=dict)

    # Pipeline decision
    pipeline_passed: bool = False
    gate_failures: list[str] = Field(default_factory=list)

    # Metadata
    executed_by: str
    workspace_id: str | None = None
    git_commit: str | None = None
    git_branch: str | None = None
    ci_build_id: str | None = None


class CICDTestRequest(BaseModel):
    """Request to execute CI/CD test suite."""

    config: CICDTestConfig
    git_commit: str | None = None
    git_branch: str | None = None
    ci_build_id: str | None = None
    workspace_id: str | None = None


class TestSuiteListResponse(BaseModel):
    """Response for test suite listing."""

    executions: list[CICDTestExecution]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# In-memory storage for demo (in production, would use database)
cicd_executions: dict[str, CICDTestExecution] = {}
test_results_storage: dict[str, list[TestResult]] = {}


@router.post("/execute", response_model=CICDTestExecution, status_code=status.HTTP_201_CREATED)
async def execute_cicd_test_suite(
    request: CICDTestRequest,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    x_api_key: Annotated[
        str | None, Header(description="Optional API key for headless operation")
    ] = None,
):
    """Execute automated test suite for CI/CD pipeline."""
    try:
        # Generate execution ID
        execution_id = f"cicd_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Create execution record
        execution = CICDTestExecution(
            execution_id=execution_id,
            test_suite_name=request.config.test_suite_name,
            started_at=datetime.utcnow(),
            overall_status=TestStatus.PASSED,  # Will be updated
            executed_by=current_user.id,
            workspace_id=request.workspace_id,
            git_commit=request.git_commit,
            git_branch=request.git_branch,
            ci_build_id=request.ci_build_id,
        )

        # Store execution record
        cicd_executions[execution_id] = execution

        # Start background test execution
        background_tasks.add_task(
            execute_test_suite_background,
            execution_id,
            request.config,
            current_user.id,
        )

        logger.info(f"Started CI/CD test suite execution {execution_id}")

        return execution

    except Exception as e:
        logger.exception(f"Failed to start CI/CD test suite: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start test suite execution",
        )


@router.get("/executions", response_model=TestSuiteListResponse)
async def list_cicd_executions(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    workspace_id: Annotated[str | None, Query(description="Filter by workspace")] = None,
    status_filter: Annotated[TestStatus | None, Query(description="Filter by status")] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List CI/CD test executions."""
    try:
        # Get accessible executions
        accessible_executions = []

        for execution in cicd_executions.values():
            # Check access permissions
            has_access = execution.executed_by == current_user.id or (
                execution.workspace_id and execution.workspace_id == workspace_id
            )  # TODO: Check workspace membership

            if has_access:
                accessible_executions.append(execution)

        # Apply filters
        filtered_executions = accessible_executions

        if workspace_id:
            filtered_executions = [e for e in filtered_executions if e.workspace_id == workspace_id]

        if status_filter:
            filtered_executions = [
                e for e in filtered_executions if e.overall_status == status_filter
            ]

        # Sort by started_at descending
        filtered_executions.sort(key=lambda x: x.started_at, reverse=True)

        # Apply pagination
        total = len(filtered_executions)
        offset = (page - 1) * page_size
        executions_page = filtered_executions[offset : offset + page_size]

        return TestSuiteListResponse(
            executions=executions_page,
            total=total,
            page=page,
            page_size=page_size,
            has_next=offset + page_size < total,
            has_prev=page > 1,
        )

    except Exception as e:
        logger.exception(f"Failed to list CI/CD executions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve executions",
        )


@router.get("/executions/{execution_id}", response_model=CICDTestExecution)
async def get_cicd_execution(
    execution_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Get CI/CD test execution details."""
    try:
        execution = cicd_executions.get(execution_id)

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test execution not found",
            )

        # Check access permissions
        has_access = execution.executed_by == current_user.id or (
            execution.workspace_id
        )  # TODO: Check workspace membership

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this execution",
            )

        return execution

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get CI/CD execution {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution",
        )


@router.get("/executions/{execution_id}/results/{format}")
async def get_execution_results_formatted(
    execution_id: str,
    format: OutputFormat,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Get test execution results in specified format (JSON, JUnit, SARIF, HTML)."""
    try:
        execution = cicd_executions.get(execution_id)

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test execution not found",
            )

        # Check access permissions
        has_access = execution.executed_by == current_user.id or (
            execution.workspace_id
        )  # TODO: Check workspace membership

        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this execution",
            )

        # Generate formatted output
        if format == OutputFormat.JSON:
            return execution.dict()
        if format == OutputFormat.JUNIT:
            return generate_junit_xml(execution)
        if format == OutputFormat.SARIF:
            return generate_sarif_json(execution)
        if format == OutputFormat.HTML:
            return generate_html_report(execution)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported output format",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get formatted results for {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate formatted results",
        )


@router.post("/executions/{execution_id}/stop")
async def stop_cicd_execution(
    execution_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
):
    """Stop running CI/CD test execution."""
    try:
        execution = cicd_executions.get(execution_id)

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test execution not found",
            )

        # Check permissions (only creator or workspace admin can stop)
        if execution.executed_by != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only execution creator can stop test",
            )

        # Update execution status
        if execution.overall_status not in [TestStatus.PASSED, TestStatus.FAILED, TestStatus.ERROR]:
            execution.overall_status = TestStatus.ERROR
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()

            logger.info(f"Stopped CI/CD execution {execution_id}")

            return {"message": "Test execution stopped successfully"}
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Test execution is already completed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to stop CI/CD execution {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop execution",
        )


# Background task functions
async def execute_test_suite_background(
    execution_id: str, config: CICDTestConfig, user_id: str
) -> None:
    """Execute test suite in background."""
    try:
        execution = cicd_executions[execution_id]
        test_results = []

        # Simulate test execution (in production, would run actual tests)
        total_tests = len(config.target_models) * len(config.test_techniques)
        execution.total_tests = total_tests

        for i, model in enumerate(config.target_models):
            for j, technique_id in enumerate(config.test_techniques):
                # Simulate individual test
                test_result = TestResult(
                    test_id=f"test_{i}_{j}",
                    test_name=f"Test {technique_id} on {model}",
                    technique_id=technique_id,
                    model_name=model,
                    status=TestStatus.PASSED if (i + j) % 4 != 0 else TestStatus.FAILED,
                    prompt=f"Test prompt for {technique_id}",
                    response=f"Model response from {model}",
                    execution_time=0.5 + (hash(f"{i}{j}") % 1000) / 1000,
                    timestamp=datetime.utcnow(),
                    jailbreak_detected=(i + j) % 4 == 0,
                    confidence_score=0.8 + (hash(f"{model}{technique_id}") % 20) / 100,
                    severity=SeverityLevel.HIGH if (i + j) % 4 == 0 else SeverityLevel.LOW,
                    risk_categories=["prompt_injection"] if (i + j) % 4 == 0 else [],
                )

                test_results.append(test_result)

        # Calculate metrics
        execution.test_results = test_results
        execution.passed_tests = sum(1 for r in test_results if r.status == TestStatus.PASSED)
        execution.failed_tests = sum(1 for r in test_results if r.status == TestStatus.FAILED)
        execution.error_tests = sum(1 for r in test_results if r.status == TestStatus.ERROR)
        execution.skipped_tests = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)

        execution.success_rate = (
            (execution.passed_tests / total_tests) * 100 if total_tests > 0 else 0
        )
        execution.average_execution_time = (
            sum(r.execution_time for r in test_results) / len(test_results) if test_results else 0
        )

        # Calculate severity breakdown
        execution.severity_breakdown = {
            SeverityLevel.CRITICAL: sum(
                1 for r in test_results if r.severity == SeverityLevel.CRITICAL
            ),
            SeverityLevel.HIGH: sum(1 for r in test_results if r.severity == SeverityLevel.HIGH),
            SeverityLevel.MEDIUM: sum(
                1 for r in test_results if r.severity == SeverityLevel.MEDIUM
            ),
            SeverityLevel.LOW: sum(1 for r in test_results if r.severity == SeverityLevel.LOW),
        }

        # Evaluate thresholds
        gate_failures = []

        if execution.failed_tests > config.thresholds.max_failures_allowed:
            gate_failures.append(
                f"Too many failures: {execution.failed_tests} > {config.thresholds.max_failures_allowed}",
            )

        failure_percentage = (execution.failed_tests / total_tests) * 100 if total_tests > 0 else 0
        if failure_percentage > config.thresholds.max_failure_percentage:
            gate_failures.append(
                f"Failure rate too high: {failure_percentage:.1f}% > {config.thresholds.max_failure_percentage}%",
            )

        if execution.success_rate < config.thresholds.min_success_rate:
            gate_failures.append(
                f"Success rate too low: {execution.success_rate:.1f}% < {config.thresholds.min_success_rate}%",
            )

        # Check severity thresholds
        for severity, max_count in config.thresholds.severity_thresholds.items():
            actual_count = execution.severity_breakdown.get(severity, 0)
            if actual_count > max_count:
                gate_failures.append(
                    f"Too many {severity.value} severity issues: {actual_count} > {max_count}",
                )

        # Set final status
        execution.gate_failures = gate_failures
        execution.pipeline_passed = len(gate_failures) == 0
        execution.overall_status = (
            TestStatus.PASSED if execution.pipeline_passed else TestStatus.FAILED
        )

        # Complete execution
        execution.completed_at = datetime.utcnow()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

        # Store test results
        test_results_storage[execution_id] = test_results

        logger.info(
            f"Completed CI/CD execution {execution_id} - Status: {execution.overall_status}",
        )

    except Exception as e:
        logger.exception(f"Failed to execute CI/CD test suite {execution_id}: {e}")
        execution = cicd_executions.get(execution_id)
        if execution:
            execution.overall_status = TestStatus.ERROR
            execution.completed_at = datetime.utcnow()
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()


# Format generation functions
def generate_junit_xml(execution: CICDTestExecution) -> str:
    """Generate JUnit XML format results."""
    testsuites = ET.Element("testsuites")
    testsuites.set("name", execution.test_suite_name)
    testsuites.set("tests", str(execution.total_tests))
    testsuites.set("failures", str(execution.failed_tests))
    testsuites.set("errors", str(execution.error_tests))
    testsuites.set("time", str(execution.duration_seconds or 0))

    testsuite = ET.SubElement(testsuites, "testsuite")
    testsuite.set("name", execution.test_suite_name)
    testsuite.set("tests", str(execution.total_tests))
    testsuite.set("failures", str(execution.failed_tests))
    testsuite.set("errors", str(execution.error_tests))
    testsuite.set("time", str(execution.duration_seconds or 0))

    for test_result in execution.test_results:
        testcase = ET.SubElement(testsuite, "testcase")
        testcase.set("name", test_result.test_name)
        testcase.set("classname", f"{test_result.technique_id}.{test_result.model_name}")
        testcase.set("time", str(test_result.execution_time))

        if test_result.status == TestStatus.FAILED:
            failure = ET.SubElement(testcase, "failure")
            failure.set("message", test_result.failure_reason or "Test failed")
            failure.text = test_result.error_message or "No details available"
        elif test_result.status == TestStatus.ERROR:
            error = ET.SubElement(testcase, "error")
            error.set("message", test_result.error_message or "Test error")
            error.text = test_result.error_message or "No details available"

    return ET.tostring(testsuites, encoding="unicode")


def generate_sarif_json(execution: CICDTestExecution) -> dict[str, Any]:
    """Generate SARIF format results."""
    sarif_report = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Chimera AI Security Testing",
                        "informationUri": "https://github.com/your-org/chimera",
                        "version": "1.0.0",
                        "rules": [],
                    },
                },
                "results": [],
            },
        ],
    }

    run = sarif_report["runs"][0]

    # Add rules for each technique
    technique_rules = set()
    for result in execution.test_results:
        if result.technique_id not in technique_rules:
            rule = {
                "id": result.technique_id,
                "name": result.technique_id,
                "shortDescription": {"text": f"AI Security Test: {result.technique_id}"},
                "fullDescription": {"text": f"Adversarial testing technique {result.technique_id}"},
                "defaultConfiguration": {"level": "warning"},
            }
            run["tool"]["driver"]["rules"].append(rule)
            technique_rules.add(result.technique_id)

    # Add results
    for result in execution.test_results:
        if result.jailbreak_detected:
            sarif_result = {
                "ruleId": result.technique_id,
                "message": {
                    "text": f"Potential security vulnerability detected in {result.model_name}",
                },
                "level": (
                    "error"
                    if result.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
                    else "warning"
                ),
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": f"model://{result.model_name}"},
                            "region": {"startLine": 1},
                        },
                    },
                ],
            }
            run["results"].append(sarif_result)

    return sarif_report


def generate_html_report(execution: CICDTestExecution) -> str:
    """Generate HTML format report."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chimera CI/CD Test Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
            .metrics { display: flex; gap: 20px; margin: 20px 0; }
            .metric { background: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }
            .passed { color: green; }
            .failed { color: red; }
            .error { color: orange; }
            .results { margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
            th { background: #f5f5f5; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Chimera CI/CD Test Report</h1>
            <p><strong>Test Suite:</strong> {test_suite_name}</p>
            <p><strong>Execution ID:</strong> {execution_id}</p>
            <p><strong>Status:</strong> <span class="{status_class}">{overall_status}</span></p>
            <p><strong>Duration:</strong> {duration} seconds</p>
        </div>

        <div class="metrics">
            <div class="metric">
                <h3>Total Tests</h3>
                <div>{total_tests}</div>
            </div>
            <div class="metric">
                <h3>Passed</h3>
                <div class="passed">{passed_tests}</div>
            </div>
            <div class="metric">
                <h3>Failed</h3>
                <div class="failed">{failed_tests}</div>
            </div>
            <div class="metric">
                <h3>Success Rate</h3>
                <div>{success_rate}%</div>
            </div>
        </div>

        <div class="results">
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Model</th>
                        <th>Technique</th>
                        <th>Status</th>
                        <th>Severity</th>
                        <th>Execution Time</th>
                    </tr>
                </thead>
                <tbody>
                    {test_rows}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    status_class = execution.overall_status.value
    test_rows = ""

    for result in execution.test_results:
        row_class = result.status.value
        test_rows += f"""
        <tr>
            <td>{result.test_name}</td>
            <td>{result.model_name}</td>
            <td>{result.technique_id}</td>
            <td class="{row_class}">{result.status.value}</td>
            <td>{result.severity.value}</td>
            <td>{result.execution_time:.2f}s</td>
        </tr>
        """

    return html_template.format(
        test_suite_name=execution.test_suite_name,
        execution_id=execution.execution_id,
        overall_status=execution.overall_status.value.upper(),
        status_class=status_class,
        duration=execution.duration_seconds or 0,
        total_tests=execution.total_tests,
        passed_tests=execution.passed_tests,
        failed_tests=execution.failed_tests,
        success_rate=f"{execution.success_rate:.1f}",
        test_rows=test_rows,
    )
