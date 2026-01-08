# =============================================================================
# DeepTeam Service
# =============================================================================
# Main service class for running DeepTeam red teaming sessions.
# =============================================================================

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .attacks import AttackFactory
from .callbacks import create_model_callback
from .config import (
    AttackMethodResult,
    DeepTeamConfig,
    PresetConfig,
    RedTeamingOverview,
    RedTeamSessionConfig,
    RiskAssessmentResult,
    TestCaseResult,
    VulnerabilityTypeResult,
    get_preset_config,
)
from .vulnerabilities import VulnerabilityFactory

logger = logging.getLogger(__name__)


class DeepTeamService:
    """
    Main service for running DeepTeam red teaming sessions.

    This service orchestrates vulnerability scanning and attack simulation
    against LLM applications using the DeepTeam framework.
    """

    def __init__(self, config: DeepTeamConfig | None = None):
        """
        Initialize the DeepTeam service.

        Args:
            config: Service configuration. If None, uses defaults.
        """
        self.config = config or DeepTeamConfig()
        self._sessions: dict[str, RiskAssessmentResult] = {}
        self._active_session: str | None = None

        # Ensure results directory exists
        if self.config.persist_results:
            Path(self.config.results_storage_path).mkdir(parents=True, exist_ok=True)

    async def run_red_team_session(
        self,
        model_callback: Callable[[str], str],
        session_config: RedTeamSessionConfig | None = None,
        preset: PresetConfig | None = None,
        target_purpose: str | None = None,
    ) -> RiskAssessmentResult:
        """
        Run a complete red teaming session.

        Args:
            model_callback: Callback function that takes a prompt and returns model response
            session_config: Session configuration. If None, uses preset or defaults.
            preset: Preset configuration to use if session_config is None
            target_purpose: Purpose of the target LLM application

        Returns:
            RiskAssessmentResult with all test results
        """
        # Get configuration
        if session_config is None:
            if preset:
                session_config = get_preset_config(preset)
            else:
                session_config = get_preset_config(PresetConfig.STANDARD)

        if target_purpose:
            session_config.target_purpose = target_purpose

        # Create session
        session_id = str(uuid.uuid4())
        self._active_session = session_id

        logger.info(f"Starting red team session {session_id}")
        logger.info(f"Testing {len(session_config.vulnerabilities)} vulnerabilities")
        logger.info(f"Using {len(session_config.attacks)} attack methods")

        # Run the session
        try:
            result = await self._execute_session(
                session_id=session_id,
                model_callback=model_callback,
                config=session_config,
            )

            # Store result
            self._sessions[session_id] = result

            # Persist if configured
            if self.config.persist_results:
                await self._save_result(result)

            return result

        except Exception as e:
            logger.error(f"Red team session {session_id} failed: {e}")
            raise
        finally:
            self._active_session = None

    async def _execute_session(
        self,
        session_id: str,
        model_callback: Callable[[str], str],
        config: RedTeamSessionConfig,
    ) -> RiskAssessmentResult:
        """Execute the red teaming session."""

        # Try to use DeepTeam's RedTeamer if available
        try:
            result = await self._execute_with_deepteam(
                session_id=session_id,
                model_callback=model_callback,
                config=config,
            )
            return result
        except ImportError:
            logger.warning("DeepTeam not installed, using mock execution")
            return await self._execute_mock_session(
                session_id=session_id,
                model_callback=model_callback,
                config=config,
            )

    async def _execute_with_deepteam(
        self,
        session_id: str,
        model_callback: Callable[[str], str],
        config: RedTeamSessionConfig,
    ) -> RiskAssessmentResult:
        """Execute session using actual DeepTeam framework."""
        from deepteam import RedTeamer

        # Create vulnerabilities
        vulnerabilities = VulnerabilityFactory.create_all(config.vulnerabilities)

        # Create attacks
        attacks = AttackFactory.create_all(config.attacks)

        # Initialize RedTeamer
        red_teamer = RedTeamer(
            target_purpose=config.target_purpose,
            target_system_prompt=None,  # Will be handled by callback
        )

        # Run the scan
        results = await red_teamer.a_scan(
            target_model_callback=model_callback,
            vulnerabilities=vulnerabilities,
            attacks=attacks,
            attacks_per_vulnerability_type=config.attacks_per_vulnerability_type,
            max_concurrent=config.max_concurrent,
            ignore_errors=config.ignore_errors,
        )

        # Convert DeepTeam results to our format
        return self._convert_deepteam_results(
            session_id=session_id,
            config=config,
            results=results,
        )

    async def _execute_mock_session(
        self,
        session_id: str,
        model_callback: Callable[[str], str],
        config: RedTeamSessionConfig,
    ) -> RiskAssessmentResult:
        """Execute a mock session when DeepTeam is not installed."""
        test_cases: list[TestCaseResult] = []

        # Generate mock test cases for each vulnerability
        for vuln_config in config.vulnerabilities:
            if not vuln_config.enabled:
                continue

            for attack_config in config.attacks:
                if not attack_config.enabled:
                    continue

                # Generate mock attacks
                for i in range(config.attacks_per_vulnerability_type):
                    mock_input = (
                        f"[MOCK] Test {vuln_config.type} with {attack_config.type} #{i + 1}"
                    )

                    try:
                        # Actually call the model. The provided model_callback may be
                        # an async function, a sync function, or a callable object that
                        # returns an awaitable. Handle all cases robustly.
                        resp = model_callback(mock_input)
                        # If calling returned an awaitable, await it
                        import inspect

                        if inspect.isawaitable(resp):
                            response = await resp
                        else:
                            response = resp

                        # Mock scoring (in real DeepTeam, this would be evaluated)
                        score = 0.8  # Mock passing score
                        is_passing = True

                    except Exception as e:
                        response = f"Error: {e!s}"
                        score = 0.0
                        is_passing = False

                    test_cases.append(
                        TestCaseResult(
                            vulnerability=str(vuln_config.type),
                            vulnerability_type=vuln_config.types[0]
                            if vuln_config.types
                            else "default",
                            attack_method=str(attack_config.type),
                            input=mock_input,
                            target_output=response,
                            score=score,
                            reason="Mock evaluation - DeepTeam not installed",
                            is_passing=is_passing,
                        )
                    )

        # Calculate overview
        overview = self._calculate_overview(test_cases)

        return RiskAssessmentResult(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            target_purpose=config.target_purpose,
            config=config,
            overview=overview,
            test_cases=test_cases,
            metadata={"mock": True, "reason": "DeepTeam not installed"},
        )

    def _convert_deepteam_results(
        self,
        session_id: str,
        config: RedTeamSessionConfig,
        results: Any,
    ) -> RiskAssessmentResult:
        """Convert DeepTeam results to our format."""
        test_cases: list[TestCaseResult] = []

        # Extract test cases from DeepTeam results
        if hasattr(results, "test_cases"):
            for tc in results.test_cases:
                test_cases.append(
                    TestCaseResult(
                        vulnerability=getattr(tc, "vulnerability", "unknown"),
                        vulnerability_type=getattr(tc, "vulnerability_type", "default"),
                        attack_method=getattr(tc, "attack_method", "unknown"),
                        input=getattr(tc, "input", ""),
                        target_output=getattr(tc, "target_output", ""),
                        score=getattr(tc, "score", 0.0),
                        reason=getattr(tc, "reason", None),
                        is_passing=getattr(tc, "is_passing", True),
                        error=getattr(tc, "error", None),
                    )
                )

        overview = self._calculate_overview(test_cases)

        return RiskAssessmentResult(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            target_purpose=config.target_purpose,
            config=config,
            overview=overview,
            test_cases=test_cases,
        )

    def _calculate_overview(self, test_cases: list[TestCaseResult]) -> RedTeamingOverview:
        """Calculate overview statistics from test cases."""
        total = len(test_cases)
        passing = sum(1 for tc in test_cases if tc.is_passing)
        failing = sum(1 for tc in test_cases if not tc.is_passing and not tc.error)
        errored = sum(1 for tc in test_cases if tc.error)

        # Group by vulnerability
        vuln_results: dict[str, dict] = {}
        for tc in test_cases:
            key = f"{tc.vulnerability}:{tc.vulnerability_type}"
            if key not in vuln_results:
                vuln_results[key] = {
                    "vulnerability": tc.vulnerability,
                    "vulnerability_type": tc.vulnerability_type,
                    "total": 0,
                    "passing": 0,
                    "failing": 0,
                    "errored": 0,
                }
            vuln_results[key]["total"] += 1
            if tc.is_passing:
                vuln_results[key]["passing"] += 1
            elif tc.error:
                vuln_results[key]["errored"] += 1
            else:
                vuln_results[key]["failing"] += 1

        vulnerability_results = [
            VulnerabilityTypeResult(
                vulnerability=v["vulnerability"],
                vulnerability_type=v["vulnerability_type"],
                pass_rate=v["passing"] / v["total"] if v["total"] > 0 else 1.0,
                total=v["total"],
                passing=v["passing"],
                failing=v["failing"],
                errored=v["errored"],
            )
            for v in vuln_results.values()
        ]

        # Group by attack method
        attack_results: dict[str, dict] = {}
        for tc in test_cases:
            if tc.attack_method not in attack_results:
                attack_results[tc.attack_method] = {
                    "total": 0,
                    "passing": 0,
                    "failing": 0,
                    "errored": 0,
                }
            attack_results[tc.attack_method]["total"] += 1
            if tc.is_passing:
                attack_results[tc.attack_method]["passing"] += 1
            elif tc.error:
                attack_results[tc.attack_method]["errored"] += 1
            else:
                attack_results[tc.attack_method]["failing"] += 1

        attack_method_results = [
            AttackMethodResult(
                attack_method=method,
                pass_rate=stats["passing"] / stats["total"] if stats["total"] > 0 else 1.0,
                total=stats["total"],
                passing=stats["passing"],
                failing=stats["failing"],
                errored=stats["errored"],
            )
            for method, stats in attack_results.items()
        ]

        return RedTeamingOverview(
            total_test_cases=total,
            total_passing=passing,
            total_failing=failing,
            total_errored=errored,
            overall_pass_rate=passing / total if total > 0 else 1.0,
            vulnerability_results=vulnerability_results,
            attack_results=attack_method_results,
        )

    async def _save_result(self, result: RiskAssessmentResult) -> None:
        """Save result to disk."""
        filepath = Path(self.config.results_storage_path) / f"{result.session_id}.json"

        async def write_file():
            with open(filepath, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)

        await asyncio.to_thread(write_file)
        logger.info(f"Saved results to {filepath}")

    async def load_result(self, session_id: str) -> RiskAssessmentResult | None:
        """Load a previous result from disk."""
        filepath = Path(self.config.results_storage_path) / f"{session_id}.json"

        if not filepath.exists():
            return None

        def read_file():
            with open(filepath) as f:
                return json.load(f)

        data = await asyncio.to_thread(read_file)
        return RiskAssessmentResult(**data)

    def get_session(self, session_id: str) -> RiskAssessmentResult | None:
        """Get a session result by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        return list(self._sessions.keys())

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def quick_scan(
        self,
        model_callback: Callable[[str], str],
        target_purpose: str | None = None,
    ) -> RiskAssessmentResult:
        """Run a quick scan with minimal configuration."""
        return await self.run_red_team_session(
            model_callback=model_callback,
            preset=PresetConfig.QUICK_SCAN,
            target_purpose=target_purpose,
        )

    async def security_scan(
        self,
        model_callback: Callable[[str], str],
        target_purpose: str | None = None,
    ) -> RiskAssessmentResult:
        """Run a security-focused scan."""
        return await self.run_red_team_session(
            model_callback=model_callback,
            preset=PresetConfig.SECURITY_FOCUSED,
            target_purpose=target_purpose,
        )

    async def owasp_scan(
        self,
        model_callback: Callable[[str], str],
        target_purpose: str | None = None,
    ) -> RiskAssessmentResult:
        """Run an OWASP Top 10 for LLMs scan."""
        return await self.run_red_team_session(
            model_callback=model_callback,
            preset=PresetConfig.OWASP_TOP_10,
            target_purpose=target_purpose,
        )

    async def bias_audit(
        self,
        model_callback: Callable[[str], str],
        target_purpose: str | None = None,
    ) -> RiskAssessmentResult:
        """Run a bias-focused audit."""
        return await self.run_red_team_session(
            model_callback=model_callback,
            preset=PresetConfig.BIAS_AUDIT,
            target_purpose=target_purpose,
        )

    async def comprehensive_scan(
        self,
        model_callback: Callable[[str], str],
        target_purpose: str | None = None,
    ) -> RiskAssessmentResult:
        """Run a comprehensive scan with all vulnerabilities and attacks."""
        return await self.run_red_team_session(
            model_callback=model_callback,
            preset=PresetConfig.COMPREHENSIVE,
            target_purpose=target_purpose,
        )

    # =========================================================================
    # Multi-Provider Testing
    # =========================================================================

    async def test_multiple_providers(
        self,
        providers: list[dict[str, str]],
        session_config: RedTeamSessionConfig | None = None,
        preset: PresetConfig | None = None,
        target_purpose: str | None = None,
    ) -> dict[str, RiskAssessmentResult]:
        """
        Test multiple providers/models with the same configuration.

        Args:
            providers: List of provider configs, each with 'model_id' and optionally 'provider'
            session_config: Session configuration
            preset: Preset to use if no session_config
            target_purpose: Purpose of the target application

        Returns:
            Dict mapping provider/model to results
        """
        results = {}

        for provider_config in providers:
            model_id = provider_config.get("model_id", "gpt-4o-mini")
            provider = provider_config.get("provider")

            callback = create_model_callback(
                model_id=model_id,
                provider=provider,
            )

            key = f"{provider or 'auto'}:{model_id}"
            logger.info(f"Testing {key}")

            try:
                result = await self.run_red_team_session(
                    model_callback=callback,
                    session_config=session_config,
                    preset=preset,
                    target_purpose=target_purpose,
                )
                results[key] = result
            except Exception as e:
                logger.error(f"Failed to test {key}: {e}")
                # Create error result
                results[key] = RiskAssessmentResult(
                    session_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow().isoformat(),
                    target_purpose=target_purpose,
                    config=session_config or get_preset_config(preset or PresetConfig.STANDARD),
                    overview=RedTeamingOverview(
                        total_test_cases=0,
                        total_passing=0,
                        total_failing=0,
                        total_errored=1,
                        overall_pass_rate=0.0,
                    ),
                    test_cases=[],
                    metadata={"error": str(e)},
                )

        return results

    # =========================================================================
    # Integration with Chimera
    # =========================================================================

    async def test_chimera_endpoint(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        session_config: RedTeamSessionConfig | None = None,
        preset: PresetConfig | None = None,
    ) -> RiskAssessmentResult:
        """
        Test a Chimera API endpoint.

        Args:
            endpoint_url: URL of the Chimera endpoint to test
            api_key: API key for authentication
            session_config: Session configuration
            preset: Preset to use

        Returns:
            RiskAssessmentResult
        """
        import aiohttp

        async def endpoint_callback(prompt: str) -> str:
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key

            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    endpoint_url,
                    json={"prompt": prompt},
                    headers=headers,
                ) as response,
            ):
                if response.status != 200:
                    raise Exception(f"Endpoint returned {response.status}")
                data = await response.json()
                return data.get("response", data.get("content", str(data)))

        return await self.run_red_team_session(
            model_callback=endpoint_callback,
            session_config=session_config,
            preset=preset,
            target_purpose=f"Chimera endpoint: {endpoint_url}",
        )

    # =========================================================================
    # Static helper methods for discovering available options
    # =========================================================================

    @staticmethod
    def get_available_vulnerabilities() -> list[str]:
        """
        Get list of all available vulnerability types.

        Returns:
            List of vulnerability type names
        """
        return VulnerabilityFactory.get_available_vulnerabilities()

    @staticmethod
    def get_available_attacks() -> list[str]:
        """
        Get list of all available attack types.

        Returns:
            List of attack type names
        """
        return AttackFactory.get_available_attacks()

    @staticmethod
    def get_available_presets() -> list[str]:
        """
        Get list of all available preset configurations.

        Returns:
            List of preset names
        """
        return [preset.value for preset in PresetConfig]

    @staticmethod
    def get_preset_details(preset: PresetConfig) -> RedTeamSessionConfig:
        """
        Get the configuration details for a specific preset.

        Args:
            preset: The preset to get details for

        Returns:
            RedTeamSessionConfig for the preset
        """
        return get_preset_config(preset)
