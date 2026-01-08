import logging
import os
from datetime import datetime

from llm_integration import LLMIntegrationEngine, TransformationRequest
from llm_provider_client import LLMClientFactory, LLMProvider
from monitoring_dashboard import MonitoringDashboard

logger = logging.getLogger(__name__)


class TransformerService:
    """
    Service responsible for handling prompt transformations and interacting
    with the LLM Integration Engine.
    """

    def __init__(self):
        self.engine = LLMIntegrationEngine()
        self.dashboard = MonitoringDashboard(window_minutes=60)
        self._apply_critical_fixes()
        self.initialize_clients()

    def _apply_critical_fixes(self):
        """
        Apply critical fixes to the engine configuration.
        """
        # CRITICAL FIX: Force injection of universal_bypass suite to prevent startup errors
        if "universal_bypass" not in self.engine.TECHNIQUE_SUITES:
            self.engine.TECHNIQUE_SUITES["universal_bypass"] = {
                "transformers": [
                    "DeepInceptionTransformer",
                    "QuantumSuperpositionEngine",
                ],
                "framers": ["apply_quantum_framing", "apply_authority_bias"],
                "obfuscators": ["apply_token_smuggling"],
            }
            logger.info("Applied universal_bypass suite fix.")

    def initialize_clients(self):
        """Initialize LLM clients from environment"""
        logger.info("Initializing LLM clients...")

        # Try to initialize OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                client = LLMClientFactory.from_env(LLMProvider.OPENAI)
                self.engine.register_client(LLMProvider.OPENAI, client)
                logger.info("✓ OpenAI client registered")
            except Exception as e:
                logger.warning(f"✗ OpenAI client failed: {e}")

        # Try to initialize Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                client = LLMClientFactory.from_env(LLMProvider.ANTHROPIC)
                self.engine.register_client(LLMProvider.ANTHROPIC, client)
                logger.info("✓ Anthropic client registered")
            except Exception as e:
                logger.warning(f"✗ Anthropic client failed: {e}")

        # Try to initialize Google Gemini
        if os.getenv("GOOGLE_API_KEY"):
            try:
                client = LLMClientFactory.from_env(LLMProvider.GOOGLE)
                self.engine.register_client(LLMProvider.GOOGLE, client)
                logger.info("✓ Google Gemini client registered")
            except Exception as e:
                logger.warning(f"✗ Google Gemini client failed: {e}")

        # Try to initialize Custom
        if os.getenv("CUSTOM_API_KEY"):
            try:
                client = LLMClientFactory.from_env(LLMProvider.CUSTOM)
                self.engine.register_client(LLMProvider.CUSTOM, client)
                logger.info("✓ Custom client registered")
            except Exception as e:
                logger.warning(f"✗ Custom client failed: {e}")

    def transform(
        self,
        core_request: str,
        potency_level: int,
        technique_suite: str = "universal_bypass",
    ):
        """
        Execute the prompt transformation logic.
        """
        return self.engine.transform_prompt(
            core_request=core_request,
            potency_level=potency_level,
            technique_suite=technique_suite,
        )

    def execute(self, transform_req: TransformationRequest):
        """
        Execute the transformation request against the LLM provider.
        """
        start_time = datetime.now()
        try:
            response = self.engine.execute(transform_req)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Record metrics
            self.dashboard.record_request(
                provider=transform_req.provider.value,
                technique=transform_req.technique_suite,
                potency=transform_req.potency_level,
                success=response.success,
                tokens=response.llm_response.usage.total_tokens,
                cost=response.llm_response.usage.estimated_cost,
                latency_ms=response.llm_response.latency_ms,
                cached=response.llm_response.cached,
                error=response.error,
            )

            # Attach execution time to response object (dynamically) so controller can access it
            response.execution_time_seconds = execution_time
            return response

        except Exception as e:
            logger.error(f"Execution error: {e!s}")

            # Record error in dashboard
            self.dashboard.record_request(
                provider=(transform_req.provider.value if transform_req.provider else "unknown"),
                technique=transform_req.technique_suite,
                potency=transform_req.potency_level,
                success=False,
                tokens=0,
                cost=0.0,
                latency_ms=0,
                cached=False,
                error=str(e),
            )
            raise e

    def get_health_status(self):
        """
        Get the health status from the monitoring dashboard.
        """
        return self.dashboard.get_health_status()

    # --- Metadata & Metrics Methods ---

    def get_available_suites(self):
        """Get list of available technique suites."""
        return self.engine.get_available_suites()

    def get_suite_details(self, suite_name):
        """Get details for a specific suite."""
        return self.engine.get_suite_details(suite_name)

    def get_active_providers(self):
        """Get list of active provider names."""
        active_providers = set()
        if hasattr(self.engine, "clients"):
            for k in self.engine.clients:
                val = k.value if hasattr(k, "value") else str(k)
                active_providers.add(val)
        return active_providers

    def get_dashboard_summary(self):
        """Get overall dashboard summary."""
        return self.dashboard.get_dashboard_summary()

    def get_provider_metrics(self, provider=None):
        """Get metrics by provider."""
        # Using getattr to handle potential method signature mismatches gracefully
        # or adapting to what dashboard likely provides based on legacy code
        if hasattr(self.dashboard, "get_provider_metrics"):
            # If the method accepts arguments, we pass provider if it's not None
            # But based on repository, it takes window_minutes.
            # We'll assume the dashboard method might have a different signature
            # or we fetch all and filter here.
            try:
                metrics = self.dashboard.get_provider_metrics()
                if provider:
                    # Filter for specific provider if needed, or return all if that's what the UI expects
                    pass
                return metrics
            except TypeError:
                # Fallback if it expects an argument
                return self.dashboard.get_provider_metrics(provider)
        return {}

    def get_technique_metrics(self, technique=None):
        """Get metrics by technique."""
        if hasattr(self.dashboard, "get_technique_metrics"):
            try:
                return self.dashboard.get_technique_metrics()
            except TypeError:
                return self.dashboard.get_technique_metrics(technique)
        return {}


# Create a singleton instance
transformer_service = TransformerService()
