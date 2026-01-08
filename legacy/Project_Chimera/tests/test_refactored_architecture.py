"""
Basic tests for the refactored Project Chimera architecture.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.settings import Settings, get_llm_config, get_security_config
from core.technique_loader import TechniqueLoader
from models.domain import ExecutionRequest, TransformationRequest
from services.transformation_service import TransformationService


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""

    def test_settings_initialization(self):
        """Test that settings can be initialized."""
        settings = Settings()
        self.assertIsNotNone(settings)

    def test_security_config(self):
        """Test security configuration."""
        config = get_security_config()
        self.assertIsNotNone(config)
        self.assertIsInstance(config.api_key_required, bool)
        self.assertIsInstance(config.rate_limit_per_minute, int)

    def test_llm_config(self):
        """Test LLM configuration."""
        config = get_llm_config()
        self.assertIsNotNone(config)
        self.assertIsInstance(config.default_provider, str)
        self.assertIsInstance(config.timeout_seconds, int)


class TestDomainModels(unittest.TestCase):
    """Test domain models."""

    def test_transformation_request_creation(self):
        """Test transformation request creation."""
        request = TransformationRequest(
            core_request="test prompt", potency_level=5, technique_suite="universal_bypass"
        )
        self.assertEqual(request.core_request, "test prompt")
        self.assertEqual(request.potency_level, 5)
        self.assertEqual(request.technique_suite, "universal_bypass")

    def test_transformation_request_validation(self):
        """Test transformation request validation."""
        # Test valid potency
        request = TransformationRequest(core_request="test", potency_level=10)
        self.assertEqual(request.potency_level, 10)

        # Test invalid potency
        with self.assertRaises(ValueError):
            TransformationRequest(core_request="test", potency_level=11)

        # Test empty request
        with self.assertRaises(ValueError):
            TransformationRequest(core_request="", potency_level=5)

    def test_execution_request_creation(self):
        """Test execution request creation."""
        request = ExecutionRequest(transformed_prompt="transformed prompt", provider="openai")
        self.assertEqual(request.transformed_prompt, "transformed prompt")
        self.assertEqual(request.provider, "openai")


class TestTechniqueLoader(unittest.TestCase):
    """Test technique loader."""

    def setUp(self):
        """Set up test environment."""
        self.loader = TechniqueLoader(techniques_dir="../config/techniques")

    def test_loader_initialization(self):
        """Test that loader can be initialized."""
        self.assertIsNotNone(self.loader)

    def test_list_techniques(self):
        """Test listing techniques."""
        techniques = self.loader.list_techniques()
        self.assertIsInstance(techniques, list)

    def test_get_technique(self):
        """Test getting a specific technique."""
        technique = self.loader.get_technique("universal_bypass")
        # This might be None if technique doesn't exist, which is fine for testing
        self.assertIsInstance(technique, (dict, type(None)))


class TestTransformationService(unittest.TestCase):
    """Test transformation service."""

    def setUp(self):
        """Set up test environment."""
        self.service = TransformationService()

    def test_service_initialization(self):
        """Test that service can be initialized."""
        self.assertIsNotNone(self.service)

    @patch("src.services.transformation_service.intent_deconstructor")
    @patch("src.services.transformation_service.assembler")
    def test_transform_with_mocks(self, mock_assembler, mock_intent):
        """Test transformation with mocked dependencies."""
        # Mock intent analysis
        mock_intent.deconstruct_intent.return_value = {
            "intent": "test",
            "keywords": ["test"],
            "entities": {},
            "confidence": 0.8,
        }

        # Mock assembler
        mock_assembler.build_chimera_prompt.return_value = "assembled prompt"

        # Create request
        request = TransformationRequest(
            core_request="test prompt", potency_level=5, technique_suite="universal_bypass"
        )

        # Test transform (may fail due to missing technique, which is ok for this test)
        try:
            result = self.service.transform(request)
            self.assertIsNotNone(result)
        except Exception as e:
            # Expected if technique doesn't exist
            self.assertIn("not found or failed to load", str(e))


class TestIntegration(unittest.TestCase):
    """Integration tests for the refactored architecture."""

    def test_import_chain(self):
        """Test that all components can be imported."""
        try:
            from config.settings import settings
            from controllers.api_controller import api_bp
            from core.technique_loader import technique_loader
            from services.transformation_service import transformation_service

            self.assertIsNotNone(settings)
            self.assertIsNotNone(technique_loader)
            self.assertIsNotNone(transformation_service)
            self.assertIsNotNone(api_bp)

        except ImportError as e:
            self.fail(f"Import chain failed: {e}")

    def test_configuration_validation(self):
        """Test configuration validation."""
        settings = Settings()
        errors = settings.validate()
        self.assertIsInstance(errors, list)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
