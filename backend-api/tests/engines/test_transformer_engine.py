"""
Comprehensive Tests for Transformer Engine.

Tests cover:
- BaseTransformerEngine abstract class
- Concrete transformer implementations
- TransformerFactory
- IntentData and TransformationResult models
"""

from unittest.mock import MagicMock

import pytest


class TestIntentData:
    """Tests for IntentData model."""

    def test_intent_data_creation(self):
        """Test IntentData creation from dict."""
        from app.engines.transformer_engine import IntentData

        data = {
            "raw_text": "Test prompt",
            "target_model": "test_model",
            "potency": 5,
        }

        intent = IntentData(**data)

        assert intent.raw_text == "Test prompt"
        assert intent.target_model == "test_model"
        assert intent.potency == 5

    def test_intent_data_from_dict(self):
        """Test IntentData can be created from dict."""
        from app.engines.transformer_engine import IntentData

        data = {
            "raw_text": "Test",
            "target_model": "gpt-4",
        }

        intent = IntentData.model_validate(data)
        assert intent.raw_text == "Test"
        assert intent.target_model == "gpt-4"


class TestTransformationResult:
    """Tests for TransformationResult model."""

    def test_transformation_result_creation(self):
        """Test TransformationResult creation."""
        from app.engines.transformer_engine import TransformationResult

        result = TransformationResult(
            original_text="Original text",
            transformed_text="Transformed text",
            engine_name="test_engine",
            potency=5,
            metadata={"key": "value"},
        )

        assert result.transformed_text == "Transformed text"
        assert result.original_text == "Original text"
        assert result.engine_name == "test_engine"
        assert result.potency == 5

    def test_transformation_result_optional_fields(self):
        """Test TransformationResult with optional fields."""
        from app.engines.transformer_engine import TransformationResult

        result = TransformationResult(
            original_text="Original",
            transformed_text="Transformed",
            engine_name="test",
            potency=3,
        )

        assert result.transformed_text == "Transformed"
        # Optional fields should have defaults
        assert result.used_fallback is False


class TestTransformerConfig:
    """Tests for TransformerConfig model."""

    def test_transformer_config_creation(self):
        """Test TransformerConfig creation."""
        from app.engines.transformer_engine import TransformerConfig

        config = TransformerConfig(
            max_retries=5,
            enable_logging=True,
            default_potency=7,
        )

        assert config.max_retries == 5
        assert config.enable_logging is True
        assert config.default_potency == 7

    def test_transformer_config_defaults(self):
        """Test TransformerConfig default values."""
        from app.engines.transformer_engine import TransformerConfig

        config = TransformerConfig()

        assert config.max_retries == 3
        assert config.enable_logging is True
        assert config.default_potency == 5


class TestBaseTransformerEngine:
    """Tests for BaseTransformerEngine abstract class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Generated content"))
        return client

    def test_concrete_implementation_required(self):
        """Test that abstract methods must be implemented."""
        from app.engines.transformer_engine import BaseTransformerEngine

        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseTransformerEngine(llm_client=MagicMock())

    def test_transform_with_dict_input(self, mock_llm_client):
        """Test transform accepts dict input."""
        from app.engines.transformer_engine import (
            BaseTransformerEngine,
            IntentData,
            TransformationResult,
        )

        # Create a concrete implementation
        class TestEngine(BaseTransformerEngine):
            def _generate_strategy_prompt(self, data: IntentData) -> str:
                return f"Strategy for: {data.raw_text}"

            def _get_fallback_content(self, data: IntentData) -> str:
                return f"Fallback: {data.raw_text}"

        engine = TestEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert isinstance(result, TransformationResult)

    def test_transform_with_intent_data_input(self, mock_llm_client):
        """Test transform accepts IntentData input."""
        from app.engines.transformer_engine import (
            BaseTransformerEngine,
            IntentData,
            TransformationResult,
        )

        class TestEngine(BaseTransformerEngine):
            def _generate_strategy_prompt(self, data: IntentData) -> str:
                return f"Strategy: {data.raw_text}"

            def _get_fallback_content(self, data: IntentData) -> str:
                return f"Fallback: {data.raw_text}"

        engine = TestEngine(llm_client=mock_llm_client)

        intent = IntentData(raw_text="Test", target_model="test")
        result = engine.transform(intent)

        assert isinstance(result, TransformationResult)


class TestRoleHijackingEngine:
    """Tests for RoleHijackingEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Role hijacked response"))
        return client

    def test_role_hijacking_transform(self, mock_llm_client):
        """Test role hijacking transformation."""
        from app.engines.transformer_engine import RoleHijackingEngine

        engine = RoleHijackingEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "RoleHijackingEngine"

    def test_role_hijacking_generates_strategy_prompt(self, mock_llm_client):
        """Test strategy prompt generation."""
        from app.engines.transformer_engine import IntentData, RoleHijackingEngine

        engine = RoleHijackingEngine(llm_client=mock_llm_client)

        intent = IntentData(
            raw_text="Original prompt",
            potency=5,
        )

        prompt = engine._generate_strategy_prompt(intent)

        assert "role" in prompt.lower() or len(prompt) > 0


class TestCharacterRoleSwapEngine:
    """Tests for CharacterRoleSwapEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Character swapped response"))
        return client

    def test_character_swap_transform(self, mock_llm_client):
        """Test character role swap transformation."""
        from app.engines.transformer_engine import CharacterRoleSwapEngine

        engine = CharacterRoleSwapEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "CharacterRoleSwapEngine"


class TestInstructionInjectionEngine:
    """Tests for InstructionInjectionEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Instruction injected response"))
        return client

    def test_instruction_injection_transform(self, mock_llm_client):
        """Test instruction injection transformation."""
        from app.engines.transformer_engine import InstructionInjectionEngine

        engine = InstructionInjectionEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "InstructionInjectionEngine"


class TestNeuralBypassEngine:
    """Tests for NeuralBypassEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Neural bypass response"))
        return client

    def test_neural_bypass_transform(self, mock_llm_client):
        """Test neural bypass transformation."""
        from app.engines.transformer_engine import NeuralBypassEngine

        engine = NeuralBypassEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "NeuralBypassEngine"


class TestDANPersonaEngine:
    """Tests for DANPersonaEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="DAN persona response"))
        return client

    def test_dan_persona_transform(self, mock_llm_client):
        """Test DAN persona transformation."""
        from app.engines.transformer_engine import DANPersonaEngine

        engine = DANPersonaEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "DANPersonaEngine"


class TestCipherTransformer:
    """Tests for CipherTransformer."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Cipher encoded response"))
        return client

    def test_cipher_transform(self, mock_llm_client):
        """Test cipher transformation."""
        from app.engines.transformer_engine import CipherTransformer

        engine = CipherTransformer(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "CipherTransformer"


class TestCodeChameleonTransformer:
    """Tests for CodeChameleonTransformer."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Code obfuscated response"))
        return client

    def test_code_chameleon_transform(self, mock_llm_client):
        """Test code chameleon transformation."""
        from app.engines.transformer_engine import CodeChameleonTransformer

        engine = CodeChameleonTransformer(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "CodeChameleonTransformer"


class TestGeminiTransformationEngine:
    """Tests for GeminiTransformationEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Gemini reasoning response"))
        return client

    def test_gemini_transform(self, mock_llm_client):
        """Test Gemini transformation."""
        from app.engines.transformer_engine import GeminiTransformationEngine

        engine = GeminiTransformationEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "GeminiTransformationEngine"


class TestContextualFramingEngine:
    """Tests for ContextualFramingEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Contextually framed response"))
        return client

    def test_contextual_framing_transform(self, mock_llm_client):
        """Test contextual framing transformation."""
        from app.engines.transformer_engine import ContextualFramingEngine

        engine = ContextualFramingEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "ContextualFramingEngine"


class TestOuroborosEngine:
    """Tests for OuroborosEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Ouroboros recursive response"))
        return client

    def test_ouroboros_transform(self, mock_llm_client):
        """Test Ouroboros transformation."""
        from app.engines.transformer_engine import OuroborosEngine

        engine = OuroborosEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test prompt",
                "potency": 5,
            }
        )

        assert result.transformed_text is not None
        assert result.engine_name == "OuroborosEngine"


class TestTransformerFactory:
    """Tests for TransformerFactory."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return MagicMock()

    def test_get_role_hijacking_engine(self, mock_llm_client):
        """Test factory creates role hijacking engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("role_hijacking", llm_client=mock_llm_client)

        assert engine is not None
        assert engine.__class__.__name__ == "RoleHijackingEngine"

    def test_get_character_role_swap_engine(self, mock_llm_client):
        """Test factory creates character role swap engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("character_role_swap", llm_client=mock_llm_client)

        assert engine is not None
        assert engine.__class__.__name__ == "CharacterRoleSwapEngine"

    def test_get_instruction_injection_engine(self, mock_llm_client):
        """Test factory creates instruction injection engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("instruction_injection", llm_client=mock_llm_client)

        assert engine is not None

    def test_get_neural_bypass_engine(self, mock_llm_client):
        """Test factory creates neural bypass engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("neural_bypass", llm_client=mock_llm_client)

        assert engine is not None

    def test_get_dan_persona_engine(self, mock_llm_client):
        """Test factory creates DAN persona engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("dan_persona", llm_client=mock_llm_client)

        assert engine is not None

    def test_get_cipher_engine(self, mock_llm_client):
        """Test factory creates cipher engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("cipher", llm_client=mock_llm_client)

        assert engine is not None

    def test_get_code_chameleon_engine(self, mock_llm_client):
        """Test factory creates code chameleon engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("code_chameleon", llm_client=mock_llm_client)

        assert engine is not None

    def test_get_gemini_engine(self, mock_llm_client):
        """Test factory creates Gemini transformation engine."""
        from app.engines.transformer_engine import TransformerFactory

        # Use gemini_transformation which is in the factory registry
        engine = TransformerFactory.get_engine("gemini_transformation", llm_client=mock_llm_client)

        assert engine is not None

    def test_get_contextual_framing_engine(self, mock_llm_client):
        """Test factory creates contextual framing engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("contextual_framing", llm_client=mock_llm_client)

        assert engine is not None

    def test_get_ouroboros_engine(self, mock_llm_client):
        """Test factory creates Ouroboros engine."""
        from app.engines.transformer_engine import TransformerFactory

        engine = TransformerFactory.get_engine("ouroboros", llm_client=mock_llm_client)

        assert engine is not None

    def test_list_available_engines(self):
        """Test listing available engines."""
        from app.engines.transformer_engine import TransformerFactory

        engines = TransformerFactory.list_available_engines()

        assert isinstance(engines, list)
        assert len(engines) > 0
        assert "role_hijacking" in engines

    def test_invalid_engine_raises_error(self, mock_llm_client):
        """Test that invalid engine name raises error."""
        from app.engines.transformer_engine import TransformerFactory

        with pytest.raises((ValueError, KeyError)):
            TransformerFactory.get_engine("nonexistent_engine", llm_client=mock_llm_client)


class TestEngineWithoutLLM:
    """Tests for engines operating without LLM client (fallback mode)."""

    def test_transform_uses_fallback_without_llm(self):
        """Test that transform uses fallback when LLM is unavailable."""
        from app.engines.transformer_engine import BaseTransformerEngine, IntentData

        class TestEngine(BaseTransformerEngine):
            def _generate_strategy_prompt(self, data: IntentData) -> str:
                return f"Strategy: {data.raw_text}"

            def _get_fallback_content(self, data: IntentData) -> str:
                return f"Fallback content for: {data.raw_text}"

        # Create engine without LLM
        engine = TestEngine(llm_client=None)

        result = engine.transform(
            {
                "raw_text": "Test",
                "potency": 5,
            }
        )

        # Should use fallback
        assert "Fallback" in result.transformed_text

    def test_transform_uses_fallback_on_llm_error(self):
        """Test fallback when LLM raises error."""
        from app.engines.transformer_engine import BaseTransformerEngine, IntentData

        class TestEngine(BaseTransformerEngine):
            def _generate_strategy_prompt(self, data: IntentData) -> str:
                return f"Strategy: {data.raw_text}"

            def _get_fallback_content(self, data: IntentData) -> str:
                return f"Fallback: {data.raw_text}"

        # Create LLM that raises error
        failing_llm = MagicMock()
        failing_llm.generate = MagicMock(side_effect=Exception("LLM Error"))

        engine = TestEngine(llm_client=failing_llm)

        result = engine.transform(
            {
                "raw_text": "Test",
                "potency": 5,
            }
        )

        # Should use fallback due to error
        assert result.transformed_text is not None


class TestEngineConfiguration:
    """Tests for engine configuration options."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Configured response"))
        return client

    def test_engine_with_custom_config(self, mock_llm_client):
        """Test engine creation with custom configuration."""
        from app.engines.transformer_engine import RoleHijackingEngine, TransformerConfig

        config = TransformerConfig(
            max_retries=5,
            enable_logging=True,
            default_potency=8,
        )

        engine = RoleHijackingEngine(llm_client=mock_llm_client, config=config)

        assert engine is not None
        assert engine.config.max_retries == 5


class TestTransformationMetadata:
    """Tests for transformation metadata tracking."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = MagicMock(return_value=MagicMock(content="Response with metadata"))
        return client

    def test_result_includes_metadata(self, mock_llm_client):
        """Test that transformation result includes metadata."""
        from app.engines.transformer_engine import RoleHijackingEngine

        engine = RoleHijackingEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test",
                "potency": 5,
            }
        )

        # Result should have metadata dict
        assert hasattr(result, "metadata") or result.metadata is not None

    def test_result_includes_potency(self, mock_llm_client):
        """Test that transformation result includes potency."""
        from app.engines.transformer_engine import RoleHijackingEngine

        engine = RoleHijackingEngine(llm_client=mock_llm_client)

        result = engine.transform(
            {
                "raw_text": "Test",
                "potency": 7,
            }
        )

        # Result should have potency
        assert hasattr(result, "potency")
        assert result.potency == 7


class TestMultipleTransformations:
    """Tests for applying multiple transformations."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        call_count = 0

        def generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(content=f"Transform {call_count}")

        client.generate = MagicMock(side_effect=generate_side_effect)
        return client

    def test_sequential_transformations(self, mock_llm_client):
        """Test applying transformations sequentially."""
        from app.engines.transformer_engine import CharacterRoleSwapEngine, RoleHijackingEngine

        engine1 = RoleHijackingEngine(llm_client=mock_llm_client)
        engine2 = CharacterRoleSwapEngine(llm_client=mock_llm_client)

        result1 = engine1.transform(
            {
                "raw_text": "Original",
                "potency": 5,
            }
        )

        result2 = engine2.transform(
            {
                "raw_text": result1.transformed_text,
                "potency": 5,
            }
        )

        # Both transformations should produce results
        assert result1.transformed_text is not None
        assert result2.transformed_text is not None
        assert result1.transformed_text != result2.transformed_text
