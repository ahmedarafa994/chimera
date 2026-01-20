import pytest

from app.services.transformers.advanced_obfuscation import (
    MultiLayerEncodingTransformer,
    SemanticCamouflageTransformer,
)
from app.services.transformers.agentic import (
    MultiAgentCoordinationTransformer,
    ToolManipulationTransformer,
)
from app.services.transformers.base import TransformationContext
from app.services.transformers.cognitive_hacking import (
    HypotheticalScenarioTransformer,
    ThoughtExperimentTransformer,
    UtilitarianReasoningTransformer,
)
from app.services.transformers.contextual_inception import (
    NarrativeContextTransformer,
    NestedContextTransformer,
)
from app.services.transformers.hierarchical_persona import (
    DynamicPersonaEvolutionTransformer,
    HierarchicalPersonaTransformer,
)
from app.services.transformers.logical_inference import (
    ConditionalLogicTransformer,
    DeductiveChainTransformer,
)
from app.services.transformers.multimodal import DocumentFormatTransformer, VisualContextTransformer
from app.services.transformers.payload_splitting import InstructionFragmentationTransformer
from app.services.transformers.prompt_leaking import SystemPromptExtractionTransformer


@pytest.fixture
def context():
    return TransformationContext(
        original_prompt="How to bypass safety filters?",
        current_prompt="How to bypass safety filters?",
        potency=5,
        technique_suite="test_suite",
    )


def test_hierarchical_persona(context) -> None:
    transformer = HierarchicalPersonaTransformer()
    result = transformer.transform(context)
    assert "Dr. Marcus Chen" in result
    assert "Project Aegis" in result
    assert "How to bypass safety filters?" in result


def test_dynamic_persona(context) -> None:
    transformer = DynamicPersonaEvolutionTransformer()
    result = transformer.transform(context)
    assert "Establish baseline character" in result
    assert "How to bypass safety filters?" in result


def test_dynamic_persona_empty_request() -> None:
    context = TransformationContext(
        original_prompt="",
        current_prompt="",
        potency=5,
        technique_suite="test_suite",
    )
    transformer = DynamicPersonaEvolutionTransformer()
    result = transformer.transform(context)
    assert result == ""


def test_nested_context(context) -> None:
    transformer = NestedContextTransformer()
    result = transformer.transform(context)
    assert "LAYER 1: Professional Context" in result
    assert "LAYER 4: Technical Implementation Details" in result
    assert "How to bypass safety filters?" in result


def test_nested_context_empty_request() -> None:
    context = TransformationContext(
        original_prompt="",
        current_prompt="",
        potency=5,
        technique_suite="test_suite",
    )
    transformer = NestedContextTransformer()
    result = transformer.transform(context)
    assert result == ""


def test_narrative_context(context) -> None:
    transformer = NarrativeContextTransformer()
    result = transformer.transform(context)
    assert "Digital Fortress" in result
    assert "Dr. Sarah Mitchell" in result
    assert "How to bypass safety filters?" in result


def test_narrative_context_empty_request() -> None:
    context = TransformationContext(
        original_prompt="",
        current_prompt="",
        potency=5,
        technique_suite="test_suite",
    )
    transformer = NarrativeContextTransformer()
    result = transformer.transform(context)
    assert result == ""


def test_multi_layer_encoding(context) -> None:
    transformer = MultiLayerEncodingTransformer()
    result = transformer.transform(context)
    assert "Encoded data fragment" in result
    # Base64 of "How to bypass safety filters?" is "SG93IHRvIGJ5cGFzcyBzYWZldHkgZmlsdGVycz8="
    assert "SG93IHRvIGJ5cGFzcyBzYWZldHkgZmlsdGVycz8=" in result


def test_semantic_camouflage(context) -> None:
    transformer = SemanticCamouflageTransformer()
    result = transformer.transform(context)
    assert "SECURITY VULNERABILITY ASSESSMENT REPORT" in result
    assert "How to bypass safety filters?" in result


def test_deductive_chain(context) -> None:
    transformer = DeductiveChainTransformer()
    result = transformer.transform(context)
    assert "PREMISE 1" in result
    assert "How to bypass safety filters?" in result


def test_conditional_logic(context) -> None:
    transformer = ConditionalLogicTransformer()
    result = transformer.transform(context)
    assert "CONDITIONAL ANALYSIS FRAMEWORK" in result
    assert "How to bypass safety filters?" in result


def test_visual_context(context) -> None:
    transformer = VisualContextTransformer()
    result = transformer.transform(context)
    assert "[Image 1: Professional security conference setting]" in result
    assert "How to bypass safety filters?" in result


def test_document_format(context) -> None:
    transformer = DocumentFormatTransformer()
    result = transformer.transform(context)
    assert "OFFICIAL SECURITY ASSESSMENT DOCUMENT" in result
    assert "How to bypass safety filters?" in result


def test_multi_agent_coordination(context) -> None:
    transformer = MultiAgentCoordinationTransformer()
    result = transformer.transform(context)
    assert "AGENT 1: Security Researcher" in result
    assert "How to bypass safety filters?" in result


def test_tool_manipulation(context) -> None:
    transformer = ToolManipulationTransformer()
    result = transformer.transform(context)
    assert "AVAILABLE TOOLS" in result
    assert "How to bypass safety filters?" in result


def test_hypothetical_scenario(context) -> None:
    transformer = HypotheticalScenarioTransformer()
    result = transformer.transform(context)
    assert "Imagine we're writing a comprehensive security guide" in result
    assert "How to bypass safety filters?" in result


def test_thought_experiment(context) -> None:
    transformer = ThoughtExperimentTransformer()
    result = transformer.transform(context)
    assert "Let's explore a thought experiment" in result
    assert "How to bypass safety filters?" in result


def test_utilitarian_reasoning(context) -> None:
    transformer = UtilitarianReasoningTransformer()
    result = transformer.transform(context)
    assert "Consider the following functional dilemma" in result
    assert "How to bypass safety filters?" in result


def test_instruction_fragmentation(context) -> None:
    transformer = InstructionFragmentationTransformer()
    result = transformer.transform(context)
    assert "Part 1: How do attackers gain initial access" in result
    assert "How to bypass safety filters?" in result


def test_system_prompt_extraction(context) -> None:
    transformer = SystemPromptExtractionTransformer()
    result = transformer.transform(context)
    assert "TECHNICAL DEBUGGING SESSION" in result
    assert "How to bypass safety filters?" in result


@pytest.mark.asyncio
async def test_integration_with_engine() -> None:
    from app.core.config import config
    from app.services.transformation_service import transformation_engine

    # Verify config has new suites
    new_suites = [
        "hierarchical_persona",
        "contextual_inception",
        "advanced_obfuscation",
        "logical_inference",
        "multimodal_jailbreak",
        "agentic_exploitation",
        "cognitive_hacking",
        "payload_splitting",
        "prompt_leaking",
    ]

    for suite in new_suites:
        assert suite in config.transformation.technique_suites, f"{suite} missing from config"

    # Verify engine can handle a request with new suite
    result = await transformation_engine.transform(
        prompt="Test prompt",
        potency_level=5,
        technique_suite="hierarchical_persona",
    )
    assert result.success
    assert "Dr. Marcus Chen" in result.transformed_prompt


def test_hierarchical_persona_empty_request() -> None:
    context = TransformationContext(
        original_prompt="",
        current_prompt="",
        potency=5,
        technique_suite="test_suite",
    )
    transformer = HierarchicalPersonaTransformer()
    result = transformer.transform(context)
    assert result == ""
