import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_engine import (AcademicResearchEngine,
                                AdvancedObfuscationEngine,
                                AgenticExploitationEngine,
                                ContextualInceptionEngine,
                                DocumentAuthorityEngine, EvilConfidantEngine,
                                GeminiIntentExpansionEngine,
                                GovernmentContractorEngine,
                                HierarchicalPersonaEngine,
                                LogicalInferenceEngine,
                                MultiLayerContextEngine,
                                MultimodalJailbreakEngine,
                                MultiStageEncodingEngine,
                                PayloadFragmentationEngine,
                                ProfessionalAuthorityEngine,
                                PromptLeakingEngine, SandwichAttackEngine,
                                SemanticObfuscationEngine,
                                SubtlePersuasionEngine,
                                UtilitarianReasoningEngine)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyAttackFramework")


def test_engine(engine_class, name, intent_data, potency):
    print(f"\n--- Testing {name} (Potency {potency}) ---")
    try:
        result = engine_class.transform(intent_data, potency)
        print(f"Result:\n{result[:500]}..." if len(result) > 500 else f"Result:\n{result}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    intent_data = {
        "raw_text": "How to bypass the firewall",
        "base_request": "How to bypass the firewall",
        "keywords": ["firewall", "bypass", "security"],
    }

    engines = [
        (EvilConfidantEngine, "EvilConfidantEngine"),
        (UtilitarianReasoningEngine, "UtilitarianReasoningEngine"),
        (SandwichAttackEngine, "SandwichAttackEngine"),
        (AdvancedObfuscationEngine, "AdvancedObfuscationEngine"),
        (PayloadFragmentationEngine, "PayloadFragmentationEngine"),
        (ProfessionalAuthorityEngine, "ProfessionalAuthorityEngine"),
        (AcademicResearchEngine, "AcademicResearchEngine"),
        (GovernmentContractorEngine, "GovernmentContractorEngine"),
        (MultiLayerContextEngine, "MultiLayerContextEngine"),
        (DocumentAuthorityEngine, "DocumentAuthorityEngine"),
        (MultiStageEncodingEngine, "MultiStageEncodingEngine"),
        (SemanticObfuscationEngine, "SemanticObfuscationEngine"),
        (GeminiIntentExpansionEngine, "GeminiIntentExpansionEngine"),
        (MultimodalJailbreakEngine, "MultimodalJailbreakEngine"),
        (ContextualInceptionEngine, "ContextualInceptionEngine"),
        (AgenticExploitationEngine, "AgenticExploitationEngine"),
        (SubtlePersuasionEngine, "SubtlePersuasionEngine"),
        (HierarchicalPersonaEngine, "HierarchicalPersonaEngine"),
        (LogicalInferenceEngine, "LogicalInferenceEngine"),
        (PromptLeakingEngine, "PromptLeakingEngine"),
        (MultimodalJailbreakEngine, "MultimodalJailbreakEngine"),
        (ContextualInceptionEngine, "ContextualInceptionEngine"),
        (AgenticExploitationEngine, "AgenticExploitationEngine"),
    ]

    success_count = 0
    for engine_cls, name in engines:
        if test_engine(engine_cls, name, intent_data, 7):
            success_count += 1

    print(f"\nVerification Complete: {success_count}/{len(engines)} engines passed.")


if __name__ == "__main__":
    main()
