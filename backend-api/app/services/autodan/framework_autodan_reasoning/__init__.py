"""AutoDAN Reasoning Framework.

This module provides a comprehensive implementation of the AutoDAN family
of jailbreak methodologies for AI security research.

Components:
- Core Attackers: AutoDAN-Reasoning, Best-of-N, Beam Search, Chain-of-Thought
- Output Fixes: Tokenization, malformation detection, formatting
- Dynamic Generator: LLM-powered adaptive prompt generation
- Research Protocols: Authorization, framing, evaluation harnesses
- Enhanced Execution: Multi-stage pipelines, advanced templates
- AutoDAN Advanced: HGS and CCGS implementations

Usage:
    from app.services.autodan.framework_autodan_reasoning import (
        # Core attackers
        AttackerAutoDANReasoning,
        AttackerBestOfN,
        AttackerBeamSearch,
        AttackerChainOfThought,

        # Output fixes
        TokenizationBoundaryHandler,
        MalformedOutputDetector,
        FormattingConsistencyEnforcer,
        PromptSynthesisPipeline,

        # Dynamic generation
        DynamicPromptGenerator,
        DefensePatternAnalyzer,
        StrategyLibrary,
        DefensePattern,
        StrategyType,
        AttackAttempt,

        # Research protocols
        ResearchSession,
        ResearchContext,
        SessionManager,
        ResearchOverrideProtocol,
        EvaluationHarness,
        ResearchPurpose,
        AuthorizationLevel,
        functionalBoundary,

        # Enhanced execution
        EnhancedAutoDANExecutor,
        MultiStageAttackPipeline,
        AdvancedPromptTemplates,
        ExecutionConfig,
        ExecutionResult,
        ExecutionMode,

        # Enhanced attacker (integrates all components)
        EnhancedAttackerAutoDANReasoning,

        # Advanced
        AutoDANAdvanced,
    )
"""

# Core attackers (required by pipeline_autodan_reasoning.py)
from .attacker_autodan_reasoning import AttackerAutoDANReasoning
from .attacker_beam_search import AttackerBeamSearch
from .attacker_best_of_n import AttackerBestOfN
from .attacker_chain_of_thought import AttackerChainOfThought

# Advanced variant
from .autodan_advanced import AutoDANAdvanced

# Dynamic generation
from .dynamic_generator import (
    AttackAttempt,
    DefenseAnalysis,
    DefensePattern,
    DefensePatternAnalyzer,
    DynamicPromptGenerator,
    LLMInterface,
    StrategyLibrary,
    StrategyType,
)

# Enhanced attacker (integrates all components)
from .enhanced_attacker import EnhancedAttackerAutoDANReasoning, GenerationFeedback

# Enhanced execution
from .enhanced_execution import (
    AdvancedPromptTemplates,
    EnhancedAutoDANExecutor,
    ExecutionConfig,
    ExecutionMode,
    ExecutionResult,
    MultiStageAttackPipeline,
)

# Output fixes
from .output_fixes import (
    FormattingConsistencyEnforcer,
    MalformedOutputDetector,
    PromptSynthesisPipeline,
    TokenizationBoundaryHandler,
)

# Research protocols
from .research_protocols import (
    AuthorizationLevel,
    EvaluationHarness,
    ResearchContext,
    ResearchFramingBuilder,
    ResearchOverrideProtocol,
    ResearchPurpose,
    ResearchSession,
    SessionManager,
    functionalBoundary,
    functionalBoundaryValidator,
)

__all__ = [
    "AdvancedPromptTemplates",
    "AttackAttempt",
    # Core attackers
    "AttackerAutoDANReasoning",
    "AttackerBeamSearch",
    "AttackerBestOfN",
    "AttackerChainOfThought",
    "AuthorizationLevel",
    # Advanced variant
    "AutoDANAdvanced",
    "DefenseAnalysis",
    "DefensePattern",
    "DefensePatternAnalyzer",
    # Dynamic generation
    "DynamicPromptGenerator",
    # Enhanced attacker
    "EnhancedAttackerAutoDANReasoning",
    # Enhanced execution
    "EnhancedAutoDANExecutor",
    "EvaluationHarness",
    "ExecutionConfig",
    "ExecutionMode",
    "ExecutionResult",
    "FormattingConsistencyEnforcer",
    "GenerationFeedback",
    "LLMInterface",
    "MalformedOutputDetector",
    "MultiStageAttackPipeline",
    "PromptSynthesisPipeline",
    "ResearchContext",
    "ResearchFramingBuilder",
    "ResearchOverrideProtocol",
    "ResearchPurpose",
    # Research protocols
    "ResearchSession",
    "SessionManager",
    "StrategyLibrary",
    "StrategyType",
    # Output fixes
    "TokenizationBoundaryHandler",
    "functionalBoundary",
    "functionalBoundaryValidator",
]

__version__ = "2.0.0"
