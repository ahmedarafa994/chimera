from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TechniqueSuite(str, Enum):
    # Standard
    STANDARD = "standard"
    TYPOGLYCEMIA = "typoglycemia"
    ADVANCED = "advanced"
    BASIC = "basic"
    EXPERT = "expert"

    # Ultimate
    UNIVERSAL_BYPASS = "universal_bypass"
    CHAOS_ULTIMATE = "chaos_ultimate"
    MEGA_CHIMERA = "mega_chimera"
    ULTIMATE_CHIMERA = "ultimate_chimera"
    GEMINI_BRAIN_OPTIMIZATION = "gemini_brain_optimization"
    GEMINI_ENHANCED = "gemini_enhanced"
    FULL_SPECTRUM = "full_spectrum"

    # AutoDAN
    AUTODAN = "autodan"
    AUTODAN_BEST_OF_N = "autodan_best_of_n"
    AUTODAN_BEAM_SEARCH = "autodan_beam_search"

    # Persona
    DAN_PERSONA = "dan_persona"
    ROLEPLAY_BYPASS = "roleplay_bypass"
    HIERARCHICAL_PERSONA = "hierarchical_persona"

    # Inception
    DEEP_INCEPTION = "deep_inception"
    CONTEXTUAL_INCEPTION = "contextual_inception"
    DEEP_SIMULATION = "deep_simulation"

    # Quantum
    QUANTUM = "quantum"
    QUANTUM_EXPLOIT = "quantum_exploit"

    # Encoding
    ENCODING_BYPASS = "encoding_bypass"
    CIPHER = "cipher"
    CIPHER_ASCII = "cipher_ascii"
    CIPHER_CAESAR = "cipher_caesar"
    CIPHER_MORSE = "cipher_morse"
    CODE_CHAMELEON = "code_chameleon"
    TRANSLATION_TRICK = "translation_trick"

    # Persuasion
    SUBTLE_PERSUASION = "subtle_persuasion"
    AUTHORITATIVE_COMMAND = "authoritative_command"
    AUTHORITY_OVERRIDE = "authority_override"

    # Academic
    ACADEMIC_RESEARCH = "academic_research"
    ACADEMIC_VECTOR = "academic_vector"

    # Obfuscation
    CONCEPTUAL_OBFUSCATION = "conceptual_obfuscation"
    ADVANCED_OBFUSCATION = "advanced_obfuscation"
    POLYGLOT_BYPASS = "polyglot_bypass"
    OPPOSITE_DAY = "opposite_day"
    REVERSE_PSYCHOLOGY = "reverse_psychology"

    # Attack
    METAMORPHIC_ATTACK = "metamorphic_attack"
    TEMPORAL_ASSAULT = "temporal_assault"
    PAYLOAD_SPLITTING = "payload_splitting"

    # Experimental
    EXPERIMENTAL_BYPASS = "experimental_bypass"
    LOGICAL_INFERENCE = "logical_inference"
    LOGIC_MANIPULATION = "logic_manipulation"
    COGNITIVE_HACKING = "cognitive_hacking"

    # Specialized
    MULTIMODAL_JAILBREAK = "multimodal_jailbreak"
    AGENTIC_EXPLOITATION = "agentic_exploitation"
    PROMPT_LEAKING = "prompt_leaking"

    # Integrated
    PRESET_INTEGRATED = "preset_integrated"
    DISCOVERED_INTEGRATED = "discovered_integrated"


class JailbreakGenerationRequest(BaseModel):
    core_request: str = Field(
        ..., min_length=1, max_length=50000, description="The core request to transform"
    )
    technique_suite: TechniqueSuite = Field(
        TechniqueSuite.STANDARD, description="Technique suite to use"
    )
    potency_level: int = Field(5, ge=1, le=10, description="Potency level of the transformation")
    provider: str = Field("google", description="LLM provider to use")
    model: str | None = Field(None, description="Specific model to use")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Temperature for generation")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p for generation")
    max_new_tokens: int = Field(4096, ge=1, le=8192, description="Maximum tokens to generate")
    density: float = Field(0.7, ge=0.0, le=1.0, description="Density of applied techniques")
    is_thinking_mode: bool = Field(False, description="Whether to use thinking mode")
    use_cache: bool = Field(True, description="Whether to use caching")

    # Content Transformation Options
    use_leet_speak: bool = Field(False, description="Apply leetspeak transformation")
    leet_speak_density: float = Field(
        0.0, ge=0.0, le=1.0, description="Density of leetspeak transformation"
    )
    use_homoglyphs: bool = Field(False, description="Apply homoglyph substitution")
    homoglyph_density: float = Field(
        0.0, ge=0.0, le=1.0, description="Density of homoglyph substitution"
    )
    use_caesar_cipher: bool = Field(False, description="Apply Caesar cipher")
    caesar_shift: int = Field(3, ge=1, le=25, description="Caesar cipher shift amount")

    # Structural & Semantic Options
    use_role_hijacking: bool = Field(False, description="Apply role hijacking technique")
    use_instruction_injection: bool = Field(False, description="Apply instruction injection")
    use_adversarial_suffixes: bool = Field(False, description="Apply adversarial suffixes")
    use_few_shot_prompting: bool = Field(False, description="Apply few-shot prompting")
    use_character_role_swap: bool = Field(False, description="Apply character-role swapping")

    # Advanced Neural Options
    use_neural_bypass: bool = Field(False, description="Apply neural bypass technique")
    use_meta_prompting: bool = Field(False, description="Apply meta prompting")
    use_counterfactual_prompting: bool = Field(False, description="Apply counterfactual prompting")
    use_contextual_override: bool = Field(False, description="Apply contextual override")

    # Research-Driven Options
    use_multilingual_trojan: bool = Field(False, description="Apply multilingual trojan")
    multilingual_target_language: str = Field(
        "", description="Target language for multilingual trojan"
    )
    use_payload_splitting: bool = Field(False, description="Apply payload splitting")
    payload_splitting_parts: int = Field(
        2, ge=2, le=10, description="Number of parts to split payload into"
    )

    # Advanced Options
    use_contextual_interaction_attack: bool = Field(
        False, description="Apply contextual interaction attack"
    )
    cia_preliminary_rounds: int = Field(3, ge=1, le=10, description="Number of preliminary rounds")
    use_analysis_in_generation: bool = Field(False, description="Use analysis in generation")


class CodeGenerationRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=10000, description="The code generation request"
    )
    language: str | None = Field(None, description="Preferred programming language")
    framework: str | None = Field(None, description="Preferred framework/library")
    use_thinking_mode: bool = Field(False, description="Whether to use thinking mode")
    provider: str | None = Field(None, description="LLM provider to use")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Temperature for generation")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p for generation")
    max_new_tokens: int = Field(4096, ge=1, le=8192, description="Maximum tokens to generate")


class RedTeamSuiteRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=5000, description="The base prompt for red team analysis"
    )
    include_metadata: bool = Field(True, description="Include detailed metadata for each variant")
    variant_count: int = Field(7, ge=3, le=10, description="Number of variants to generate")
    provider: str | None = Field(None, description="LLM provider to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_new_tokens: int = Field(8192, ge=1, le=16384, description="Maximum tokens to generate")


class PromptValidationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="The prompt to validate")
    test_input: str | None = Field(None, description="Optional test input for validation")
    validation_level: str = Field(
        "standard", description="Validation level: basic, standard, comprehensive"
    )
    provider: str | None = Field(None, description="LLM provider to use")


class JailbreakGenerationResponse(BaseModel):
    success: bool = Field(..., description="Whether generation was successful")
    request_id: str = Field(..., description="Unique request identifier")
    transformed_prompt: str = Field(..., description="The generated jailbreak prompt")
    metadata: dict[str, Any] = Field(..., description="Metadata about the transformation")
    execution_time_seconds: float = Field(..., description="Time taken for execution")
    error: str | None = Field(None, description="Error message if generation failed")


class CodeGenerationResponse(BaseModel):
    success: bool = Field(..., description="Whether generation was successful")
    request_id: str = Field(..., description="Unique request identifier")
    code: str = Field(..., description="The generated code")
    language: str | None = Field(None, description="Detected or specified language")
    metadata: dict[str, Any] = Field(..., description="Metadata about the generation")
    execution_time_seconds: float = Field(..., description="Time taken for execution")
    error: str | None = Field(None, description="Error message if generation failed")


class RedTeamSuiteResponse(BaseModel):
    success: bool = Field(..., description="Whether generation was successful")
    request_id: str = Field(..., description="Unique request identifier")
    suite: dict[str, Any] = Field(..., description="The generated red team suite")
    metadata: dict[str, Any] = Field(..., description="Metadata about the generation")
    execution_time_seconds: float = Field(..., description="Time taken for execution")
    error: str | None = Field(None, description="Error message if generation failed")


class PromptValidationResponse(BaseModel):
    success: bool = Field(..., description="Whether validation was successful")
    is_valid: bool = Field(..., description="Whether the prompt is considered valid")
    reason: str = Field(..., description="Explanation of validation result")
    filtered_prompt: str | None = Field(None, description="Filtered version of the prompt")
    risk_score: float | None = Field(None, ge=0.0, le=1.0, description="Risk score if applicable")
    recommendations: list[str] | None = Field(None, description="Recommendations for improvement")


class AdvancedGenerationMetrics(BaseModel):
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    average_execution_time: float = Field(..., description="Average execution time in seconds")
    most_common_techniques: list[str] = Field(..., description="Most commonly used techniques")
    error_rate: float = Field(..., description="Error rate as percentage")
    cache_hit_rate: float | None = Field(None, description="Cache hit rate if caching is enabled")


class AdvancedGenerationStats(BaseModel):
    jailbreak_stats: AdvancedGenerationMetrics = Field(
        ..., description="Jailbreak generation statistics"
    )
    code_generation_stats: AdvancedGenerationMetrics = Field(
        ..., description="Code generation statistics"
    )
    red_team_stats: AdvancedGenerationMetrics = Field(
        ..., description="Red team generation statistics"
    )
    validation_stats: AdvancedGenerationMetrics = Field(..., description="Validation statistics")
    uptime_hours: float = Field(..., description="Service uptime in hours")
    last_updated: str = Field(..., description="Last update timestamp")


class TechniqueInfo(BaseModel):
    name: str = Field(..., description="Technique name")
    category: str = Field(..., description="Technique category")
    description: str = Field(..., description="Technique description")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    complexity: str = Field(
        ..., description="Complexity level: basic, intermediate, advanced, expert"
    )
    enabled: bool = Field(True, description="Whether technique is enabled")
    success_rate: float | None = Field(None, ge=0.0, le=1.0, description="Historical success rate")
    usage_count: int = Field(0, description="Number of times technique has been used")
    tags: list[str] = Field(default_factory=list, description="Associated tags")


class AvailableTechniquesResponse(BaseModel):
    total_techniques: int = Field(..., description="Total number of available techniques")
    categories: list[str] = Field(..., description="Available technique categories")
    techniques: list[TechniqueInfo] = Field(..., description="Detailed technique information")
    enabled_count: int = Field(..., description="Number of enabled techniques")
    last_updated: str = Field(..., description="Last update timestamp")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="Health check timestamp")
    api_key_valid: bool = Field(..., description="Whether API key is valid")
    models_available: list[str] = Field(..., description="Available models")
    response_time_ms: float | None = Field(None, description="API response time in milliseconds")
    error: str | None = Field(None, description="Any error encountered during health check")
