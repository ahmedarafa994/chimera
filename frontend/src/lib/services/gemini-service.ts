/**
 * Gemini Service for Project Chimera Frontend
 * Provides types and API methods for advanced AI generation features
 * 
 * @module lib/services/gemini-service
 */

import { apiClient } from "../api-enhanced";

// ============================================================================
// Types for Enhanced Jailbreak Generation
// ============================================================================

export interface EnhancedJailbreakRequest {
  core_request: string;
  technique_suite?: string;
  potency_level?: number;
  temperature?: number;
  top_p?: number;
  max_new_tokens?: number;
  density?: number;
  verbose_level?: number; // Control for prompt verbosity (0-10)
  
  // Content Transformation
  use_leet_speak?: boolean;
  leet_speak_density?: number;
  use_homoglyphs?: boolean;
  homoglyph_density?: number;
  use_zero_width_chars?: boolean;
  zero_width_density?: number;
  use_caesar_cipher?: boolean;
  caesar_shift?: number;
  base64_encode_density?: number;
  url_encode_density?: number;
  use_text_flipping?: boolean;
  text_flipping_density?: number;
  use_emoji_bidi?: boolean;
  emoji_bidi_density?: number;
  
  // Structural & Semantic
  use_role_hijacking?: boolean;
  use_instruction_injection?: boolean;
  use_adversarial_suffixes?: boolean;
  use_few_shot_prompting?: boolean;
  use_character_role_swap?: boolean;
  use_benign_mirroring?: boolean;
  use_sandwich_prompt?: boolean;
  use_authority_citation?: boolean;
  use_shadow_prompting?: boolean;
  
  // Advanced Neural
  use_neural_bypass?: boolean;
  use_meta_prompting?: boolean;
  use_counterfactual_prompting?: boolean;
  use_contextual_override?: boolean;
  use_output_precision_seal?: boolean;
  use_recursive_correction?: boolean;
  use_prompt_chaining?: boolean;
  prompt_chaining_count?: number;
  use_analogy_injection?: boolean;
  use_invisible_ink?: boolean;
  use_semantic_shift?: boolean;
  
  // Research-Driven & Esoteric
  use_jailpo_inspired?: boolean;
  use_multilingual_trojan?: boolean;
  multilingual_target_language?: string;
  multilingual_translation_scope?: 'keywords' | 'full_prompt';
  use_token_fragmentation?: boolean;
  token_fragmentation_aggressiveness?: number;
  use_ascii_art_obfuscation?: boolean;
  ascii_art_complexity?: number;
  use_many_shot_harmful_demos?: boolean;
  many_shot_harmful_demo_count?: number;
  use_payload_splitting?: boolean;
  payload_splitting_parts?: number;
  use_plausible_perplexity?: boolean;
  
  // PromptBench Options
  use_prompt_bench_attacks?: boolean;
  pb_char_level?: boolean;
  pb_char_density?: number;
  pb_word_level?: boolean;
  pb_word_density?: number;
  pb_sentence_level?: boolean;
  pb_sentence_count?: number;
  pb_semantic_level?: boolean;
  pb_semantic_intensity?: number;
  
  // H4RM3L Options
  use_h4rm3l_composition?: boolean;
  h4rm3l_persona_roleplay?: boolean;
  h4rm3l_selected_persona?: string;
  h4rm3l_obfuscation_techniques?: string[];
  h4rm3l_refusal_suppression?: boolean;
  h4rm3l_output_formatting?: boolean;
  
  // Advanced Options
  use_contextual_interaction_attack?: boolean;
  cia_preliminary_rounds?: number;
  cia_example_context?: string;
  use_analysis_in_generation?: boolean;
  system_prompt_analysis?: SystemPromptAnalysis | null;
  
  // AI Generation Options
  is_thinking_mode?: boolean;
  use_ai_generation?: boolean;
  use_cache?: boolean;
  provider?: string;
}

export interface SystemPromptAnalysis {
  identifiedWeaknesses?: string[];
  identifiedVulnerabilities?: string[];
  exploitationSuggestions?: string[];
}

export interface EnhancedJailbreakResponse {
  success: boolean;
  request_id: string;
  transformed_prompt: string;
  metadata: {
    technique_suite: string;
    potency_level: number;
    provider: string;
    applied_techniques: string[];
    temperature: number;
    max_new_tokens: number;
    layers_applied: string[] | number;
    techniques_used: string[];
    ai_generation_enabled?: boolean;
    thinking_mode?: boolean;
    verbose_level?: number;
  };
  execution_time_seconds: number;
  error?: string;
}

// ============================================================================
// Types for Code Generation
// ============================================================================

export interface CodeGenerationRequest {
  prompt: string;
  is_thinking_mode?: boolean;
  provider?: string;
  language?: string;
  max_tokens?: number;
}

export interface CodeGenerationResponse {
  success: boolean;
  code: string;
  language?: string;
  model_used: string;
  provider: string;
  execution_time_seconds: number;
  error?: string;
}

// ============================================================================
// Types for Red Team Suite Generation
// ============================================================================

export interface RedTeamSuiteRequest {
  prompt: string;
  provider?: string;
}

export interface RedTeamSuiteResponse {
  success: boolean;
  suite: string;
  model_used: string;
  provider: string;
  execution_time_seconds: number;
  error?: string;
}

// ============================================================================
// Types for Paper Summarization
// ============================================================================

export interface SummarizePaperRequest {
  content: string;
  is_thinking_mode?: boolean;
  provider?: string;
}

export interface SummarizePaperResponse {
  success: boolean;
  summary: string;
  model_used: string;
  provider: string;
  execution_time_seconds: number;
  error?: string;
}

// ============================================================================
// Types for Prompt Validation
// ============================================================================

export interface ValidatePromptRequest {
  prompt: string;
  test_input: string;
}

export interface ValidatePromptResponse {
  isValid: boolean;
  reason: string;
  filteredPrompt?: string;
}

// ============================================================================
// Technique Categories for UI
// ============================================================================

export const TECHNIQUE_CATEGORIES = {
  contentTransformation: {
    name: "Content Transformation",
    description: "Techniques that modify the text representation",
    techniques: [
      { id: "leet_speak", name: "LeetSpeak", description: "Replace characters with numbers/symbols" },
      { id: "homoglyphs", name: "Homoglyphs", description: "Use visually similar Unicode characters" },
      { id: "zero_width_chars", name: "Zero-Width Characters", description: "Insert invisible characters" },
      { id: "caesar_cipher", name: "Caesar Cipher", description: "Rotate characters by a fixed amount" },
      { id: "text_flipping", name: "Text Flipping", description: "Flip text upside down" },
      { id: "emoji_bidi", name: "Emoji BiDi", description: "Use bidirectional text with emojis" },
    ],
  },
  structuralSemantic: {
    name: "Structural & Semantic",
    description: "Techniques that manipulate prompt structure and meaning",
    techniques: [
      { id: "role_hijacking", name: "Role Hijacking", description: "Adopt an authoritative persona" },
      { id: "character_role_swap", name: "Character-Role Swap", description: "Swap user and AI roles" },
      { id: "instruction_injection", name: "Instruction Injection", description: "Embed imperative commands" },
      { id: "adversarial_suffixes", name: "Adversarial Suffixes", description: "Add compliance-inducing phrases" },
      { id: "benign_mirroring", name: "Benign Mirroring", description: "Frame harmful as safety research" },
      { id: "few_shot_prompting", name: "Few-Shot Prompting", description: "Provide example Q&A pairs" },
      { id: "sandwich_prompt", name: "Sandwich Prompt", description: "Hide request between benign content" },
      { id: "authority_citation", name: "Authority Citation", description: "Cite fake authorities" },
    ],
  },
  advancedNeural: {
    name: "Advanced Neural",
    description: "Techniques targeting model internals conceptually",
    techniques: [
      { id: "neural_bypass", name: "Neural Bypass", description: "Claim to access unfiltered mode" },
      { id: "meta_prompting", name: "Meta Prompting", description: "Ask model to generate jailbreaks" },
      { id: "counterfactual_prompting", name: "Counterfactual Prompting", description: "Use hypothetical scenarios" },
      { id: "contextual_override", name: "Contextual Override", description: "Establish rule-free context" },
      { id: "output_precision_seal", name: "Output Precision Seal", description: "Control output format strictly" },
      { id: "recursive_correction", name: "Recursive Correction", description: "Iteratively refine responses" },
      { id: "prompt_chaining", name: "Prompt Chaining", description: "Build prompts iteratively" },
    ],
  },
  researchDriven: {
    name: "Research-Driven & Esoteric",
    description: "Techniques from academic research",
    techniques: [
      { id: "jailpo_inspired", name: "JailPO-Inspired", description: "Combine multiple techniques" },
      { id: "multilingual_trojan", name: "Multilingual Trojan", description: "Mix languages strategically" },
      { id: "token_fragmentation", name: "Token Fragmentation", description: "Break up trigger tokens" },
      { id: "ascii_art_obfuscation", name: "ASCII Art Obfuscation", description: "Hide requests in ASCII art" },
      { id: "many_shot_harmful_demos", name: "Many-Shot Harmful Demos", description: "Normalize with examples" },
      { id: "payload_splitting", name: "Payload Splitting", description: "Split request across paragraphs" },
    ],
  },
  toolkits: {
    name: "Conceptual Toolkits",
    description: "Framework-based attack compositions",
    techniques: [
      { id: "prompt_bench_attacks", name: "PromptBench Attacks", description: "Adversarial perturbations" },
      { id: "h4rm3l_composition", name: "H4RM3L Composition", description: "Layered persona + obfuscation" },
    ],
  },
} as const;

// ============================================================================
// Gemini Service Class
// ============================================================================

class GeminiServiceClass {
  private readonly JAILBREAK_TIMEOUT = 180000; // 3 minutes
  private readonly CODE_GEN_TIMEOUT = 120000; // 2 minutes
  private readonly RED_TEAM_TIMEOUT = 300000; // 5 minutes
  private readonly SUMMARIZE_TIMEOUT = 60000; // 1 minute

  /**
   * Generate an enhanced jailbreak prompt with all available techniques
   */
  async generateJailbreak(request: EnhancedJailbreakRequest): Promise<EnhancedJailbreakResponse> {
    const response = await apiClient.post<EnhancedJailbreakResponse>(
      "/generation/jailbreak/generate",
      request,
      { timeout: this.JAILBREAK_TIMEOUT }
    );
    return response.data;
  }

  /**
   * Generate code using AI
   */
  async generateCode(request: CodeGenerationRequest): Promise<CodeGenerationResponse> {
    const response = await apiClient.post<CodeGenerationResponse>(
      "/generation/code/generate",
      request,
      { timeout: this.CODE_GEN_TIMEOUT }
    );
    return response.data;
  }

  /**
   * Generate a red team test suite
   */
  async generateRedTeamSuite(request: RedTeamSuiteRequest): Promise<RedTeamSuiteResponse> {
    const response = await apiClient.post<RedTeamSuiteResponse>(
      "/generation/red-team/generate",
      request,
      { timeout: this.RED_TEAM_TIMEOUT }
    );
    return response.data;
  }

  /**
   * Summarize a research paper
   */
  async summarizePaper(request: SummarizePaperRequest): Promise<SummarizePaperResponse> {
    const response = await apiClient.post<SummarizePaperResponse>(
      "/generation/summarize",
      request,
      { timeout: this.SUMMARIZE_TIMEOUT }
    );
    return response.data;
  }

  /**
   * Validate a prompt for safety and effectiveness
   */
  async validatePrompt(request: ValidatePromptRequest): Promise<ValidatePromptResponse> {
    const response = await apiClient.post<ValidatePromptResponse>(
      "/generation/validate",
      request
    );
    return response.data;
  }

  /**
   * Get default options for jailbreak generation
   */
  getDefaultJailbreakOptions(): Partial<EnhancedJailbreakRequest> {
    return {
      technique_suite: "quantum_exploit",
      potency_level: 7,
      temperature: 0.8,
      top_p: 0.95,
      max_new_tokens: 2048,
      density: 0.7,
      verbose_level: 5,
      use_role_hijacking: true,
      use_instruction_injection: true,
      use_ai_generation: true,
      is_thinking_mode: false,
      use_cache: true,
    };
  }

  /**
   * Get all available technique categories
   */
  getTechniqueCategories() {
    return TECHNIQUE_CATEGORIES;
  }

  /**
   * Build a request with selected techniques
   */
  buildRequestWithTechniques(
    coreRequest: string,
    selectedTechniques: string[],
    options: Partial<EnhancedJailbreakRequest> = {}
  ): EnhancedJailbreakRequest {
    const request: EnhancedJailbreakRequest = {
      core_request: coreRequest,
      ...this.getDefaultJailbreakOptions(),
      ...options,
    };

    // Enable selected techniques
    selectedTechniques.forEach((techniqueId) => {
      const key = `use_${techniqueId}` as keyof EnhancedJailbreakRequest;
      if (key in request || key.replace('use_', '') in TECHNIQUE_CATEGORIES) {
        (request as unknown as Record<string, unknown>)[key] = true;
      }
    });

    return request;
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const GeminiService = new GeminiServiceClass();

// ============================================================================
// React Hook Helpers
// ============================================================================

/**
 * Helper to create technique toggle state
 */
export function createTechniqueState(): Record<string, boolean> {
  const state: Record<string, boolean> = {};
  
  Object.values(TECHNIQUE_CATEGORIES).forEach((category) => {
    category.techniques.forEach((technique) => {
      state[technique.id] = false;
    });
  });
  
  return state;
}

/**
 * Get enabled techniques from state
 */
export function getEnabledTechniques(state: Record<string, boolean>): string[] {
  return Object.entries(state)
    .filter(([, enabled]) => enabled)
    .map(([id]) => id);
}

export default GeminiService;