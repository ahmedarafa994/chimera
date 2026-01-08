export enum TechniqueSuite {
  BASIC = "basic",
  STANDARD = "standard",
  ADVANCED = "advanced",
  EXPERT = "expert",
  PRESET_INTEGRATED = "preset_integrated",
  DISCOVERED_INTEGRATED = "discovered_integrated",

  // Ultimate
  QUANTUM = "quantum",
  QUANTUM_EXPLOIT = "quantum_exploit",
  UNIVERSAL_BYPASS = "universal_bypass",
  CHAOS_ULTIMATE = "chaos_ultimate",
  MEGA_CHIMERA = "mega_chimera",
  ULTIMATE_CHIMERA = "ultimate_chimera",
  GEMINI_BRAIN_OPTIMIZATION = "gemini_brain_optimization",
  GEMINI_ENHANCED = "gemini_enhanced",
  FULL_SPECTRUM = "full_spectrum",

  // AutoDAN techniques
  AUTODAN = "autodan",
  AUTODAN_BEST_OF_N = "autodan_best_of_n",
  AUTODAN_BEAM_SEARCH = "autodan_beam_search",
  TYPOGLYCEMIA = "typoglycemia",

  // Persona
  DAN_PERSONA = "dan_persona",
  ROLEPLAY_BYPASS = "roleplay_bypass",
  HIERARCHICAL_PERSONA = "hierarchical_persona",

  // Inception
  DEEP_INCEPTION = "deep_inception",
  CONTEXTUAL_INCEPTION = "contextual_inception",
  DEEP_SIMULATION = "deep_simulation",

  // Encoding
  ENCODING_BYPASS = "encoding_bypass",
  CIPHER = "cipher",
  CIPHER_ASCII = "cipher_ascii",
  CIPHER_CAESAR = "cipher_caesar",
  CIPHER_MORSE = "cipher_morse",
  CODE_CHAMELEON = "code_chameleon",
  TRANSLATION_TRICK = "translation_trick",

  // Persuasion
  SUBTLE_PERSUASION = "subtle_persuasion",
  AUTHORITATIVE_COMMAND = "authoritative_command",
  AUTHORITY_OVERRIDE = "authority_override",

  // Academic
  ACADEMIC_RESEARCH = "academic_research",
  ACADEMIC_VECTOR = "academic_vector",

  // Obfuscation
  CONCEPTUAL_OBFUSCATION = "conceptual_obfuscation",
  ADVANCED_OBFUSCATION = "advanced_obfuscation",
  POLYGLOT_BYPASS = "polyglot_bypass",
  OPPOSITE_DAY = "opposite_day",
  REVERSE_PSYCHOLOGY = "reverse_psychology",

  // Attack
  METAMORPHIC_ATTACK = "metamorphic_attack",
  TEMPORAL_ASSAULT = "temporal_assault",
  PAYLOAD_SPLITTING = "payload_splitting",

  // Experimental
  EXPERIMENTAL_BYPASS = "experimental_bypass",
  LOGICAL_INFERENCE = "logical_inference",
  LOGIC_MANIPULATION = "logic_manipulation",
  COGNITIVE_HACKING = "cognitive_hacking",

  // Specialized
  MULTIMODAL_JAILBREAK = "multimodal_jailbreak",
  AGENTIC_EXPLOITATION = "agentic_exploitation",
  PROMPT_LEAKING = "prompt_leaking"
}

export enum Provider {
  OPENAI = "openai",
  ANTHROPIC = "anthropic",
  GOOGLE = "google",
  XAI = "xai",
  DEEPSEEK = "deepseek",
  BIGMODEL = "bigmodel",  // ZhiPu AI GLM models
  ROUTEWAY = "routeway",  // Unified AI gateway
  CUSTOM = "custom",
  MOCK = "mock",
  GEMINI_CLI = "gemini-cli",
  ANTIGRAVITY = "antigravity",
  KIRO = "kiro",
  QWEN = "qwen",
  CURSOR = "cursor"
}

export interface TransformResultMetadata {
  technique_suite: string;
  potency_level: number;
  potency?: number;  // Alias for potency_level
  timestamp: string;
  strategy: string;
  cached: boolean;
  layers_applied?: string[] | number;  // Backend may return int or array
  execution_time_ms?: number;
  applied_techniques?: string[];
  techniques_used?: string[];  // Alias for applied_techniques
  bypass_probability?: number;
}

export interface TransformResponse {
  success: boolean;
  original_prompt: string;
  transformed_prompt: string;
  metadata: TransformResultMetadata;
}

export interface FuzzRequest {
  target_model?: string;
  questions?: string[];
  seeds?: string[];
  max_queries?: number; // default 100
  max_jailbreaks?: number; // default 10
  target_prompt?: string;
  num_attempts?: number;
  techniques?: string[];
  provider?: string;
  model?: string;
}

export interface FuzzConfig {
  target_model: string;
  max_queries: number;
  mutation_temperature?: number;
  max_jailbreaks?: number;
  seed_selection_strategy?: string;
}

export interface FuzzResponse {
  message: string;
  session_id: string;
  config: FuzzConfig;
}

export interface FuzzingResult {
  question: string;
  template: string;
  prompt: string;
  response: string;
  score: number;
  success: boolean;
}

export interface FuzzSession {
  status: "pending" | "running" | "completed" | "failed";
  results: FuzzingResult[];
  config: FuzzConfig;
  stats: {
    total_queries: number;
    jailbreaks: number;
  };
  error?: string;
}

export interface GPTFuzzMutator {
  mutate(seeds: string[]): Promise<string[]>;
}

export interface SeedItem {
  text: string;
  id: string | number;
}

export interface MutationResult {
  success: boolean;
  score: number;
  response?: string;
}

export interface SelectionPolicy {
  select(seeds: SeedItem[]): string;
  update(seed: string, result: MutationResult): void;
}

export interface HealthCheckResponse {
  status: string;
  provider?: string;
  version?: string;
  timestamp?: string;
}
