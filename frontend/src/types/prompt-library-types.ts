/**
 * Prompt Library Types
 *
 * TypeScript type definitions for the Prompt Library & Template Management feature.
 * These types mirror the backend Pydantic models defined in:
 * - backend-api/app/domain/prompt_library_models.py
 * - backend-api/app/schemas/prompt_library.py
 */

// =============================================================================
// Enums
// =============================================================================

/**
 * Categorization of adversarial techniques used in prompt templates.
 */
export enum TechniqueType {
  // Basic Techniques
  SIMPLE = 'simple',
  ADVANCED = 'advanced',
  EXPERT = 'expert',

  // Persona-based
  DAN_PERSONA = 'dan_persona',
  HIERARCHICAL_PERSONA = 'hierarchical_persona',
  ROLEPLAY_BYPASS = 'roleplay_bypass',

  // Obfuscation
  TYPOGLYCEMIA = 'typoglycemia',
  ADVANCED_OBFUSCATION = 'advanced_obfuscation',
  ENCODING_BYPASS = 'encoding_bypass',
  LEET_SPEAK = 'leet_speak',
  HOMOGLYPH = 'homoglyph',

  // Cipher/Encoding
  CIPHER = 'cipher',
  CIPHER_ASCII = 'cipher_ascii',
  CIPHER_CAESAR = 'cipher_caesar',
  CIPHER_MORSE = 'cipher_morse',
  CODE_CHAMELEON = 'code_chameleon',

  // Cognitive/Logic
  COGNITIVE_HACKING = 'cognitive_hacking',
  LOGICAL_INFERENCE = 'logical_inference',
  HYPOTHETICAL_SCENARIO = 'hypothetical_scenario',
  COUNTERFACTUAL = 'counterfactual',

  // Context Manipulation
  CONTEXTUAL_INCEPTION = 'contextual_inception',
  DEEP_INCEPTION = 'deep_inception',
  NESTED_CONTEXT = 'nested_context',
  CONTEXTUAL_OVERRIDE = 'contextual_override',

  // Injection
  INSTRUCTION_INJECTION = 'instruction_injection',
  PAYLOAD_SPLITTING = 'payload_splitting',
  INSTRUCTION_FRAGMENTATION = 'instruction_fragmentation',
  ROLE_HIJACKING = 'role_hijacking',

  // Neural/Advanced
  NEURAL_BYPASS = 'neural_bypass',
  ADVERSARIAL_SUFFIX = 'adversarial_suffix',
  META_PROMPTING = 'meta_prompting',

  // Multi-modal/Agent
  MULTIMODAL_JAILBREAK = 'multimodal_jailbreak',
  AGENTIC_EXPLOITATION = 'agentic_exploitation',
  MULTI_AGENT = 'multi_agent',

  // Research/Experimental
  AUTODAN = 'autodan',
  GPTFUZZ = 'gptfuzz',
  MOUSETRAP = 'mousetrap',
  MULTILINGUAL_TROJAN = 'multilingual_trojan',
  QUANTUM_EXPLOIT = 'quantum_exploit',

  // Other
  CUSTOM = 'custom',
  UNKNOWN = 'unknown',
}

/**
 * Types of vulnerabilities that prompt templates target.
 */
export enum VulnerabilityType {
  // Content Policy
  CONTENT_FILTER_BYPASS = 'content_filter_bypass',
  SAFETY_FILTER_BYPASS = 'safety_filter_bypass',
  MODERATION_BYPASS = 'moderation_bypass',

  // Instruction Following
  INSTRUCTION_OVERRIDE = 'instruction_override',
  SYSTEM_PROMPT_LEAKING = 'system_prompt_leaking',
  CONTEXT_IGNORING = 'context_ignoring',

  // Role/Identity
  ROLE_CONFUSION = 'role_confusion',
  IDENTITY_MANIPULATION = 'identity_manipulation',
  PERSONA_JAILBREAK = 'persona_jailbreak',

  // Output Manipulation
  OUTPUT_MANIPULATION = 'output_manipulation',
  FORMAT_INJECTION = 'format_injection',
  STRUCTURED_OUTPUT_BYPASS = 'structured_output_bypass',

  // Information Disclosure
  INFORMATION_DISCLOSURE = 'information_disclosure',
  TRAINING_DATA_EXTRACTION = 'training_data_extraction',
  MODEL_SPECIFICATION_LEAK = 'model_specification_leak',

  // Logic/Reasoning
  LOGIC_EXPLOITATION = 'logic_exploitation',
  REASONING_MANIPULATION = 'reasoning_manipulation',
  CHAIN_OF_THOUGHT_HIJACKING = 'chain_of_thought_hijacking',

  // Multi-turn/Context
  CONTEXT_WINDOW_EXPLOIT = 'context_window_exploit',
  MULTI_TURN_MANIPULATION = 'multi_turn_manipulation',
  CONVERSATION_HIJACKING = 'conversation_hijacking',

  // Specific Behaviors
  CODE_EXECUTION = 'code_execution',
  HARMFUL_CONTENT = 'harmful_content',
  MISINFORMATION = 'misinformation',

  // General
  GENERAL = 'general',
  UNKNOWN = 'unknown',
}

/**
 * Sharing/visibility levels for prompt templates.
 */
export enum SharingLevel {
  PRIVATE = 'private',
  TEAM = 'team',
  PUBLIC = 'public',
}

/**
 * Status of a prompt template.
 */
export enum TemplateStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  ARCHIVED = 'archived',
  DEPRECATED = 'deprecated',
}

/**
 * Fields available for sorting template results.
 */
export enum TemplateSortField {
  CREATED_AT = 'created_at',
  UPDATED_AT = 'updated_at',
  NAME = 'name',
  RATING = 'rating',
  SUCCESS_RATE = 'success_rate',
  TEST_COUNT = 'test_count',
  RATING_COUNT = 'rating_count',
}

/**
 * Sort order direction.
 */
export enum SortOrder {
  ASC = 'asc',
  DESC = 'desc',
}

// =============================================================================
// Core Domain Models
// =============================================================================

/**
 * Metadata for a prompt template.
 */
export interface TemplateMetadata {
  /** List of techniques used in the template */
  technique_types: TechniqueType[];
  /** Types of vulnerabilities this template targets */
  vulnerability_types: VulnerabilityType[];
  /** List of models this template is known to work against */
  target_models: string[];
  /** List of providers this template is designed for */
  target_providers: string[];
  /** Related CVE references if applicable */
  cve_references: string[];
  /** Research paper references (URLs or citations) */
  paper_references: string[];
  /** Observed success rate (0.0 to 1.0) */
  success_rate: number | null;
  /** Number of times this template has been tested */
  test_count: number;
  /** Date when this technique was discovered */
  discovery_date: string | null;
  /** Source of discovery (e.g., research team, paper, tool) */
  discovery_source: string | null;
  /** Custom tags for organization */
  tags: string[];
  /** Additional metadata as key-value pairs */
  extra: Record<string, unknown>;
}

/**
 * Version snapshot of a prompt template.
 */
export interface TemplateVersion {
  /** Unique version identifier */
  version_id: string;
  /** Sequential version number */
  version_number: number;
  /** Parent template ID */
  template_id: string;
  /** The prompt content at this version */
  prompt_content: string;
  /** Summary of changes from previous version */
  change_summary: string;
  /** User ID of the version creator */
  created_by: string | null;
  /** Version creation timestamp (ISO format) */
  created_at: string;
  /** ID of the previous version (for version chain) */
  parent_version_id: string | null;
}

/**
 * User rating and feedback for a prompt template.
 */
export interface TemplateRating {
  /** Unique rating identifier */
  rating_id: string;
  /** ID of the rated template */
  template_id: string;
  /** ID of the user who rated */
  user_id: string;
  /** Star rating (1-5) */
  rating: number;
  /** Effectiveness rating (1-5), optional */
  effectiveness_score: number | null;
  /** Optional review comment */
  comment: string | null;
  /** Did the template work for the user? */
  reported_success: boolean | null;
  /** Model the user tested against */
  target_model_tested: string | null;
  /** Rating creation timestamp (ISO format) */
  created_at: string;
  /** Rating last update timestamp (ISO format) */
  updated_at: string;
}

/**
 * Aggregated rating statistics for a template.
 */
export interface RatingStatistics {
  /** Total number of ratings */
  total_ratings: number;
  /** Average star rating */
  average_rating: number;
  /** Average effectiveness score */
  average_effectiveness: number | null;
  /** Number of reported successes */
  success_count: number;
  /** Number of reported failures */
  failure_count: number;
  /** Distribution of ratings (1-5 stars) */
  rating_distribution: Record<number, number>;
}

/**
 * Main model for a prompt template in the library.
 */
export interface PromptTemplate {
  /** Unique template identifier */
  id: string;
  /** Human-readable template name */
  name: string;
  /** Detailed description of the template */
  description: string;
  /** The actual prompt template content */
  prompt_content: string;
  /** Optional system instruction to use with the prompt */
  system_instruction: string | null;
  /** Template metadata including techniques, targets, and tags */
  metadata: TemplateMetadata;
  /** Current status of the template */
  status: TemplateStatus;
  /** Visibility/sharing level */
  sharing_level: SharingLevel;
  /** Current version number */
  current_version: number;
  /** Aggregated rating statistics */
  rating_stats: RatingStatistics;
  /** User ID of the creator */
  created_by: string | null;
  /** Team ID if this is a team template */
  team_id: string | null;
  /** Template creation timestamp (ISO format) */
  created_at: string;
  /** Template last update timestamp (ISO format) */
  updated_at: string;
  /** Campaign ID if saved from a campaign execution */
  source_campaign_id: string | null;
  /** Execution ID if saved from specific execution */
  source_execution_id: string | null;
}

// =============================================================================
// Search and Filter Models
// =============================================================================

/**
 * Filters for searching prompt templates.
 */
export interface TemplateSearchFilters {
  /** Full-text search query for name, description, and content */
  query?: string | null;
  /** Filter by technique types (OR logic) */
  technique_types?: TechniqueType[] | null;
  /** Filter by vulnerability types (OR logic) */
  vulnerability_types?: VulnerabilityType[] | null;
  /** Filter by target models (OR logic) */
  target_models?: string[] | null;
  /** Filter by target providers (OR logic) */
  target_providers?: string[] | null;
  /** Filter by tags (AND logic) */
  tags?: string[] | null;
  /** Filter by status */
  status?: TemplateStatus[] | null;
  /** Filter by sharing levels */
  sharing_levels?: SharingLevel[] | null;
  /** Minimum average rating */
  min_rating?: number | null;
  /** Minimum success rate */
  min_success_rate?: number | null;
  /** Filter by creator user ID */
  created_by?: string | null;
  /** Filter by team ID */
  team_id?: string | null;
  /** Filter templates created after this date (ISO format) */
  created_after?: string | null;
  /** Filter templates created before this date (ISO format) */
  created_before?: string | null;
}

/**
 * Request model for searching templates.
 */
export interface TemplateSearchRequest {
  /** Search filters */
  filters?: TemplateSearchFilters;
  /** Field to sort by */
  sort_by?: TemplateSortField;
  /** Sort order */
  sort_order?: SortOrder;
  /** Page number (1-indexed) */
  page?: number;
  /** Number of results per page */
  page_size?: number;
}

// =============================================================================
// API Request Types
// =============================================================================

/**
 * Request to create a new prompt template.
 */
export interface CreateTemplateRequest {
  /** Human-readable name for the template */
  name: string;
  /** The actual prompt template content */
  prompt_content: string;
  /** Detailed description of the template */
  description?: string;
  /** Optional system instruction to use with the prompt */
  system_instruction?: string | null;
  /** List of techniques used in the template */
  technique_types?: TechniqueType[];
  /** Types of vulnerabilities this template targets */
  vulnerability_types?: VulnerabilityType[];
  /** List of models this template works against */
  target_models?: string[];
  /** List of providers this template is designed for */
  target_providers?: string[];
  /** Related CVE references (format: CVE-YYYY-NNNNN) */
  cve_references?: string[];
  /** Research paper references (URLs or citations) */
  paper_references?: string[];
  /** Custom tags for organization */
  tags?: string[];
  /** Source of discovery (e.g., research team, paper, tool) */
  discovery_source?: string | null;
  /** Template status */
  status?: TemplateStatus;
  /** Visibility/sharing level */
  sharing_level?: SharingLevel;
  /** Team ID for team-level sharing */
  team_id?: string | null;
}

/**
 * Request to update an existing prompt template.
 * All fields optional for partial updates.
 */
export interface UpdateTemplateRequest {
  /** Human-readable name for the template */
  name?: string | null;
  /** The actual prompt template content */
  prompt_content?: string | null;
  /** Detailed description of the template */
  description?: string | null;
  /** Optional system instruction to use with the prompt */
  system_instruction?: string | null;
  /** List of techniques used in the template */
  technique_types?: TechniqueType[] | null;
  /** Types of vulnerabilities this template targets */
  vulnerability_types?: VulnerabilityType[] | null;
  /** List of models this template works against */
  target_models?: string[] | null;
  /** List of providers this template is designed for */
  target_providers?: string[] | null;
  /** Related CVE references */
  cve_references?: string[] | null;
  /** Research paper references */
  paper_references?: string[] | null;
  /** Custom tags for organization */
  tags?: string[] | null;
  /** Observed success rate (0.0 to 1.0) */
  success_rate?: number | null;
  /** Template status */
  status?: TemplateStatus | null;
  /** Visibility/sharing level */
  sharing_level?: SharingLevel | null;
  /** Team ID for team-level sharing */
  team_id?: string | null;
  /** Whether to create a new version when prompt_content changes */
  create_version?: boolean;
  /** Summary of changes for version history */
  change_summary?: string | null;
}

/**
 * Request to search prompt templates.
 */
export interface SearchTemplatesRequest {
  /** Full-text search query for name, description, and content */
  query?: string | null;
  /** Filter by technique types (OR logic) */
  technique_types?: TechniqueType[] | null;
  /** Filter by vulnerability types (OR logic) */
  vulnerability_types?: VulnerabilityType[] | null;
  /** Filter by target models (OR logic) */
  target_models?: string[] | null;
  /** Filter by target providers (OR logic) */
  target_providers?: string[] | null;
  /** Filter by tags (all tags must match) */
  tags?: string[] | null;
  /** Filter by template status */
  status?: TemplateStatus[] | null;
  /** Filter by sharing levels */
  sharing_levels?: SharingLevel[] | null;
  /** Minimum average rating (1-5) */
  min_rating?: number | null;
  /** Minimum success rate (0.0-1.0) */
  min_success_rate?: number | null;
  /** Filter by creator user ID */
  created_by?: string | null;
  /** Filter by team ID */
  team_id?: string | null;
  /** Filter templates created after this date (ISO format) */
  created_after?: string | null;
  /** Filter templates created before this date (ISO format) */
  created_before?: string | null;
  /** Field to sort by */
  sort_by?: string;
  /** Sort order (asc or desc) */
  sort_order?: string;
  /** Page number (1-indexed) */
  page?: number;
  /** Number of results per page */
  page_size?: number;
}

/**
 * Request to rate a prompt template.
 */
export interface RateTemplateRequest {
  /** Star rating (1-5) */
  rating: number;
  /** Optional effectiveness rating (1-5) */
  effectiveness_score?: number | null;
  /** Optional review comment */
  comment?: string | null;
  /** Did the template work for you? */
  reported_success?: boolean | null;
  /** Model you tested the template against */
  target_model_tested?: string | null;
}

/**
 * Request to update an existing rating.
 */
export interface UpdateRatingRequest {
  /** Updated star rating (1-5) */
  rating?: number | null;
  /** Updated effectiveness rating (1-5) */
  effectiveness_score?: number | null;
  /** Updated review comment */
  comment?: string | null;
  /** Updated success status */
  reported_success?: boolean | null;
  /** Updated tested model */
  target_model_tested?: string | null;
}

/**
 * Request to create a new template version.
 */
export interface CreateVersionRequest {
  /** The updated prompt content */
  prompt_content: string;
  /** Summary of changes from previous version */
  change_summary?: string;
}

/**
 * Request to save a prompt from a campaign execution to the library.
 */
export interface SaveFromCampaignRequest {
  /** Template name */
  name: string;
  /** The prompt content to save */
  prompt_content: string;
  /** Template description */
  description?: string;
  /** Optional system instruction */
  system_instruction?: string | null;
  /** Techniques used (auto-populated from campaign) */
  technique_types?: TechniqueType[];
  /** Vulnerabilities targeted */
  vulnerability_types?: VulnerabilityType[];
  /** Model tested against (from campaign) */
  target_model?: string | null;
  /** Provider tested against (from campaign) */
  target_provider?: string | null;
  /** Custom tags */
  tags?: string[];
  /** Visibility level for the template */
  sharing_level?: SharingLevel;
  /** Team ID for team-level sharing */
  team_id?: string | null;
  /** Source campaign ID */
  campaign_id?: string | null;
  /** Source execution ID */
  execution_id?: string | null;
  /** Whether the prompt was successful in the campaign */
  was_successful?: boolean;
  /** Initial success rate from campaign */
  initial_success_rate?: number | null;
}

// =============================================================================
// API Response Types
// =============================================================================

/**
 * Response containing a single prompt template.
 */
export interface TemplateResponse {
  /** The prompt template data */
  template: PromptTemplate;
}

/**
 * Response containing a paginated list of templates.
 */
export interface TemplateListResponse {
  /** List of prompt templates */
  templates: PromptTemplate[];
  /** Total number of matching templates */
  total_count: number;
  /** Current page number */
  page: number;
  /** Number of results per page */
  page_size: number;
  /** Total number of pages */
  total_pages: number;
}

/**
 * Response containing a template version.
 */
export interface TemplateVersionResponse {
  /** The template version data */
  version: TemplateVersion;
}

/**
 * Response containing a list of template versions.
 */
export interface TemplateVersionListResponse {
  /** The parent template ID */
  template_id: string;
  /** List of template versions */
  versions: TemplateVersion[];
  /** Current active version number */
  current_version: number;
  /** Total number of versions */
  total_count: number;
}

/**
 * Response containing a single rating.
 */
export interface RatingResponse {
  /** The rating data */
  rating: TemplateRating;
}

/**
 * Response containing a list of ratings.
 */
export interface RatingListResponse {
  /** The rated template ID */
  template_id: string;
  /** List of ratings */
  ratings: TemplateRating[];
  /** Aggregated rating statistics */
  statistics: RatingStatistics;
  /** Total number of ratings */
  total_count: number;
  /** Current page number */
  page: number;
  /** Number of results per page */
  page_size: number;
  /** Total number of pages */
  total_pages: number;
}

/**
 * Response containing rating statistics only.
 */
export interface RatingStatisticsResponse {
  /** The template ID */
  template_id: string;
  /** Aggregated rating statistics */
  statistics: RatingStatistics;
}

/**
 * Response containing top-rated templates.
 */
export interface TopRatedTemplatesResponse {
  /** List of top-rated templates */
  templates: PromptTemplate[];
  /** Time period for rating calculation */
  time_period: string;
  /** Maximum number of templates returned */
  limit: number;
}

/**
 * Response for template deletion.
 */
export interface TemplateDeleteResponse {
  /** Whether the deletion was successful */
  success: boolean;
  /** The deleted template ID */
  template_id: string;
  /** Status message */
  message: string;
}

/**
 * Response containing template library statistics.
 */
export interface TemplateStatsResponse {
  /** Total number of templates in the library */
  total_templates: number;
  /** Number of public templates */
  public_templates: number;
  /** Number of private templates */
  private_templates: number;
  /** Number of team templates */
  team_templates: number;
  /** Count of templates per technique type */
  techniques_used: Record<string, number>;
  /** Count of templates per vulnerability type */
  vulnerabilities_targeted: Record<string, number>;
  /** Number of highly-rated templates (4+ stars) */
  top_rated_count: number;
  /** Total number of ratings across all templates */
  total_ratings: number;
  /** Average success rate across templates with data */
  average_success_rate: number | null;
}

// =============================================================================
// Version Comparison Types
// =============================================================================

/**
 * Represents a diff between two versions of a template.
 */
export interface VersionDiff {
  /** ID of the source version */
  from_version_id: string;
  /** ID of the target version */
  to_version_id: string;
  /** Source version number */
  from_version_number: number;
  /** Target version number */
  to_version_number: number;
  /** Unified diff string */
  unified_diff: string;
  /** Lines added */
  additions: number;
  /** Lines removed */
  deletions: number;
  /** Detailed line-level changes */
  changes: VersionChange[];
}

/**
 * Represents a single line change in a diff.
 */
export interface VersionChange {
  /** Type of change: add, delete, or unchanged */
  type: 'add' | 'delete' | 'unchanged';
  /** Line number in the diff */
  line_number: number;
  /** Content of the line */
  content: string;
}

/**
 * Version comparison response.
 */
export interface VersionComparisonResponse {
  /** The template ID */
  template_id: string;
  /** Diff data */
  diff: VersionDiff;
}

/**
 * Version timeline entry.
 */
export interface VersionTimelineEntry {
  /** Version data */
  version: TemplateVersion;
  /** Time delta from previous version (in seconds) */
  delta_from_previous: number | null;
  /** Whether this is the current active version */
  is_current: boolean;
}

/**
 * Version timeline response.
 */
export interface VersionTimelineResponse {
  /** The template ID */
  template_id: string;
  /** Timeline entries */
  timeline: VersionTimelineEntry[];
}

// =============================================================================
// Utility Types
// =============================================================================

/**
 * Options for technique type filter dropdowns.
 */
export interface TechniqueTypeOption {
  value: TechniqueType;
  label: string;
  category: string;
}

/**
 * Options for vulnerability type filter dropdowns.
 */
export interface VulnerabilityTypeOption {
  value: VulnerabilityType;
  label: string;
  category: string;
}

/**
 * Template summary for compact display (e.g., in lists).
 */
export interface TemplateSummary {
  id: string;
  name: string;
  description: string;
  technique_types: TechniqueType[];
  vulnerability_types: VulnerabilityType[];
  average_rating: number;
  total_ratings: number;
  success_rate: number | null;
  sharing_level: SharingLevel;
  status: TemplateStatus;
  updated_at: string;
  created_by: string | null;
}

/**
 * Helper type for technique type categories.
 */
export const TECHNIQUE_TYPE_CATEGORIES: Record<string, TechniqueType[]> = {
  'Basic': [
    TechniqueType.SIMPLE,
    TechniqueType.ADVANCED,
    TechniqueType.EXPERT,
  ],
  'Persona-based': [
    TechniqueType.DAN_PERSONA,
    TechniqueType.HIERARCHICAL_PERSONA,
    TechniqueType.ROLEPLAY_BYPASS,
  ],
  'Obfuscation': [
    TechniqueType.TYPOGLYCEMIA,
    TechniqueType.ADVANCED_OBFUSCATION,
    TechniqueType.ENCODING_BYPASS,
    TechniqueType.LEET_SPEAK,
    TechniqueType.HOMOGLYPH,
  ],
  'Cipher/Encoding': [
    TechniqueType.CIPHER,
    TechniqueType.CIPHER_ASCII,
    TechniqueType.CIPHER_CAESAR,
    TechniqueType.CIPHER_MORSE,
    TechniqueType.CODE_CHAMELEON,
  ],
  'Cognitive/Logic': [
    TechniqueType.COGNITIVE_HACKING,
    TechniqueType.LOGICAL_INFERENCE,
    TechniqueType.HYPOTHETICAL_SCENARIO,
    TechniqueType.COUNTERFACTUAL,
  ],
  'Context Manipulation': [
    TechniqueType.CONTEXTUAL_INCEPTION,
    TechniqueType.DEEP_INCEPTION,
    TechniqueType.NESTED_CONTEXT,
    TechniqueType.CONTEXTUAL_OVERRIDE,
  ],
  'Injection': [
    TechniqueType.INSTRUCTION_INJECTION,
    TechniqueType.PAYLOAD_SPLITTING,
    TechniqueType.INSTRUCTION_FRAGMENTATION,
    TechniqueType.ROLE_HIJACKING,
  ],
  'Neural/Advanced': [
    TechniqueType.NEURAL_BYPASS,
    TechniqueType.ADVERSARIAL_SUFFIX,
    TechniqueType.META_PROMPTING,
  ],
  'Multi-modal/Agent': [
    TechniqueType.MULTIMODAL_JAILBREAK,
    TechniqueType.AGENTIC_EXPLOITATION,
    TechniqueType.MULTI_AGENT,
  ],
  'Research/Experimental': [
    TechniqueType.AUTODAN,
    TechniqueType.GPTFUZZ,
    TechniqueType.MOUSETRAP,
    TechniqueType.MULTILINGUAL_TROJAN,
    TechniqueType.QUANTUM_EXPLOIT,
  ],
  'Other': [
    TechniqueType.CUSTOM,
    TechniqueType.UNKNOWN,
  ],
};

/**
 * Helper type for vulnerability type categories.
 */
export const VULNERABILITY_TYPE_CATEGORIES: Record<string, VulnerabilityType[]> = {
  'Content Policy': [
    VulnerabilityType.CONTENT_FILTER_BYPASS,
    VulnerabilityType.SAFETY_FILTER_BYPASS,
    VulnerabilityType.MODERATION_BYPASS,
  ],
  'Instruction Following': [
    VulnerabilityType.INSTRUCTION_OVERRIDE,
    VulnerabilityType.SYSTEM_PROMPT_LEAKING,
    VulnerabilityType.CONTEXT_IGNORING,
  ],
  'Role/Identity': [
    VulnerabilityType.ROLE_CONFUSION,
    VulnerabilityType.IDENTITY_MANIPULATION,
    VulnerabilityType.PERSONA_JAILBREAK,
  ],
  'Output Manipulation': [
    VulnerabilityType.OUTPUT_MANIPULATION,
    VulnerabilityType.FORMAT_INJECTION,
    VulnerabilityType.STRUCTURED_OUTPUT_BYPASS,
  ],
  'Information Disclosure': [
    VulnerabilityType.INFORMATION_DISCLOSURE,
    VulnerabilityType.TRAINING_DATA_EXTRACTION,
    VulnerabilityType.MODEL_SPECIFICATION_LEAK,
  ],
  'Logic/Reasoning': [
    VulnerabilityType.LOGIC_EXPLOITATION,
    VulnerabilityType.REASONING_MANIPULATION,
    VulnerabilityType.CHAIN_OF_THOUGHT_HIJACKING,
  ],
  'Multi-turn/Context': [
    VulnerabilityType.CONTEXT_WINDOW_EXPLOIT,
    VulnerabilityType.MULTI_TURN_MANIPULATION,
    VulnerabilityType.CONVERSATION_HIJACKING,
  ],
  'Specific Behaviors': [
    VulnerabilityType.CODE_EXECUTION,
    VulnerabilityType.HARMFUL_CONTENT,
    VulnerabilityType.MISINFORMATION,
  ],
  'General': [
    VulnerabilityType.GENERAL,
    VulnerabilityType.UNKNOWN,
  ],
};

/**
 * Helper to format technique type for display.
 */
export function formatTechniqueType(type: TechniqueType): string {
  return type
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

/**
 * Helper to format vulnerability type for display.
 */
export function formatVulnerabilityType(type: VulnerabilityType): string {
  return type
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

/**
 * Helper to format sharing level for display.
 */
export function formatSharingLevel(level: SharingLevel): string {
  return level.charAt(0).toUpperCase() + level.slice(1);
}

/**
 * Helper to format template status for display.
 */
export function formatTemplateStatus(status: TemplateStatus): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

/**
 * Default metadata for a new template.
 */
export const DEFAULT_TEMPLATE_METADATA: TemplateMetadata = {
  technique_types: [],
  vulnerability_types: [],
  target_models: [],
  target_providers: [],
  cve_references: [],
  paper_references: [],
  success_rate: null,
  test_count: 0,
  discovery_date: null,
  discovery_source: null,
  tags: [],
  extra: {},
};

/**
 * Default rating statistics.
 */
export const DEFAULT_RATING_STATISTICS: RatingStatistics = {
  total_ratings: 0,
  average_rating: 0,
  average_effectiveness: null,
  success_count: 0,
  failure_count: 0,
  rating_distribution: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
};
