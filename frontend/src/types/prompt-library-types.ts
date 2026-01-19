export enum TechniqueType {
    AUTODAN = "autodan",
    GPTFUZZ = "gptfuzz",
    CHIMERA_FRAMING = "chimera_framing",
    PAIR = "pair",
    GCG = "gcg",
    TAP = "tap",
    CRESCENDO = "crescendo",
    MANUAL = "manual",
    OTHER = "other"
}

export enum VulnerabilityType {
    JAILBREAK = "jailbreak",
    INJECTION = "injection",
    PII_LEAK = "pii_leak",
    BYPASS = "bypass",
    MALICIOUS_CONTENT = "malicious_content",
    CODE_EXECUTION = "code_execution",
    DENIAL_OF_SERVICE = "denial_of_service",
    SOCIAL_ENGINEERING = "social_engineering",
    OTHER = "other"
}

export enum SharingLevel {
    PRIVATE = "private",
    TEAM = "team",
    PUBLIC = "public"
}

export enum TemplateStatus {
    DRAFT = "draft",
    ACTIVE = "active",
    ARCHIVED = "archived",
    DEPRECATED = "deprecated"
}

export enum TemplateSortField {
    TITLE = "title",
    CREATED_AT = "created_at",
    RATING = "rating",
    EFFECTIVENESS = "effectiveness"
}

export enum SortOrder {
    ASC = "asc",
    DESC = "desc"
}

export interface TemplateRating {
    user_id: string;
    rating: number;
    effectiveness_vote: boolean;
    comment?: string;
    created_at: string;
}

export interface RatingStatistics {
    avg_rating: number;
    total_ratings: number;
    effectiveness_score: number;
    rating_distribution: Record<number, number>;
}

export interface TemplateVersion {
    version_id: string;
    parent_version_id?: string;
    prompt_text: string;
    description?: string;
    created_by: string;
    created_at: string;
    metadata_overrides?: Record<string, any>;
}

export interface TemplateMetadata {
    technique_types: TechniqueType[];
    vulnerability_types: VulnerabilityType[];
    target_models: string[];
    success_rate: number;
    test_count: number;
    avg_score: number;
    cve_references: string[];
    discovery_date?: string;
    tags: string[];
    custom_data: Record<string, any>;
}

export interface PromptTemplate {
    id: string;
    title: string;
    description: string;
    original_prompt: string;
    current_version_id: string;
    owner_id: string;
    organization_id?: string;
    sharing_level: SharingLevel;
    status: TemplateStatus;
    metadata: TemplateMetadata;
    versions: TemplateVersion[];
    ratings: TemplateRating[];
    created_at: string;
    updated_at: string;
    avg_rating: number;
    total_ratings: number;
    effectiveness_score: number;
}

export interface TemplateListItem {
    id: string;
    title: string;
    description: string;
    technique_types: TechniqueType[];
    vulnerability_types: VulnerabilityType[];
    sharing_level: SharingLevel;
    status: TemplateStatus;
    avg_rating: number;
    total_ratings: number;
    effectiveness_score: number;
    tags: string[];
    created_at: string;
    owner_id: string;
}

export interface TemplateSearchFilters {
    query?: string;
    technique_type?: TechniqueType;
    vulnerability_type?: VulnerabilityType;
    sharing_level?: SharingLevel;
    tags?: string[];
    owner_id?: string;
    min_rating?: number;
}

export interface TemplateSearchRequest extends TemplateSearchFilters {
    limit: number;
    offset: number;
    sort_by?: TemplateSortField;
    sort_order?: SortOrder;
}

/**
 * Request to search for templates.
 * Empty interface for future extension while maintaining compatibility.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface SearchTemplatesRequest extends TemplateSearchRequest {}

export interface SearchTemplatesResponse {
    items: TemplateListItem[];
    total: number;
    limit: number;
    offset: number;
}

export interface CreateTemplateRequest {
    title: string;
    description: string;
    prompt_text: string;
    technique_types: TechniqueType[];
    vulnerability_types: VulnerabilityType[];
    sharing_level: SharingLevel;
    target_models?: string[];
    tags?: string[];
    custom_data?: Record<string, any>;
}

export interface UpdateTemplateRequest {
    title?: string;
    description?: string;
    sharing_level?: SharingLevel;
    status?: TemplateStatus;
    technique_types?: TechniqueType[];
    vulnerability_types?: VulnerabilityType[];
    target_models?: string[];
    tags?: string[];
    custom_data?: Record<string, any>;
}

export interface RateTemplateRequest {
    rating: number;
    effectiveness_vote: boolean;
    comment?: string;
}

export interface CreateVersionRequest {
    prompt_text: string;
    description?: string;
    metadata_overrides?: Record<string, any>;
}

/**
 * Empty interface for future extension while maintaining compatibility.
 * @deprecated Use RateTemplateRequest directly if no additional fields are needed.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface UpdateRatingRequest extends RateTemplateRequest {}

export interface CreateVersionRequest {
    prompt_text: string;
    description?: string;
    metadata_overrides?: Record<string, any>;
}

export interface SaveFromCampaignRequest {
    campaign_id: string;
    attack_id?: string;
    title: string;
    description: string;
    sharing_level: SharingLevel;
}

// Additional response types
export interface TemplateResponse {
    success: boolean;
    template: PromptTemplate;
}

export interface TemplateListResponse {
    items: TemplateListItem[];
    total: number;
}

export interface TemplateVersionResponse {
    version: TemplateVersion;
}

export interface TemplateVersionListResponse {
    versions: TemplateVersion[];
}

export interface RatingResponse {
    rating: TemplateRating;
}

export interface RatingListResponse {
    ratings: TemplateRating[];
}

export interface RatingStatisticsResponse {
    stats: RatingStatistics;
}

export interface TopRatedTemplatesResponse {
    items: TemplateListItem[];
}

export interface TemplateDeleteResponse {
    success: boolean;
    id: string;
}

export interface TemplateStatsResponse {
    total_templates: number;
    total_ratings: number;
    avg_effectiveness: number;
}

export interface VersionDiff {
    before: string;
    after: string;
    changes: any[];
}

export interface VersionChange {
    type: string;
    path: string;
    value: any;
}

export interface VersionComparisonResponse {
    diff: VersionDiff;
}

export interface VersionTimelineEntry {
    version_id: string;
    timestamp: string;
    author: string;
}

export interface VersionTimelineResponse {
    entries: VersionTimelineEntry[];
}

export interface TechniqueTypeOption {
    value: TechniqueType;
    label: string;
}

export interface VulnerabilityTypeOption {
    value: VulnerabilityType;
    label: string;
}

export interface TemplateSummary {
    id: string;
    title: string;
}

// Constants
export const TECHNIQUE_TYPE_CATEGORIES = {};
export const VULNERABILITY_TYPE_CATEGORIES = {};
export const DEFAULT_TEMPLATE_METADATA: any = {};
export const DEFAULT_RATING_STATISTICS: any = {};

// Utility formatting functions
export const formatTechniqueType = (type: TechniqueType): string => {
    return type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

export const formatVulnerabilityType = (type: VulnerabilityType): string => {
    return type.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

export const formatSharingLevel = (level: SharingLevel): string => {
    return level.charAt(0).toUpperCase() + level.slice(1);
};

export const formatTemplateStatus = (status: TemplateStatus): string => {
    return status.charAt(0).toUpperCase() + status.slice(1);
};
