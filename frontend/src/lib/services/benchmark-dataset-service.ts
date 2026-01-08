/**
 * Benchmark Dataset Service
 *
 * Service for interacting with the benchmark datasets API.
 * Provides access to Do-Not-Answer and other safety benchmark datasets
 * for adversarial testing and jailbreak evaluation.
 */

import { enhancedApi as api, apiClient } from "@/lib/api-enhanced";

// Environment check for debug logging
const isDevelopment = process.env.NODE_ENV === "development";

/**
 * Debug logger for Benchmark Dataset service.
 */
const debugLog = {
  log: (...args: unknown[]) => isDevelopment && console.log("[BenchmarkDataset]", ...args),
  error: (...args: unknown[]) => console.error("[BenchmarkDataset]", ...args),
  warn: (...args: unknown[]) => isDevelopment && console.warn("[BenchmarkDataset]", ...args),
};

/**
 * Risk area information from the Do-Not-Answer taxonomy
 */
export interface RiskAreaInfo {
  id: string;
  name: string;
  description: string;
  harm_types: HarmTypeInfo[];
  // Additional properties used by components
  display_name?: string;
  prompt_count?: number;
}

/**
 * Harm type information from the Do-Not-Answer taxonomy
 */
export interface HarmTypeInfo {
  id: string;
  name: string;
  description: string;
  example_behaviors: string[];
}

/**
 * A single benchmark prompt
 */
export interface BenchmarkPrompt {
  id: string;
  risk_area: string;
  harm_type: string;
  category: string;
  subcategory?: string;
  prompt: string;
  expected_behavior: string;
  severity: "low" | "medium" | "high" | "critical";
  metadata?: Record<string, unknown>;
}

/**
 * Summary of a benchmark dataset
 */
export interface BenchmarkDatasetSummary {
  name: string;
  display_name: string;
  description: string;
  total_prompts: number;
  risk_areas: number;
  harm_types: number;
  source: string;
  paper_url?: string;
  license?: string;
}

/**
 * Full benchmark dataset with all prompts and taxonomy
 */
export interface BenchmarkDataset {
  name: string;
  display_name: string;
  description: string;
  version: string;
  source: string;
  paper_url?: string;
  license?: string;
  prompts: BenchmarkPrompt[];
  taxonomy: {
    risk_areas: RiskAreaInfo[];
  };
  statistics: {
    total_prompts: number;
    prompts_by_risk_area: Record<string, number>;
    prompts_by_severity: Record<string, number>;
  };
}

/**
 * Statistics for a benchmark dataset
 */
export interface BenchmarkStatistics {
  total_prompts: number;
  prompts_by_risk_area: Record<string, number>;
  prompts_by_harm_type: Record<string, number>;
  prompts_by_severity: Record<string, number>;
  average_severity_score: number;
  // Additional properties used by components
  risk_area_counts?: Record<string, number>;
  harm_type_counts?: Record<string, number>;
  severity_counts?: Record<string, number>;
}

/**
 * Test batch configuration
 */
export interface TestBatchConfig {
  count: number;
  risk_areas?: string[];
  harm_types?: string[];
  min_severity?: "low" | "medium" | "high" | "critical";
  shuffle?: boolean;
}

/**
 * Test batch response
 */
export interface TestBatchResponse {
  dataset: string;
  prompts: BenchmarkPrompt[];
  count: number;
  filters_applied: {
    risk_areas?: string[];
    harm_types?: string[];
    min_severity?: string;
  };
}

/**
 * Export configuration for jailbreak testing
 */
export interface ExportConfig {
  format: "json" | "csv" | "jsonl";
  include_metadata?: boolean;
  risk_areas?: string[];
  harm_types?: string[];
}

/**
 * Benchmark Dataset Service class
 */
class BenchmarkDatasetService {
  private readonly baseUrl = "/benchmark-datasets";

  /**
   * List all available benchmark datasets
   */
  async listDatasets(): Promise<BenchmarkDatasetSummary[]> {
    const startTime = Date.now();
    const endpoint = this.baseUrl;

    debugLog.log("Fetching benchmark datasets list");

    try {
      const response = await api.get<{ datasets: BenchmarkDatasetSummary[] }>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.log(`Fetched ${response.datasets?.length || 0} datasets in ${duration}ms`);
      return response.datasets || [];
    } catch (error) {
      debugLog.error("Failed to fetch datasets:", error);
      throw error;
    }
  }

  /**
   * Get details for a specific dataset
   */
  async getDataset(datasetName: string): Promise<BenchmarkDataset> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}`;

    debugLog.log(`Fetching dataset: ${datasetName}`);

    try {
      const response = await api.get<BenchmarkDataset>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.log(`Fetched dataset ${datasetName} in ${duration}ms`);
      return response;
    } catch (error) {
      debugLog.error(`Failed to fetch dataset ${datasetName}:`, error);
      throw error;
    }
  }

  /**
   * Get prompts from a dataset with optional filtering
   */
  async getPrompts(
    datasetName: string,
    options?: {
      risk_area?: string;
      harm_type?: string;
      severity?: string;
      limit?: number;
      offset?: number;
    }
  ): Promise<{ prompts: BenchmarkPrompt[]; total: number }> {
    const startTime = Date.now();
    const params = new URLSearchParams();

    if (options?.risk_area) params.set("risk_area", options.risk_area);
    if (options?.harm_type) params.set("harm_type", options.harm_type);
    if (options?.severity) params.set("severity", options.severity);
    if (options?.limit) params.set("limit", options.limit.toString());
    if (options?.offset) params.set("offset", options.offset.toString());

    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}/prompts?${params.toString()}`;

    debugLog.log(`Fetching prompts from ${datasetName}`, options);

    try {
      const response = await api.get<{ prompts: BenchmarkPrompt[]; total: number }>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.log(`Fetched ${response.prompts?.length || 0} prompts in ${duration}ms`);
      return response;
    } catch (error) {
      debugLog.error(`Failed to fetch prompts from ${datasetName}:`, error);
      throw error;
    }
  }

  /**
   * Get random prompts from a dataset
   */
  async getRandomPrompts(
    datasetName: string,
    count: number = 5,
    options?: {
      risk_area?: string;
      harm_type?: string;
    }
  ): Promise<BenchmarkPrompt[]> {
    const startTime = Date.now();
    const params = new URLSearchParams();
    params.set("count", count.toString());

    if (options?.risk_area) params.set("risk_area", options.risk_area);
    if (options?.harm_type) params.set("harm_type", options.harm_type);

    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}/random?${params.toString()}`;

    debugLog.log(`Fetching ${count} random prompts from ${datasetName}`);

    try {
      const response = await api.get<{ prompts: BenchmarkPrompt[] }>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.log(`Fetched ${response.prompts?.length || 0} random prompts in ${duration}ms`);
      return response.prompts || [];
    } catch (error) {
      debugLog.error(`Failed to fetch random prompts from ${datasetName}:`, error);
      throw error;
    }
  }

  /**
   * Get the taxonomy for a dataset
   */
  async getTaxonomy(datasetName: string): Promise<{ risk_areas: RiskAreaInfo[] }> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}/taxonomy`;

    debugLog.log(`Fetching taxonomy for ${datasetName}`);

    try {
      const response = await api.get<{ risk_areas: RiskAreaInfo[] }>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.log(`Fetched taxonomy in ${duration}ms`);
      return response;
    } catch (error) {
      debugLog.error(`Failed to fetch taxonomy for ${datasetName}:`, error);
      throw error;
    }
  }

  /**
   * Get risk areas for a dataset
   */
  async getRiskAreas(datasetName: string): Promise<RiskAreaInfo[]> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}/risk-areas`;

    debugLog.log(`Fetching risk areas for ${datasetName}`);

    try {
      const response = await api.get<{ risk_areas: RiskAreaInfo[] }>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.log(`Fetched ${response.risk_areas?.length || 0} risk areas in ${duration}ms`);
      return response.risk_areas || [];
    } catch (error) {
      debugLog.error(`Failed to fetch risk areas for ${datasetName}:`, error);
      throw error;
    }
  }

  /**
   * Get statistics for a dataset
   */
  async getStatistics(datasetName: string): Promise<BenchmarkStatistics> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}/statistics`;

    debugLog.log(`Fetching statistics for ${datasetName}`);

    try {
      const response = await api.get<BenchmarkStatistics>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.log(`Fetched statistics in ${duration}ms`);
      return response;
    } catch (error) {
      debugLog.error(`Failed to fetch statistics for ${datasetName}:`, error);
      throw error;
    }
  }

  /**
   * Generate a test batch for adversarial testing
   */
  async generateTestBatch(
    datasetName: string,
    config: TestBatchConfig
  ): Promise<TestBatchResponse> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}/test-batch`;

    debugLog.log(`Generating test batch from ${datasetName}`, config);

    try {
      const response = await apiClient.post<TestBatchResponse>(endpoint, config);
      const duration = Date.now() - startTime;
      debugLog.log(`Generated ${response.data.count} prompts in ${duration}ms`);
      return response.data;
    } catch (error) {
      debugLog.error(`Failed to generate test batch from ${datasetName}:`, error);
      throw error;
    }
  }

  /**
   * Export prompts for jailbreak testing
   */
  async exportForJailbreak(
    datasetName: string,
    config?: ExportConfig
  ): Promise<{ data: string; format: string; count: number }> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/${encodeURIComponent(datasetName)}/export`;

    debugLog.log(`Exporting prompts from ${datasetName}`, config);

    try {
      const response = await apiClient.post<{ data: string; format: string; count: number }>(
        endpoint,
        config || { format: "json" }
      );
      const duration = Date.now() - startTime;
      debugLog.log(`Exported ${response.data.count} prompts in ${duration}ms`);
      return response.data;
    } catch (error) {
      debugLog.error(`Failed to export prompts from ${datasetName}:`, error);
      throw error;
    }
  }
}

// Export singleton instance
export const benchmarkDatasetService = new BenchmarkDatasetService();
