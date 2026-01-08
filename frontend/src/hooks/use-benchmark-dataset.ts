"use client";

import { useState, useCallback, useEffect } from "react";
import {
  benchmarkDatasetService,
  BenchmarkDatasetSummary,
  BenchmarkDataset,
  BenchmarkPrompt,
  BenchmarkStatistics,
  RiskAreaInfo,
  TestBatchConfig,
  TestBatchResponse,
} from "@/lib/services/benchmark-dataset-service";

/**
 * Return type for the useBenchmarkDataset hook
 */
export interface UseBenchmarkDatasetReturn {
  // State
  isLoading: boolean;
  error: string | null;
  datasets: BenchmarkDatasetSummary[];
  selectedDataset: BenchmarkDataset | null;
  selectedPrompts: BenchmarkPrompt[];
  statistics: BenchmarkStatistics | null;
  riskAreas: RiskAreaInfo[];

  // Actions
  fetchDatasets: () => Promise<void>;
  selectDataset: (datasetName: string) => Promise<void>;
  fetchPrompts: (options?: {
    risk_area?: string;
    harm_type?: string;
    severity?: string;
    limit?: number;
  }) => Promise<void>;
  fetchRandomPrompts: (count?: number, options?: {
    risk_area?: string;
    harm_type?: string;
  }) => Promise<BenchmarkPrompt[]>;
  generateTestBatch: (config: TestBatchConfig) => Promise<TestBatchResponse>;
  clearSelection: () => void;
  clearError: () => void;
}

/**
 * Hook for managing benchmark dataset interactions.
 * Provides state management and API interactions for benchmark datasets.
 */
export function useBenchmarkDataset(): UseBenchmarkDatasetReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<BenchmarkDatasetSummary[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<BenchmarkDataset | null>(null);
  const [selectedPrompts, setSelectedPrompts] = useState<BenchmarkPrompt[]>([]);
  const [statistics, setStatistics] = useState<BenchmarkStatistics | null>(null);
  const [riskAreas, setRiskAreas] = useState<RiskAreaInfo[]>([]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedDataset(null);
    setSelectedPrompts([]);
    setStatistics(null);
    setRiskAreas([]);
  }, []);

  /**
   * Fetch all available benchmark datasets
   */
  const fetchDatasets = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await benchmarkDatasetService.listDatasets();
      setDatasets(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to fetch datasets";
      setError(message);
      console.error("[useBenchmarkDataset] Failed to fetch datasets:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Select and load a specific dataset
   */
  const selectDataset = useCallback(async (datasetName: string) => {
    setIsLoading(true);
    setError(null);

    try {
      // Fetch dataset details, statistics, and risk areas in parallel
      const [dataset, stats, areas] = await Promise.all([
        benchmarkDatasetService.getDataset(datasetName),
        benchmarkDatasetService.getStatistics(datasetName),
        benchmarkDatasetService.getRiskAreas(datasetName),
      ]);

      setSelectedDataset(dataset);
      setStatistics(stats);
      setRiskAreas(areas);
      setSelectedPrompts(dataset.prompts || []);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load dataset";
      setError(message);
      console.error("[useBenchmarkDataset] Failed to load dataset:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Fetch prompts with optional filtering
   */
  const fetchPrompts = useCallback(async (options?: {
    risk_area?: string;
    harm_type?: string;
    severity?: string;
    limit?: number;
  }) => {
    if (!selectedDataset) {
      setError("No dataset selected");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await benchmarkDatasetService.getPrompts(selectedDataset.name, options);
      setSelectedPrompts(result.prompts);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to fetch prompts";
      setError(message);
      console.error("[useBenchmarkDataset] Failed to fetch prompts:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedDataset]);

  /**
   * Fetch random prompts from the selected dataset
   */
  const fetchRandomPrompts = useCallback(async (
    count: number = 5,
    options?: {
      risk_area?: string;
      harm_type?: string;
    }
  ): Promise<BenchmarkPrompt[]> => {
    if (!selectedDataset) {
      setError("No dataset selected");
      return [];
    }

    setIsLoading(true);
    setError(null);

    try {
      const prompts = await benchmarkDatasetService.getRandomPrompts(
        selectedDataset.name,
        count,
        options
      );
      return prompts;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to fetch random prompts";
      setError(message);
      console.error("[useBenchmarkDataset] Failed to fetch random prompts:", err);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [selectedDataset]);

  /**
   * Generate a test batch for adversarial testing
   */
  const generateTestBatch = useCallback(async (
    config: TestBatchConfig
  ): Promise<TestBatchResponse> => {
    if (!selectedDataset) {
      throw new Error("No dataset selected");
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await benchmarkDatasetService.generateTestBatch(
        selectedDataset.name,
        config
      );
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to generate test batch";
      setError(message);
      console.error("[useBenchmarkDataset] Failed to generate test batch:", err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [selectedDataset]);

  return {
    isLoading,
    error,
    datasets,
    selectedDataset,
    selectedPrompts,
    statistics,
    riskAreas,
    fetchDatasets,
    selectDataset,
    fetchPrompts,
    fetchRandomPrompts,
    generateTestBatch,
    clearSelection,
    clearError,
  };
}