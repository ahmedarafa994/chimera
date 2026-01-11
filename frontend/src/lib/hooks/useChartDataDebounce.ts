/**
 * useChartDataDebounce Hook
 *
 * Provides debounced chart data updates to prevent excessive re-renders
 * during high-frequency telemetry event processing.
 *
 * Performance optimizations:
 * - 100ms debounce for chart data updates (per spec requirement)
 * - Uses useDeferredValue for non-blocking chart updates
 * - Maintains immediate state updates for critical metrics
 */

"use client";

import { useMemo, useRef, useEffect, useState, useDeferredValue } from "react";
import {
  SuccessRateTimeSeries,
  TokenUsageTimeSeries,
  LatencyTimeSeries,
} from "@/types/aegis-telemetry";

// ============================================================================
// Constants
// ============================================================================

/**
 * Debounce delay for chart updates in milliseconds
 * Per spec requirement: "Debounced chart updates (100ms)"
 */
export const CHART_UPDATE_DEBOUNCE_MS = 100;

/**
 * Maximum number of points to render in charts for performance
 */
export const MAX_CHART_RENDER_POINTS = 50;

// ============================================================================
// Types
// ============================================================================

export interface ChartDataState {
  successRateData: SuccessRateTimeSeries[];
  tokenUsageData: TokenUsageTimeSeries[];
  latencyData: LatencyTimeSeries[];
}

export interface UseChartDataDebounceReturn {
  /** Debounced success rate data for chart rendering */
  debouncedSuccessRateData: SuccessRateTimeSeries[];
  /** Debounced token usage data for chart rendering */
  debouncedTokenUsageData: TokenUsageTimeSeries[];
  /** Debounced latency data for chart rendering */
  debouncedLatencyData: LatencyTimeSeries[];
  /** Whether chart data is currently being debounced */
  isPending: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Downsample data for performance while preserving trends
 * Uses LTTB-like algorithm for visually representative downsampling
 */
function downsampleData<T extends { timestamp: string }>(
  data: T[],
  targetPoints: number
): T[] {
  if (data.length <= targetPoints) {
    return data;
  }

  const result: T[] = [];
  const step = (data.length - 2) / (targetPoints - 2);

  // Always include first point
  result.push(data[0]);

  // Sample middle points
  for (let i = 1; i < targetPoints - 1; i++) {
    const index = Math.round(i * step);
    if (index < data.length - 1) {
      result.push(data[index]);
    }
  }

  // Always include last point
  result.push(data[data.length - 1]);

  return result;
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Custom hook for debounced chart data updates
 *
 * Uses a combination of techniques to optimize chart performance:
 * 1. Debouncing to batch rapid updates (100ms)
 * 2. useDeferredValue for non-blocking updates
 * 3. Data downsampling for large datasets
 *
 * @param successRateData - Raw success rate time series data
 * @param tokenUsageData - Raw token usage time series data
 * @param latencyData - Raw latency time series data
 * @returns Debounced chart data ready for rendering
 *
 * @example
 * ```tsx
 * const {
 *   debouncedSuccessRateData,
 *   debouncedTokenUsageData,
 *   debouncedLatencyData,
 *   isPending,
 * } = useChartDataDebounce(
 *   successRateHistory,
 *   tokenUsageHistory,
 *   latencyHistory
 * );
 *
 * return (
 *   <SuccessRateTrendChart
 *     data={debouncedSuccessRateData}
 *     isLoading={isPending}
 *   />
 * );
 * ```
 */
export function useChartDataDebounce(
  successRateData: SuccessRateTimeSeries[],
  tokenUsageData: TokenUsageTimeSeries[],
  latencyData: LatencyTimeSeries[]
): UseChartDataDebounceReturn {
  // Use state to hold debounced values
  const [debouncedState, setDebouncedState] = useState<ChartDataState>({
    successRateData: [],
    tokenUsageData: [],
    latencyData: [],
  });

  // Track pending updates
  const [isPending, setIsPending] = useState(false);

  // Use ref for debounce timer
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const pendingDataRef = useRef<ChartDataState | null>(null);

  // Process and downsample data
  const processedData = useMemo((): ChartDataState => {
    return {
      successRateData: downsampleData(successRateData, MAX_CHART_RENDER_POINTS),
      tokenUsageData: downsampleData(tokenUsageData, MAX_CHART_RENDER_POINTS),
      latencyData: downsampleData(latencyData, MAX_CHART_RENDER_POINTS),
    };
  }, [successRateData, tokenUsageData, latencyData]);

  // Debounce the updates
  useEffect(() => {
    // Store pending data
    pendingDataRef.current = processedData;
    setIsPending(true);

    // Clear existing timer
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    // Set debounced update
    timerRef.current = setTimeout(() => {
      if (pendingDataRef.current) {
        setDebouncedState(pendingDataRef.current);
        setIsPending(false);
        pendingDataRef.current = null;
      }
    }, CHART_UPDATE_DEBOUNCE_MS);

    // Cleanup
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [processedData]);

  // Use deferred values for additional optimization
  const deferredSuccessRateData = useDeferredValue(debouncedState.successRateData);
  const deferredTokenUsageData = useDeferredValue(debouncedState.tokenUsageData);
  const deferredLatencyData = useDeferredValue(debouncedState.latencyData);

  return {
    debouncedSuccessRateData: deferredSuccessRateData,
    debouncedTokenUsageData: deferredTokenUsageData,
    debouncedLatencyData: deferredLatencyData,
    isPending,
  };
}

// ============================================================================
// Exports
// ============================================================================

export default useChartDataDebounce;
