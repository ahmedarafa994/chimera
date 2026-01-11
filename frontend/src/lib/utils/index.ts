/**
 * Utility Functions Index
 *
 * Central export for all utility modules.
 *
 * @module lib/utils
 */

// Debounce utilities
export {
  debounce,
  throttle,
  deduplicateRequest,
  generateCacheKey,
  DEBOUNCE_DELAYS,
  type DebouncedFunction,
  type DebounceOptions,
  type ThrottleOptions,
} from './debounce';

// Chart export utilities
export {
  exportChartAsPNG,
  exportChartAsSVG,
  downloadFile,
  exportAndDownloadChart,
  exportMultipleCharts,
  generateChartFilename,
  checkExportSupport,
  type ChartExportFormat,
  type PNGExportOptions,
  type SVGExportOptions,
  type ChartExportResult,
  type ChartRef,
} from './chart-export';
