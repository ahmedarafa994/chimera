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

// CSV export utilities
export {
  generateCSV,
  generateCSVAuto,
  downloadCSV,
  exportAndDownloadCSV,
  generateCSVFilename,
  checkCSVExportSupport,
  CAMPAIGN_CSV_COLUMNS,
  type CSVDateFormat,
  type CSVColumn,
  type CSVGenerateOptions,
  type CSVGenerateResult,
  type CSVDownloadResult,
  type CampaignCSVColumnSet,
} from './csv-export';

// ZIP export utilities
export {
  generateZip,
  downloadZip,
  generateAndDownloadZip,
  generateZipFilename,
  checkZipExportSupport,
  type ZipFileEntry,
  type ZipGenerateResult,
  type ZipGenerateOptions,
} from './zip-export';
