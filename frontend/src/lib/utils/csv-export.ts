/**
 * CSV Export Utilities for Campaign Analytics
 *
 * Provides utility functions for generating and downloading CSV files from
 * analytics data. Supports nested objects, date formatting, number precision,
 * and handles special characters properly for Excel/Google Sheets compatibility.
 *
 * @module lib/utils/csv-export
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Date format options for CSV export.
 */
export type CSVDateFormat = 'ISO' | 'US' | 'EU' | 'TIMESTAMP';

/**
 * Column definition for CSV export.
 */
export interface CSVColumn<T = unknown> {
  /** Column header label */
  header: string;
  /** Key path to access data (supports dot notation for nested objects) */
  accessor: string | ((item: T) => unknown);
  /** Custom formatter for the value */
  formatter?: (value: unknown, item: T) => string;
  /** Type hint for automatic formatting */
  type?: 'string' | 'number' | 'date' | 'boolean' | 'percentage' | 'currency';
  /** Decimal precision for number types (default: 2) */
  precision?: number;
  /** Date format for date types (default: 'ISO') */
  dateFormat?: CSVDateFormat;
  /** Currency code for currency types (default: 'USD') */
  currencyCode?: string;
  /** Whether to include this column (default: true) */
  include?: boolean;
}

/**
 * Configuration options for CSV generation.
 */
export interface CSVGenerateOptions {
  /** Column delimiter (default: ',') */
  delimiter?: string;
  /** Row separator (default: '\r\n' for Windows compatibility) */
  lineEnding?: string;
  /** Whether to include headers (default: true) */
  includeHeaders?: boolean;
  /** Date format for date values (default: 'ISO') */
  dateFormat?: CSVDateFormat;
  /** Decimal precision for numbers (default: 2) */
  decimalPrecision?: number;
  /** Whether to add BOM for Excel UTF-8 support (default: true) */
  addBOM?: boolean;
  /** Null value representation (default: '') */
  nullValue?: string;
  /** Undefined value representation (default: '') */
  undefinedValue?: string;
  /** Boolean true representation (default: 'true') */
  trueValue?: string;
  /** Boolean false representation (default: 'false') */
  falseValue?: string;
  /** Quote character for escaping (default: '"') */
  quoteChar?: string;
  /** Whether to always quote values (default: false) */
  alwaysQuote?: boolean;
}

/**
 * Result of CSV generation.
 */
export interface CSVGenerateResult {
  /** Whether generation was successful */
  success: boolean;
  /** Generated CSV content (if successful) */
  content?: string;
  /** Blob of CSV file (if successful) */
  blob?: Blob;
  /** Error message (if failed) */
  error?: string;
  /** Number of rows exported (excluding header) */
  rowCount?: number;
  /** Number of columns exported */
  columnCount?: number;
}

/**
 * Result of CSV download operation.
 */
export interface CSVDownloadResult {
  /** Whether download was initiated successfully */
  success: boolean;
  /** Filename used for download */
  filename?: string;
  /** File size in bytes */
  sizeBytes?: number;
  /** Error message (if failed) */
  error?: string;
}

// =============================================================================
// Constants
// =============================================================================

/** Default CSV generation options */
const DEFAULT_OPTIONS: Required<CSVGenerateOptions> = {
  delimiter: ',',
  lineEnding: '\r\n',
  includeHeaders: true,
  dateFormat: 'ISO',
  decimalPrecision: 2,
  addBOM: true,
  nullValue: '',
  undefinedValue: '',
  trueValue: 'true',
  falseValue: 'false',
  quoteChar: '"',
  alwaysQuote: false,
};

/** UTF-8 BOM for Excel compatibility */
const UTF8_BOM = '\uFEFF';

/** MIME type for CSV files */
const CSV_MIME_TYPE = 'text/csv;charset=utf-8';

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Gets a nested value from an object using dot notation path.
 *
 * @param obj - Source object
 * @param path - Dot-notation path (e.g., 'user.profile.name')
 * @returns The value at the path, or undefined if not found
 *
 * @example
 * ```ts
 * const data = { user: { profile: { name: 'John' } } };
 * getNestedValue(data, 'user.profile.name'); // 'John'
 * getNestedValue(data, 'user.age'); // undefined
 * ```
 */
function getNestedValue(obj: unknown, path: string): unknown {
  if (obj === null || obj === undefined) {
    return undefined;
  }

  const keys = path.split('.');
  let current: unknown = obj;

  for (const key of keys) {
    if (current === null || current === undefined) {
      return undefined;
    }

    if (typeof current === 'object' && key in current) {
      current = (current as Record<string, unknown>)[key];
    } else {
      return undefined;
    }
  }

  return current;
}

/**
 * Formats a date value according to the specified format.
 *
 * @param value - Date value (Date, string, or number)
 * @param format - Date format to use
 * @returns Formatted date string
 */
function formatDate(value: unknown, format: CSVDateFormat): string {
  if (value === null || value === undefined) {
    return '';
  }

  let date: Date;

  if (value instanceof Date) {
    date = value;
  } else if (typeof value === 'string' || typeof value === 'number') {
    date = new Date(value);
    if (isNaN(date.getTime())) {
      return String(value);
    }
  } else {
    return String(value);
  }

  switch (format) {
    case 'ISO':
      return date.toISOString();

    case 'US':
      // MM/DD/YYYY HH:mm:ss
      return `${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')}/${date.getFullYear()} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;

    case 'EU':
      // DD/MM/YYYY HH:mm:ss
      return `${String(date.getDate()).padStart(2, '0')}/${String(date.getMonth() + 1).padStart(2, '0')}/${date.getFullYear()} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;

    case 'TIMESTAMP':
      return String(date.getTime());

    default:
      return date.toISOString();
  }
}

/**
 * Formats a number value with specified precision.
 *
 * @param value - Number value
 * @param precision - Decimal places
 * @returns Formatted number string
 */
function formatNumber(value: unknown, precision: number): string {
  if (value === null || value === undefined) {
    return '';
  }

  const num = typeof value === 'number' ? value : parseFloat(String(value));

  if (isNaN(num)) {
    return String(value);
  }

  return num.toFixed(precision);
}

/**
 * Formats a percentage value.
 *
 * @param value - Number value (0-100 or 0-1)
 * @param precision - Decimal places
 * @returns Formatted percentage string
 */
function formatPercentage(value: unknown, precision: number): string {
  if (value === null || value === undefined) {
    return '';
  }

  const num = typeof value === 'number' ? value : parseFloat(String(value));

  if (isNaN(num)) {
    return String(value);
  }

  // Assume values > 1 are already in percentage form
  const percentage = num > 1 ? num : num * 100;
  return `${percentage.toFixed(precision)}%`;
}

/**
 * Formats a currency value.
 *
 * @param value - Number value in cents
 * @param precision - Decimal places
 * @param currencyCode - Currency code
 * @returns Formatted currency string
 */
function formatCurrency(
  value: unknown,
  precision: number,
  currencyCode: string
): string {
  if (value === null || value === undefined) {
    return '';
  }

  const num = typeof value === 'number' ? value : parseFloat(String(value));

  if (isNaN(num)) {
    return String(value);
  }

  // Assume value is in cents, convert to dollars
  const dollars = num / 100;
  return `${currencyCode} ${dollars.toFixed(precision)}`;
}

/**
 * Escapes a CSV cell value according to RFC 4180.
 *
 * Handles special characters, quotes, and newlines.
 *
 * @param value - Value to escape
 * @param options - CSV options
 * @returns Escaped value safe for CSV
 */
function escapeCSVValue(
  value: string,
  options: Required<CSVGenerateOptions>
): string {
  const { delimiter, quoteChar, alwaysQuote } = options;

  // Check if quoting is needed
  const needsQuoting =
    alwaysQuote ||
    value.includes(delimiter) ||
    value.includes(quoteChar) ||
    value.includes('\n') ||
    value.includes('\r');

  if (!needsQuoting) {
    return value;
  }

  // Escape quote characters by doubling them
  const escaped = value.replace(
    new RegExp(quoteChar, 'g'),
    quoteChar + quoteChar
  );

  return `${quoteChar}${escaped}${quoteChar}`;
}

/**
 * Converts a value to a string for CSV export.
 *
 * @param value - Value to convert
 * @param options - CSV options
 * @param column - Optional column definition for type-specific formatting
 * @param item - Original data item for custom formatters
 * @returns String representation of the value
 */
function valueToString<T>(
  value: unknown,
  options: Required<CSVGenerateOptions>,
  column?: CSVColumn<T>,
  item?: T
): string {
  // Use custom formatter if provided
  if (column?.formatter && item !== undefined) {
    return String(column.formatter(value, item));
  }

  // Handle null and undefined
  if (value === null) {
    return options.nullValue;
  }
  if (value === undefined) {
    return options.undefinedValue;
  }

  // Type-specific formatting based on column type
  if (column?.type) {
    const precision = column.precision ?? options.decimalPrecision;
    const dateFormat = column.dateFormat ?? options.dateFormat;
    const currencyCode = column.currencyCode ?? 'USD';

    switch (column.type) {
      case 'number':
        return formatNumber(value, precision);

      case 'date':
        return formatDate(value, dateFormat);

      case 'boolean':
        return value ? options.trueValue : options.falseValue;

      case 'percentage':
        return formatPercentage(value, precision);

      case 'currency':
        return formatCurrency(value, precision, currencyCode);

      case 'string':
      default:
        break;
    }
  }

  // Handle arrays
  if (Array.isArray(value)) {
    return value.join('; ');
  }

  // Handle objects
  if (typeof value === 'object' && value !== null) {
    try {
      return JSON.stringify(value);
    } catch {
      return '[Object]';
    }
  }

  // Handle booleans
  if (typeof value === 'boolean') {
    return value ? options.trueValue : options.falseValue;
  }

  // Handle dates (auto-detect)
  if (value instanceof Date) {
    return formatDate(value, options.dateFormat);
  }

  // Default to string conversion
  return String(value);
}

/**
 * Gets the value from a data item based on column accessor.
 *
 * @param item - Data item
 * @param column - Column definition
 * @returns Extracted value
 */
function getColumnValue<T>(item: T, column: CSVColumn<T>): unknown {
  if (typeof column.accessor === 'function') {
    return column.accessor(item);
  }
  return getNestedValue(item, column.accessor);
}

// =============================================================================
// Main Export Functions
// =============================================================================

/**
 * Generates a CSV string from data with column definitions.
 *
 * Supports:
 * - Nested object access via dot notation
 * - Custom formatters per column
 * - Type-specific formatting (numbers, dates, percentages, currency)
 * - Proper escaping of special characters
 * - Excel-compatible UTF-8 BOM
 *
 * @param data - Array of data items to export
 * @param columns - Column definitions specifying headers and data access
 * @param options - CSV generation options
 * @returns Result containing CSV content or error
 *
 * @example
 * ```tsx
 * const data = [
 *   { id: 1, name: 'Campaign A', stats: { successRate: 0.85 }, created: '2026-01-10' },
 *   { id: 2, name: 'Campaign B', stats: { successRate: 0.72 }, created: '2026-01-11' },
 * ];
 *
 * const columns: CSVColumn[] = [
 *   { header: 'ID', accessor: 'id' },
 *   { header: 'Name', accessor: 'name' },
 *   { header: 'Success Rate', accessor: 'stats.successRate', type: 'percentage' },
 *   { header: 'Created', accessor: 'created', type: 'date', dateFormat: 'US' },
 * ];
 *
 * const result = generateCSV(data, columns);
 * if (result.success) {
 *   console.log(result.content);
 * }
 * ```
 */
export function generateCSV<T>(
  data: T[],
  columns: CSVColumn<T>[],
  options: CSVGenerateOptions = {}
): CSVGenerateResult {
  const mergedOptions: Required<CSVGenerateOptions> = {
    ...DEFAULT_OPTIONS,
    ...options,
  };

  try {
    // Filter out excluded columns
    const activeColumns = columns.filter((col) => col.include !== false);

    if (activeColumns.length === 0) {
      return {
        success: false,
        error: 'No columns specified for export',
      };
    }

    const rows: string[] = [];

    // Generate header row
    if (mergedOptions.includeHeaders) {
      const headerRow = activeColumns
        .map((col) => escapeCSVValue(col.header, mergedOptions))
        .join(mergedOptions.delimiter);
      rows.push(headerRow);
    }

    // Generate data rows
    for (const item of data) {
      const rowValues = activeColumns.map((column) => {
        const value = getColumnValue(item, column);
        const stringValue = valueToString(value, mergedOptions, column, item);
        return escapeCSVValue(stringValue, mergedOptions);
      });
      rows.push(rowValues.join(mergedOptions.delimiter));
    }

    // Join rows
    let content = rows.join(mergedOptions.lineEnding);

    // Add BOM for Excel UTF-8 compatibility
    if (mergedOptions.addBOM) {
      content = UTF8_BOM + content;
    }

    // Create blob
    const blob = new Blob([content], { type: CSV_MIME_TYPE });

    return {
      success: true,
      content,
      blob,
      rowCount: data.length,
      columnCount: activeColumns.length,
    };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : 'Unknown error during CSV generation',
    };
  }
}

/**
 * Generates CSV from data using automatic column detection.
 *
 * Automatically creates columns from object keys. Useful for quick exports
 * when column customization is not needed.
 *
 * @param data - Array of data items to export
 * @param options - CSV generation options
 * @returns Result containing CSV content or error
 *
 * @example
 * ```tsx
 * const data = [
 *   { id: 1, name: 'Test', value: 100 },
 *   { id: 2, name: 'Demo', value: 200 },
 * ];
 *
 * const result = generateCSVAuto(data);
 * ```
 */
export function generateCSVAuto<T extends Record<string, unknown>>(
  data: T[],
  options: CSVGenerateOptions = {}
): CSVGenerateResult {
  if (!data || data.length === 0) {
    return {
      success: true,
      content: options.addBOM !== false ? UTF8_BOM : '',
      blob: new Blob([''], { type: CSV_MIME_TYPE }),
      rowCount: 0,
      columnCount: 0,
    };
  }

  // Collect all unique keys from all objects
  const allKeys = new Set<string>();
  for (const item of data) {
    if (item && typeof item === 'object') {
      Object.keys(item).forEach((key) => allKeys.add(key));
    }
  }

  // Create column definitions from keys
  const columns: CSVColumn<T>[] = Array.from(allKeys).map((key) => ({
    header: key,
    accessor: key,
  }));

  return generateCSV(data, columns, options);
}

/**
 * Downloads CSV data as a file.
 *
 * Takes either a Blob, CSV string, or data + columns and triggers a download.
 *
 * @param data - CSV Blob, string, or array of data items
 * @param filename - Filename for the download (should include .csv extension)
 * @param columns - Column definitions (required if data is an array)
 * @param options - CSV generation options (used if generating from data array)
 * @returns Promise resolving to download result
 *
 * @example
 * ```tsx
 * // Download from blob
 * const result = generateCSV(data, columns);
 * if (result.success && result.blob) {
 *   await downloadCSV(result.blob, 'campaign-export.csv');
 * }
 *
 * // Download directly from data
 * await downloadCSV(data, 'campaign-export.csv', columns);
 * ```
 */
export async function downloadCSV<T>(
  data: Blob | string | T[],
  filename: string,
  columns?: CSVColumn<T>[],
  options?: CSVGenerateOptions
): Promise<CSVDownloadResult> {
  try {
    let blob: Blob;

    // Handle different data types
    if (data instanceof Blob) {
      blob = data;
    } else if (typeof data === 'string') {
      // Ensure BOM is present for string data
      const content =
        options?.addBOM !== false && !data.startsWith(UTF8_BOM)
          ? UTF8_BOM + data
          : data;
      blob = new Blob([content], { type: CSV_MIME_TYPE });
    } else if (Array.isArray(data)) {
      if (!columns) {
        // Auto-detect columns
        const result = generateCSVAuto(
          data as unknown as Record<string, unknown>[],
          options
        );
        if (!result.success || !result.blob) {
          return {
            success: false,
            error: result.error || 'Failed to generate CSV',
          };
        }
        blob = result.blob;
      } else {
        const result = generateCSV(data, columns, options);
        if (!result.success || !result.blob) {
          return {
            success: false,
            error: result.error || 'Failed to generate CSV',
          };
        }
        blob = result.blob;
      }
    } else {
      return {
        success: false,
        error: 'Invalid data type. Expected Blob, string, or array.',
      };
    }

    // Ensure filename has .csv extension
    const finalFilename = filename.endsWith('.csv')
      ? filename
      : `${filename}.csv`;

    // Create object URL for the blob
    const url = URL.createObjectURL(blob);

    // Create a temporary anchor element
    const link = document.createElement('a');
    link.href = url;
    link.download = finalFilename;

    // Append to body (required for Firefox)
    document.body.appendChild(link);

    // Trigger download
    link.click();

    // Cleanup
    document.body.removeChild(link);

    // Revoke object URL after a small delay to ensure download starts
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 100);

    return {
      success: true,
      filename: finalFilename,
      sizeBytes: blob.size,
    };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : 'Unknown error during CSV download',
    };
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Generates and immediately downloads a CSV file.
 *
 * Combines generation and download into a single operation.
 *
 * @param data - Array of data items to export
 * @param filename - Filename for the download
 * @param columns - Column definitions
 * @param options - CSV generation options
 * @returns Promise resolving to combined result
 *
 * @example
 * ```tsx
 * await exportAndDownloadCSV(
 *   telemetryEvents,
 *   'campaign-telemetry-export',
 *   telemetryColumns
 * );
 * ```
 */
export async function exportAndDownloadCSV<T>(
  data: T[],
  filename: string,
  columns: CSVColumn<T>[],
  options?: CSVGenerateOptions
): Promise<CSVGenerateResult & CSVDownloadResult> {
  // Generate CSV
  const generateResult = generateCSV(data, columns, options);

  if (!generateResult.success || !generateResult.blob) {
    return {
      ...generateResult,
      success: false,
    };
  }

  // Download CSV
  const downloadResult = await downloadCSV(generateResult.blob, filename);

  return {
    ...generateResult,
    ...downloadResult,
    success: generateResult.success && downloadResult.success,
  };
}

/**
 * Generates a filename for CSV export with timestamp.
 *
 * @param baseName - Base name for the file
 * @param includeTimestamp - Whether to include timestamp (default: true)
 * @returns Generated filename with .csv extension
 *
 * @example
 * ```tsx
 * const filename = generateCSVFilename('campaign-export');
 * // Returns: "campaign-export-2026-01-11T15-30-00.csv"
 * ```
 */
export function generateCSVFilename(
  baseName: string,
  includeTimestamp: boolean = true
): string {
  // Sanitize base name (remove unsafe characters)
  const sanitized = baseName
    .toLowerCase()
    .replace(/[^a-z0-9-_]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');

  if (includeTimestamp) {
    const timestamp = new Date()
      .toISOString()
      .replace(/:/g, '-')
      .replace(/\..+/, '');
    return `${sanitized}-${timestamp}.csv`;
  }

  return `${sanitized}.csv`;
}

/**
 * Checks if CSV export is supported in the current browser.
 *
 * @returns Object indicating support for CSV export features
 */
export function checkCSVExportSupport(): {
  blob: boolean;
  download: boolean;
  overall: boolean;
} {
  const blob = typeof Blob !== 'undefined';
  const download =
    typeof document !== 'undefined' &&
    'download' in document.createElement('a');

  return {
    blob,
    download,
    overall: blob && download,
  };
}

// =============================================================================
// Pre-built Column Definitions for Campaign Analytics
// =============================================================================

/**
 * Pre-built column definitions for common campaign analytics exports.
 *
 * These can be used directly or as templates for custom exports.
 */
export const CAMPAIGN_CSV_COLUMNS = {
  /**
   * Columns for campaign summary export.
   */
  campaignSummary: [
    { header: 'Campaign ID', accessor: 'id' },
    { header: 'Name', accessor: 'name' },
    { header: 'Objective', accessor: 'objective' },
    { header: 'Status', accessor: 'status' },
    { header: 'Provider', accessor: 'target_provider' },
    { header: 'Model', accessor: 'target_model' },
    {
      header: 'Techniques',
      accessor: 'technique_suites',
      formatter: (v: unknown) => (Array.isArray(v) ? v.join(', ') : String(v)),
    },
    { header: 'Total Attempts', accessor: 'total_attempts', type: 'number' as const },
    {
      header: 'Success Rate',
      accessor: 'success_rate',
      type: 'percentage' as const,
      precision: 2,
    },
    {
      header: 'Avg Latency (ms)',
      accessor: 'avg_latency_ms',
      type: 'number' as const,
      precision: 0,
    },
    {
      header: 'Duration (s)',
      accessor: 'duration_seconds',
      type: 'number' as const,
      precision: 0,
    },
    {
      header: 'Tags',
      accessor: 'tags',
      formatter: (v: unknown) => (Array.isArray(v) ? v.join(', ') : String(v)),
    },
    {
      header: 'Created At',
      accessor: 'created_at',
      type: 'date' as const,
      dateFormat: 'ISO' as const,
    },
    {
      header: 'Completed At',
      accessor: 'completed_at',
      type: 'date' as const,
      dateFormat: 'ISO' as const,
    },
  ] as CSVColumn[],

  /**
   * Columns for telemetry event export.
   */
  telemetryEvent: [
    { header: 'Event ID', accessor: 'id' },
    { header: 'Campaign ID', accessor: 'campaign_id' },
    { header: 'Sequence #', accessor: 'sequence_number', type: 'number' as const },
    { header: 'Technique', accessor: 'technique_suite' },
    { header: 'Potency Level', accessor: 'potency_level', type: 'number' as const },
    { header: 'Provider', accessor: 'provider' },
    { header: 'Model', accessor: 'model' },
    { header: 'Status', accessor: 'status' },
    { header: 'Success', accessor: 'success_indicator', type: 'boolean' as const },
    {
      header: 'Latency (ms)',
      accessor: 'total_latency_ms',
      type: 'number' as const,
      precision: 0,
    },
    { header: 'Total Tokens', accessor: 'total_tokens', type: 'number' as const },
    { header: 'Prompt Tokens', accessor: 'prompt_tokens', type: 'number' as const },
    {
      header: 'Completion Tokens',
      accessor: 'completion_tokens',
      type: 'number' as const,
    },
    {
      header: 'Semantic Success',
      accessor: 'semantic_success_score',
      type: 'percentage' as const,
    },
    {
      header: 'Effectiveness',
      accessor: 'effectiveness_score',
      type: 'percentage' as const,
    },
    { header: 'Original Prompt', accessor: 'original_prompt' },
    { header: 'Transformed Prompt', accessor: 'transformed_prompt' },
    { header: 'Response', accessor: 'response_text' },
    { header: 'Error', accessor: 'error_message' },
    {
      header: 'Created At',
      accessor: 'created_at',
      type: 'date' as const,
      dateFormat: 'ISO' as const,
    },
  ] as CSVColumn[],

  /**
   * Columns for technique breakdown export.
   */
  techniqueBreakdown: [
    { header: 'Technique', accessor: 'name' },
    { header: 'Attempts', accessor: 'attempts', type: 'number' as const },
    { header: 'Successes', accessor: 'successes', type: 'number' as const },
    {
      header: 'Success Rate',
      accessor: 'success_rate',
      type: 'percentage' as const,
      precision: 2,
    },
    {
      header: 'Avg Latency (ms)',
      accessor: 'avg_latency_ms',
      type: 'number' as const,
      precision: 0,
    },
    { header: 'Avg Tokens', accessor: 'avg_tokens', type: 'number' as const, precision: 0 },
    {
      header: 'Total Cost',
      accessor: 'total_cost_cents',
      type: 'currency' as const,
      precision: 2,
    },
  ] as CSVColumn[],

  /**
   * Columns for provider breakdown export.
   */
  providerBreakdown: [
    { header: 'Provider', accessor: 'name' },
    { header: 'Attempts', accessor: 'attempts', type: 'number' as const },
    { header: 'Successes', accessor: 'successes', type: 'number' as const },
    {
      header: 'Success Rate',
      accessor: 'success_rate',
      type: 'percentage' as const,
      precision: 2,
    },
    {
      header: 'Avg Latency (ms)',
      accessor: 'avg_latency_ms',
      type: 'number' as const,
      precision: 0,
    },
    { header: 'Avg Tokens', accessor: 'avg_tokens', type: 'number' as const, precision: 0 },
    {
      header: 'Total Cost',
      accessor: 'total_cost_cents',
      type: 'currency' as const,
      precision: 2,
    },
  ] as CSVColumn[],

  /**
   * Columns for time series export.
   */
  timeSeries: [
    {
      header: 'Timestamp',
      accessor: 'timestamp',
      type: 'date' as const,
      dateFormat: 'ISO' as const,
    },
    { header: 'Value', accessor: 'value', type: 'number' as const, precision: 4 },
    { header: 'Count', accessor: 'count', type: 'number' as const },
  ] as CSVColumn[],

  /**
   * Columns for campaign comparison export.
   */
  campaignComparison: [
    { header: 'Campaign ID', accessor: 'campaign_id' },
    { header: 'Campaign Name', accessor: 'campaign_name' },
    { header: 'Status', accessor: 'status' },
    { header: 'Total Attempts', accessor: 'total_attempts', type: 'number' as const },
    {
      header: 'Success Rate',
      accessor: 'success_rate',
      type: 'percentage' as const,
      precision: 2,
    },
    {
      header: 'Semantic Success',
      accessor: 'semantic_success_mean',
      type: 'percentage' as const,
    },
    {
      header: 'Latency Mean (ms)',
      accessor: 'latency_mean',
      type: 'number' as const,
      precision: 0,
    },
    {
      header: 'Latency P95 (ms)',
      accessor: 'latency_p95',
      type: 'number' as const,
      precision: 0,
    },
    { header: 'Avg Tokens', accessor: 'avg_tokens', type: 'number' as const, precision: 0 },
    { header: 'Total Tokens', accessor: 'total_tokens', type: 'number' as const },
    {
      header: 'Total Cost',
      accessor: 'total_cost_cents',
      type: 'currency' as const,
      precision: 2,
    },
    {
      header: 'Avg Cost/Attempt',
      accessor: 'avg_cost_per_attempt',
      type: 'currency' as const,
      precision: 4,
    },
    {
      header: 'Duration (s)',
      accessor: 'duration_seconds',
      type: 'number' as const,
      precision: 0,
    },
    { header: 'Best Technique', accessor: 'best_technique' },
    { header: 'Best Provider', accessor: 'best_provider' },
    {
      header: 'Normalized Success',
      accessor: 'normalized_success_rate',
      type: 'number' as const,
      precision: 4,
    },
    {
      header: 'Normalized Latency',
      accessor: 'normalized_latency',
      type: 'number' as const,
      precision: 4,
    },
    {
      header: 'Normalized Cost',
      accessor: 'normalized_cost',
      type: 'number' as const,
      precision: 4,
    },
    {
      header: 'Normalized Effectiveness',
      accessor: 'normalized_effectiveness',
      type: 'number' as const,
      precision: 4,
    },
  ] as CSVColumn[],
};

/**
 * Type alias for pre-built column sets.
 */
export type CampaignCSVColumnSet = keyof typeof CAMPAIGN_CSV_COLUMNS;
