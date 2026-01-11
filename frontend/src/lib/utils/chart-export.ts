/**
 * Chart Export Utilities for Campaign Analytics
 *
 * Provides utility functions for exporting Recharts components as PNG, SVG,
 * and downloading files. Handles Recharts ref extraction and canvas conversion
 * for research publication exports.
 *
 * @module lib/utils/chart-export
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Supported export formats for charts.
 */
export type ChartExportFormat = 'png' | 'svg';

/**
 * Configuration options for PNG export.
 */
export interface PNGExportOptions {
  /** Scale factor for higher resolution (default: 2 for retina) */
  scale?: number;
  /** Background color for the exported image (default: white) */
  backgroundColor?: string;
  /** Image quality for JPEG/WebP (0-1, not used for PNG) */
  quality?: number;
  /** Padding around the chart in pixels (default: 20) */
  padding?: number;
  /** Custom width override */
  width?: number;
  /** Custom height override */
  height?: number;
}

/**
 * Configuration options for SVG export.
 */
export interface SVGExportOptions {
  /** Whether to inline styles (default: true) */
  inlineStyles?: boolean;
  /** Custom CSS to inject into the SVG */
  customCSS?: string;
  /** Whether to optimize the SVG output (default: false) */
  optimize?: boolean;
  /** Padding around the chart in pixels (default: 20) */
  padding?: number;
}

/**
 * Result of a chart export operation.
 */
export interface ChartExportResult {
  /** Whether the export was successful */
  success: boolean;
  /** Blob of the exported file (if successful) */
  blob?: Blob;
  /** Data URL of the exported file (if successful) */
  dataUrl?: string;
  /** Error message (if failed) */
  error?: string;
  /** File format */
  format: ChartExportFormat;
  /** Dimensions of the exported chart */
  dimensions?: {
    width: number;
    height: number;
  };
}

/**
 * Reference to a chart element for export.
 */
export interface ChartRef {
  /** DOM element containing the chart */
  current: HTMLElement | null;
}

// =============================================================================
// Constants
// =============================================================================

/** Default PNG export options */
const DEFAULT_PNG_OPTIONS: Required<PNGExportOptions> = {
  scale: 2,
  backgroundColor: '#ffffff',
  quality: 1,
  padding: 20,
  width: 0,
  height: 0,
};

/** Default SVG export options */
const DEFAULT_SVG_OPTIONS: Required<SVGExportOptions> = {
  inlineStyles: true,
  customCSS: '',
  optimize: false,
  padding: 20,
};

/** MIME types for export formats */
const MIME_TYPES: Record<ChartExportFormat, string> = {
  png: 'image/png',
  svg: 'image/svg+xml',
};

/** File extensions for export formats */
const FILE_EXTENSIONS: Record<ChartExportFormat, string> = {
  png: '.png',
  svg: '.svg',
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Extracts the SVG element from a Recharts container.
 *
 * Recharts wraps its charts in a ResponsiveContainer which contains an SVG.
 * This function finds and extracts that SVG element.
 *
 * @param container - The container element holding the Recharts chart
 * @returns The SVG element or null if not found
 */
function extractSVGFromRecharts(container: HTMLElement): SVGSVGElement | null {
  // Recharts renders SVG elements inside the container
  // The SVG is usually a direct child or nested within wrapper divs
  const svg = container.querySelector('svg');

  if (svg && svg instanceof SVGSVGElement) {
    return svg;
  }

  return null;
}

/**
 * Clones an SVG element with all computed styles inlined.
 *
 * @param svg - The original SVG element
 * @param options - SVG export options
 * @returns Cloned SVG with inline styles
 */
function cloneSVGWithStyles(
  svg: SVGSVGElement,
  options: Required<SVGExportOptions>
): SVGSVGElement {
  // Deep clone the SVG
  const clone = svg.cloneNode(true) as SVGSVGElement;

  if (options.inlineStyles) {
    inlineStyles(clone, svg);
  }

  // Add custom CSS if provided
  if (options.customCSS) {
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = options.customCSS;
    clone.insertBefore(style, clone.firstChild);
  }

  // Ensure viewBox is set for proper scaling
  if (!clone.getAttribute('viewBox')) {
    const width = svg.clientWidth || parseFloat(svg.getAttribute('width') || '0');
    const height = svg.clientHeight || parseFloat(svg.getAttribute('height') || '0');
    if (width && height) {
      clone.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }
  }

  // Set explicit dimensions
  const bbox = svg.getBBox();
  const padding = options.padding;
  clone.setAttribute('width', String(bbox.width + padding * 2));
  clone.setAttribute('height', String(bbox.height + padding * 2));

  return clone;
}

/**
 * Recursively inlines computed styles onto SVG elements.
 *
 * @param clone - The cloned element to apply styles to
 * @param original - The original element to extract computed styles from
 */
function inlineStyles(clone: Element, original: Element): void {
  // Get computed styles from the original element
  const computedStyle = window.getComputedStyle(original);

  // Properties relevant for SVG styling
  const svgStyleProperties = [
    'fill',
    'stroke',
    'stroke-width',
    'stroke-dasharray',
    'stroke-linecap',
    'stroke-linejoin',
    'opacity',
    'fill-opacity',
    'stroke-opacity',
    'font-family',
    'font-size',
    'font-weight',
    'font-style',
    'text-anchor',
    'dominant-baseline',
    'visibility',
    'display',
    'color',
  ];

  // Apply inline styles
  let styleString = '';
  for (const prop of svgStyleProperties) {
    const value = computedStyle.getPropertyValue(prop);
    if (value && value !== 'none') {
      styleString += `${prop}:${value};`;
    }
  }

  if (styleString && clone instanceof SVGElement) {
    const existingStyle = clone.getAttribute('style') || '';
    clone.setAttribute('style', existingStyle + styleString);
  }

  // Recursively process children
  const cloneChildren = clone.children;
  const originalChildren = original.children;

  for (let i = 0; i < cloneChildren.length && i < originalChildren.length; i++) {
    inlineStyles(cloneChildren[i], originalChildren[i]);
  }
}

/**
 * Converts an SVG element to a canvas for PNG export.
 *
 * @param svg - The SVG element to convert
 * @param options - PNG export options
 * @returns Promise resolving to a canvas element
 */
async function svgToCanvas(
  svg: SVGSVGElement,
  options: Required<PNGExportOptions>
): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    // Get dimensions from the SVG
    const svgWidth = svg.clientWidth || parseFloat(svg.getAttribute('width') || '0');
    const svgHeight = svg.clientHeight || parseFloat(svg.getAttribute('height') || '0');

    if (!svgWidth || !svgHeight) {
      reject(new Error('Could not determine SVG dimensions'));
      return;
    }

    // Calculate canvas dimensions with scale and padding
    const width = (options.width || svgWidth + options.padding * 2) * options.scale;
    const height = (options.height || svgHeight + options.padding * 2) * options.scale;

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      reject(new Error('Could not create canvas context'));
      return;
    }

    // Fill background
    ctx.fillStyle = options.backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Clone SVG and inline styles for export
    const clonedSVG = cloneSVGWithStyles(svg, {
      ...DEFAULT_SVG_OPTIONS,
      padding: options.padding,
    });

    // Serialize SVG to string
    const svgData = new XMLSerializer().serializeToString(clonedSVG);
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const svgUrl = URL.createObjectURL(svgBlob);

    // Create image and draw to canvas
    const img = new Image();
    img.onload = () => {
      // Scale and center the image with padding
      ctx.scale(options.scale, options.scale);
      ctx.drawImage(img, options.padding, options.padding);
      URL.revokeObjectURL(svgUrl);
      resolve(canvas);
    };
    img.onerror = (error) => {
      URL.revokeObjectURL(svgUrl);
      reject(new Error(`Failed to load SVG image: ${error}`));
    };
    img.src = svgUrl;
  });
}

// =============================================================================
// Main Export Functions
// =============================================================================

/**
 * Exports a chart as a PNG image.
 *
 * Extracts the SVG from a Recharts container, converts it to a canvas,
 * and generates a PNG blob.
 *
 * @param chartRef - React ref to the chart container element
 * @param options - PNG export options
 * @returns Promise resolving to export result
 *
 * @example
 * ```tsx
 * const chartRef = useRef<HTMLDivElement>(null);
 *
 * const handleExport = async () => {
 *   const result = await exportChartAsPNG(chartRef, { scale: 3 });
 *   if (result.success && result.blob) {
 *     downloadFile(result.blob, 'campaign-chart.png');
 *   }
 * };
 * ```
 */
export async function exportChartAsPNG(
  chartRef: ChartRef,
  options: PNGExportOptions = {}
): Promise<ChartExportResult> {
  const mergedOptions: Required<PNGExportOptions> = {
    ...DEFAULT_PNG_OPTIONS,
    ...options,
  };

  try {
    // Validate ref
    if (!chartRef.current) {
      return {
        success: false,
        error: 'Chart reference is null or undefined',
        format: 'png',
      };
    }

    // Extract SVG from Recharts container
    const svg = extractSVGFromRecharts(chartRef.current);
    if (!svg) {
      return {
        success: false,
        error: 'Could not find SVG element in chart container',
        format: 'png',
      };
    }

    // Convert SVG to canvas
    const canvas = await svgToCanvas(svg, mergedOptions);

    // Convert canvas to blob
    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(
        (b) => resolve(b),
        MIME_TYPES.png,
        mergedOptions.quality
      );
    });

    if (!blob) {
      return {
        success: false,
        error: 'Failed to create PNG blob from canvas',
        format: 'png',
      };
    }

    // Generate data URL
    const dataUrl = canvas.toDataURL(MIME_TYPES.png, mergedOptions.quality);

    return {
      success: true,
      blob,
      dataUrl,
      format: 'png',
      dimensions: {
        width: canvas.width,
        height: canvas.height,
      },
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error during PNG export',
      format: 'png',
    };
  }
}

/**
 * Exports a chart as an SVG file.
 *
 * Extracts the SVG from a Recharts container, clones it with inline styles,
 * and generates an SVG blob.
 *
 * @param chartRef - React ref to the chart container element
 * @param options - SVG export options
 * @returns Promise resolving to export result
 *
 * @example
 * ```tsx
 * const chartRef = useRef<HTMLDivElement>(null);
 *
 * const handleExport = async () => {
 *   const result = await exportChartAsSVG(chartRef, { inlineStyles: true });
 *   if (result.success && result.blob) {
 *     downloadFile(result.blob, 'campaign-chart.svg');
 *   }
 * };
 * ```
 */
export async function exportChartAsSVG(
  chartRef: ChartRef,
  options: SVGExportOptions = {}
): Promise<ChartExportResult> {
  const mergedOptions: Required<SVGExportOptions> = {
    ...DEFAULT_SVG_OPTIONS,
    ...options,
  };

  try {
    // Validate ref
    if (!chartRef.current) {
      return {
        success: false,
        error: 'Chart reference is null or undefined',
        format: 'svg',
      };
    }

    // Extract SVG from Recharts container
    const svg = extractSVGFromRecharts(chartRef.current);
    if (!svg) {
      return {
        success: false,
        error: 'Could not find SVG element in chart container',
        format: 'svg',
      };
    }

    // Clone SVG with inline styles
    const clonedSVG = cloneSVGWithStyles(svg, mergedOptions);

    // Add XML namespace for standalone SVG
    clonedSVG.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clonedSVG.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');

    // Serialize to string
    const serializer = new XMLSerializer();
    let svgString = serializer.serializeToString(clonedSVG);

    // Add XML declaration for standalone file
    svgString = '<?xml version="1.0" encoding="UTF-8"?>\n' + svgString;

    // Create blob
    const blob = new Blob([svgString], { type: MIME_TYPES.svg });

    // Create data URL
    const dataUrl = `data:${MIME_TYPES.svg};base64,${btoa(unescape(encodeURIComponent(svgString)))}`;

    // Get dimensions
    const width = parseFloat(clonedSVG.getAttribute('width') || '0');
    const height = parseFloat(clonedSVG.getAttribute('height') || '0');

    return {
      success: true,
      blob,
      dataUrl,
      format: 'svg',
      dimensions: {
        width,
        height,
      },
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error during SVG export',
      format: 'svg',
    };
  }
}

/**
 * Downloads a Blob as a file to the user's device.
 *
 * Creates a temporary download link and triggers it programmatically.
 * Handles cleanup automatically.
 *
 * @param blob - The Blob to download
 * @param filename - The name for the downloaded file
 * @returns Promise resolving to true if download was initiated successfully
 *
 * @example
 * ```tsx
 * const result = await exportChartAsPNG(chartRef);
 * if (result.success && result.blob) {
 *   await downloadFile(result.blob, 'my-chart.png');
 * }
 * ```
 */
export async function downloadFile(blob: Blob, filename: string): Promise<boolean> {
  try {
    // Create object URL for the blob
    const url = URL.createObjectURL(blob);

    // Create a temporary anchor element
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;

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

    return true;
  } catch (error) {
    return false;
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Exports a chart and immediately triggers download.
 *
 * Combines export and download into a single operation.
 *
 * @param chartRef - React ref to the chart container element
 * @param filename - Filename without extension (extension will be added based on format)
 * @param format - Export format ('png' or 'svg')
 * @param options - Export options (PNG or SVG options based on format)
 * @returns Promise resolving to export result with download success status
 *
 * @example
 * ```tsx
 * await exportAndDownloadChart(chartRef, 'success-rate-analysis', 'png', { scale: 3 });
 * ```
 */
export async function exportAndDownloadChart(
  chartRef: ChartRef,
  filename: string,
  format: ChartExportFormat = 'png',
  options: PNGExportOptions | SVGExportOptions = {}
): Promise<ChartExportResult & { downloaded: boolean }> {
  // Export based on format
  const result = format === 'png'
    ? await exportChartAsPNG(chartRef, options as PNGExportOptions)
    : await exportChartAsSVG(chartRef, options as SVGExportOptions);

  // Download if successful
  let downloaded = false;
  if (result.success && result.blob) {
    const fullFilename = filename.endsWith(FILE_EXTENSIONS[format])
      ? filename
      : filename + FILE_EXTENSIONS[format];
    downloaded = await downloadFile(result.blob, fullFilename);
  }

  return {
    ...result,
    downloaded,
  };
}

/**
 * Exports multiple charts as separate files.
 *
 * Useful for bulk export of dashboard charts.
 *
 * @param charts - Array of chart references with metadata
 * @param format - Export format to use for all charts
 * @param options - Shared export options
 * @returns Promise resolving to array of export results
 *
 * @example
 * ```tsx
 * const charts = [
 *   { id: 'success-rate', name: 'Success Rate', ref: successRateRef },
 *   { id: 'latency', name: 'Latency Distribution', ref: latencyRef },
 * ];
 *
 * const results = await exportMultipleCharts(charts, 'png', { scale: 2 });
 * ```
 */
export async function exportMultipleCharts(
  charts: Array<{
    id: string;
    name: string;
    ref: ChartRef;
  }>,
  format: ChartExportFormat = 'png',
  options: PNGExportOptions | SVGExportOptions = {}
): Promise<Array<{ id: string; name: string; result: ChartExportResult }>> {
  const results: Array<{ id: string; name: string; result: ChartExportResult }> = [];

  for (const chart of charts) {
    const result = format === 'png'
      ? await exportChartAsPNG(chart.ref, options as PNGExportOptions)
      : await exportChartAsSVG(chart.ref, options as SVGExportOptions);

    results.push({
      id: chart.id,
      name: chart.name,
      result,
    });
  }

  return results;
}

/**
 * Generates a filename for chart export with timestamp.
 *
 * @param baseName - Base name for the file
 * @param format - Export format
 * @param includeTimestamp - Whether to include timestamp (default: true)
 * @returns Generated filename
 *
 * @example
 * ```tsx
 * const filename = generateChartFilename('campaign-analysis', 'png');
 * // Returns: "campaign-analysis-2026-01-11T15-30-00.png"
 * ```
 */
export function generateChartFilename(
  baseName: string,
  format: ChartExportFormat,
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
    return `${sanitized}-${timestamp}${FILE_EXTENSIONS[format]}`;
  }

  return `${sanitized}${FILE_EXTENSIONS[format]}`;
}

/**
 * Checks if chart export is supported in the current browser.
 *
 * @returns Object indicating support for various export features
 */
export function checkExportSupport(): {
  canvas: boolean;
  svg: boolean;
  blob: boolean;
  download: boolean;
  overall: boolean;
} {
  const canvas = typeof HTMLCanvasElement !== 'undefined' &&
    !!document.createElement('canvas').getContext;
  const svg = typeof SVGSVGElement !== 'undefined';
  const blob = typeof Blob !== 'undefined';
  const download = typeof document !== 'undefined' &&
    'download' in document.createElement('a');

  return {
    canvas,
    svg,
    blob,
    download,
    overall: canvas && svg && blob && download,
  };
}
