/**
 * Asset Optimization Utilities for Project Chimera
 * 
 * Provides comprehensive asset optimization including:
 * - Image optimization and responsive loading
 * - Font loading strategies
 * - Script and stylesheet optimization
 * - Resource prioritization
 * 
 * @module lib/optimization/asset-optimizer
 */

// ============================================================================
// Types
// ============================================================================

export interface ImageOptimizationConfig {
  /** Enable WebP format */
  webp?: boolean;
  /** Enable AVIF format */
  avif?: boolean;
  /** Quality (1-100) */
  quality?: number;
  /** Enable lazy loading */
  lazy?: boolean;
  /** Blur placeholder */
  placeholder?: 'blur' | 'empty' | 'color';
  /** Placeholder color (for color placeholder) */
  placeholderColor?: string;
  /** Responsive breakpoints */
  breakpoints?: number[];
  /** Device pixel ratios to support */
  devicePixelRatios?: number[];
}

export interface FontConfig {
  /** Font family name */
  family: string;
  /** Font weights to load */
  weights?: number[];
  /** Font styles to load */
  styles?: ('normal' | 'italic')[];
  /** Font display strategy */
  display?: 'auto' | 'block' | 'swap' | 'fallback' | 'optional';
  /** Preload critical fonts */
  preload?: boolean;
  /** Unicode ranges to subset */
  unicodeRange?: string;
}

export interface ScriptConfig {
  /** Script source URL */
  src: string;
  /** Loading strategy */
  strategy?: 'async' | 'defer' | 'module' | 'blocking';
  /** Preload hint */
  preload?: boolean;
  /** Integrity hash */
  integrity?: string;
  /** Cross-origin setting */
  crossOrigin?: 'anonymous' | 'use-credentials';
}

export interface StylesheetConfig {
  /** Stylesheet URL */
  href: string;
  /** Media query */
  media?: string;
  /** Critical CSS (inline) */
  critical?: boolean;
  /** Preload hint */
  preload?: boolean;
}

// ============================================================================
// Image Optimization
// ============================================================================

const DEFAULT_IMAGE_CONFIG: ImageOptimizationConfig = {
  webp: true,
  avif: true,
  quality: 80,
  lazy: true,
  placeholder: 'blur',
  breakpoints: [640, 768, 1024, 1280, 1536],
  devicePixelRatios: [1, 2],
};

/**
 * Generate responsive image srcset
 */
export function generateSrcSet(
  baseSrc: string,
  config: ImageOptimizationConfig = {}
): string {
  const { breakpoints, devicePixelRatios } = { ...DEFAULT_IMAGE_CONFIG, ...config };
  
  const srcSetEntries: string[] = [];
  
  for (const bp of breakpoints || []) {
    for (const dpr of devicePixelRatios || []) {
      const width = bp * dpr;
      const url = transformImageUrl(baseSrc, { width, quality: config.quality });
      srcSetEntries.push(`${url} ${width}w`);
    }
  }
  
  return srcSetEntries.join(', ');
}

/**
 * Generate responsive image sizes attribute
 */
export function generateSizes(
  breakpoints: { maxWidth: number; size: string }[],
  defaultSize: string
): string {
  const sizeEntries = breakpoints.map(
    ({ maxWidth, size }) => `(max-width: ${maxWidth}px) ${size}`
  );
  
  return [...sizeEntries, defaultSize].join(', ');
}

/**
 * Transform image URL with optimization parameters
 */
export function transformImageUrl(
  src: string,
  options: {
    width?: number;
    height?: number;
    quality?: number;
    format?: 'webp' | 'avif' | 'jpeg' | 'png';
  }
): string {
  // For Next.js Image Optimization API
  const params = new URLSearchParams();
  
  if (options.width) params.set('w', options.width.toString());
  if (options.height) params.set('h', options.height.toString());
  if (options.quality) params.set('q', options.quality.toString());
  
  // Check if it's an external URL or internal
  if (src.startsWith('http://') || src.startsWith('https://')) {
    params.set('url', src);
    return `/_next/image?${params.toString()}`;
  }
  
  return `/_next/image?url=${encodeURIComponent(src)}&${params.toString()}`;
}

/**
 * Generate blur placeholder data URL
 */
export function generateBlurPlaceholder(
  width: number = 10,
  height: number = 10,
  color: string = '#e5e7eb'
): string {
  // Simple SVG blur placeholder
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}">
      <filter id="b" color-interpolation-filters="sRGB">
        <feGaussianBlur stdDeviation="20"/>
      </filter>
      <rect width="100%" height="100%" fill="${color}" filter="url(#b)"/>
    </svg>
  `;
  
  return `data:image/svg+xml;base64,${btoa(svg.trim())}`;
}

/**
 * Check if browser supports modern image formats
 */
export function checkImageFormatSupport(): {
  webp: boolean;
  avif: boolean;
} {
  if (typeof window === 'undefined') {
    return { webp: true, avif: true }; // Assume support on server
  }

  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;

  return {
    webp: canvas.toDataURL('image/webp').startsWith('data:image/webp'),
    avif: canvas.toDataURL('image/avif').startsWith('data:image/avif'),
  };
}

/**
 * Preload critical images
 */
export function preloadImage(
  src: string,
  options: {
    as?: 'image';
    type?: string;
    media?: string;
    fetchPriority?: 'high' | 'low' | 'auto';
  } = {}
): void {
  if (typeof window === 'undefined') return;

  const existingLink = document.querySelector(`link[rel="preload"][href="${src}"]`);
  if (existingLink) return;

  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = options.as || 'image';
  link.href = src;
  
  if (options.type) link.type = options.type;
  if (options.media) link.media = options.media;
  if (options.fetchPriority) {
    link.setAttribute('fetchpriority', options.fetchPriority);
  }

  document.head.appendChild(link);
}

// ============================================================================
// Font Optimization
// ============================================================================

const FONT_DISPLAY_STRATEGIES = {
  // Invisible text for short period, then fallback
  swap: 'swap',
  // Invisible text for very short period, then fallback forever
  fallback: 'fallback',
  // Use fallback immediately, swap when ready
  optional: 'optional',
  // Block rendering until font loads
  block: 'block',
  // Browser decides
  auto: 'auto',
} as const;

/**
 * Load a web font with optimization
 */
export async function loadFont(config: FontConfig): Promise<void> {
  const {
    family,
    weights = [400],
    styles = ['normal'],
    display = 'swap',
    preload = false,
    unicodeRange,
  } = config;

  // Create @font-face rules
  const fontFaces: FontFace[] = [];

  for (const weight of weights) {
    for (const style of styles) {
      const fontFace = new FontFace(family, `local("${family}")`, {
        weight: weight.toString(),
        style,
        display: FONT_DISPLAY_STRATEGIES[display],
        unicodeRange,
      });

      fontFaces.push(fontFace);
    }
  }

  // Load all font faces
  await Promise.all(fontFaces.map((face) => face.load()));

  // Add to document fonts
  fontFaces.forEach((face) => document.fonts.add(face));

  // Add preload hint if requested
  if (preload) {
    preloadFont(family, weights[0], styles[0]);
  }
}

/**
 * Preload a font file
 */
export function preloadFont(
  family: string,
  weight: number = 400,
  style: string = 'normal'
): void {
  if (typeof window === 'undefined') return;

  // This would need the actual font URL
  // For Google Fonts or self-hosted fonts
  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = 'font';
  link.type = 'font/woff2';
  link.crossOrigin = 'anonymous';
  // URL would be constructed based on font service
  document.head.appendChild(link);
}

/**
 * Generate font-face CSS with subsetting
 */
export function generateFontFaceCSS(
  family: string,
  src: string,
  options: {
    weight?: number;
    style?: string;
    display?: string;
    unicodeRange?: string;
  } = {}
): string {
  const {
    weight = 400,
    style = 'normal',
    display = 'swap',
    unicodeRange,
  } = options;

  return `
@font-face {
  font-family: '${family}';
  src: url('${src}') format('woff2');
  font-weight: ${weight};
  font-style: ${style};
  font-display: ${display};
  ${unicodeRange ? `unicode-range: ${unicodeRange};` : ''}
}
  `.trim();
}

/**
 * Subset unicode ranges for common character sets
 */
export const UNICODE_RANGES = {
  latin: 'U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD',
  latinExtended: 'U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF',
  cyrillic: 'U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116',
  greek: 'U+0370-03FF',
  arabic: 'U+0600-06FF, U+200C-200E, U+2010-2011, U+204F, U+2E41, U+FB50-FDFF, U+FE80-FEFC',
} as const;

// ============================================================================
// Script Optimization
// ============================================================================

/**
 * Load a script with optimization
 */
export function loadScript(config: ScriptConfig): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof window === 'undefined') {
      resolve();
      return;
    }

    const existingScript = document.querySelector(`script[src="${config.src}"]`);
    if (existingScript) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = config.src;

    switch (config.strategy) {
      case 'async':
        script.async = true;
        break;
      case 'defer':
        script.defer = true;
        break;
      case 'module':
        script.type = 'module';
        break;
      case 'blocking':
      default:
        // No special attributes
        break;
    }

    if (config.integrity) {
      script.integrity = config.integrity;
    }

    if (config.crossOrigin) {
      script.crossOrigin = config.crossOrigin;
    }

    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${config.src}`));

    // Add preload hint if requested
    if (config.preload) {
      preloadScript(config.src);
    }

    document.body.appendChild(script);
  });
}

/**
 * Preload a script
 */
export function preloadScript(src: string): void {
  if (typeof window === 'undefined') return;

  const existingLink = document.querySelector(`link[rel="preload"][href="${src}"]`);
  if (existingLink) return;

  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = 'script';
  link.href = src;
  document.head.appendChild(link);
}

/**
 * Load multiple scripts in order
 */
export async function loadScriptsInOrder(scripts: ScriptConfig[]): Promise<void> {
  for (const script of scripts) {
    await loadScript(script);
  }
}

/**
 * Load multiple scripts in parallel
 */
export async function loadScriptsParallel(scripts: ScriptConfig[]): Promise<void> {
  await Promise.all(scripts.map(loadScript));
}

// ============================================================================
// Stylesheet Optimization
// ============================================================================

/**
 * Load a stylesheet with optimization
 */
export function loadStylesheet(config: StylesheetConfig): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof window === 'undefined') {
      resolve();
      return;
    }

    const existingLink = document.querySelector(`link[href="${config.href}"]`);
    if (existingLink) {
      resolve();
      return;
    }

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = config.href;

    if (config.media) {
      link.media = config.media;
    }

    // For non-critical CSS, load asynchronously
    if (!config.critical) {
      link.media = 'print';
      link.onload = () => {
        link.media = config.media || 'all';
        resolve();
      };
    } else {
      link.onload = () => resolve();
    }

    link.onerror = () => reject(new Error(`Failed to load stylesheet: ${config.href}`));

    // Add preload hint if requested
    if (config.preload) {
      preloadStylesheet(config.href);
    }

    document.head.appendChild(link);
  });
}

/**
 * Preload a stylesheet
 */
export function preloadStylesheet(href: string): void {
  if (typeof window === 'undefined') return;

  const existingLink = document.querySelector(`link[rel="preload"][href="${href}"]`);
  if (existingLink) return;

  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = 'style';
  link.href = href;
  document.head.appendChild(link);
}

/**
 * Inline critical CSS
 */
export function inlineCriticalCSS(css: string, id: string = 'critical-css'): void {
  if (typeof window === 'undefined') return;

  const existingStyle = document.getElementById(id);
  if (existingStyle) return;

  const style = document.createElement('style');
  style.id = id;
  style.textContent = css;
  document.head.insertBefore(style, document.head.firstChild);
}

// ============================================================================
// Resource Prioritization
// ============================================================================

export type ResourcePriority = 'highest' | 'high' | 'medium' | 'low' | 'lowest';

export interface ResourceHint {
  type: 'preload' | 'prefetch' | 'preconnect' | 'dns-prefetch' | 'prerender';
  href: string;
  as?: string;
  crossOrigin?: 'anonymous' | 'use-credentials';
  media?: string;
  type_attr?: string;
}

/**
 * Add resource hints to the document
 */
export function addResourceHints(hints: ResourceHint[]): void {
  if (typeof window === 'undefined') return;

  hints.forEach((hint) => {
    const existingLink = document.querySelector(
      `link[rel="${hint.type}"][href="${hint.href}"]`
    );
    if (existingLink) return;

    const link = document.createElement('link');
    link.rel = hint.type;
    link.href = hint.href;

    if (hint.as) link.setAttribute('as', hint.as);
    if (hint.crossOrigin) link.crossOrigin = hint.crossOrigin;
    if (hint.media) link.media = hint.media;
    if (hint.type_attr) link.type = hint.type_attr;

    document.head.appendChild(link);
  });
}

/**
 * Preconnect to critical origins
 */
export function preconnectOrigins(origins: string[]): void {
  addResourceHints(
    origins.map((href) => ({
      type: 'preconnect',
      href,
      crossOrigin: 'anonymous',
    }))
  );
}

/**
 * DNS prefetch for non-critical origins
 */
export function dnsPrefetchOrigins(origins: string[]): void {
  addResourceHints(
    origins.map((href) => ({
      type: 'dns-prefetch',
      href,
    }))
  );
}

// ============================================================================
// Compression Utilities
// ============================================================================

/**
 * Check if browser supports compression
 */
export function checkCompressionSupport(): {
  gzip: boolean;
  brotli: boolean;
  deflate: boolean;
} {
  if (typeof window === 'undefined') {
    return { gzip: true, brotli: true, deflate: true };
  }

  // Check Accept-Encoding support via feature detection
  // This is a simplified check; actual support depends on server
  return {
    gzip: true, // Universally supported
    brotli: 'CompressionStream' in window,
    deflate: true, // Universally supported
  };
}

/**
 * Compress data using CompressionStream (if available)
 */
export async function compressData(
  data: string,
  format: 'gzip' | 'deflate' = 'gzip'
): Promise<Uint8Array> {
  if (typeof CompressionStream === 'undefined') {
    throw new Error('CompressionStream not supported');
  }

  const encoder = new TextEncoder();
  const stream = new CompressionStream(format);
  const writer = stream.writable.getWriter();
  
  writer.write(encoder.encode(data));
  writer.close();

  const reader = stream.readable.getReader();
  const chunks: Uint8Array[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }

  const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;

  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  return result;
}

/**
 * Decompress data using DecompressionStream (if available)
 */
export async function decompressData(
  data: Uint8Array,
  format: 'gzip' | 'deflate' = 'gzip'
): Promise<string> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('DecompressionStream not supported');
  }

  const stream = new DecompressionStream(format);
  const writer = stream.writable.getWriter();
  
  writer.write(data);
  writer.close();

  const reader = stream.readable.getReader();
  const decoder = new TextDecoder();
  let result = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    result += decoder.decode(value, { stream: true });
  }

  return result;
}

// ============================================================================
// Cache Control
// ============================================================================

export interface CacheConfig {
  /** Cache name */
  name: string;
  /** Max age in seconds */
  maxAge?: number;
  /** Stale while revalidate */
  staleWhileRevalidate?: number;
  /** Cache strategy */
  strategy?: 'cache-first' | 'network-first' | 'stale-while-revalidate';
}

/**
 * Generate Cache-Control header value
 */
export function generateCacheControl(config: {
  public?: boolean;
  private?: boolean;
  maxAge?: number;
  sMaxAge?: number;
  staleWhileRevalidate?: number;
  staleIfError?: number;
  noCache?: boolean;
  noStore?: boolean;
  mustRevalidate?: boolean;
  immutable?: boolean;
}): string {
  const directives: string[] = [];

  if (config.public) directives.push('public');
  if (config.private) directives.push('private');
  if (config.maxAge !== undefined) directives.push(`max-age=${config.maxAge}`);
  if (config.sMaxAge !== undefined) directives.push(`s-maxage=${config.sMaxAge}`);
  if (config.staleWhileRevalidate !== undefined) {
    directives.push(`stale-while-revalidate=${config.staleWhileRevalidate}`);
  }
  if (config.staleIfError !== undefined) {
    directives.push(`stale-if-error=${config.staleIfError}`);
  }
  if (config.noCache) directives.push('no-cache');
  if (config.noStore) directives.push('no-store');
  if (config.mustRevalidate) directives.push('must-revalidate');
  if (config.immutable) directives.push('immutable');

  return directives.join(', ');
}

/**
 * Recommended cache configurations for different asset types
 */
export const CACHE_CONFIGS = {
  // Static assets (images, fonts, etc.) - long cache with immutable
  static: generateCacheControl({
    public: true,
    maxAge: 31536000, // 1 year
    immutable: true,
  }),
  
  // HTML pages - short cache with revalidation
  html: generateCacheControl({
    public: true,
    maxAge: 0,
    mustRevalidate: true,
  }),
  
  // API responses - no cache by default
  api: generateCacheControl({
    private: true,
    noCache: true,
    noStore: true,
  }),
  
  // API responses with caching
  apiCached: generateCacheControl({
    private: true,
    maxAge: 60,
    staleWhileRevalidate: 300,
  }),
  
  // JavaScript/CSS bundles - long cache with versioning
  bundles: generateCacheControl({
    public: true,
    maxAge: 31536000,
    immutable: true,
  }),
} as const;