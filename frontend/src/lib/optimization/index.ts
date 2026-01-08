/**
 * Performance Optimization Module for Project Chimera
 * 
 * Provides utilities for:
 * - Lazy loading components and routes
 * - Code splitting strategies
 * - Asset compression and optimization
 * - Performance monitoring and metrics
 * 
 * @module lib/optimization
 */

/**
 * Performance Optimization Module for Project Chimera
 *
 * Comprehensive optimization utilities including:
 * - Lazy loading with intersection observer
 * - Code splitting strategies
 * - Asset optimization (images, fonts, scripts)
 * - Performance monitoring (Core Web Vitals)
 * - Intelligent prefetching
 *
 * @module lib/optimization
 */

// Lazy Loading
export {
  createLazyComponent,
  createPreloadableLazyComponent,
  useLazyLoad,
  LazyImage,
  LazySection,
  PreloadManager,
  preloadManager,
  useRoutePreload,
  dynamicImportWithRetry,
  preloadChunk,
  prefetchChunk,
  LoadingSkeleton,
  type LazyComponentOptions,
  type LazyLoadOptions,
  type LazyLoadConfig,
  type LazyLoadState,
  type LazyImageProps,
  type LazySectionProps,
  type LoadingSkeletonProps,
} from './lazy-loader';

// Re-export prefetch-manager from .tsx file

// Code Splitting
export {
  createRouteComponent,
  createRouteComponents,
  splitComponent,
  createConditionalComponent,
  createFeatureFlaggedComponent,
  generateVendorChunksConfig,
  CHIMERA_VENDOR_CHUNKS,
  importNamed,
  importMultiple,
  batchImport,
  addResourceHint,
  removeResourceHint,
  preconnect,
  dnsPrefetch,
  loadChunkWithStrategy,
  loadRemoteModule,
  injectCriticalCSS,
  loadAsyncCSS,
  type ChunkConfig,
  type SplitPointConfig,
  type RouteChunk,
  type VendorChunkConfig,
  type RemoteModuleConfig,
  type ChunkLoadingStrategy,
} from './code-splitting';

// Asset Optimization
export {
  generateSrcSet,
  generateSizes,
  transformImageUrl,
  generateBlurPlaceholder,
  checkImageFormatSupport,
  preloadImage,
  loadFont,
  preloadFont,
  generateFontFaceCSS,
  UNICODE_RANGES,
  loadScript,
  preloadScript,
  loadScriptsInOrder,
  loadScriptsParallel,
  loadStylesheet,
  preloadStylesheet,
  inlineCriticalCSS,
  addResourceHints,
  preconnectOrigins,
  dnsPrefetchOrigins,
  checkCompressionSupport,
  compressData,
  decompressData,
  generateCacheControl,
  CACHE_CONFIGS,
  type ImageOptimizationConfig,
  type FontConfig,
  type ScriptConfig,
  type StylesheetConfig,
  type ResourceHint,
  type ResourcePriority,
  type CacheConfig,
} from './asset-optimizer';

// Performance Monitoring
export {
  PerformanceMonitor,
  performanceMonitor,
  usePerformanceMonitor,
  measureExecutionTime,
  measureAsyncExecutionTime,
  withPerformanceTracking,
  debounceWithTracking,
  throttleWithTracking,
  benchmark,
  compareBenchmarks,
  checkPerformanceBudget,
  DEFAULT_PERFORMANCE_BUDGET,
  type WebVitalsMetric,
  type PerformanceMetrics,
  type ResourceMetrics,
  type LongTaskMetrics,
  type PerformanceReport,
  type MetricCallback,
  type ReportCallback,
  type BenchmarkResult,
  type PerformanceBudget,
} from './performance-monitor';

// Prefetch Management
export {
  PrefetchManager,
  prefetchManager,
  usePrefetchOnVisible,
  usePrefetchOnHover,
  usePrefetchData,
  usePrefetchComponent,
  usePrefetchManager,
  PrefetchLink,
  PredictivePrefetchManager,
  predictivePrefetch,
  usePredictivePrefetch,
  type PrefetchPriority,
  type PrefetchItem,
  type PrefetchConfig,
  type NetworkInfo,
  type PrefetchLinkProps,
  type NavigationPattern,
} from './prefetch-manager';