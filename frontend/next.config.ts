import type { NextConfig } from "next";
import { execSync } from 'child_process';
import withBundleAnalyzer from '@next/bundle-analyzer';

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,

  // PERF-005 FIX: CDN Asset Prefix for production deployments
  // Set CDN_URL environment variable to enable CDN (e.g., https://cdn.example.com)
  assetPrefix: process.env.CDN_URL || '',

  // Enable standalone output for optimized Docker builds
  output: 'standalone',

  // Explicit build directory
  distDir: '.next',

  // Consistent trailing slash behavior for routing
  trailingSlash: false,

  // Enable compression for better performance
  compress: true,

  // Optimize images for better Core Web Vitals
  images: {
    formats: ['image/avif', 'image/webp'],
    minimumCacheTTL: 60 * 60 * 24 * 30, // 30 days cache
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048],
    imageSizes: [16, 32, 48, 64, 96, 128, 256],
  },

  // Modularize imports for better tree-shaking (PERF-001)
  modularizeImports: {
    '@radix-ui/react-icons': {
      transform: '@radix-ui/react-icons/dist/{{member}}',
    },
  },

  // Generate deterministic build ID from git commit (Task 3.2)
  generateBuildId: async () => {
    try {
      const gitHash = execSync('git rev-parse --short HEAD').toString().trim();
      return `build-${gitHash}`;
    } catch {
      // Fallback for non-git environments
      return `build-${Date.now()}`;
    }
  },

  // API proxy configuration to avoid CORS issues in development
  // NOTE: /api/v1/:path* rewrite removed to allow requests to flow through
  // the custom API route handler at frontend/src/app/api/v1/[...path]/route.ts
  // which has proper 10-minute timeouts configured via undici
  async rewrites() {
    return [
      {
        source: "/api/proxy/:path*",
        destination: "http://localhost:8001/api/v1/:path*",
      },
    ];
  },

  // Turbopack configuration with memory optimizations
  turbopack: {
    // Explicit root directory to silence warning
    root: '../',
    // Turbopack-specific resolve aliases if needed
    resolveAlias: {
      // Add any module aliases here if needed
    },
  },

  // Server external packages (moved from experimental)
  serverExternalPackages: ['sharp', 'onnxruntime-node'],

  // Typed routes (moved from experimental)
  typedRoutes: false,

  // Experimental features
  experimental: {
    // PERF-003 FIX: Optimize CSS for smaller bundles
    optimizeCss: true,
    // PERF-003 FIX: Optimize server components
    optimizeServerReact: true,
    // Improve HMR reliability
    serverActions: {
      bodySizeLimit: "2mb",
    },
    // PERF-003 FIX: Enable scroll performance optimizations
    scrollRestoration: true,
    // PERF-003 FIX: Enable parallel route builds for faster builds (Removed as invalid)
    // PERF-005 FIX: Edge runtime support for edge deployments (Removed invalid runtime config)
    // runtime: 'nodejs', // Moved or removed as it is not valid in experimental
    // PERF-005 FIX: Enable ISR (Incremental Static Regeneration) (Removed invalid config)
    // isrMemoryCacheSize: 50, // MB
    // Optimize package imports for smaller bundles (PERF-002)
    optimizePackageImports: [
      '@radix-ui/react-avatar',
      '@radix-ui/react-checkbox',
      '@radix-ui/react-collapsible',
      '@radix-ui/react-dialog',
      '@radix-ui/react-dropdown-menu',
      '@radix-ui/react-label',
      '@radix-ui/react-popover',
      '@radix-ui/react-progress',
      '@radix-ui/react-radio-group',
      '@radix-ui/react-scroll-area',
      '@radix-ui/react-select',
      '@radix-ui/react-separator',
      '@radix-ui/react-slider',
      '@radix-ui/react-slot',
      '@radix-ui/react-switch',
      '@radix-ui/react-tabs',
      '@radix-ui/react-tooltip',
      'recharts',
    ],
  },

  // Webpack fallback for when using --webpack flag
  webpack: (config, { isServer, dev, webpack: _webpack }) => {
    // PERF-003 FIX: Add webpack cache for faster rebuilds in development
    if (!isServer && dev) {
      config.cache = {
        type: 'filesystem',
        cacheDirectory: require.resolve('.next/cache'),
        maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
        compression: 'gzip',
      };
    }

    // Handle potential module resolution issues
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };

      // Production optimizations (PERF-003)
      if (!dev) {
        // Enable more aggressive tree shaking
        config.optimization = {
          ...config.optimization,
          usedExports: true,
          sideEffects: false, // More aggressive tree shaking
          splitChunks: {
            chunks: 'all',
            cacheGroups: {
              vendor: {
                test: /[\\/]node_modules[\\/]/,
                name: 'vendors',
                chunks: 'all',
                priority: 10,
                reuseExistingChunk: true,
              },
              radixui: {
                test: /[\\/]node_modules[\\/]@radix-ui[\\/]/,
                name: 'radixui',
                chunks: 'all',
                priority: 20,
              },
              recharts: {
                test: /[\\/]node_modules[\\/]recharts[\\/]/,
                name: 'recharts',
                chunks: 'all',
                priority: 20,
              },
              common: {
                name: 'common',
                minChunks: 2,
                chunks: 'all',
                priority: 5,
                reuseExistingChunk: true,
              },
            },
          },
          // PERF-003 FIX: Add module concatenation to reduce bundle overhead
          concatenateModules: true,
          // PERF-003 FIX: Minimize bundle size
          minimize: true,
        };
      }
    }

    // PERF-003 FIX: Production source maps for debugging (disabled for size)
    if (!dev) {
      config.devtool = false; // Disable source maps for production to reduce bundle size
    }

    return config;
  },

  // PERF-005 FIX: CDN & Edge Optimization - Advanced caching headers
  async headers() {
    const CDN_HEADERS = {
      // Immutable static assets - cached at CDN edge for 1 year
      'Cache-Control': 'public, max-age=31536000, immutable',
      // CORS for CDN access
      'Access-Control-Allow-Origin': '*',
      // CDN-specific headers
      'CDN-Cache-Control': 'public, max-age=31536000, immutable',
      // Edge server caching
      'Edge-Cache-Tag': 'static-assets',
    };

    const API_HEADERS = {
      // Short cache for API responses with revalidation
      'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=30',
      // CORS for API
      'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGINS || '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
    };

    return [
      // Static assets (images, fonts, icons) - immutable caching
      {
        source: '/:all*(svg|jpg|jpeg|png|gif|ico|webp|avif|woff|woff2|ttf|otf|eot)',
        headers: Object.entries(CDN_HEADERS).map(([key, value]) => ({ key, value })),
      },
      // Next.js static chunks - hashed filenames are immutable
      {
        source: '/_next/static/:path*',
        headers: Object.entries(CDN_HEADERS).map(([key, value]) => ({ key, value })),
      },
      // Next.js image optimization - cache at CDN
      {
        source: '/_next/image/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
          {
            key: 'CDN-Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      // API responses - edge caching with revalidation
      {
        source: '/api/:path*',
        headers: Object.entries(API_HEADERS).map(([key, value]) => ({ key, value })),
      },
      // Health endpoints - aggressive edge caching
      {
        source: '/health/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, s-maxage=10, stale-while-revalidate=5',
          },
        ],
      },
      // Provider/Model endpoints - cache for 5 minutes at edge
      {
        source: '/api/:path*(providers|models|session)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, s-maxage=300, stale-while-revalidate=60',
          },
          {
            key: 'CDN-Cache-Control',
            value: 'public, max-age=300, stale-while-revalidate=60',
          },
        ],
      },
      // Security headers for CDN
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
        ],
      },
    ];
  },
};

const bundleAnalyzer = withBundleAnalyzer({
  enabled: process.env.ANALYZE === 'true',
})

export default bundleAnalyzer(nextConfig);
