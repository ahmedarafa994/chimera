#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Performance audit configuration
const PERFORMANCE_CONFIG = {
  // Bundle size thresholds (KB)
  bundles: {
    total: 500,
    js: 300,
    css: 50,
    vendor: 200,
  },
  // Core Web Vitals thresholds (ms)
  vitals: {
    LCP: 2500, // Largest Contentful Paint
    FID: 100,  // First Input Delay
    CLS: 0.1,  // Cumulative Layout Shift
    FCP: 1800, // First Contentful Paint
    TTFB: 800, // Time to First Byte
  },
  // Build performance thresholds
  build: {
    duration: 120000, // 2 minutes
    memoryUsage: 4096, // 4GB
  }
};

// Colors for console output
const colors = {
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

class PerformanceAuditor {
  constructor() {
    this.results = {
      bundles: {},
      buildTime: 0,
      errors: [],
      warnings: [],
      suggestions: []
    };
  }

  log(message, color = 'reset') {
    console.log(colors[color] + message + colors.reset);
  }

  error(message) {
    this.log(`❌ ERROR: ${message}`, 'red');
    this.results.errors.push(message);
  }

  warning(message) {
    this.log(`⚠️  WARNING: ${message}`, 'yellow');
    this.results.warnings.push(message);
  }

  success(message) {
    this.log(`✅ ${message}`, 'green');
  }

  info(message) {
    this.log(`ℹ️  ${message}`, 'blue');
  }

  suggestion(message) {
    this.log(`💡 SUGGESTION: ${message}`, 'yellow');
    this.results.suggestions.push(message);
  }

  // Check if required files exist
  checkFiles() {
    this.log('\n🔍 Checking project structure...', 'bold');

    const requiredFiles = [
      'package.json',
      'next.config.ts',
      'src/app/layout.tsx',
      'tailwind.config.ts',
      '.env.template'
    ];

    for (const file of requiredFiles) {
      const filePath = path.join(process.cwd(), file);
      if (!fs.existsSync(filePath)) {
        this.error(`Missing required file: ${file}`);
      } else {
        this.success(`Found: ${file}`);
      }
    }
  }

  // Analyze package.json for performance issues
  analyzePackageJson() {
    this.log('\n📦 Analyzing package.json...', 'bold');

    try {
      const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf-8'));

      // Check for heavy dependencies
      const heavyDeps = {
        'moment': 'Use date-fns or dayjs instead',
        'lodash': 'Use specific lodash functions or native JS',
        '@emotion/styled': 'Consider lighter CSS-in-JS solutions',
        'material-ui': 'Consider lighter UI libraries'
      };

      const allDeps = { ...packageJson.dependencies, ...packageJson.devDependencies };

      for (const [dep, suggestion] of Object.entries(heavyDeps)) {
        if (allDeps[dep]) {
          this.warning(`Heavy dependency detected: ${dep}`);
          this.suggestion(suggestion);
        }
      }

      // Check for missing performance dependencies
      const perfDeps = ['web-vitals', '@next/bundle-analyzer'];
      for (const dep of perfDeps) {
        if (!allDeps[dep]) {
          this.warning(`Missing performance dependency: ${dep}`);
        }
      }

      this.success('Package.json analysis complete');
    } catch (error) {
      this.error(`Failed to analyze package.json: ${error.message}`);
    }
  }

  // Run build and measure performance
  async runBuildAnalysis() {
    this.log('\n🏗️  Running build analysis...', 'bold');

    try {
      // Check if build directory exists and clean it
      const buildDir = path.join(process.cwd(), '.next');
      if (fs.existsSync(buildDir)) {
        this.info('Cleaning previous build...');
        fs.rmSync(buildDir, { recursive: true, force: true });
      }

      // Measure build time
      const buildStart = Date.now();
      this.info('Starting build...');

      try {
        // Run build with bundle analyzer
        execSync('npm run build', {
          stdio: 'inherit',
          env: { ...process.env, ANALYZE: 'true' },
          timeout: PERFORMANCE_CONFIG.build.duration
        });

        const buildTime = Date.now() - buildStart;
        this.results.buildTime = buildTime;

        if (buildTime > PERFORMANCE_CONFIG.build.duration) {
          this.warning(`Build time exceeded threshold: ${buildTime}ms > ${PERFORMANCE_CONFIG.build.duration}ms`);
        } else {
          this.success(`Build completed in ${buildTime}ms`);
        }

      } catch (buildError) {
        this.error(`Build failed: ${buildError.message}`);
        return false;
      }

      // Analyze build output
      await this.analyzeBuildOutput();

    } catch (error) {
      this.error(`Build analysis failed: ${error.message}`);
      return false;
    }

    return true;
  }

  // Analyze build output files
  async analyzeBuildOutput() {
    this.log('\n📊 Analyzing build output...', 'bold');

    const buildDir = path.join(process.cwd(), '.next/static');
    if (!fs.existsSync(buildDir)) {
      this.error('Build directory not found');
      return;
    }

    try {
      // Get chunk files
      const getFileSize = (filePath) => {
        if (fs.existsSync(filePath)) {
          return fs.statSync(filePath).size;
        }
        return 0;
      };

      // Find all JS and CSS files
      const findFiles = (dir, ext) => {
        const files = [];
        const items = fs.readdirSync(dir, { withFileTypes: true });

        for (const item of items) {
          const fullPath = path.join(dir, item.name);
          if (item.isDirectory()) {
            files.push(...findFiles(fullPath, ext));
          } else if (item.name.endsWith(ext)) {
            files.push(fullPath);
          }
        }

        return files;
      };

      const jsFiles = findFiles(buildDir, '.js');
      const cssFiles = findFiles(buildDir, '.css');

      // Calculate sizes
      let totalJsSize = 0;
      let totalCssSize = 0;
      let vendorSize = 0;

      jsFiles.forEach(file => {
        const size = getFileSize(file);
        totalJsSize += size;

        if (file.includes('vendor') || file.includes('framework') || file.includes('webpack')) {
          vendorSize += size;
        }
      });

      cssFiles.forEach(file => {
        totalCssSize += getFileSize(file);
      });

      const totalSize = totalJsSize + totalCssSize;

      // Store results
      this.results.bundles = {
        total: Math.round(totalSize / 1024),
        js: Math.round(totalJsSize / 1024),
        css: Math.round(totalCssSize / 1024),
        vendor: Math.round(vendorSize / 1024)
      };

      // Check against thresholds
      this.checkBundleThresholds();

    } catch (error) {
      this.error(`Failed to analyze build output: ${error.message}`);
    }
  }

  // Check bundle sizes against thresholds
  checkBundleThresholds() {
    this.log('\n📏 Checking bundle size thresholds...', 'bold');

    const { bundles } = this.results;
    const thresholds = PERFORMANCE_CONFIG.bundles;

    for (const [type, size] of Object.entries(bundles)) {
      const threshold = thresholds[type];
      if (size > threshold) {
        this.warning(`${type.toUpperCase()} bundle size exceeded: ${size}KB > ${threshold}KB`);
        this.suggestion(this.getBundleOptimizationSuggestion(type));
      } else {
        this.success(`${type.toUpperCase()} bundle size OK: ${size}KB <= ${threshold}KB`);
      }
    }
  }

  // Get optimization suggestions for bundle types
  getBundleOptimizationSuggestion(bundleType) {
    const suggestions = {
      total: 'Consider code splitting, tree shaking, and dynamic imports',
      js: 'Use dynamic imports for heavy components and implement code splitting',
      css: 'Remove unused CSS, use CSS purging, and consider CSS-in-JS optimization',
      vendor: 'Split vendor chunks, consider smaller alternatives to heavy dependencies'
    };

    return suggestions[bundleType] || 'Optimize bundle size';
  }

  // Generate performance report
  generateReport() {
    this.log('\n📋 Performance Audit Report', 'bold');
    this.log('='.repeat(50), 'blue');

    // Build Performance
    this.log('\n🏗️  Build Performance:', 'bold');
    this.log(`Build Time: ${this.results.buildTime}ms`);

    // Bundle Sizes
    this.log('\n📦 Bundle Sizes:', 'bold');
    for (const [type, size] of Object.entries(this.results.bundles)) {
      const threshold = PERFORMANCE_CONFIG.bundles[type];
      const status = size <= threshold ? '✅' : '❌';
      this.log(`${status} ${type.toUpperCase()}: ${size}KB (threshold: ${threshold}KB)`);
    }

    // Issues Summary
    this.log('\n🔍 Issues Summary:', 'bold');
    this.log(`Errors: ${this.results.errors.length}`, this.results.errors.length > 0 ? 'red' : 'green');
    this.log(`Warnings: ${this.results.warnings.length}`, this.results.warnings.length > 0 ? 'yellow' : 'green');

    if (this.results.errors.length > 0) {
      this.log('\n❌ Errors:', 'red');
      this.results.errors.forEach(error => this.log(`  • ${error}`));
    }

    if (this.results.warnings.length > 0) {
      this.log('\n⚠️  Warnings:', 'yellow');
      this.results.warnings.forEach(warning => this.log(`  • ${warning}`));
    }

    if (this.results.suggestions.length > 0) {
      this.log('\n💡 Suggestions:', 'yellow');
      this.results.suggestions.forEach(suggestion => this.log(`  • ${suggestion}`));
    }

    // Overall Score
    const score = this.calculateOverallScore();
    this.log(`\n🎯 Overall Performance Score: ${score}/100`, score >= 80 ? 'green' : score >= 60 ? 'yellow' : 'red');

    // Save report to file
    this.saveReport();
  }

  // Calculate overall performance score
  calculateOverallScore() {
    let score = 100;

    // Deduct points for errors
    score -= this.results.errors.length * 20;

    // Deduct points for warnings
    score -= this.results.warnings.length * 10;

    // Deduct points for bundle size overages
    const { bundles } = this.results;
    const thresholds = PERFORMANCE_CONFIG.bundles;

    for (const [type, size] of Object.entries(bundles)) {
      const threshold = thresholds[type];
      if (size > threshold) {
        const overage = (size - threshold) / threshold;
        score -= Math.min(overage * 15, 15);
      }
    }

    // Deduct points for slow build
    if (this.results.buildTime > PERFORMANCE_CONFIG.build.duration) {
      score -= 10;
    }

    return Math.max(0, Math.round(score));
  }

  // Save report to file
  saveReport() {
    const reportData = {
      timestamp: new Date().toISOString(),
      score: this.calculateOverallScore(),
      buildTime: this.results.buildTime,
      bundles: this.results.bundles,
      errors: this.results.errors,
      warnings: this.results.warnings,
      suggestions: this.results.suggestions,
      thresholds: PERFORMANCE_CONFIG
    };

    const reportPath = path.join(process.cwd(), 'performance-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(reportData, null, 2));
    this.success(`Report saved to: ${reportPath}`);
  }

  // Run complete audit
  async runAudit() {
    this.log('🚀 Starting Chimera Performance Audit', 'bold');
    this.log('='.repeat(50), 'blue');

    this.checkFiles();
    this.analyzePackageJson();

    const buildSuccess = await this.runBuildAnalysis();

    this.generateReport();

    if (this.results.errors.length > 0) {
      process.exit(1);
    }

    this.log('\n🎉 Performance audit completed!', 'green');
  }
}

// Run audit if called directly
if (require.main === module) {
  const auditor = new PerformanceAuditor();
  auditor.runAudit().catch(error => {
    console.error('Performance audit failed:', error);
    process.exit(1);
  });
}

module.exports = PerformanceAuditor;