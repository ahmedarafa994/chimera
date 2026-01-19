# Chimera Optimization Strategy - Part 2

## 5. Testing Strategies (Continued)

### 5.2 Integration Tests

**Location:** `frontend/src/__tests__/integration/`

#### WebSocket Integration Tests

```typescript
// frontend/src/__tests__/integration/websocket.test.ts

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WebSocketManager } from '@/lib/sync';
import { createMockWebSocketServer } from '../mocks/websocket-server';

describe('WebSocket Integration', () => {
  let server: MockWebSocketServer;
  let manager: WebSocketManager;

  beforeAll(async () => {
    server = await createMockWebSocketServer(8080);
    manager = new WebSocketManager({
      url: 'ws://localhost:8080',
      reconnect: { maxAttempts: 3 },
    });
  });

  afterAll(async () => {
    manager.disconnect();
    await server.close();
  });

  it('should establish connection', async () => {
    await manager.connect();
    expect(manager.getState()).toBe('connected');
  });

  it('should handle request/response pattern', async () => {
    await manager.connect();
    
    server.onMessage('ping', () => ({ pong: true }));
    
    const response = await manager.request('ping', {});
    expect(response).toEqual({ pong: true });
  });

  it('should reconnect after disconnection', async () => {
    await manager.connect();
    
    // Simulate server disconnect
    server.disconnectAll();
    
    // Wait for reconnection
    await new Promise((resolve) => setTimeout(resolve, 2000));
    
    expect(manager.getState()).toBe('connected');
  });

  it('should queue messages during disconnection', async () => {
    await manager.connect();
    
    server.disconnectAll();
    
    // Send while disconnected
    const sendPromise = manager.send('test', { data: 'queued' });
    
    // Reconnect
    await new Promise((resolve) => setTimeout(resolve, 2000));
    
    // Message should be delivered
    await sendPromise;
    expect(server.receivedMessages).toContainEqual(
      expect.objectContaining({ type: 'test', payload: { data: 'queued' } })
    );
  });
});
```

#### State Sync Integration Tests

```typescript
// frontend/src/__tests__/integration/state-sync.test.ts

import { describe, it, expect } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useSyncedState } from '@/lib/sync';
import { createMockSyncServer } from '../mocks/sync-server';

describe('State Sync Integration', () => {
  let server: MockSyncServer;

  beforeAll(async () => {
    server = await createMockSyncServer();
  });

  afterAll(async () => {
    await server.close();
  });

  it('should sync state changes to server', async () => {
    const { result } = renderHook(() => 
      useSyncedState('test-key', { count: 0 })
    );

    act(() => {
      result.current[1]({ count: 1 });
    });

    await waitFor(() => {
      expect(server.getState('test-key')).toEqual({ count: 1 });
    });
  });

  it('should handle conflict resolution', async () => {
    const { result } = renderHook(() => 
      useSyncedState('conflict-key', { value: 'initial' }, {
        conflictResolution: 'server-wins',
      })
    );

    // Simulate server update
    server.setState('conflict-key', { value: 'server-value' });

    // Local update
    act(() => {
      result.current[1]({ value: 'local-value' });
    });

    // Server should win
    await waitFor(() => {
      expect(result.current[0]).toEqual({ value: 'server-value' });
    });
  });

  it('should work offline and sync when online', async () => {
    const { result } = renderHook(() => 
      useSyncedState('offline-key', { data: 'initial' })
    );

    // Go offline
    server.goOffline();

    act(() => {
      result.current[1]({ data: 'offline-change' });
    });

    // Local state should update
    expect(result.current[0]).toEqual({ data: 'offline-change' });

    // Go online
    server.goOnline();

    // Should sync to server
    await waitFor(() => {
      expect(server.getState('offline-key')).toEqual({ data: 'offline-change' });
    });
  });
});
```

### 5.3 End-to-End Tests

**Location:** `frontend/e2e/`

#### Performance E2E Tests

```typescript
// frontend/e2e/performance.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Performance', () => {
  test('should meet LCP target', async ({ page }) => {
    await page.goto('/');
    
    const lcp = await page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          resolve(lastEntry.startTime);
        }).observe({ type: 'largest-contentful-paint', buffered: true });
      });
    });

    expect(lcp).toBeLessThan(2500); // 2.5s target
  });

  test('should meet CLS target', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to stabilize
    await page.waitForTimeout(3000);
    
    const cls = await page.evaluate(() => {
      return new Promise((resolve) => {
        let clsValue = 0;
        new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (!entry.hadRecentInput) {
              clsValue += entry.value;
            }
          }
          resolve(clsValue);
        }).observe({ type: 'layout-shift', buffered: true });
        
        setTimeout(() => resolve(clsValue), 1000);
      });
    });

    expect(cls).toBeLessThan(0.1); // 0.1 target
  });

  test('should load critical resources efficiently', async ({ page }) => {
    const resourceTimings: any[] = [];
    
    page.on('response', (response) => {
      resourceTimings.push({
        url: response.url(),
        status: response.status(),
        timing: response.timing(),
      });
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Check JavaScript bundle sizes
    const jsResources = resourceTimings.filter(r => 
      r.url.endsWith('.js') && r.status === 200
    );
    
    for (const resource of jsResources) {
      const size = await page.evaluate(async (url) => {
        const response = await fetch(url);
        const blob = await response.blob();
        return blob.size;
      }, resource.url);
      
      // Each JS chunk should be under 200KB
      expect(size).toBeLessThan(200 * 1024);
    }
  });
});
```

#### Real-time Features E2E Tests

```typescript
// frontend/e2e/realtime.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Real-time Features', () => {
  test('should establish WebSocket connection', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Wait for WebSocket connection
    const wsConnected = await page.evaluate(() => {
      return new Promise((resolve) => {
        const checkConnection = setInterval(() => {
          const indicator = document.querySelector('[data-testid="connection-status"]');
          if (indicator?.textContent === 'Connected') {
            clearInterval(checkConnection);
            resolve(true);
          }
        }, 100);
        
        setTimeout(() => {
          clearInterval(checkConnection);
          resolve(false);
        }, 5000);
      });
    });

    expect(wsConnected).toBe(true);
  });

  test('should handle optimistic updates', async ({ page }) => {
    await page.goto('/todos');
    
    // Add a new todo
    await page.fill('[data-testid="todo-input"]', 'New Todo');
    await page.click('[data-testid="add-todo-button"]');
    
    // Should appear immediately (optimistic)
    await expect(page.locator('text=New Todo')).toBeVisible();
    
    // Should persist after reload
    await page.reload();
    await expect(page.locator('text=New Todo')).toBeVisible();
  });

  test('should handle offline mode', async ({ page, context }) => {
    await page.goto('/dashboard');
    
    // Go offline
    await context.setOffline(true);
    
    // Should show offline indicator
    await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();
    
    // Should still allow interactions
    await page.click('[data-testid="action-button"]');
    
    // Should queue the action
    await expect(page.locator('[data-testid="pending-actions"]')).toContainText('1');
    
    // Go online
    await context.setOffline(false);
    
    // Should sync pending actions
    await expect(page.locator('[data-testid="pending-actions"]')).toContainText('0');
  });

  test('should recover from connection loss', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Wait for initial connection
    await page.waitForSelector('[data-testid="connection-status"]:has-text("Connected")');
    
    // Simulate connection loss by blocking WebSocket
    await page.route('**/ws', (route) => route.abort());
    
    // Should show reconnecting state
    await expect(page.locator('[data-testid="connection-status"]')).toContainText('Reconnecting');
    
    // Restore connection
    await page.unroute('**/ws');
    
    // Should reconnect
    await expect(page.locator('[data-testid="connection-status"]')).toContainText('Connected');
  });
});
```

### 5.4 Visual Regression Tests

```typescript
// frontend/e2e/visual.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Visual Regression', () => {
  test('dashboard should match snapshot', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Wait for animations to complete
    await page.waitForTimeout(1000);
    
    await expect(page).toHaveScreenshot('dashboard.png', {
      maxDiffPixels: 100,
    });
  });

  test('components should render correctly across viewports', async ({ page }) => {
    const viewports = [
      { width: 375, height: 667, name: 'mobile' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 1280, height: 800, name: 'desktop' },
      { width: 1920, height: 1080, name: 'large' },
    ];

    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/components');
      await page.waitForLoadState('networkidle');
      
      await expect(page).toHaveScreenshot(`components-${viewport.name}.png`, {
        maxDiffPixels: 50,
      });
    }
  });

  test('loading states should display correctly', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Capture loading state
    await expect(page.locator('[data-testid="skeleton"]')).toHaveScreenshot('loading-skeleton.png');
    
    // Wait for content
    await page.waitForSelector('[data-testid="content"]');
    
    // Capture loaded state
    await expect(page.locator('[data-testid="content"]')).toHaveScreenshot('loaded-content.png');
  });
});
```

### 5.5 Accessibility Tests

```typescript
// frontend/e2e/accessibility.spec.ts

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility', () => {
  test('should have no accessibility violations on dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Tab through interactive elements
    await page.keyboard.press('Tab');
    const firstFocused = await page.evaluate(() => document.activeElement?.tagName);
    expect(['BUTTON', 'A', 'INPUT']).toContain(firstFocused);
    
    // Continue tabbing
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');
      const focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(focused).not.toBe('BODY'); // Should always have focus on interactive element
    }
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check buttons have accessible names
    const buttons = await page.locator('button').all();
    for (const button of buttons) {
      const accessibleName = await button.getAttribute('aria-label') || 
                            await button.textContent();
      expect(accessibleName).toBeTruthy();
    }
    
    // Check images have alt text
    const images = await page.locator('img').all();
    for (const image of images) {
      const alt = await image.getAttribute('alt');
      expect(alt).toBeTruthy();
    }
  });

  test('should support reduced motion', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/dashboard');
    
    // Check that animations are disabled
    const animatedElement = page.locator('[data-animated]').first();
    const animationDuration = await animatedElement.evaluate((el) => 
      getComputedStyle(el).animationDuration
    );
    
    expect(animationDuration).toBe('0s');
  });
});
```

### 5.6 Network Condition Tests

```typescript
// frontend/e2e/network.spec.ts

import { test, expect } from '@playwright/test';

test.describe('Network Conditions', () => {
  test('should handle slow 3G', async ({ page, context }) => {
    // Simulate slow 3G
    await context.route('**/*', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 500));
      await route.continue();
    });

    const startTime = Date.now();
    await page.goto('/dashboard');
    await page.waitForLoadState('domcontentloaded');
    const loadTime = Date.now() - startTime;

    // Should still be usable within 10 seconds
    expect(loadTime).toBeLessThan(10000);
    
    // Should show loading states
    await expect(page.locator('[data-testid="skeleton"]')).toBeVisible();
  });

  test('should handle intermittent failures', async ({ page }) => {
    let requestCount = 0;
    
    await page.route('**/api/**', async (route) => {
      requestCount++;
      if (requestCount % 3 === 0) {
        // Fail every 3rd request
        await route.abort('failed');
      } else {
        await route.continue();
      }
    });

    await page.goto('/dashboard');
    
    // Should show error state
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    
    // Should offer retry
    await page.click('[data-testid="retry-button"]');
    
    // Should eventually succeed
    await expect(page.locator('[data-testid="content"]')).toBeVisible();
  });

  test('should cache resources appropriately', async ({ page }) => {
    // First visit
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    const firstVisitRequests: string[] = [];
    page.on('request', (request) => {
      firstVisitRequests.push(request.url());
    });
    
    // Second visit
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    const secondVisitRequests: string[] = [];
    page.on('request', (request) => {
      secondVisitRequests.push(request.url());
    });
    
    // Static assets should be cached
    const staticAssets = secondVisitRequests.filter(url => 
      url.includes('.js') || url.includes('.css')
    );
    
    // Should have fewer requests on second visit due to caching
    expect(secondVisitRequests.length).toBeLessThan(firstVisitRequests.length);
  });
});
```

---

## 6. Implementation Guide

### 6.1 Quick Start

```bash
# Install dependencies
npm run install:all

# Start development server
npm run dev

# Run tests
npm run test           # Unit tests
npm run test:e2e       # E2E tests
npm run test:perf      # Performance tests
```

### 6.2 Module Integration

#### Performance Optimization

```typescript
// frontend/src/app/layout.tsx

import { PerformanceMonitor } from '@/lib/optimization';
import { QueryProvider } from '@/providers/query-provider';

// Initialize performance monitoring
if (typeof window !== 'undefined') {
  PerformanceMonitor.getInstance().startMonitoring({
    reportEndpoint: '/api/metrics',
    sampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1,
  });
}

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <QueryProvider>
          {children}
        </QueryProvider>
      </body>
    </html>
  );
}
```

#### Sync Module

```typescript
// frontend/src/providers/sync-provider.tsx

import { createContext, useContext, useEffect } from 'react';
import { 
  getWebSocketManager, 
  getEventBus, 
  getErrorHandler 
} from '@/lib/sync';

const SyncContext = createContext(null);

export function SyncProvider({ children }) {
  useEffect(() => {
    const wsManager = getWebSocketManager();
    const eventBus = getEventBus();
    const errorHandler = getErrorHandler();

    // Configure error handler
    errorHandler.configure({
      logErrors: true,
      reportErrors: process.env.NODE_ENV === 'production',
      reportEndpoint: '/api/errors',
    });

    // Connect WebSocket
    wsManager.connect();

    return () => {
      wsManager.disconnect();
    };
  }, []);

  return (
    <SyncContext.Provider value={{}}>
      {children}
    </SyncContext.Provider>
  );
}
```

### 6.3 Configuration

#### Next.js Configuration

```typescript
// frontend/next.config.ts

import { getOptimizedSplitChunks } from '@/lib/optimization';

const nextConfig = {
  experimental: {
    optimizeCss: true,
  },
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.optimization.splitChunks = getOptimizedSplitChunks();
    }
    return config;
  },
};

export default nextConfig;
```

#### Environment Variables

```bash
# frontend/.env.example

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_WS_URL=ws://localhost:8001/ws

# Performance
NEXT_PUBLIC_PERFORMANCE_SAMPLE_RATE=0.1
NEXT_PUBLIC_ERROR_REPORTING_ENABLED=true

# Feature Flags
NEXT_PUBLIC_ENABLE_PREFETCH=true
NEXT_PUBLIC_ENABLE_OFFLINE_MODE=true
```

### 6.4 Deployment Checklist

- [ ] Run `npm run build` and verify bundle sizes
- [ ] Run `npm run test` and ensure all tests pass
- [ ] Run `npm run test:e2e` for E2E validation
- [ ] Run Lighthouse audit and verify Core Web Vitals
- [ ] Verify WebSocket connections work in production
- [ ] Test offline functionality
- [ ] Verify error reporting is configured
- [ ] Check CDN caching headers
- [ ] Validate CORS configuration
- [ ] Test on multiple devices and browsers

---

## 7. File Structure Summary

```
frontend/src/
├── lib/
│   ├── optimization/
│   │   ├── index.ts                 # Barrel export
│   │   ├── lazy-loader.ts           # Lazy loading utilities
│   │   ├── code-splitting.ts        # Code splitting helpers
│   │   ├── asset-optimizer.ts       # Asset optimization
│   │   ├── performance-monitor.ts   # Core Web Vitals tracking
│   │   └── prefetch-manager.ts      # Intelligent prefetching
│   │
│   └── sync/
│       ├── index.ts                 # Barrel export
│       ├── types.ts                 # TypeScript definitions
│       ├── websocket-manager.ts     # WebSocket with reconnection
│       ├── sse-manager.ts           # Server-Sent Events
│       ├── event-bus.ts             # Pub/sub event system
│       ├── state-sync.ts            # State synchronization
│       ├── optimistic-updates.ts    # Optimistic UI patterns
│       └── error-handling.ts        # Error classification & recovery
│
├── components/
│   └── enhanced/
│       ├── AnimatedCard.tsx         # Cards with micro-interactions
│       ├── AnimatedButton.tsx       # Buttons with effects
│       ├── ResponsiveLayout.tsx     # Responsive grid system
│       ├── LoadingStates.tsx        # Skeletons & progress
│       ├── MicroInteractions.tsx    # Animation components
│       ├── FeedbackComponents.tsx   # Alerts & status
│       └── NavigationEnhanced.tsx   # Navigation components
│
└── __tests__/
    ├── optimization/                # Performance module tests
    ├── sync/                        # Sync module tests
    └── integration/                 # Integration tests

docs/
└── OPTIMIZATION_STRATEGY.md         # This document
```

---

## 8. Conclusion

This optimization strategy provides a comprehensive foundation for building a high-performance, user-friendly web application with robust real-time capabilities. The implementation includes:

1. **Performance Optimization**: Lazy loading, code splitting, asset compression, and monitoring
2. **UI/UX Improvements**: Enhanced components with micro-interactions and accessibility
3. **Real-time Sync**: WebSocket, SSE, event bus, and state synchronization
4. **Resilience**: Optimistic updates, error handling, retry mechanisms, and circuit breakers
5. **Testing**: Unit, integration, E2E, visual regression, and accessibility tests

All modules are production-ready and integrate seamlessly with the existing Chimera architecture.