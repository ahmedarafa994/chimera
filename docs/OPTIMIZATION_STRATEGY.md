# Chimera Web Application Optimization Strategy

## Executive Summary

This document outlines a comprehensive optimization strategy for the Chimera web application, addressing frontend performance, UI/UX improvements, and backend-frontend synchronization. The implementation includes production-ready code modules, performance benchmarks, and testing strategies.

---

## Table of Contents

1. [Frontend Performance Optimization](#1-frontend-performance-optimization)
2. [UI/UX Improvements](#2-uiux-improvements)
3. [Backend-Frontend Synchronization](#3-backend-frontend-synchronization)
4. [Performance Benchmarks](#4-performance-benchmarks)
5. [Testing Strategies](#5-testing-strategies)
6. [Implementation Guide](#6-implementation-guide)

---

## 1. Frontend Performance Optimization

### 1.1 Lazy Loading

**Location:** [`frontend/src/lib/optimization/lazy-loader.ts`](../frontend/src/lib/optimization/lazy-loader.ts)

#### Implementation

```typescript
import { LazyLoader, useLazyLoad, useLazyImage } from '@/lib/optimization';

// Component lazy loading with intersection observer
function Gallery() {
  const { ref, isLoaded, isInView } = useLazyLoad({
    threshold: 0.1,
    rootMargin: '100px',
  });

  return (
    <div ref={ref}>
      {isInView && <ExpensiveComponent />}
    </div>
  );
}

// Image lazy loading with blur placeholder
function ProductImage({ src, alt }) {
  const { imgRef, isLoaded, currentSrc } = useLazyImage(src, {
    placeholder: '/placeholder.webp',
    threshold: 0.1,
  });

  return (
    <img
      ref={imgRef}
      src={currentSrc}
      alt={alt}
      className={isLoaded ? 'opacity-100' : 'opacity-0 blur-sm'}
    />
  );
}
```

#### Features
- Intersection Observer-based visibility detection
- Configurable thresholds and root margins
- Native lazy loading fallback
- Blur-up image loading pattern
- Memory-efficient observer cleanup

### 1.2 Code Splitting

**Location:** [`frontend/src/lib/optimization/code-splitting.ts`](../frontend/src/lib/optimization/code-splitting.ts)

#### Route-Based Splitting

```typescript
import { createLazyRoute, createLazyComponent } from '@/lib/optimization';

// Lazy route with loading state
const DashboardPage = createLazyRoute(
  () => import('@/app/dashboard/page'),
  { ssr: false }
);

// Lazy component with custom loading
const HeavyChart = createLazyComponent(
  () => import('@/components/HeavyChart'),
  {
    loading: <ChartSkeleton />,
    ssr: false,
  }
);
```

#### Vendor Chunk Configuration

```typescript
// next.config.ts
import { getVendorChunks, getOptimizedSplitChunks } from '@/lib/optimization';

const nextConfig = {
  webpack: (config) => {
    config.optimization.splitChunks = getOptimizedSplitChunks();
    return config;
  },
};
```

#### Recommended Chunk Strategy

| Chunk | Contents | Max Size |
|-------|----------|----------|
| `vendor-react` | React, React DOM | 150KB |
| `vendor-ui` | Radix UI, Shadcn | 100KB |
| `vendor-query` | TanStack Query | 50KB |
| `vendor-utils` | Lodash, date-fns | 80KB |
| `commons` | Shared utilities | 50KB |

### 1.3 Asset Compression

**Location:** [`frontend/src/lib/optimization/asset-optimizer.ts`](../frontend/src/lib/optimization/asset-optimizer.ts)

#### Image Optimization

```typescript
import { ImageOptimizer, useOptimizedImage } from '@/lib/optimization';

// Automatic format selection and responsive sizing
function ProductImage({ src, alt }) {
  const optimizedSrc = useOptimizedImage(src, {
    width: 800,
    quality: 85,
    format: 'auto', // WebP with JPEG fallback
  });

  return <img src={optimizedSrc} alt={alt} loading="lazy" />;
}
```

#### Font Optimization

```typescript
import { FontOptimizer } from '@/lib/optimization';

// Preload critical fonts
FontOptimizer.preloadFont('/fonts/inter-var.woff2', {
  crossOrigin: 'anonymous',
  type: 'font/woff2',
});

// Font display swap for FOUT prevention
const fontConfig = {
  display: 'swap',
  preload: true,
  fallback: ['system-ui', 'sans-serif'],
};
```

#### Compression Targets

| Asset Type | Target Size | Compression |
|------------|-------------|-------------|
| JavaScript | < 200KB gzipped | Terser + gzip |
| CSS | < 50KB gzipped | cssnano + gzip |
| Images | < 100KB | WebP @ 85% quality |
| Fonts | < 50KB per weight | WOFF2 subset |

### 1.4 Performance Monitoring

**Location:** [`frontend/src/lib/optimization/performance-monitor.ts`](../frontend/src/lib/optimization/performance-monitor.ts)

#### Core Web Vitals Tracking

```typescript
import { PerformanceMonitor, usePerformanceMetrics } from '@/lib/optimization';

// Initialize monitoring
const monitor = PerformanceMonitor.getInstance();
monitor.startMonitoring({
  reportEndpoint: '/api/metrics',
  sampleRate: 0.1, // 10% sampling
});

// React hook for component metrics
function App() {
  const metrics = usePerformanceMetrics();
  
  useEffect(() => {
    if (metrics.LCP > 2500) {
      console.warn('LCP exceeds threshold');
    }
  }, [metrics]);
}
```

#### Benchmark Utilities

```typescript
import { benchmark, measureRender } from '@/lib/optimization';

// Measure function performance
const result = await benchmark(
  'API Call',
  () => fetchData(),
  { iterations: 100 }
);
console.log(`Average: ${result.average}ms, P95: ${result.p95}ms`);

// Measure component render
const renderTime = measureRender('Dashboard', () => {
  renderDashboard();
});
```

### 1.5 Prefetching

**Location:** [`frontend/src/lib/optimization/prefetch-manager.ts`](../frontend/src/lib/optimization/prefetch-manager.ts)

#### Intelligent Prefetching

```typescript
import { PrefetchManager, usePrefetch } from '@/lib/optimization';

// Prefetch on hover
function NavLink({ href, children }) {
  const prefetch = usePrefetch();

  return (
    <a
      href={href}
      onMouseEnter={() => prefetch.route(href)}
      onFocus={() => prefetch.route(href)}
    >
      {children}
    </a>
  );
}

// Predictive prefetching based on navigation patterns
PrefetchManager.getInstance().enablePredictive({
  maxPrefetches: 3,
  minConfidence: 0.7,
});
```

---

## 2. UI/UX Improvements

### 2.1 Component Library

**Location:** [`frontend/src/components/enhanced/`](../frontend/src/components/enhanced/)

All enhanced components wrap Shadcn/Radix primitives for accessibility and stability.

#### AnimatedCard

```tsx
import { AnimatedCard } from '@/components/enhanced/AnimatedCard';

<AnimatedCard
  variant="elevated"
  hoverEffect="lift"
  clickable
  onClick={handleClick}
>
  <CardHeader>
    <CardTitle>Feature</CardTitle>
  </CardHeader>
  <CardContent>
    Content with micro-interactions
  </CardContent>
</AnimatedCard>
```

#### AnimatedButton

```tsx
import { AnimatedButton } from '@/components/enhanced/AnimatedButton';

<AnimatedButton
  variant="primary"
  size="lg"
  loading={isSubmitting}
  success={isSuccess}
  ripple
  magnetic
>
  Submit
</AnimatedButton>
```

#### ResponsiveLayout

```tsx
import { Container, Grid, Stack, Show, Hide } from '@/components/enhanced/ResponsiveLayout';

<Container maxWidth="xl" padding="responsive">
  <Grid cols={{ base: 1, md: 2, lg: 3 }} gap={6}>
    <Show above="md">
      <Sidebar />
    </Show>
    <Stack direction="column" gap={4}>
      <MainContent />
    </Stack>
  </Grid>
</Container>
```

### 2.2 Micro-Interactions

**Location:** [`frontend/src/components/enhanced/MicroInteractions.tsx`](../frontend/src/components/enhanced/MicroInteractions.tsx)

#### Available Effects

| Effect | Use Case | Performance Impact |
|--------|----------|-------------------|
| `HoverScale` | Buttons, cards | Low (transform) |
| `HoverGlow` | CTAs, highlights | Medium (box-shadow) |
| `ScrollReveal` | Content sections | Low (intersection) |
| `Parallax` | Hero sections | Medium (scroll listener) |
| `Typewriter` | Headlines | Low (text manipulation) |
| `CountUp` | Statistics | Low (number animation) |
| `Ripple` | Touch feedback | Low (pseudo-element) |

```tsx
import { HoverScale, ScrollReveal, Typewriter } from '@/components/enhanced/MicroInteractions';

<ScrollReveal animation="fade-up" delay={200}>
  <HoverScale scale={1.02}>
    <Card>
      <Typewriter text="Welcome to Chimera" speed={50} />
    </Card>
  </HoverScale>
</ScrollReveal>
```

### 2.3 Loading States

**Location:** [`frontend/src/components/enhanced/LoadingStates.tsx`](../frontend/src/components/enhanced/LoadingStates.tsx)

#### Skeleton Patterns

```tsx
import { Skeleton, CardSkeleton, TableSkeleton } from '@/components/enhanced/LoadingStates';

// Card skeleton with shimmer
<CardSkeleton lines={3} hasImage hasActions />

// Table skeleton
<TableSkeleton rows={10} columns={5} />

// Custom skeleton composition
<div className="space-y-4">
  <Skeleton variant="text" width="60%" />
  <Skeleton variant="rectangular" height={200} />
  <Skeleton variant="circular" size={48} />
</div>
```

#### Progress Indicators

```tsx
import { Progress, CircularProgress, StepProgress } from '@/components/enhanced/LoadingStates';

// Linear progress with label
<Progress value={75} showLabel variant="gradient" />

// Circular progress
<CircularProgress value={60} size="lg" strokeWidth={4} />

// Step progress
<StepProgress
  steps={['Upload', 'Process', 'Complete']}
  currentStep={1}
  variant="numbered"
/>
```

### 2.4 Feedback Components

**Location:** [`frontend/src/components/enhanced/FeedbackComponents.tsx`](../frontend/src/components/enhanced/FeedbackComponents.tsx)

```tsx
import { AlertBanner, StatusIndicator, EmptyState } from '@/components/enhanced/FeedbackComponents';

// Alert banner with actions
<AlertBanner
  variant="warning"
  title="Session Expiring"
  description="Your session will expire in 5 minutes"
  actions={[
    { label: 'Extend', onClick: extendSession },
    { label: 'Logout', onClick: logout, variant: 'outline' },
  ]}
  dismissible
/>

// Status indicator
<StatusIndicator status="online" label="Connected" pulse />

// Empty state
<EmptyState
  icon={<SearchIcon />}
  title="No results found"
  description="Try adjusting your search criteria"
  action={{ label: 'Clear filters', onClick: clearFilters }}
/>
```

### 2.5 Navigation Components

**Location:** [`frontend/src/components/enhanced/NavigationEnhanced.tsx`](../frontend/src/components/enhanced/NavigationEnhanced.tsx)

```tsx
import { 
  Breadcrumbs, 
  TabsEnhanced, 
  Pagination,
  CommandPalette 
} from '@/components/enhanced/NavigationEnhanced';

// Breadcrumbs with icons
<Breadcrumbs
  items={[
    { label: 'Home', href: '/', icon: <HomeIcon /> },
    { label: 'Projects', href: '/projects' },
    { label: 'Chimera', current: true },
  ]}
  separator="chevron"
/>

// Enhanced tabs with badges
<TabsEnhanced
  tabs={[
    { id: 'all', label: 'All', badge: 42 },
    { id: 'active', label: 'Active', badge: 12 },
    { id: 'archived', label: 'Archived' },
  ]}
  activeTab={activeTab}
  onChange={setActiveTab}
  variant="pills"
/>

// Command palette (Cmd+K)
<CommandPalette
  commands={commands}
  placeholder="Search commands..."
  shortcut="mod+k"
/>
```

---

## 3. Backend-Frontend Synchronization

### 3.1 WebSocket Manager

**Location:** [`frontend/src/lib/sync/websocket-manager.ts`](../frontend/src/lib/sync/websocket-manager.ts)

#### Features
- Auto-reconnection with exponential backoff
- Heartbeat/ping-pong for connection health
- Message queuing during disconnection
- Request/response correlation
- Binary message support

```typescript
import { useWebSocket } from '@/lib/sync';

function RealtimeComponent() {
  const { 
    state, 
    send, 
    subscribe, 
    request 
  } = useWebSocket('wss://api.chimera.dev/ws');

  useEffect(() => {
    const unsubscribe = subscribe('update', (data) => {
      console.log('Received update:', data);
    });
    return unsubscribe;
  }, [subscribe]);

  // Request/response pattern
  const handleAction = async () => {
    const response = await request('action', { type: 'generate' });
    console.log('Response:', response);
  };

  return (
    <div>
      <StatusIndicator status={state === 'connected' ? 'online' : 'offline'} />
      <Button onClick={handleAction}>Execute</Button>
    </div>
  );
}
```

### 3.2 Server-Sent Events

**Location:** [`frontend/src/lib/sync/sse-manager.ts`](../frontend/src/lib/sync/sse-manager.ts)

#### Streaming Text Support

```typescript
import { useSSE, useStreamingText } from '@/lib/sync';

function StreamingResponse() {
  const { text, isStreaming, error } = useStreamingText(
    '/api/generate/stream',
    { autoStart: false }
  );

  return (
    <div>
      <p className={isStreaming ? 'animate-pulse' : ''}>
        {text}
        {isStreaming && <span className="cursor">|</span>}
      </p>
    </div>
  );
}
```

### 3.3 Event Bus

**Location:** [`frontend/src/lib/sync/event-bus.ts`](../frontend/src/lib/sync/event-bus.ts)

#### Typed Events

```typescript
import { useEventBus, useChannel } from '@/lib/sync';

// Subscribe to typed events
function NotificationListener() {
  const { subscribe } = useEventBus();

  useEffect(() => {
    return subscribe('notification:new', (notification) => {
      toast(notification.message);
    });
  }, [subscribe]);
}

// Channel-based communication
function ChatRoom({ roomId }) {
  const { messages, send } = useChannel(`chat:${roomId}`);

  return (
    <div>
      {messages.map((msg) => (
        <Message key={msg.id} {...msg} />
      ))}
      <Input onSubmit={(text) => send({ type: 'message', text })} />
    </div>
  );
}
```

### 3.4 State Synchronization

**Location:** [`frontend/src/lib/sync/state-sync.ts`](../frontend/src/lib/sync/state-sync.ts)

#### Conflict Resolution Strategies

| Strategy | Use Case | Behavior |
|----------|----------|----------|
| `client-wins` | User preferences | Local changes override server |
| `server-wins` | Shared data | Server is source of truth |
| `merge` | Collaborative editing | Deep merge with timestamps |
| `manual` | Critical data | User resolves conflicts |

```typescript
import { useSyncedState, useOfflineSync } from '@/lib/sync';

function CollaborativeEditor() {
  const [document, setDocument] = useSyncedState('document', initialDoc, {
    conflictResolution: 'merge',
    syncInterval: 1000,
    onConflict: (local, remote) => {
      // Custom merge logic
      return mergeDocuments(local, remote);
    },
  });

  return <Editor value={document} onChange={setDocument} />;
}

// Offline support
function OfflineCapableForm() {
  const { 
    data, 
    setData, 
    pendingChanges, 
    syncStatus 
  } = useOfflineSync('form-data');

  return (
    <form>
      <Input value={data.name} onChange={(e) => setData({ name: e.target.value })} />
      {pendingChanges > 0 && (
        <Badge>{pendingChanges} pending changes</Badge>
      )}
      <StatusIndicator status={syncStatus} />
    </form>
  );
}
```

### 3.5 Optimistic Updates

**Location:** [`frontend/src/lib/sync/optimistic-updates.ts`](../frontend/src/lib/sync/optimistic-updates.ts)

#### React Query Integration

```typescript
import { useQueryOptimisticMutation, useResilientMutation } from '@/lib/sync';

function TodoList() {
  const addTodo = useQueryOptimisticMutation(
    ['todos'],
    (newTodo) => api.createTodo(newTodo),
    {
      optimisticUpdate: (newTodo, oldTodos) => [...(oldTodos || []), newTodo],
      onError: (error) => toast.error('Failed to add todo'),
    }
  );

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      addTodo.mutate({ title: inputValue, completed: false });
    }}>
      <Input value={inputValue} onChange={setInputValue} />
      <Button type="submit" loading={addTodo.isPending}>
        Add
      </Button>
    </form>
  );
}
```

#### Resilient Mutations with Circuit Breaker

```typescript
import { useResilientMutation } from '@/lib/sync';

function CriticalAction() {
  const mutation = useResilientMutation(
    ['critical-data'],
    (data) => api.criticalUpdate(data),
    {
      optimisticUpdate: (data, old) => ({ ...old, ...data }),
      retry: {
        maxAttempts: 3,
        baseDelay: 1000,
        backoffMultiplier: 2,
      },
      circuitBreaker: {
        failureThreshold: 5,
        resetTimeout: 30000,
      },
    }
  );

  return (
    <div>
      <Button 
        onClick={() => mutation.mutate(data)}
        disabled={!mutation.canExecute}
      >
        Update
      </Button>
      {mutation.circuitState.state === 'open' && (
        <Alert variant="warning">
          Service temporarily unavailable. Retrying in {mutation.circuitState.nextAttempt}...
        </Alert>
      )}
    </div>
  );
}
```

### 3.6 Error Handling

**Location:** [`frontend/src/lib/sync/error-handling.ts`](../frontend/src/lib/sync/error-handling.ts)

#### Error Classification

```typescript
import { useErrorHandler, useAsyncError } from '@/lib/sync';

function DataFetcher() {
  const { error, handleError, clearError, displayConfig } = useErrorHandler();

  const fetchData = async () => {
    try {
      const data = await api.getData();
      return data;
    } catch (err) {
      handleError(err, { component: 'DataFetcher' });
    }
  };

  if (error && displayConfig) {
    return (
      <AlertBanner
        variant={displayConfig.severity}
        title={displayConfig.title}
        description={displayConfig.message}
        actions={displayConfig.actions}
        onDismiss={displayConfig.dismissible ? clearError : undefined}
      />
    );
  }

  return <DataDisplay />;
}
```

#### Recovery Strategies

```typescript
import { 
  createNetworkRecoveryStrategy, 
  createAuthRecoveryStrategy,
  attemptRecovery 
} from '@/lib/sync';

// Configure recovery strategies
const recoveryStrategies = [
  createNetworkRecoveryStrategy(() => {
    // Retry failed requests when back online
    queryClient.refetchQueries();
  }),
  createAuthRecoveryStrategy(async () => {
    // Attempt token refresh
    return await refreshAuthToken();
  }),
];

// Attempt automatic recovery
async function handleError(error: AppError) {
  const recovered = await attemptRecovery(error, recoveryStrategies);
  if (!recovered) {
    // Show error to user
    showErrorNotification(error);
  }
}
```

---

## 4. Performance Benchmarks

### 4.1 Target Metrics

#### Core Web Vitals

| Metric | Target | Good | Needs Improvement | Poor |
|--------|--------|------|-------------------|------|
| **LCP** (Largest Contentful Paint) | < 2.0s | < 2.5s | 2.5s - 4.0s | > 4.0s |
| **FID** (First Input Delay) | < 50ms | < 100ms | 100ms - 300ms | > 300ms |
| **CLS** (Cumulative Layout Shift) | < 0.05 | < 0.1 | 0.1 - 0.25 | > 0.25 |
| **INP** (Interaction to Next Paint) | < 100ms | < 200ms | 200ms - 500ms | > 500ms |
| **TTFB** (Time to First Byte) | < 200ms | < 800ms | 800ms - 1800ms | > 1800ms |

#### Application-Specific Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Initial Bundle Size | < 200KB gzipped | `npm run build && gzip -9` |
| Time to Interactive | < 3.0s | Lighthouse |
| First Contentful Paint | < 1.5s | Lighthouse |
| API Response Time (P95) | < 500ms | Backend metrics |
| WebSocket Reconnection | < 2.0s | Custom monitoring |
| Optimistic Update Latency | < 50ms | Performance API |

### 4.2 Benchmark Scripts

```typescript
// frontend/src/lib/optimization/benchmarks.ts

import { benchmark, PerformanceMonitor } from '@/lib/optimization';

export async function runBenchmarks() {
  const results = {
    componentRender: await benchmarkComponentRender(),
    apiCalls: await benchmarkApiCalls(),
    stateUpdates: await benchmarkStateUpdates(),
    webSocket: await benchmarkWebSocket(),
  };

  return results;
}

async function benchmarkComponentRender() {
  return benchmark('Component Render', async () => {
    const root = createRoot(container);
    root.render(<ComplexComponent />);
    await waitForRender();
    root.unmount();
  }, { iterations: 100 });
}

async function benchmarkApiCalls() {
  return benchmark('API Call', async () => {
    await fetch('/api/health');
  }, { iterations: 50 });
}

async function benchmarkStateUpdates() {
  return benchmark('State Update', () => {
    store.dispatch(updateAction(largePayload));
  }, { iterations: 1000 });
}

async function benchmarkWebSocket() {
  return benchmark('WebSocket Message', async () => {
    await wsManager.request('ping', {});
  }, { iterations: 100 });
}
```

### 4.3 Monitoring Dashboard

```typescript
// frontend/src/components/PerformanceDashboard.tsx

import { usePerformanceMetrics } from '@/lib/optimization';

export function PerformanceDashboard() {
  const metrics = usePerformanceMetrics();

  return (
    <Grid cols={4} gap={4}>
      <MetricCard
        label="LCP"
        value={metrics.LCP}
        target={2000}
        unit="ms"
        status={metrics.LCP < 2500 ? 'good' : 'poor'}
      />
      <MetricCard
        label="FID"
        value={metrics.FID}
        target={50}
        unit="ms"
        status={metrics.FID < 100 ? 'good' : 'poor'}
      />
      <MetricCard
        label="CLS"
        value={metrics.CLS}
        target={0.05}
        status={metrics.CLS < 0.1 ? 'good' : 'poor'}
      />
      <MetricCard
        label="INP"
        value={metrics.INP}
        target={100}
        unit="ms"
        status={metrics.INP < 200 ? 'good' : 'poor'}
      />
    </Grid>
  );
}
```

---

## 5. Testing Strategies

### 5.1 Unit Tests

**Location:** `frontend/src/__tests__/`

#### Performance Module Tests

```typescript
// frontend/src/__tests__/optimization/lazy-loader.test.ts

import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useLazyLoad, useLazyImage } from '@/lib/optimization';

describe('useLazyLoad', () => {
  it('should detect when element enters viewport', async () => {
    const mockIntersectionObserver = vi.fn();
    mockIntersectionObserver.mockReturnValue({
      observe: vi.fn(),
      unobserve: vi.fn(),
      disconnect: vi.fn(),
    });
    window.IntersectionObserver = mockIntersectionObserver;

    const { result } = renderHook(() => useLazyLoad());

    expect(result.current.isInView).toBe(false);

    // Simulate intersection
    act(() => {
      const callback = mockIntersectionObserver.mock.calls[0][0];
      callback([{ isIntersecting: true }]);
    });

    expect(result.current.isInView).toBe(true);
  });

  it('should respect threshold option', () => {
    const { result } = renderHook(() => 
      useLazyLoad({ threshold: 0.5 })
    );

    expect(window.IntersectionObserver).toHaveBeenCalledWith(
      expect.any(Function),
      expect.objectContaining({ threshold: 0.5 })
    );
  });
});
```

#### Sync Module Tests

```typescript
// frontend/src/__tests__/sync/optimistic-updates.test.ts

import { describe, it, expect, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useQueryOptimisticMutation } from '@/lib/sync';

describe('useQueryOptimisticMutation', () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const wrapper = ({ children }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );

  beforeEach(() => {
    queryClient.clear();
  });

  it('should apply optimistic update immediately', async () => {
    const mutationFn = vi.fn().mockResolvedValue({ id: 1, name: 'Test' });
    
    queryClient.setQueryData(['items'], []);

    const { result } = renderHook(
      () => useQueryOptimisticMutation(
        ['items'],
        mutationFn,
        {
          optimisticUpdate: (newItem, old) => [...(old || []), newItem],
        }
      ),
      { wrapper }
    );

    act(() => {
      result.current.mutate({ id: 1, name: 'Test' });
    });

    // Optimistic update should be applied immediately
    expect(queryClient.getQueryData(['items'])).toEqual([
      { id: 1, name: 'Test' },
    ]);
  });

  it('should rollback on error', async () => {
    const mutationFn = vi.fn().mockRejectedValue(new Error('Failed'));
    
    queryClient.setQueryData(['items'], [{ id: 0, name: 'Existing' }]);

    const { result } = renderHook(
      () => useQueryOptimisticMutation(
        ['items'],
        mutationFn,
        {
          optimisticUpdate: (newItem, old) => [...(old || []), newItem],
        }
      ),
      { wrapper }
    );

    act(() => {
      result.current.mutate({ id: 1, name: 'Test' });
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    // Should rollback to original state
    expect(queryClient.getQueryData(['items'])).toEqual([
      { id: 0, name: 'Existing' },
    ]);
  });
});
```

#### Error Handling Tests

```typescript
// frontend/src/__tests__/sync/error-handling.test.ts

import { describe, it, expect } from 'vitest';
import { createAppError, isAppError, ERROR_CODES } from '@/lib/sync';

describe('createAppError', () => {
  it('should classify network errors correctly', () => {
    const error = new TypeError('Failed to fetch');
    const appError = createAppError(error);

    expect(appError.code).toBe(ERROR_CODES.NETWORK_ERROR);
    expect(appError.category).toBe('network');
    expect(appError.retryable).toBe(true);
  });

  it('should classify HTTP errors by status code', () => {
    const axiosError = {
      response: { status: 401, data: { message: 'Unauthorized' } },
      config: { url: '/api/data', method: 'GET' },
      message: 'Request failed',
    };

    const appError = createAppError(axiosError);

    expect(appError.code).toBe(ERROR_CODES.UNAUTHORIZED);
    expect(appError.category).toBe('authentication');
    expect(appError.retryable).toBe(false);
  });

  it('should preserve context', () => {
    const error = new Error('Test error');
    const appError = createAppError(error, { userId: '123' });

    expect(appError.context).toEqual({ userId: '123' });
  });
});