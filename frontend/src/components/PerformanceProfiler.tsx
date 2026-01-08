// React performance profiler component for identifying rendering bottlenecks
import { Profiler, ProfilerOnRenderCallback, ReactNode } from 'react';
import { performanceMonitor } from '@/lib/performance-monitor';

interface PerformanceProfilerProps {
  id: string;
  children: ReactNode;
  enabled?: boolean;
}

interface RenderMetrics {
  id: string;
  phase: 'mount' | 'update' | 'nested-update';
  actualDuration: number;
  baseDuration: number;
  startTime: number;
  commitTime: number;
  interactions: Set<any>;
}

class ReactPerformanceAnalyzer {
  private renderMetrics: Map<string, RenderMetrics[]> = new Map();
  private slowComponents: Set<string> = new Set();
  private renderThreshold = 16; // 16ms for 60fps

  public onRender: ProfilerOnRenderCallback = (
    id,
    phase,
    actualDuration,
    baseDuration,
    startTime,
    commitTime
  ) => {
    const metrics: RenderMetrics = {
      id,
      phase,
      actualDuration,
      baseDuration,
      startTime,
      commitTime,
      interactions: new Set() // Default empty set for React 19 compatibility
    };

    // Store metrics
    if (!this.renderMetrics.has(id)) {
      this.renderMetrics.set(id, []);
    }
    this.renderMetrics.get(id)!.push(metrics);

    // Track slow renders
    if (actualDuration > this.renderThreshold) {
      this.slowComponents.add(id);
      performanceMonitor.trackReactComponent(id, actualDuration);
    }

    // Send to performance monitor
    this.sendMetricsToMonitor(metrics);
  };

  private sendMetricsToMonitor(metrics: RenderMetrics) {
    fetch('/api/performance/react-renders', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sessionId: performanceMonitor.getSessionMetrics().sessionId,
        ...metrics
      })
    }).catch(error => {
      console.warn('Failed to send React render metrics:', error);
    });
  }

  public getSlowComponents(): string[] {
    return Array.from(this.slowComponents);
  }

  public getComponentStats(componentId: string) {
    const metrics = this.renderMetrics.get(componentId) || [];

    if (metrics.length === 0) return null;

    const durations = metrics.map(m => m.actualDuration);
    const phases = metrics.reduce((acc, m) => {
      acc[m.phase] = (acc[m.phase] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      totalRenders: metrics.length,
      averageDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
      maxDuration: Math.max(...durations),
      minDuration: Math.min(...durations),
      phases,
      slowRenders: durations.filter(d => d > this.renderThreshold).length
    };
  }

  public generateReport(): string {
    const report = {
      timestamp: new Date().toISOString(),
      totalComponents: this.renderMetrics.size,
      slowComponents: this.slowComponents.size,
      componentStats: Object.fromEntries(
        Array.from(this.renderMetrics.keys()).map(id => [
          id,
          this.getComponentStats(id)
        ])
      )
    };

    return JSON.stringify(report, null, 2);
  }
}

export const reactAnalyzer = new ReactPerformanceAnalyzer();

export const PerformanceProfiler: React.FC<PerformanceProfilerProps> = ({
  id,
  children,
  enabled = process.env.NODE_ENV === 'development'
}) => {
  if (!enabled) {
    return <>{children}</>;
  }

  return (
    <Profiler id={id} onRender={reactAnalyzer.onRender}>
      {children}
    </Profiler>
  );
};

export default PerformanceProfiler;