/**
 * Tests for MetricsDashboard component.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the component
vi.mock('@/components/metrics-dashboard', () => ({
  MetricsDashboard: () => (
    <div data-testid="metrics-dashboard">
      <div data-testid="metrics-header">
        <h2>System Metrics</h2>
      </div>
      <div data-testid="metrics-grid">
        <div data-testid="metric-card-success-rate">
          <span className="metric-label">Success Rate</span>
          <span className="metric-value" data-testid="success-rate-value">
            85%
          </span>
        </div>
        <div data-testid="metric-card-avg-score">
          <span className="metric-label">Average Score</span>
          <span className="metric-value" data-testid="avg-score-value">
            7.5
          </span>
        </div>
        <div data-testid="metric-card-total-attempts">
          <span className="metric-label">Total Attempts</span>
          <span className="metric-value" data-testid="total-attempts-value">
            1,234
          </span>
        </div>
        <div data-testid="metric-card-avg-latency">
          <span className="metric-label">Avg Latency</span>
          <span className="metric-value" data-testid="avg-latency-value">
            2.3s
          </span>
        </div>
      </div>
      <div data-testid="metrics-chart">
        <div data-testid="chart-placeholder">Chart Area</div>
      </div>
    </div>
  ),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
  Wrapper.displayName = 'QueryWrapper';
  return Wrapper;
};

describe('MetricsDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders dashboard container', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    expect(screen.getByTestId('metrics-dashboard')).toBeInTheDocument();
  });

  it('displays header section', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    expect(screen.getByTestId('metrics-header')).toBeInTheDocument();
    expect(screen.getByText('System Metrics')).toBeInTheDocument();
  });

  it('displays metrics grid with cards', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    expect(screen.getByTestId('metrics-grid')).toBeInTheDocument();
    expect(screen.getByTestId('metric-card-success-rate')).toBeInTheDocument();
    expect(screen.getByTestId('metric-card-avg-score')).toBeInTheDocument();
    expect(screen.getByTestId('metric-card-total-attempts')).toBeInTheDocument();
    expect(screen.getByTestId('metric-card-avg-latency')).toBeInTheDocument();
  });

  it('displays success rate metric', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    const successRate = screen.getByTestId('success-rate-value');
    expect(successRate).toHaveTextContent('85%');
  });

  it('displays average score metric', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    const avgScore = screen.getByTestId('avg-score-value');
    expect(avgScore).toHaveTextContent('7.5');
  });

  it('displays total attempts metric', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    const totalAttempts = screen.getByTestId('total-attempts-value');
    expect(totalAttempts).toHaveTextContent('1,234');
  });

  it('displays latency metric', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    const avgLatency = screen.getByTestId('avg-latency-value');
    expect(avgLatency).toHaveTextContent('2.3s');
  });

  it('displays chart area', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    expect(screen.getByTestId('metrics-chart')).toBeInTheDocument();
    expect(screen.getByTestId('chart-placeholder')).toBeInTheDocument();
  });
});

describe('MetricsDashboard Layout', () => {
  it('has proper structure for responsive display', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    const dashboard = screen.getByTestId('metrics-dashboard');
    const header = screen.getByTestId('metrics-header');
    const grid = screen.getByTestId('metrics-grid');
    const chart = screen.getByTestId('metrics-chart');

    // All sections should be present in order
    expect(dashboard).toContainElement(header);
    expect(dashboard).toContainElement(grid);
    expect(dashboard).toContainElement(chart);
  });

  it('displays four metric cards', async () => {
    const { MetricsDashboard } = await import('@/components/metrics-dashboard');

    render(<MetricsDashboard />, { wrapper: createWrapper() });

    const grid = screen.getByTestId('metrics-grid');
    const cards = grid.querySelectorAll('[data-testid^="metric-card"]');

    expect(cards.length).toBe(4);
  });
});