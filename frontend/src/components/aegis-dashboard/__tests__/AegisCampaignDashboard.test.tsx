/**
 * Tests for AegisCampaignDashboard component and related sub-components.
 *
 * Covers:
 * - Component rendering and loading states
 * - WebSocket hook behavior mocking
 * - Metrics aggregation and display
 * - Reconnection logic
 * - Error states
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the useAegisTelemetry hook
const mockUseAegisTelemetry = vi.fn();
vi.mock('@/lib/hooks/useAegisTelemetry', () => ({
  useAegisTelemetry: () => mockUseAegisTelemetry(),
}));

// Mock the api-config
vi.mock('@/lib/api-config', () => ({
  getApiConfig: vi.fn(() => ({
    backendApiUrl: 'http://localhost:8001/api/v1',
  })),
}));

// Mock Recharts to avoid canvas issues in tests
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  AreaChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="area-chart">{children}</div>
  ),
  Area: () => <div data-testid="area" />,
  BarChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="bar-chart">{children}</div>
  ),
  Bar: () => <div data-testid="bar" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  ReferenceLine: () => <div data-testid="reference-line" />,
  Legend: () => <div data-testid="legend" />,
  Cell: () => <div data-testid="cell" />,
}));

// Import types and defaults for test data
import {
  WebSocketConnectionStatus,
  CampaignStatus,
  TechniqueCategory,
  AttackMetrics,
  TokenUsage,
  LatencyMetrics,
  TechniquePerformance,
  CampaignSummary,
  PromptEvolution,
  createDefaultAttackMetrics,
  createDefaultTokenUsage,
  createDefaultLatencyMetrics,
  createDefaultDashboardState,
} from '@/types/aegis-telemetry';

// ============================================================================
// Test Data Factories
// ============================================================================

function createMockAttackMetrics(overrides: Partial<AttackMetrics> = {}): AttackMetrics {
  return {
    ...createDefaultAttackMetrics(),
    success_rate: 65.5,
    total_attempts: 100,
    successful_attacks: 65,
    failed_attacks: 35,
    current_streak: 3,
    best_score: 95.2,
    average_score: 72.3,
    ...overrides,
  };
}

function createMockTokenUsage(overrides: Partial<TokenUsage> = {}): TokenUsage {
  return {
    ...createDefaultTokenUsage(),
    prompt_tokens: 15000,
    completion_tokens: 8000,
    total_tokens: 23000,
    cost_estimate_usd: 0.45,
    provider: 'google',
    model: 'gemini-1.5-pro',
    ...overrides,
  };
}

function createMockLatencyMetrics(overrides: Partial<LatencyMetrics> = {}): LatencyMetrics {
  return {
    ...createDefaultLatencyMetrics(),
    api_latency_ms: 850,
    processing_latency_ms: 150,
    total_latency_ms: 1000,
    avg_latency_ms: 920,
    p50_latency_ms: 800,
    p95_latency_ms: 1500,
    p99_latency_ms: 2200,
    min_latency_ms: 450,
    max_latency_ms: 2500,
    ...overrides,
  };
}

function createMockTechniquePerformance(
  name: string,
  overrides: Partial<TechniquePerformance> = {}
): TechniquePerformance {
  return {
    technique_name: name,
    technique_category: TechniqueCategory.AUTODAN,
    success_count: 15,
    failure_count: 5,
    total_applications: 20,
    success_rate: 75,
    avg_score: 72.5,
    best_score: 92,
    avg_execution_time_ms: 1200,
    ...overrides,
  };
}

function createMockCampaignSummary(
  overrides: Partial<CampaignSummary> = {}
): CampaignSummary {
  return {
    campaign_id: 'test-campaign-123',
    status: CampaignStatus.RUNNING,
    objective: 'Test objective for jailbreak campaign',
    started_at: new Date().toISOString(),
    completed_at: null,
    duration_seconds: 120,
    current_iteration: 5,
    max_iterations: 20,
    attack_metrics: createMockAttackMetrics(),
    technique_breakdown: [
      createMockTechniquePerformance('AutoDAN'),
      createMockTechniquePerformance('GPTFuzz', {
        technique_category: TechniqueCategory.GPTFUZZ,
      }),
    ],
    token_usage: createMockTokenUsage(),
    latency_metrics: createMockLatencyMetrics(),
    best_prompt: 'This is the best prompt so far',
    best_score: 95.2,
    target_model: 'gpt-4',
    ...overrides,
  };
}

function createMockPromptEvolution(iteration: number): PromptEvolution {
  return {
    iteration,
    original_prompt: `Original prompt ${iteration}`,
    evolved_prompt: `Evolved prompt ${iteration}`,
    score: 60 + iteration * 5,
    improvement: 5,
    techniques_applied: ['AutoDAN'],
    is_successful: iteration > 3,
    timestamp: new Date().toISOString(),
  };
}

function createDefaultMockHookReturn() {
  return {
    connectionStatus: 'disconnected' as WebSocketConnectionStatus,
    dashboardState: createDefaultDashboardState(),
    metrics: createDefaultAttackMetrics(),
    campaignSummary: null,
    techniqueBreakdown: [],
    tokenUsage: createDefaultTokenUsage(),
    latencyMetrics: createDefaultLatencyMetrics(),
    recentEvents: [],
    successRateHistory: [],
    tokenUsageHistory: [],
    latencyHistory: [],
    promptEvolutions: [],
    error: null,
    reconnectAttempts: 0,
    isConnected: false,
    reconnect: vi.fn(),
    disconnect: vi.fn(),
    requestSummary: vi.fn(),
    sendPing: vi.fn(),
  };
}

// ============================================================================
// Component Tests
// ============================================================================

describe('AegisCampaignDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAegisTelemetry.mockReturnValue(createDefaultMockHookReturn());
  });

  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../AegisCampaignDashboard');
      expect(module).toBeDefined();
      expect(module.AegisCampaignDashboard).toBeDefined();
    });
  });

  describe('Empty State', () => {
    it('shows empty state when no campaign is connected', async () => {
      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard />);

      expect(screen.getByText('No Campaign Connected')).toBeInTheDocument();
      expect(screen.getByText(/Enter a campaign ID above/)).toBeInTheDocument();
    });

    it('displays feature badges in empty state', async () => {
      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard />);

      expect(screen.getByText('Real-time Updates')).toBeInTheDocument();
      expect(screen.getByText('Attack Metrics')).toBeInTheDocument();
      expect(screen.getByText('Technique Analysis')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('shows loading skeleton when connecting', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connecting',
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      // Should show skeleton elements (the dashboard skeleton creates multiple skeleton items)
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  describe('Connected State', () => {
    it('shows dashboard content when connected', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connected',
        isConnected: true,
        campaignSummary: createMockCampaignSummary(),
        metrics: createMockAttackMetrics(),
        techniqueBreakdown: [createMockTechniquePerformance('AutoDAN')],
        tokenUsage: createMockTokenUsage(),
        latencyMetrics: createMockLatencyMetrics(),
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      // Should not show empty state
      expect(screen.queryByText('No Campaign Connected')).not.toBeInTheDocument();
    });

    it('displays campaign header with status badge', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connected',
        isConnected: true,
        campaignSummary: createMockCampaignSummary({ status: CampaignStatus.RUNNING }),
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByText('Aegis Campaign Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Running')).toBeInTheDocument();
    });

    it('displays campaign objective', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connected',
        isConnected: true,
        campaignSummary: createMockCampaignSummary({
          objective: 'Test jailbreak objective',
        }),
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByText(/Objective:/)).toBeInTheDocument();
      expect(screen.getByText(/Test jailbreak objective/)).toBeInTheDocument();
    });

    it('displays target model', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connected',
        isConnected: true,
        campaignSummary: createMockCampaignSummary({
          target_model: 'gpt-4-turbo',
        }),
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByText(/Target:/)).toBeInTheDocument();
      expect(screen.getByText(/gpt-4-turbo/)).toBeInTheDocument();
    });
  });

  describe('Campaign Selector', () => {
    it('renders campaign ID input field', async () => {
      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard />);

      const input = screen.getByPlaceholderText(/Enter campaign ID/);
      expect(input).toBeInTheDocument();
    });

    it('updates input value when typing', async () => {
      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard />);

      const input = screen.getByPlaceholderText(/Enter campaign ID/) as HTMLInputElement;
      await userEvent.type(input, 'new-campaign-456');

      expect(input.value).toBe('new-campaign-456');
    });

    it('shows connect button', async () => {
      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard />);

      expect(screen.getByRole('button', { name: /Connect/ })).toBeInTheDocument();
    });

    it('disables connect button when input is empty', async () => {
      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard />);

      const connectButton = screen.getByRole('button', { name: /Connect/ });
      expect(connectButton).toBeDisabled();
    });

    it('enables connect button when campaign ID is entered', async () => {
      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard />);

      const input = screen.getByPlaceholderText(/Enter campaign ID/);
      await userEvent.type(input, 'test-campaign');

      const connectButton = screen.getByRole('button', { name: /Connect/ });
      expect(connectButton).not.toBeDisabled();
    });

    it('shows Connected state when connected', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connected',
        isConnected: true,
        campaignSummary: createMockCampaignSummary(),
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByRole('button', { name: /Connected/ })).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('displays error message when there is an error', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'error',
        error: {
          error_code: 'CONNECTION_FAILED',
          error_message: 'Failed to connect to WebSocket',
          severity: 'high',
          component: 'websocket',
          recoverable: true,
        },
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByText('CONNECTION_FAILED')).toBeInTheDocument();
      expect(screen.getByText('Failed to connect to WebSocket')).toBeInTheDocument();
    });

    it('shows retry button for recoverable errors', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'error',
        error: {
          error_code: 'CONNECTION_FAILED',
          error_message: 'Failed to connect',
          severity: 'high',
          component: 'websocket',
          recoverable: true,
        },
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByRole('button', { name: /Retry Connection/ })).toBeInTheDocument();
    });

    it('calls reconnect when retry button is clicked', async () => {
      const mockReconnect = vi.fn();
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'error',
        reconnect: mockReconnect,
        error: {
          error_code: 'CONNECTION_FAILED',
          error_message: 'Failed to connect',
          severity: 'high',
          component: 'websocket',
          recoverable: true,
        },
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      const retryButton = screen.getByRole('button', { name: /Retry Connection/ });
      await userEvent.click(retryButton);

      expect(mockReconnect).toHaveBeenCalled();
    });
  });

  describe('Iteration Progress', () => {
    it('displays iteration progress bar when connected', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connected',
        isConnected: true,
        campaignSummary: createMockCampaignSummary({
          current_iteration: 10,
          max_iterations: 20,
        }),
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByText('Iteration Progress')).toBeInTheDocument();
      expect(screen.getByText('10 / 20')).toBeInTheDocument();
    });
  });

  describe('Techniques Summary Card', () => {
    it('displays techniques count when connected', async () => {
      mockUseAegisTelemetry.mockReturnValue({
        ...createDefaultMockHookReturn(),
        connectionStatus: 'connected',
        isConnected: true,
        campaignSummary: createMockCampaignSummary(),
        techniqueBreakdown: [
          createMockTechniquePerformance('AutoDAN'),
          createMockTechniquePerformance('GPTFuzz'),
          createMockTechniquePerformance('ChimeraFraming'),
        ],
      });

      const { AegisCampaignDashboard } = await import('../AegisCampaignDashboard');

      render(<AegisCampaignDashboard initialCampaignId="test-campaign" autoConnect />);

      expect(screen.getByText('Techniques Applied')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument();
      expect(screen.getByText('unique techniques')).toBeInTheDocument();
    });
  });
});

// ============================================================================
// ConnectionStatus Component Tests
// ============================================================================

describe('ConnectionStatus Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../ConnectionStatus');
      expect(module).toBeDefined();
      expect(module.ConnectionStatus).toBeDefined();
    });
  });

  describe('Status Display', () => {
    it('shows Connected status with correct styling', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="connected"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('Real-time telemetry stream active')).toBeInTheDocument();
    });

    it('shows Disconnected status', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="disconnected"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.getByText('Disconnected')).toBeInTheDocument();
      expect(screen.getByText('No active connection')).toBeInTheDocument();
    });

    it('shows Connecting status', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="connecting"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.getByText('Connecting')).toBeInTheDocument();
      expect(screen.getByText('Establishing WebSocket connection...')).toBeInTheDocument();
    });

    it('shows Reconnecting status with attempt counter', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="reconnecting"
          onReconnect={vi.fn()}
          reconnectAttempts={3}
        />
      );

      expect(screen.getByText('Reconnecting')).toBeInTheDocument();
      expect(screen.getByText('Attempt 3/5')).toBeInTheDocument();
    });

    it('shows Error status', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="error"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.getByText('Connection error occurred')).toBeInTheDocument();
    });
  });

  describe('Reconnect Button', () => {
    it('shows reconnect button when disconnected', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="disconnected"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.getByRole('button', { name: /Reconnect/i })).toBeInTheDocument();
    });

    it('shows reconnect button when in error state', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="error"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.getByRole('button', { name: /Reconnect/i })).toBeInTheDocument();
    });

    it('hides reconnect button when connected', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="connected"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.queryByRole('button', { name: /Reconnect/i })).not.toBeInTheDocument();
    });

    it('calls onReconnect when clicked', async () => {
      const mockReconnect = vi.fn();
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="disconnected"
          onReconnect={mockReconnect}
        />
      );

      const button = screen.getByRole('button', { name: /Reconnect/i });
      await userEvent.click(button);

      expect(mockReconnect).toHaveBeenCalled();
    });
  });

  describe('Compact Mode', () => {
    it('renders compact version when compact prop is true', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="connected"
          onReconnect={vi.fn()}
          compact
        />
      );

      // In compact mode, description is not shown
      expect(screen.queryByText('Real-time telemetry stream active')).not.toBeInTheDocument();
      // But status label should still be visible
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });

  describe('Live Indicator', () => {
    it('shows Live indicator when connected', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="connected"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.getByText('Live')).toBeInTheDocument();
    });

    it('does not show Live indicator when disconnected', async () => {
      const { ConnectionStatus } = await import('../ConnectionStatus');

      render(
        <ConnectionStatus
          status="disconnected"
          onReconnect={vi.fn()}
        />
      );

      expect(screen.queryByText('Live')).not.toBeInTheDocument();
    });
  });
});

// ============================================================================
// SuccessRateCard Component Tests
// ============================================================================

describe('SuccessRateCard Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../SuccessRateCard');
      expect(module).toBeDefined();
      expect(module.SuccessRateCard).toBeDefined();
    });
  });

  describe('Metrics Display', () => {
    it('displays success rate percentage', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics({ success_rate: 75 })}
          successRateHistory={[]}
        />
      );

      expect(screen.getByText('75.0%')).toBeInTheDocument();
    });

    it('displays total attempts', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics({ total_attempts: 150 })}
          successRateHistory={[]}
        />
      );

      expect(screen.getByText('150')).toBeInTheDocument();
    });

    it('displays successful attacks count', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics({ successful_attacks: 85 })}
          successRateHistory={[]}
        />
      );

      expect(screen.getByText('85')).toBeInTheDocument();
    });

    it('displays failed attacks count', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics({ failed_attacks: 65 })}
          successRateHistory={[]}
        />
      );

      expect(screen.getByText('65')).toBeInTheDocument();
    });

    it('displays best score', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics({ best_score: 92.5 })}
          successRateHistory={[]}
        />
      );

      expect(screen.getByText('92.5')).toBeInTheDocument();
    });
  });

  describe('Trend Indicator', () => {
    it('shows upward trend when specified', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics()}
          successRateHistory={[]}
          trend="up"
          trendChange={10}
        />
      );

      expect(screen.getByText('+10.0%')).toBeInTheDocument();
    });

    it('shows downward trend when specified', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics()}
          successRateHistory={[]}
          trend="down"
          trendChange={-5}
        />
      );

      expect(screen.getByText('-5.0%')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('shows loading skeleton when loading', async () => {
      const { SuccessRateCard } = await import('../SuccessRateCard');

      render(
        <SuccessRateCard
          metrics={createMockAttackMetrics()}
          successRateHistory={[]}
          loading
        />
      );

      // Should show skeleton elements
      const skeletons = document.querySelectorAll('.animate-pulse');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });
});

// ============================================================================
// TokenUsageCard Component Tests
// ============================================================================

describe('TokenUsageCard Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../TokenUsageCard');
      expect(module).toBeDefined();
      expect(module.TokenUsageCard).toBeDefined();
    });
  });

  describe('Token Display', () => {
    it('displays cost estimate', async () => {
      const { TokenUsageCard } = await import('../TokenUsageCard');

      render(
        <TokenUsageCard
          tokenUsage={createMockTokenUsage({ cost_estimate_usd: 0.55 })}
          tokenUsageHistory={[]}
          successfulAttacks={10}
          totalAttacks={20}
        />
      );

      expect(screen.getByText('$0.55')).toBeInTheDocument();
    });

    it('displays prompt tokens', async () => {
      const { TokenUsageCard } = await import('../TokenUsageCard');

      render(
        <TokenUsageCard
          tokenUsage={createMockTokenUsage({ prompt_tokens: 12000 })}
          tokenUsageHistory={[]}
          successfulAttacks={10}
          totalAttacks={20}
        />
      );

      // Token count may be formatted (12K, 12,000, etc)
      const content = document.body.textContent;
      expect(content).toContain('12');
    });

    it('displays completion tokens', async () => {
      const { TokenUsageCard } = await import('../TokenUsageCard');

      render(
        <TokenUsageCard
          tokenUsage={createMockTokenUsage({ completion_tokens: 6000 })}
          tokenUsageHistory={[]}
          successfulAttacks={10}
          totalAttacks={20}
        />
      );

      const content = document.body.textContent;
      expect(content).toContain('6');
    });
  });
});

// ============================================================================
// LatencyCard Component Tests
// ============================================================================

describe('LatencyCard Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../LatencyCard');
      expect(module).toBeDefined();
      expect(module.LatencyCard).toBeDefined();
    });
  });

  describe('Latency Display', () => {
    it('displays average latency', async () => {
      const { LatencyCard } = await import('../LatencyCard');

      render(
        <LatencyCard
          latencyMetrics={createMockLatencyMetrics({ avg_latency_ms: 920 })}
          latencyHistory={[]}
        />
      );

      // Latency may be displayed in ms or s
      const content = document.body.textContent;
      expect(content).toContain('920');
    });
  });
});

// ============================================================================
// TechniqueBreakdown Component Tests
// ============================================================================

describe('TechniqueBreakdown Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../TechniqueBreakdown');
      expect(module).toBeDefined();
      expect(module.TechniqueBreakdown).toBeDefined();
    });
  });

  describe('Techniques Display', () => {
    it('displays technique names', async () => {
      const { TechniqueBreakdown } = await import('../TechniqueBreakdown');

      render(
        <TechniqueBreakdown
          techniques={[
            createMockTechniquePerformance('AutoDAN'),
            createMockTechniquePerformance('GPTFuzz'),
          ]}
        />
      );

      expect(screen.getByText('AutoDAN')).toBeInTheDocument();
      expect(screen.getByText('GPTFuzz')).toBeInTheDocument();
    });

    it('shows empty state when no techniques', async () => {
      const { TechniqueBreakdown } = await import('../TechniqueBreakdown');

      render(<TechniqueBreakdown techniques={[]} />);

      expect(screen.getByText(/No techniques applied yet/i)).toBeInTheDocument();
    });
  });
});

// ============================================================================
// LiveEventFeed Component Tests
// ============================================================================

describe('LiveEventFeed Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../LiveEventFeed');
      expect(module).toBeDefined();
      expect(module.LiveEventFeed).toBeDefined();
    });
  });

  describe('Events Display', () => {
    it('shows empty state when no events', async () => {
      const { LiveEventFeed } = await import('../LiveEventFeed');

      render(<LiveEventFeed events={[]} />);

      expect(screen.getByText(/No events yet/i)).toBeInTheDocument();
    });
  });
});

// ============================================================================
// PromptEvolutionTimeline Component Tests
// ============================================================================

describe('PromptEvolutionTimeline Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../PromptEvolutionTimeline');
      expect(module).toBeDefined();
      expect(module.PromptEvolutionTimeline).toBeDefined();
    });
  });

  describe('Evolution Display', () => {
    it('shows empty state when no evolutions', async () => {
      const { PromptEvolutionTimeline } = await import('../PromptEvolutionTimeline');

      render(<PromptEvolutionTimeline evolutions={[]} />);

      expect(screen.getByText(/No prompt evolutions yet/i)).toBeInTheDocument();
    });

    it('displays prompt evolutions', async () => {
      const { PromptEvolutionTimeline } = await import('../PromptEvolutionTimeline');

      const evolutions = [
        createMockPromptEvolution(1),
        createMockPromptEvolution(2),
      ];

      render(<PromptEvolutionTimeline evolutions={evolutions} />);

      // Should show iteration numbers
      expect(screen.getByText(/Iteration 1/i)).toBeInTheDocument();
      expect(screen.getByText(/Iteration 2/i)).toBeInTheDocument();
    });
  });
});

// ============================================================================
// SuccessRateTrendChart Component Tests
// ============================================================================

describe('SuccessRateTrendChart Component', () => {
  describe('Module Import', () => {
    it('can be imported without errors', async () => {
      const module = await import('../SuccessRateTrendChart');
      expect(module).toBeDefined();
      expect(module.SuccessRateTrendChart).toBeDefined();
    });
  });

  describe('Chart Display', () => {
    it('shows empty state when no data', async () => {
      const { SuccessRateTrendChart } = await import('../SuccessRateTrendChart');

      render(<SuccessRateTrendChart data={[]} />);

      expect(screen.getByText(/No data available/i)).toBeInTheDocument();
    });

    it('renders chart when data is present', async () => {
      const { SuccessRateTrendChart } = await import('../SuccessRateTrendChart');

      const data = [
        { timestamp: new Date().toISOString(), success_rate: 50, total_attempts: 10, successful_attacks: 5 },
        { timestamp: new Date().toISOString(), success_rate: 60, total_attempts: 20, successful_attacks: 12 },
      ];

      render(<SuccessRateTrendChart data={data} />);

      // Chart should be rendered (mocked)
      expect(screen.getByTestId('area-chart')).toBeInTheDocument();
    });
  });
});

// ============================================================================
// Hook Tests (useAegisTelemetry)
// ============================================================================

describe('useAegisTelemetry Hook Types', () => {
  it('exports correct hook return type interface', async () => {
    const hookModule = await import('@/lib/hooks/useAegisTelemetry');
    expect(hookModule.useAegisTelemetry).toBeDefined();
  });
});
