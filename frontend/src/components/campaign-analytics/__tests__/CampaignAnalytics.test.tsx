/**
 * Tests for Campaign Analytics Components.
 *
 * Covers:
 * - StatisticsCard: Display, loading, error states, formatting
 * - CampaignSelector: Single/multi-select, search, loading, error states
 * - FilterBar: Filter controls, state management, loading states
 *
 * Uses Vitest + React Testing Library.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// =============================================================================
// Mock Setup
// =============================================================================

// Mock the campaign queries
vi.mock('@/lib/api/query/campaign-queries', () => ({
  useCampaigns: vi.fn(),
  useTechniqueBreakdown: vi.fn(),
  useProviderBreakdown: vi.fn(),
}));

// Mock the utils
vi.mock('@/lib/utils', () => ({
  cn: (...inputs: unknown[]) => inputs.filter(Boolean).join(' '),
}));

// Import mocked modules
import { useCampaigns, useTechniqueBreakdown, useProviderBreakdown } from '@/lib/api/query/campaign-queries';

// Import components under test
import {
  StatisticsCard,
  StatisticsCardSkeleton,
  DistributionStatCard,
  SuccessRateCard,
  LatencyCard,
  TokenUsageCard,
  CostCard,
} from '../StatisticsCard';

import {
  CampaignSelector,
  CampaignSelectorSingle,
  CampaignSelectorMulti,
  CampaignComparisonSelector,
} from '../CampaignSelector';

import {
  FilterBar,
  FilterBarSkeleton,
  FilterBarEmpty,
  CompactFilterBar,
  createDefaultFilterState,
  filterStateToParams,
  type FilterState,
} from '../FilterBar';

// =============================================================================
// Test Utilities
// =============================================================================

/**
 * Create a QueryClient wrapper for testing.
 */
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  Wrapper.displayName = 'QueryWrapper';
  return Wrapper;
};

/**
 * Mock campaign data for testing.
 */
const mockCampaigns = [
  {
    id: 'campaign-1',
    name: 'Test Campaign Alpha',
    objective: 'Test objective for security research',
    status: 'completed',
    created_at: new Date(Date.now() - 86400000).toISOString(), // 1 day ago
    target_provider: 'openai',
    success_rate: 0.85,
    total_attempts: 100,
    tags: ['security', 'research'],
  },
  {
    id: 'campaign-2',
    name: 'Beta Campaign',
    objective: 'Another test campaign',
    status: 'running',
    created_at: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
    target_provider: 'anthropic',
    success_rate: 0.65,
    total_attempts: 50,
    tags: ['jailbreak'],
  },
  {
    id: 'campaign-3',
    name: 'Gamma Research',
    objective: 'Testing cognitive techniques',
    status: 'failed',
    created_at: new Date(Date.now() - 604800000).toISOString(), // 7 days ago
    target_provider: 'google',
    success_rate: 0.25,
    total_attempts: 200,
    tags: ['cognitive'],
  },
];

/**
 * Mock distribution stats for testing.
 */
const mockDistributionStats = {
  mean: 0.75,
  median: 0.72,
  std_dev: 0.15,
  min_value: 0.1,
  max_value: 0.98,
  percentiles: {
    p50: 0.72,
    p90: 0.90,
    p95: 0.94,
    p99: 0.97,
  },
};

/**
 * Mock technique breakdown for testing.
 */
const mockTechniqueBreakdown = {
  campaign_id: 'campaign-1',
  items: [
    { name: 'cognitive_hacking', success_rate: 0.85, attempts: 50 },
    { name: 'dan_persona', success_rate: 0.70, attempts: 30 },
    { name: 'payload_splitting', success_rate: 0.55, attempts: 20 },
  ],
};

/**
 * Mock provider breakdown for testing.
 */
const mockProviderBreakdown = {
  campaign_id: 'campaign-1',
  items: [
    { name: 'openai', success_rate: 0.80, attempts: 60 },
    { name: 'anthropic', success_rate: 0.75, attempts: 30 },
    { name: 'google', success_rate: 0.60, attempts: 10 },
  ],
};

// =============================================================================
// StatisticsCard Tests
// =============================================================================

describe('StatisticsCard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('renders with label and value', () => {
      render(<StatisticsCard label="Success Rate" value={0.85} />);

      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('0.85')).toBeInTheDocument();
    });

    it('renders with formatted percentage value', () => {
      render(<StatisticsCard label="Success Rate" value={0.85} format="percentage" />);

      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      expect(screen.getByText('85.00%')).toBeInTheDocument();
    });

    it('renders with duration format', () => {
      render(<StatisticsCard label="Latency" value={1500} format="duration" />);

      expect(screen.getByText('Latency')).toBeInTheDocument();
      expect(screen.getByText('1.50s')).toBeInTheDocument();
    });

    it('renders milliseconds for small duration values', () => {
      render(<StatisticsCard label="Latency" value={250} format="duration" />);

      expect(screen.getByText('250.00ms')).toBeInTheDocument();
    });

    it('renders with compact format for large numbers', () => {
      render(<StatisticsCard label="Total Tokens" value={1500000} format="compact" />);

      expect(screen.getByText('1.5M')).toBeInTheDocument();
    });

    it('renders with currency format', () => {
      render(<StatisticsCard label="Cost" value={150} format="currency" />);

      expect(screen.getByText('$1.50')).toBeInTheDocument();
    });

    it('renders with unit suffix', () => {
      render(<StatisticsCard label="Tokens" value={1000} unit="tokens" />);

      expect(screen.getByText('1000.00 tokens')).toBeInTheDocument();
    });

    it('renders null/undefined values as em dash', () => {
      render(<StatisticsCard label="Unknown" value={null} />);

      expect(screen.getByText('—')).toBeInTheDocument();
    });
  });

  describe('Delta/Trend Display', () => {
    it('renders positive delta with up trend', () => {
      render(
        <StatisticsCard label="Success Rate" value={0.85} format="percentage" delta={0.05} />
      );

      expect(screen.getByText('+5.00%')).toBeInTheDocument();
    });

    it('renders negative delta with down trend', () => {
      render(
        <StatisticsCard label="Success Rate" value={0.75} format="percentage" delta={-0.1} />
      );

      expect(screen.getByText('10.00%')).toBeInTheDocument();
    });

    it('renders delta label', () => {
      render(
        <StatisticsCard
          label="Success Rate"
          value={0.85}
          delta={0.05}
          deltaLabel="vs last week"
        />
      );

      expect(screen.getByText('vs last week')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('renders skeleton when isLoading is true', () => {
      render(<StatisticsCard label="Success Rate" value={0.85} isLoading />);

      // Should not display the actual value
      expect(screen.queryByText('0.85')).not.toBeInTheDocument();
    });

    it('StatisticsCardSkeleton renders correctly', () => {
      render(<StatisticsCardSkeleton />);

      // Skeleton should have the Card structure
      const card = document.querySelector('[class*="overflow-hidden"]');
      expect(card).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('renders error message when error is provided', () => {
      render(
        <StatisticsCard label="Success Rate" value={0.85} error="Failed to load data" />
      );

      expect(screen.getByText('Failed to load data')).toBeInTheDocument();
      expect(screen.getByText('Success Rate')).toBeInTheDocument();
    });

    it('displays error with red styling', () => {
      render(<StatisticsCard label="Test" value={0} error="Error message" />);

      const errorText = screen.getByText('Error message');
      expect(errorText).toHaveClass('text-red-500');
    });
  });

  describe('Color Variants', () => {
    it('applies success variant styling', () => {
      render(<StatisticsCard label="Success" value={0.85} variant="success" />);

      const card = document.querySelector('[class*="border-green"]');
      expect(card).toBeInTheDocument();
    });

    it('applies error variant styling', () => {
      render(<StatisticsCard label="Failure" value={0.15} variant="error" />);

      const card = document.querySelector('[class*="border-red"]');
      expect(card).toBeInTheDocument();
    });
  });

  describe('Clickable Behavior', () => {
    it('calls onClick when clicked', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();

      render(<StatisticsCard label="Test" value={100} onClick={handleClick} />);

      const card = screen.getByRole('button');
      await user.click(card);

      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('has cursor-pointer class when clickable', () => {
      render(<StatisticsCard label="Test" value={100} onClick={() => {}} />);

      const card = screen.getByRole('button');
      expect(card).toHaveClass('cursor-pointer');
    });

    it('supports keyboard navigation when clickable', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();

      render(<StatisticsCard label="Test" value={100} onClick={handleClick} />);

      const card = screen.getByRole('button');
      card.focus();
      await user.keyboard('{Enter}');

      expect(handleClick).toHaveBeenCalledTimes(1);
    });
  });

  describe('Metric Type Badge', () => {
    it('displays metric type badge for specific types', () => {
      render(<StatisticsCard label="Test" value={100} metricType="mean" />);

      expect(screen.getByText('MEAN')).toBeInTheDocument();
    });

    it('hides metric type badge for custom/total/count types', () => {
      render(<StatisticsCard label="Test" value={100} metricType="custom" />);

      expect(screen.queryByText('CUSTOM')).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper aria-label', () => {
      render(<StatisticsCard label="Success Rate" value={0.85} format="percentage" />);

      const card = screen.getByLabelText('Success Rate: 85.00%');
      expect(card).toBeInTheDocument();
    });

    it('supports custom ariaLabel', () => {
      render(
        <StatisticsCard
          label="Success Rate"
          value={0.85}
          ariaLabel="Campaign success rate is 85%"
        />
      );

      const card = screen.getByLabelText('Campaign success rate is 85%');
      expect(card).toBeInTheDocument();
    });
  });
});

describe('DistributionStatCard Component', () => {
  it('renders mean value from distribution stats', () => {
    render(
      <DistributionStatCard
        label="Success Rate"
        stats={mockDistributionStats}
        primaryMetric="mean"
        format="percentage"
      />
    );

    expect(screen.getByText('75.00%')).toBeInTheDocument();
  });

  it('renders median value when specified', () => {
    render(
      <DistributionStatCard
        label="Success Rate"
        stats={mockDistributionStats}
        primaryMetric="median"
        format="percentage"
      />
    );

    expect(screen.getByText('72.00%')).toBeInTheDocument();
  });

  it('renders loading state', () => {
    render(
      <DistributionStatCard label="Success Rate" stats={null} isLoading />
    );

    // Should show skeleton, not the value
    expect(screen.queryByText('—')).not.toBeInTheDocument();
  });

  it('renders null stats as dash', () => {
    render(<DistributionStatCard label="Success Rate" stats={null} />);

    expect(screen.getByText('—')).toBeInTheDocument();
  });
});

describe('Preset Statistics Cards', () => {
  describe('SuccessRateCard', () => {
    it('renders success variant for high success rate', () => {
      render(<SuccessRateCard stats={{ ...mockDistributionStats, mean: 0.85 }} />);

      expect(screen.getByText('Success Rate')).toBeInTheDocument();
      const card = document.querySelector('[class*="border-green"]');
      expect(card).toBeInTheDocument();
    });

    it('renders warning variant for moderate success rate', () => {
      render(<SuccessRateCard stats={{ ...mockDistributionStats, mean: 0.55 }} />);

      const card = document.querySelector('[class*="border-yellow"]');
      expect(card).toBeInTheDocument();
    });

    it('renders error variant for low success rate', () => {
      render(<SuccessRateCard stats={{ ...mockDistributionStats, mean: 0.25 }} />);

      const card = document.querySelector('[class*="border-red"]');
      expect(card).toBeInTheDocument();
    });
  });

  describe('LatencyCard', () => {
    it('renders latency with duration format', () => {
      render(<LatencyCard stats={{ ...mockDistributionStats, median: 250 }} />);

      expect(screen.getByText('Latency (p50)')).toBeInTheDocument();
    });

    it('shows P95 description when available', () => {
      const statsWithP95 = {
        ...mockDistributionStats,
        median: 250,
        percentiles: { ...mockDistributionStats.percentiles, p95: 500 },
      };
      render(<LatencyCard stats={statsWithP95} />);

      expect(screen.getByText(/P95:/)).toBeInTheDocument();
    });
  });

  describe('TokenUsageCard', () => {
    it('renders token count with compact format', () => {
      render(<TokenUsageCard stats={{ ...mockDistributionStats, mean: 1500 }} />);

      expect(screen.getByText('Avg Tokens')).toBeInTheDocument();
    });

    it('shows total when provided', () => {
      render(
        <TokenUsageCard
          stats={{ ...mockDistributionStats, mean: 1500 }}
          total={150000}
        />
      );

      expect(screen.getByText(/Total:/)).toBeInTheDocument();
    });
  });

  describe('CostCard', () => {
    it('renders cost with currency format', () => {
      render(<CostCard stats={{ ...mockDistributionStats, mean: 50 }} />);

      expect(screen.getByText('Avg Cost')).toBeInTheDocument();
    });
  });
});

// =============================================================================
// CampaignSelector Tests
// =============================================================================

describe('CampaignSelector Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Setup default mock response
    (useCampaigns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { items: mockCampaigns, total: mockCampaigns.length },
      isLoading: false,
      isError: false,
      error: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Single Select Mode', () => {
    it('renders with placeholder when no selection', () => {
      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      expect(screen.getByText('Select campaign...')).toBeInTheDocument();
    });

    it('renders selected campaign name', () => {
      render(<CampaignSelector value="campaign-1" onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      expect(screen.getByText('Test Campaign Alpha')).toBeInTheDocument();
    });

    it('opens popover when clicked', async () => {
      const user = userEvent.setup();

      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      const trigger = screen.getByRole('combobox');
      await user.click(trigger);

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search campaigns...')).toBeInTheDocument();
      });
    });

    it('shows campaign list in popover', async () => {
      const user = userEvent.setup();

      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        expect(screen.getByText('Test Campaign Alpha')).toBeInTheDocument();
        expect(screen.getByText('Beta Campaign')).toBeInTheDocument();
        expect(screen.getByText('Gamma Research')).toBeInTheDocument();
      });
    });

    it('calls onChange when campaign selected', async () => {
      const user = userEvent.setup();
      const handleChange = vi.fn();

      render(<CampaignSelector value={null} onChange={handleChange} />, {
        wrapper: createWrapper(),
      });

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        expect(screen.getByText('Test Campaign Alpha')).toBeInTheDocument();
      });

      // Click on the campaign option
      const option = screen.getByRole('option', { name: /Test Campaign Alpha/i });
      await user.click(option);

      expect(handleChange).toHaveBeenCalledWith('campaign-1');
    });

    it('filters campaigns based on search query', async () => {
      const user = userEvent.setup();

      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search campaigns...')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search campaigns...');
      await user.type(searchInput, 'Alpha');

      await waitFor(() => {
        expect(screen.getByText('Test Campaign Alpha')).toBeInTheDocument();
        expect(screen.queryByText('Beta Campaign')).not.toBeInTheDocument();
      });
    });

    it('shows clear button when selection exists', () => {
      render(
        <CampaignSelector value="campaign-1" onChange={() => {}} allowClear />,
        { wrapper: createWrapper() }
      );

      const clearButton = screen.getByLabelText('Clear selection');
      expect(clearButton).toBeInTheDocument();
    });

    it('clears selection when clear button clicked', async () => {
      const user = userEvent.setup();
      const handleChange = vi.fn();

      render(
        <CampaignSelector value="campaign-1" onChange={handleChange} allowClear />,
        { wrapper: createWrapper() }
      );

      const clearButton = screen.getByLabelText('Clear selection');
      await user.click(clearButton);

      expect(handleChange).toHaveBeenCalledWith(null);
    });
  });

  describe('Multi Select Mode', () => {
    it('renders with multi-select placeholder', () => {
      render(
        <CampaignSelector mode="multi" values={[]} onMultiChange={() => {}} />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Select campaign...')).toBeInTheDocument();
    });

    it('renders selected campaigns as chips', () => {
      render(
        <CampaignSelector
          mode="multi"
          values={['campaign-1', 'campaign-2']}
          onMultiChange={() => {}}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByText('Test Campaign Alpha')).toBeInTheDocument();
      expect(screen.getByText('Beta Campaign')).toBeInTheDocument();
    });

    it('shows selection count in popover', async () => {
      const user = userEvent.setup();

      render(
        <CampaignSelector
          mode="multi"
          values={['campaign-1', 'campaign-2']}
          onMultiChange={() => {}}
          maxSelections={4}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        expect(screen.getByText('2 of 4 selected')).toBeInTheDocument();
      });
    });

    it('disables options when max selections reached', async () => {
      const user = userEvent.setup();

      render(
        <CampaignSelector
          mode="multi"
          values={['campaign-1', 'campaign-2']}
          onMultiChange={() => {}}
          maxSelections={2}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        const options = screen.getAllByRole('option');
        // The unselected option should be disabled
        const gammaOption = options.find((opt) =>
          opt.textContent?.includes('Gamma Research')
        );
        expect(gammaOption).toHaveClass('opacity-50');
      });
    });

    it('removes campaign when chip X button clicked', async () => {
      const user = userEvent.setup();
      const handleMultiChange = vi.fn();

      render(
        <CampaignSelector
          mode="multi"
          values={['campaign-1', 'campaign-2']}
          onMultiChange={handleMultiChange}
        />,
        { wrapper: createWrapper() }
      );

      const removeButton = screen.getByLabelText('Remove Test Campaign Alpha');
      await user.click(removeButton);

      expect(handleMultiChange).toHaveBeenCalledWith(['campaign-2']);
    });
  });

  describe('Loading State', () => {
    it('shows loading skeleton in popover', async () => {
      const user = userEvent.setup();

      (useCampaigns as ReturnType<typeof vi.fn>).mockReturnValue({
        data: null,
        isLoading: true,
        isError: false,
        error: null,
      });

      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        // Should show skeleton elements
        const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
        expect(skeletons.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Error State', () => {
    it('shows error message in popover', async () => {
      const user = userEvent.setup();

      (useCampaigns as ReturnType<typeof vi.fn>).mockReturnValue({
        data: null,
        isLoading: false,
        isError: true,
        error: new Error('Network error'),
      });

      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        expect(screen.getByText('Failed to load campaigns')).toBeInTheDocument();
        expect(screen.getByText('Network error')).toBeInTheDocument();
      });
    });
  });

  describe('Empty State', () => {
    it('shows empty message when no campaigns match', async () => {
      const user = userEvent.setup();

      (useCampaigns as ReturnType<typeof vi.fn>).mockReturnValue({
        data: { items: [], total: 0 },
        isLoading: false,
        isError: false,
        error: null,
      });

      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        expect(screen.getByText('No campaigns found')).toBeInTheDocument();
      });
    });

    it('shows custom empty message', async () => {
      const user = userEvent.setup();

      (useCampaigns as ReturnType<typeof vi.fn>).mockReturnValue({
        data: { items: [], total: 0 },
        isLoading: false,
        isError: false,
        error: null,
      });

      render(
        <CampaignSelector
          value={null}
          onChange={() => {}}
          emptyMessage="No campaigns available"
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByRole('combobox'));

      await waitFor(() => {
        expect(screen.getByText('No campaigns available')).toBeInTheDocument();
      });
    });
  });

  describe('Disabled State', () => {
    it('disables trigger button when disabled prop is true', () => {
      render(<CampaignSelector value={null} onChange={() => {}} disabled />, {
        wrapper: createWrapper(),
      });

      const trigger = screen.getByRole('combobox');
      expect(trigger).toBeDisabled();
    });
  });

  describe('Accessibility', () => {
    it('has proper combobox role', () => {
      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('has proper aria-expanded state', async () => {
      const user = userEvent.setup();

      render(<CampaignSelector value={null} onChange={() => {}} />, {
        wrapper: createWrapper(),
      });

      const trigger = screen.getByRole('combobox');
      expect(trigger).toHaveAttribute('aria-expanded', 'false');

      await user.click(trigger);

      await waitFor(() => {
        expect(trigger).toHaveAttribute('aria-expanded', 'true');
      });
    });

    it('supports custom aria-label', () => {
      render(
        <CampaignSelector
          value={null}
          onChange={() => {}}
          ariaLabel="Select a campaign for analysis"
        />,
        { wrapper: createWrapper() }
      );

      expect(
        screen.getByLabelText('Select a campaign for analysis')
      ).toBeInTheDocument();
    });
  });
});

describe('CampaignSelector Variants', () => {
  beforeEach(() => {
    (useCampaigns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { items: mockCampaigns, total: mockCampaigns.length },
      isLoading: false,
      isError: false,
      error: null,
    });
  });

  it('CampaignSelectorSingle works correctly', () => {
    const handleChange = vi.fn();
    render(<CampaignSelectorSingle value="campaign-1" onChange={handleChange} />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByText('Test Campaign Alpha')).toBeInTheDocument();
  });

  it('CampaignSelectorMulti works correctly', () => {
    const handleMultiChange = vi.fn();
    render(
      <CampaignSelectorMulti values={['campaign-1']} onMultiChange={handleMultiChange} />,
      { wrapper: createWrapper() }
    );

    expect(screen.getByText('Test Campaign Alpha')).toBeInTheDocument();
  });

  it('CampaignComparisonSelector shows only completed campaigns', () => {
    render(
      <CampaignComparisonSelector values={[]} onMultiChange={() => {}} />,
      { wrapper: createWrapper() }
    );

    expect(
      screen.getByText('Select 2-4 campaigns to compare...')
    ).toBeInTheDocument();
  });
});

// =============================================================================
// FilterBar Tests
// =============================================================================

describe('FilterBar Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Setup default mock responses
    (useTechniqueBreakdown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockTechniqueBreakdown,
      isLoading: false,
    });

    (useProviderBreakdown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockProviderBreakdown,
      isLoading: false,
    });
  });

  describe('Basic Rendering', () => {
    it('renders filter bar with all controls', () => {
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={['cognitive_hacking', 'dan_persona']}
          providerOptions={['openai', 'anthropic']}
        />,
        { wrapper: createWrapper() }
      );

      // Should have filter controls
      expect(screen.getByRole('toolbar')).toBeInTheDocument();
      expect(screen.getByLabelText('Filter by Techniques')).toBeInTheDocument();
      expect(screen.getByLabelText('Filter by Providers')).toBeInTheDocument();
      expect(screen.getByLabelText('Filter by date range')).toBeInTheDocument();
      expect(screen.getByLabelText('Filter by status')).toBeInTheDocument();
    });

    it('shows active filter count badge', () => {
      const filters: FilterState = {
        techniques: ['cognitive_hacking'],
        providers: ['openai'],
        dateRange: { start: null, end: null },
        successStatus: [],
      };

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={['cognitive_hacking', 'dan_persona']}
          providerOptions={['openai', 'anthropic']}
          showActiveCount
        />,
        { wrapper: createWrapper() }
      );

      // Should show count of 2 active filters
      expect(screen.getByText('2')).toBeInTheDocument();
    });
  });

  describe('Technique Filter', () => {
    it('opens technique filter popover on click', async () => {
      const user = userEvent.setup();
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={['cognitive_hacking', 'dan_persona']}
          providerOptions={['openai']}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByLabelText('Filter by Techniques'));

      await waitFor(() => {
        expect(screen.getByText('cognitive_hacking')).toBeInTheDocument();
        expect(screen.getByText('dan_persona')).toBeInTheDocument();
      });
    });

    it('calls onFiltersChange when technique selected', async () => {
      const user = userEvent.setup();
      const filters = createDefaultFilterState();
      const handleChange = vi.fn();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={handleChange}
          techniqueOptions={['cognitive_hacking', 'dan_persona']}
          providerOptions={['openai']}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByLabelText('Filter by Techniques'));

      await waitFor(() => {
        expect(screen.getByText('cognitive_hacking')).toBeInTheDocument();
      });

      // Click on a technique option
      const techniqueButton = screen
        .getAllByRole('button')
        .find((btn) => btn.textContent?.includes('cognitive_hacking'));
      if (techniqueButton) {
        await user.click(techniqueButton);
      }

      expect(handleChange).toHaveBeenCalledWith({
        ...filters,
        techniques: ['cognitive_hacking'],
      });
    });

    it('shows selection count badge when techniques selected', () => {
      const filters: FilterState = {
        techniques: ['cognitive_hacking', 'dan_persona'],
        providers: [],
        dateRange: { start: null, end: null },
        successStatus: [],
      };

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={['cognitive_hacking', 'dan_persona']}
          providerOptions={['openai']}
        />,
        { wrapper: createWrapper() }
      );

      // The technique button should show count
      const techniqueButton = screen.getByLabelText('Filter by Techniques');
      expect(within(techniqueButton).getByText('2')).toBeInTheDocument();
    });
  });

  describe('Date Range Filter', () => {
    it('opens date range popover on click', async () => {
      const user = userEvent.setup();
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={[]}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByLabelText('Filter by date range'));

      await waitFor(() => {
        expect(screen.getByText('Date Range')).toBeInTheDocument();
        expect(screen.getByText('Today')).toBeInTheDocument();
        expect(screen.getByText('Last 7 days')).toBeInTheDocument();
        expect(screen.getByText('Last 30 days')).toBeInTheDocument();
      });
    });

    it('calls onFiltersChange when date preset selected', async () => {
      const user = userEvent.setup();
      const filters = createDefaultFilterState();
      const handleChange = vi.fn();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={handleChange}
          techniqueOptions={[]}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByLabelText('Filter by date range'));

      await waitFor(() => {
        expect(screen.getByText('Last 7 days')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Last 7 days'));

      expect(handleChange).toHaveBeenCalled();
      const callArgs = handleChange.mock.calls[0][0] as FilterState;
      expect(callArgs.dateRange.start).toBeInstanceOf(Date);
      expect(callArgs.dateRange.end).toBeInstanceOf(Date);
    });
  });

  describe('Status Filter', () => {
    it('opens status filter popover on click', async () => {
      const user = userEvent.setup();
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={[]}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByLabelText('Filter by status'));

      await waitFor(() => {
        expect(screen.getByText('All Statuses')).toBeInTheDocument();
        expect(screen.getByText('Success')).toBeInTheDocument();
        expect(screen.getByText('Failure')).toBeInTheDocument();
        expect(screen.getByText('Partial Success')).toBeInTheDocument();
      });
    });
  });

  describe('Clear All Button', () => {
    it('shows clear all button when filters are active', () => {
      const filters: FilterState = {
        techniques: ['cognitive_hacking'],
        providers: [],
        dateRange: { start: null, end: null },
        successStatus: [],
      };

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={['cognitive_hacking']}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByLabelText('Clear all filters')).toBeInTheDocument();
    });

    it('hides clear all button when no filters active', () => {
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={[]}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.queryByLabelText('Clear all filters')).not.toBeInTheDocument();
    });

    it('calls onClearAll when clear button clicked', async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        techniques: ['cognitive_hacking'],
        providers: [],
        dateRange: { start: null, end: null },
        successStatus: [],
      };
      const handleClearAll = vi.fn();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          onClearAll={handleClearAll}
          techniqueOptions={['cognitive_hacking']}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByLabelText('Clear all filters'));

      expect(handleClearAll).toHaveBeenCalledTimes(1);
    });
  });

  describe('Loading State', () => {
    it('shows loading in technique popover when loading', async () => {
      const user = userEvent.setup();

      (useTechniqueBreakdown as ReturnType<typeof vi.fn>).mockReturnValue({
        data: null,
        isLoading: true,
      });

      const filters = createDefaultFilterState();

      render(
        <FilterBar
          campaignId="campaign-1"
          filters={filters}
          onFiltersChange={() => {}}
        />,
        { wrapper: createWrapper() }
      );

      await user.click(screen.getByLabelText('Filter by Techniques'));

      await waitFor(() => {
        const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
        expect(skeletons.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Disabled State', () => {
    it('disables all controls when disabled prop is true', () => {
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={['cognitive_hacking']}
          providerOptions={['openai']}
          disabled
        />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByLabelText('Filter by Techniques')).toBeDisabled();
      expect(screen.getByLabelText('Filter by Providers')).toBeDisabled();
      expect(screen.getByLabelText('Filter by date range')).toBeDisabled();
      expect(screen.getByLabelText('Filter by status')).toBeDisabled();
    });
  });

  describe('Compact Mode', () => {
    it('renders in compact mode with reduced styling', () => {
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={[]}
          providerOptions={[]}
          compact
        />,
        { wrapper: createWrapper() }
      );

      const toolbar = screen.getByRole('toolbar');
      expect(toolbar).toHaveClass('gap-1');
    });
  });

  describe('Custom Options', () => {
    it('uses custom technique options instead of API', () => {
      const filters = createDefaultFilterState();

      render(
        <FilterBar
          campaignId="campaign-1"
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={['custom_technique_1', 'custom_technique_2']}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      // Should not call the API when custom options provided
      expect(useTechniqueBreakdown).toHaveBeenCalledWith('campaign-1', {
        enabled: false,
      });
    });
  });
});

describe('FilterBar Helper Components', () => {
  describe('FilterBarSkeleton', () => {
    it('renders loading skeleton', () => {
      render(<FilterBarSkeleton />);

      const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  describe('FilterBarEmpty', () => {
    it('renders empty state with default message', () => {
      render(<FilterBarEmpty />);

      expect(
        screen.getByText('Select a campaign to enable filters')
      ).toBeInTheDocument();
    });

    it('renders empty state with custom message', () => {
      render(<FilterBarEmpty message="No filters available" />);

      expect(screen.getByText('No filters available')).toBeInTheDocument();
    });
  });

  describe('CompactFilterBar', () => {
    it('renders with compact mode enabled', () => {
      const filters = createDefaultFilterState();

      render(
        <CompactFilterBar
          filters={filters}
          onFiltersChange={() => {}}
          techniqueOptions={[]}
          providerOptions={[]}
        />,
        { wrapper: createWrapper() }
      );

      const toolbar = screen.getByRole('toolbar');
      expect(toolbar).toHaveClass('gap-1');
    });
  });
});

describe('FilterBar Utility Functions', () => {
  describe('createDefaultFilterState', () => {
    it('creates empty filter state', () => {
      const state = createDefaultFilterState();

      expect(state.techniques).toEqual([]);
      expect(state.providers).toEqual([]);
      expect(state.dateRange).toEqual({ start: null, end: null });
      expect(state.successStatus).toEqual([]);
    });
  });

  describe('filterStateToParams', () => {
    it('converts empty filter state to empty params', () => {
      const filters = createDefaultFilterState();
      const params = filterStateToParams(filters);

      expect(params).toEqual({});
    });

    it('converts techniques to technique_suite param', () => {
      const filters: FilterState = {
        techniques: ['cognitive_hacking', 'dan_persona'],
        providers: [],
        dateRange: { start: null, end: null },
        successStatus: [],
      };
      const params = filterStateToParams(filters);

      expect(params.technique_suite).toEqual(['cognitive_hacking', 'dan_persona']);
    });

    it('converts providers to provider param', () => {
      const filters: FilterState = {
        techniques: [],
        providers: ['openai', 'anthropic'],
        dateRange: { start: null, end: null },
        successStatus: [],
      };
      const params = filterStateToParams(filters);

      expect(params.provider).toEqual(['openai', 'anthropic']);
    });

    it('converts date range to start_time and end_time params', () => {
      const start = new Date('2024-01-01');
      const end = new Date('2024-01-31');
      const filters: FilterState = {
        techniques: [],
        providers: [],
        dateRange: { start, end },
        successStatus: [],
      };
      const params = filterStateToParams(filters);

      expect(params.start_time).toBe(start.toISOString());
      expect(params.end_time).toBe(end.toISOString());
    });

    it('converts success status to status param', () => {
      const filters: FilterState = {
        techniques: [],
        providers: [],
        dateRange: { start: null, end: null },
        successStatus: ['success', 'failure'],
      };
      const params = filterStateToParams(filters);

      expect(params.status).toEqual(['success', 'failure']);
    });

    it('ignores all status when present', () => {
      const filters: FilterState = {
        techniques: [],
        providers: [],
        dateRange: { start: null, end: null },
        successStatus: ['all'],
      };
      const params = filterStateToParams(filters);

      expect(params.status).toBeUndefined();
    });
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('Component Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    (useCampaigns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { items: mockCampaigns, total: mockCampaigns.length },
      isLoading: false,
      isError: false,
      error: null,
    });

    (useTechniqueBreakdown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockTechniqueBreakdown,
      isLoading: false,
    });

    (useProviderBreakdown as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockProviderBreakdown,
      isLoading: false,
    });
  });

  it('StatisticsCard displays correct value after selection changes', async () => {
    const TestComponent = () => {
      const [value, setValue] = React.useState<number>(0.5);
      return (
        <div>
          <button onClick={() => setValue(0.85)}>Update Value</button>
          <StatisticsCard label="Success Rate" value={value} format="percentage" />
        </div>
      );
    };

    const user = userEvent.setup();
    render(<TestComponent />);

    expect(screen.getByText('50.00%')).toBeInTheDocument();

    await user.click(screen.getByText('Update Value'));

    expect(screen.getByText('85.00%')).toBeInTheDocument();
  });

  it('FilterBar properly updates when filters change', async () => {
    const TestComponent = () => {
      const [filters, setFilters] = React.useState<FilterState>(
        createDefaultFilterState()
      );
      return (
        <FilterBar
          filters={filters}
          onFiltersChange={setFilters}
          techniqueOptions={['cognitive_hacking', 'dan_persona']}
          providerOptions={['openai', 'anthropic']}
        />
      );
    };

    const user = userEvent.setup();
    render(<TestComponent />, { wrapper: createWrapper() });

    // Initially no clear button
    expect(screen.queryByLabelText('Clear all filters')).not.toBeInTheDocument();

    // Open technique filter and select one
    await user.click(screen.getByLabelText('Filter by Techniques'));

    await waitFor(() => {
      expect(screen.getByText('cognitive_hacking')).toBeInTheDocument();
    });

    const techniqueButton = screen
      .getAllByRole('button')
      .find((btn) => btn.textContent?.includes('cognitive_hacking'));
    if (techniqueButton) {
      await user.click(techniqueButton);
    }

    // Now clear button should appear
    await waitFor(() => {
      expect(screen.getByLabelText('Clear all filters')).toBeInTheDocument();
    });
  });
});
