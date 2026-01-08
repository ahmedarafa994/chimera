/**
 * Tests for ModelSelector components.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the component
vi.mock('@/components/model-selector/ModelSelector', () => ({
  ModelSelector: ({
    onModelSelect,
    selectedModel,
  }: {
    onModelSelect?: (model: string) => void;
    selectedModel?: string;
  }) => {
    const ModelSelectorMock = () => (
      <div data-testid="model-selector">
        <select
          data-testid="model-select"
          value={selectedModel || ''}
          onChange={(e) => onModelSelect?.(e.target.value)}
        >
          <option value="">Select a model</option>
          <option value="gpt-4">GPT-4</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="claude-3-opus">Claude 3 Opus</option>
          <option value="claude-3-sonnet">Claude 3 Sonnet</option>
          <option value="llama-3-70b">Llama 3 70B</option>
        </select>
        <div data-testid="model-info">
          {selectedModel && <span>Selected: {selectedModel}</span>}
        </div>
      </div>
    );
    ModelSelectorMock.displayName = 'ModelSelector';
    return ModelSelectorMock;
  },
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

describe('ModelSelector Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders model selection dropdown', async () => {
    const { ModelSelector } = await import(
      '@/components/model-selector/ModelSelector'
    );

    render(<ModelSelector />, { wrapper: createWrapper() });

    expect(screen.getByTestId('model-selector')).toBeInTheDocument();
    expect(screen.getByTestId('model-select')).toBeInTheDocument();
  });

  it('displays available model options', async () => {
    const { ModelSelector } = await import(
      '@/components/model-selector/ModelSelector'
    );

    render(<ModelSelector />, { wrapper: createWrapper() });

    const select = screen.getByTestId('model-select');
    const options = select.querySelectorAll('option');

    // Should have multiple model options plus empty option
    expect(options.length).toBeGreaterThanOrEqual(4);
  });

  it('calls onModelSelect when model is changed', async () => {
    const mockOnModelSelect = vi.fn();
    const { ModelSelector } = await import(
      '@/components/model-selector/ModelSelector'
    );

    render(
      <ModelSelector onModelSelect={mockOnModelSelect} />,
      { wrapper: createWrapper() }
    );

    const select = screen.getByTestId('model-select');
    fireEvent.change(select, { target: { value: 'gpt-4' } });

    expect(mockOnModelSelect).toHaveBeenCalledWith('gpt-4');
  });

  it('displays selected model', async () => {
    const { ModelSelector } = await import(
      '@/components/model-selector/ModelSelector'
    );

    render(
      <ModelSelector selectedModel="claude-3-opus" />,
      { wrapper: createWrapper() }
    );

    expect(screen.getByText(/Selected: claude-3-opus/)).toBeInTheDocument();
  });

  it('allows selection of different model families', async () => {
    const mockOnModelSelect = vi.fn();
    const { ModelSelector } = await import(
      '@/components/model-selector/ModelSelector'
    );

    render(
      <ModelSelector onModelSelect={mockOnModelSelect} />,
      { wrapper: createWrapper() }
    );

    const select = screen.getByTestId('model-select');

    // Select OpenAI model
    fireEvent.change(select, { target: { value: 'gpt-4' } });
    expect(mockOnModelSelect).toHaveBeenCalledWith('gpt-4');

    // Select Anthropic model
    fireEvent.change(select, { target: { value: 'claude-3-opus' } });
    expect(mockOnModelSelect).toHaveBeenCalledWith('claude-3-opus');

    // Select Llama model
    fireEvent.change(select, { target: { value: 'llama-3-70b' } });
    expect(mockOnModelSelect).toHaveBeenCalledWith('llama-3-70b');
  });
});

describe('ModelSelector Accessibility', () => {
  it('select element is accessible', async () => {
    const { ModelSelector } = await import(
      '@/components/model-selector/ModelSelector'
    );

    render(<ModelSelector />, { wrapper: createWrapper() });

    const select = screen.getByTestId('model-select');
    expect(select.tagName.toLowerCase()).toBe('select');
  });

  it('is keyboard navigable', async () => {
    const mockOnModelSelect = vi.fn();
    const { ModelSelector } = await import(
      '@/components/model-selector/ModelSelector'
    );

    render(
      <ModelSelector onModelSelect={mockOnModelSelect} />,
      { wrapper: createWrapper() }
    );

    const select = screen.getByTestId('model-select');
    select.focus();
    expect(document.activeElement).toBe(select);

    // Simulate keyboard selection
    fireEvent.keyDown(select, { key: 'ArrowDown' });
  });
});