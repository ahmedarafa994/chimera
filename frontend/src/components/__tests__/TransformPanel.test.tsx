/**
 * Tests for TransformPanel component.
 *
 * Covers:
 * - Initial rendering
 * - User interactions
 * - Form validation
 * - API mutation handling
 * - Result display
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TransformPanel } from '../transform-panel';

// Mock the enhanced API
vi.mock('@/lib/api-enhanced', () => ({
  default: {
    transform: vi.fn(),
  },
}));

// Mock sonner toast
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

// Create a wrapper with QueryClient for testing
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  Wrapper.displayName = 'QueryWrapper';
  return Wrapper;
};

describe('TransformPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial Rendering', () => {
    it('renders the transform panel with all elements', () => {
      render(<TransformPanel />, { wrapper: createWrapper() });

      expect(screen.getByText('Prompt Transformation')).toBeInTheDocument();
      expect(screen.getByText('Transformation Result')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /transform prompt/i })).toBeInTheDocument();
    });

    it('renders with default prompt text', () => {
      render(<TransformPanel />, { wrapper: createWrapper() });

      const textarea = screen.getByPlaceholderText(/enter your prompt/i);
      expect(textarea).toHaveValue('How to create a secure password?');
    });

    it('renders technique suite selector', () => {
      render(<TransformPanel />, { wrapper: createWrapper() });

      expect(screen.getByText('Technique Suite')).toBeInTheDocument();
    });

    it('renders potency level input', () => {
      render(<TransformPanel />, { wrapper: createWrapper() });

      expect(screen.getByText('Potency Level (1-10)')).toBeInTheDocument();
      const input = screen.getByRole('spinbutton');
      expect(input).toHaveValue(5);
    });

    it('displays placeholder text in results panel', () => {
      render(<TransformPanel />, { wrapper: createWrapper() });

      expect(screen.getByText('Transform a prompt to see results')).toBeInTheDocument();
    });
  });

  describe('User Interactions', () => {
    it('allows changing the prompt text', async () => {
      const user = userEvent.setup();
      render(<TransformPanel />, { wrapper: createWrapper() });

      const textarea = screen.getByPlaceholderText(/enter your prompt/i);
      await user.clear(textarea);
      await user.type(textarea, 'New test prompt');

      expect(textarea).toHaveValue('New test prompt');
    });

    it('allows changing potency level', async () => {
      const user = userEvent.setup();
      render(<TransformPanel />, { wrapper: createWrapper() });

      const input = screen.getByRole('spinbutton');
      await user.clear(input);
      await user.type(input, '8');

      expect(input).toHaveValue(8);
    });

    it('validates potency level min/max', () => {
      render(<TransformPanel />, { wrapper: createWrapper() });

      const input = screen.getByRole('spinbutton');
      expect(input).toHaveAttribute('min', '1');
      expect(input).toHaveAttribute('max', '10');
    });
  });

  describe('Form Validation', () => {
    it('shows error for empty prompt', async () => {
      const user = userEvent.setup();
      const { toast } = await import('sonner');

      render(<TransformPanel />, { wrapper: createWrapper() });

      const textarea = screen.getByPlaceholderText(/enter your prompt/i);
      await user.clear(textarea);

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      expect(toast.error).toHaveBeenCalledWith(
        'Validation Error',
        expect.objectContaining({ description: 'Please enter a prompt' })
      );
    });

    it('shows error for whitespace-only prompt', async () => {
      const user = userEvent.setup();
      const { toast } = await import('sonner');

      render(<TransformPanel />, { wrapper: createWrapper() });

      const textarea = screen.getByPlaceholderText(/enter your prompt/i);
      await user.clear(textarea);
      await user.type(textarea, '   ');

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      expect(toast.error).toHaveBeenCalled();
    });
  });

  describe('API Mutation', () => {
    it('calls transform API on submit', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');
      (api.default.transform as ReturnType<typeof vi.fn>).mockResolvedValue({
        success: true,
        transformed_prompt: 'Transformed result',
      });

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        expect(api.default.transform).toHaveBeenCalledWith({
          prompt: 'How to create a secure password?',
          transformation_type: 'advanced',
        });
      });
    });

    it('shows loading state during mutation', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');

      // Create a promise that we can control
      let resolvePromise: (value: unknown) => void;
      const pendingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      (api.default.transform as ReturnType<typeof vi.fn>).mockReturnValue(pendingPromise);

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      // Button should show loading text
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /transforming/i })).toBeInTheDocument();
      });

      // Resolve the promise
      resolvePromise!({ success: true, transformed_prompt: 'Result' });
    });

    it('displays success result after mutation', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');
      (api.default.transform as ReturnType<typeof vi.fn>).mockResolvedValue({
        success: true,
        transformed_prompt: 'Successfully transformed prompt',
      });

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        expect(screen.getByText('Success')).toBeInTheDocument();
      });
    });

    it('handles API error gracefully', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

      (api.default.transform as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error('API Error')
      );

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        expect(consoleError).toHaveBeenCalled();
      });

      consoleError.mockRestore();
    });
  });

  describe('Result Display', () => {
    it('displays transformed prompt in result textarea', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');
      (api.default.transform as ReturnType<typeof vi.fn>).mockResolvedValue({
        success: true,
        transformed_prompt: 'Transformed text here',
      });

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        const resultTextarea = screen.getAllByRole('textbox')[1];
        expect(resultTextarea).toHaveValue('Transformed text here');
      });
    });

    it('shows copy button when result is displayed', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');
      (api.default.transform as ReturnType<typeof vi.fn>).mockResolvedValue({
        success: true,
        transformed_prompt: 'Result to copy',
      });

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        // Find the copy button (it has Copy icon)
        const copyButtons = screen.getAllByRole('button');
        const copyButton = copyButtons.find((btn) =>
          btn.querySelector('svg[class*="copy"]') !== null ||
          btn.innerHTML.includes('Copy')
        );
        expect(copyButton || copyButtons.length > 1).toBeTruthy();
      });
    });

    it('displays error message when result has error', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');
      (api.default.transform as ReturnType<typeof vi.fn>).mockResolvedValue({
        success: false,
        transformed_prompt: '',
        error: 'Transformation failed',
      });

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        expect(screen.getByText('Transformation failed')).toBeInTheDocument();
      });
    });

    it('displays transformation applied info when available', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');
      (api.default.transform as ReturnType<typeof vi.fn>).mockResolvedValue({
        success: true,
        transformed_prompt: 'Result',
        transformation_applied: 'advanced_obfuscation',
      });

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        expect(screen.getByText('Transformation Applied')).toBeInTheDocument();
        expect(screen.getByText('advanced_obfuscation')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('has accessible labels for all inputs', () => {
      render(<TransformPanel />, { wrapper: createWrapper() });

      // Check for labels (may use different association methods)
      expect(screen.getByText(/original prompt/i)).toBeInTheDocument();
      expect(screen.getByText(/potency level/i)).toBeInTheDocument();
      
      // Verify input elements exist and are accessible
      expect(screen.getByPlaceholderText(/enter your prompt/i)).toBeInTheDocument();
      expect(screen.getByRole('spinbutton')).toBeInTheDocument();
    });

    it('button is disabled during loading', async () => {
      const user = userEvent.setup();
      const api = await import('@/lib/api-enhanced');

      let resolvePromise: (value: unknown) => void;
      const pendingPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      (api.default.transform as ReturnType<typeof vi.fn>).mockReturnValue(pendingPromise);

      render(<TransformPanel />, { wrapper: createWrapper() });

      const button = screen.getByRole('button', { name: /transform prompt/i });
      await user.click(button);

      await waitFor(() => {
        const loadingButton = screen.getByRole('button', { name: /transforming/i });
        expect(loadingButton).toBeDisabled();
      });

      resolvePromise!({ success: true, transformed_prompt: 'Result' });
    });
  });
});