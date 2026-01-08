import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ErrorBoundary } from '../ErrorBoundary';

// Component that throws an error
const ThrowError = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
};
ThrowError.displayName = 'ThrowError';

describe('ErrorBoundary', () => {
  it('should render children when there is no error', () => {
    render(
      <ErrorBoundary>
        <div>Test content</div>
      </ErrorBoundary>
    );

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('should render error UI when child component throws', () => {
    // Suppress console.error for this test
    const consoleError = console.error;
    console.error = vi.fn();

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();

    console.error = consoleError;
  });

  it('should render custom fallback when provided', () => {
    const consoleError = console.error;
    console.error = vi.fn();

    const customFallback = (error: Error, _reset: () => void) => (
      <div>Custom error message: {error.message}</div>
    );

    render(
      <ErrorBoundary fallback={customFallback}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/custom error message/i)).toBeInTheDocument();

    console.error = consoleError;
  });

  it('should call onError callback when error occurs', () => {
    const consoleError = console.error;
    console.error = vi.fn();

    const onError = vi.fn();

    render(
      <ErrorBoundary onError={onError}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
      })
    );

    console.error = consoleError;
  });

  it('should reset error state when reset is called', () => {
    const consoleError = console.error;
    console.error = vi.fn();

    const { rerender } = render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();

    // Click reset button
    const resetButton = screen.getByRole('button', { name: /try again/i });
    resetButton.click();

    // Rerender with no error
    rerender(
      <ErrorBoundary>
        <ThrowError shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByText('No error')).toBeInTheDocument();

    console.error = consoleError;
  });

  it('should display error details in development mode', () => {
    const consoleError = console.error;
    console.error = vi.fn();

    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    );

    // Use getAllByText since error message appears multiple times
    const errorElements = screen.getAllByText(/test error/i);
    expect(errorElements.length).toBeGreaterThan(0);

    process.env.NODE_ENV = originalEnv;
    console.error = consoleError;
  });
});