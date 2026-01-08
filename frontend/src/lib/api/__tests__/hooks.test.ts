/**
 * API Hooks Tests
 * Unit tests for React hooks
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useApi, useMutation, useQuery, usePaginatedQuery } from '../hooks/use-api';

// Mock the logger
vi.mock('../logger', () => ({
  logger: {
    logInfo: vi.fn(),
    logError: vi.fn(),
    logWarning: vi.fn(),
  },
}));

describe('useApi Hook', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize with correct default state', () => {
    const mockApiFn = vi.fn().mockResolvedValue({ data: null, status: 200 });
    
    const { result } = renderHook(() => useApi(mockApiFn));
    
    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.isSuccess).toBe(false);
    expect(result.current.isError).toBe(false);
  });

  it('should execute API call and update state', async () => {
    const mockData = { id: 1, name: 'Test' };
    const mockApiFn = vi.fn().mockResolvedValue({ data: mockData, status: 200 });
    
    const { result } = renderHook(() => useApi(mockApiFn));
    
    await act(async () => {
      await result.current.execute();
    });
    
    expect(result.current.data).toEqual(mockData);
    expect(result.current.isSuccess).toBe(true);
    expect(result.current.isLoading).toBe(false);
  });

  it('should handle errors correctly', async () => {
    const mockError = {
      message: 'Network error',
      code: 'NETWORK_ERROR',
      status: 500,
    };
    const mockApiFn = vi.fn().mockRejectedValue(mockError);
    
    const { result } = renderHook(() => useApi(mockApiFn));
    
    await act(async () => {
      await result.current.execute();
    });
    
    expect(result.current.error).toEqual(mockError);
    expect(result.current.isError).toBe(true);
    expect(result.current.isSuccess).toBe(false);
  });

  it('should reset state correctly', async () => {
    const mockData = { id: 1 };
    const mockApiFn = vi.fn().mockResolvedValue({ data: mockData, status: 200 });
    
    const { result } = renderHook(() => useApi(mockApiFn));
    
    await act(async () => {
      await result.current.execute();
    });
    
    expect(result.current.data).toEqual(mockData);
    
    act(() => {
      result.current.reset();
    });
    
    expect(result.current.data).toBeNull();
    expect(result.current.isSuccess).toBe(false);
  });

  it('should call onSuccess callback', async () => {
    const mockData = { id: 1 };
    const mockApiFn = vi.fn().mockResolvedValue({ data: mockData, status: 200 });
    const onSuccess = vi.fn();
    
    const { result } = renderHook(() => useApi(mockApiFn, { onSuccess }));
    
    await act(async () => {
      await result.current.execute();
    });
    
    expect(onSuccess).toHaveBeenCalledWith(mockData);
  });

  it('should call onError callback', async () => {
    const mockError = { message: 'Error', code: 'ERROR', status: 500 };
    const mockApiFn = vi.fn().mockRejectedValue(mockError);
    const onError = vi.fn();
    
    const { result } = renderHook(() => useApi(mockApiFn, { onError }));
    
    await act(async () => {
      await result.current.execute();
    });
    
    expect(onError).toHaveBeenCalledWith(mockError);
  });
});

describe('useMutation Hook', () => {
  it('should not execute immediately', () => {
    const mockApiFn = vi.fn().mockResolvedValue({ data: {}, status: 200 });
    
    renderHook(() => useMutation(mockApiFn));
    
    expect(mockApiFn).not.toHaveBeenCalled();
  });

  it('should execute on demand', async () => {
    const mockData = { created: true };
    const mockApiFn = vi.fn().mockResolvedValue({ data: mockData, status: 201 });
    
    const { result } = renderHook(() => useMutation(mockApiFn));
    
    await act(async () => {
      await result.current.execute({ name: 'test' });
    });
    
    expect(mockApiFn).toHaveBeenCalledWith({ name: 'test' });
    expect(result.current.data).toEqual(mockData);
  });
});

describe('useQuery Hook', () => {
  it('should execute immediately when enabled', async () => {
    const mockData = { items: [] };
    const mockApiFn = vi.fn().mockResolvedValue({ data: mockData, status: 200 });
    
    const { result } = renderHook(() => useQuery(mockApiFn, { enabled: true }));
    
    await waitFor(() => {
      expect(mockApiFn).toHaveBeenCalled();
    });
  });

  it('should not execute when disabled', () => {
    const mockApiFn = vi.fn().mockResolvedValue({ data: {}, status: 200 });
    
    renderHook(() => useQuery(mockApiFn, { enabled: false }));
    
    expect(mockApiFn).not.toHaveBeenCalled();
  });
});

describe('usePaginatedQuery Hook', () => {
  it('should handle pagination correctly', async () => {
    const mockPage1 = {
      items: [{ id: 1 }, { id: 2 }],
      total: 4,
      page: 1,
      page_size: 2,
      total_pages: 2,
    };
    
    const mockApiFn = vi.fn().mockResolvedValue({ data: mockPage1, status: 200 });
    
    const { result } = renderHook(() => usePaginatedQuery(mockApiFn));
    
    await act(async () => {
      await result.current.execute(1);
    });
    
    expect(result.current.items).toHaveLength(2);
    expect(result.current.hasMore).toBe(true);
  });
});