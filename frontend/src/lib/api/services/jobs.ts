/**
 * Background Jobs Service
 * 
 * Provides utilities for managing and monitoring background jobs.
 * SCALE-003: Frontend background jobs integration.
 */

import { getApiConfig, getActiveApiUrl, getApiHeaders } from '../core/config';

// Types
export interface JobResult {
  success: boolean;
  data: unknown;
  error: string | null;
  execution_time_seconds: number;
}

export interface BackgroundJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  progress: number;
  metadata: Record<string, unknown>;
  result: JobResult | null;
}

export interface JobListResponse {
  jobs: BackgroundJob[];
  total: number;
}

export interface JobStatsResponse {
  total_jobs: number;
  running_jobs: number;
  max_concurrent_jobs: number;
  queue_size: number;
  status_counts: Record<string, number>;
}

/**
 * List background jobs
 */
export async function listJobs(
  statusFilter?: string,
  limit: number = 100
): Promise<JobListResponse> {
  const baseUrl = getActiveApiUrl();
  const headers = getApiHeaders();
  
  const params = new URLSearchParams();
  if (statusFilter) params.append('status_filter', statusFilter);
  params.append('limit', limit.toString());
  
  const response = await fetch(`${baseUrl}/jobs?${params}`, {
    method: 'GET',
    headers: headers as HeadersInit,
    credentials: 'include',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to list jobs: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get job statistics
 */
export async function getJobStats(): Promise<JobStatsResponse> {
  const baseUrl = getActiveApiUrl();
  const headers = getApiHeaders();

  const response = await fetch(`${baseUrl}/jobs/stats`, {
    method: 'GET',
    headers: headers as HeadersInit,
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error(`Failed to get job stats: ${response.status}`);
  }

  return response.json();
}

/**
 * Get a specific job by ID
 */
export async function getJob(jobId: string): Promise<BackgroundJob> {
  const baseUrl = getActiveApiUrl();
  const headers = getApiHeaders();

  const response = await fetch(`${baseUrl}/jobs/${jobId}`, {
    method: 'GET',
    headers: headers as HeadersInit,
    credentials: 'include',
  });

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Job not found: ${jobId}`);
    }
    throw new Error(`Failed to get job: ${response.status}`);
  }

  return response.json();
}

/**
 * Cancel a pending job
 */
export async function cancelJob(jobId: string): Promise<{ success: boolean; message: string }> {
  const baseUrl = getActiveApiUrl();
  const headers = getApiHeaders();

  const response = await fetch(`${baseUrl}/jobs/${jobId}/cancel`, {
    method: 'POST',
    headers: headers as HeadersInit,
    credentials: 'include',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `Failed to cancel job: ${response.status}`);
  }

  return response.json();
}

/**
 * Poll job status until completion
 */
export async function pollJobUntilComplete(
  jobId: string,
  options: {
    pollInterval?: number;
    timeout?: number;
    onProgress?: (job: BackgroundJob) => void;
  } = {}
): Promise<BackgroundJob> {
  const { pollInterval = 2000, timeout = 600000, onProgress } = options;
  const startTime = Date.now();
  
  while (true) {
    const job = await getJob(jobId);
    
    if (onProgress) {
      onProgress(job);
    }
    
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
      return job;
    }
    
    if (Date.now() - startTime > timeout) {
      throw new Error(`Job polling timed out after ${timeout}ms`);
    }
    
    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }
}

/**
 * React hook for job management
 */
export function useBackgroundJobs() {
  return {
    listJobs,
    getJobStats,
    getJob,
    cancelJob,
    pollJobUntilComplete,
  };
}

/**
 * Job status helpers
 */
export const JobStatusHelpers = {
  isPending: (job: BackgroundJob) => job.status === 'pending',
  isRunning: (job: BackgroundJob) => job.status === 'running',
  isCompleted: (job: BackgroundJob) => job.status === 'completed',
  isFailed: (job: BackgroundJob) => job.status === 'failed',
  isCancelled: (job: BackgroundJob) => job.status === 'cancelled',
  isFinished: (job: BackgroundJob) => 
    ['completed', 'failed', 'cancelled'].includes(job.status),
  
  getStatusColor: (status: BackgroundJob['status']): string => {
    switch (status) {
      case 'pending': return 'yellow';
      case 'running': return 'blue';
      case 'completed': return 'green';
      case 'failed': return 'red';
      case 'cancelled': return 'gray';
      default: return 'gray';
    }
  },
  
  getStatusLabel: (status: BackgroundJob['status']): string => {
    switch (status) {
      case 'pending': return 'Pending';
      case 'running': return 'Running';
      case 'completed': return 'Completed';
      case 'failed': return 'Failed';
      case 'cancelled': return 'Cancelled';
      default: return 'Unknown';
    }
  },
};