/**
 * Admin Types
 * 
 * Type definitions for admin dashboard operations.
 */

/**
 * Admin statistics
 */
export interface AdminStats {
  total_users: number;
  total_prompts: number;
  total_attacks: number;
  total_evasions: number;
  active_sessions: number;
  cache_hit_rate: number;
  average_response_time_ms: number;
  uptime_seconds: number;
  last_updated: string;
}

/**
 * Data import request
 */
export interface ImportDataRequest {
  source: 'file' | 'url' | 'database' | 'api';
  data_type: 'prompts' | 'jailbreaks' | 'attacks' | 'models';
  data?: string | Record<string, unknown>;
  url?: string;
  options?: {
    overwrite?: boolean;
    validate?: boolean;
    dry_run?: boolean;
  };
}

/**
 * Data import response
 */
export interface ImportDataResponse {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  records_processed: number;
  records_imported: number;
  records_failed: number;
  errors: string[];
  warnings: string[];
  started_at: string;
  completed_at?: string;
}

/**
 * Cache statistics
 */
export interface CacheStats {
  total_entries: number;
  memory_usage_mb: number;
  hit_count: number;
  miss_count: number;
  hit_rate: number;
  eviction_count: number;
  last_cleared: string;
  cache_types: {
    type: string;
    entries: number;
    memory_mb: number;
  }[];
}

/**
 * System health status
 */
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    name: string;
    status: 'healthy' | 'degraded' | 'unhealthy';
    latency_ms?: number;
    error?: string;
  }[];
  database: {
    connected: boolean;
    latency_ms: number;
    pool_size: number;
    active_connections: number;
  };
  redis: {
    connected: boolean;
    latency_ms: number;
    memory_usage_mb: number;
  };
  celery: {
    connected: boolean;
    active_workers: number;
    queued_tasks: number;
  };
  timestamp: string;
}