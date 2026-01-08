/**
 * Activity Feed Types for Project Chimera
 * 
 * Defines the structure for real-time activity events
 * displayed in the Activity Feed component.
 */

/**
 * Activity event types supported by the system
 */
export type ActivityEventType =
  | "jailbreak_started"
  | "jailbreak_completed"
  | "jailbreak_failed"
  | "autodan_started"
  | "autodan_completed"
  | "autodan_failed"
  | "gptfuzz_started"
  | "gptfuzz_progress"
  | "gptfuzz_completed"
  | "gptfuzz_failed"
  | "houyi_started"
  | "houyi_completed"
  | "houyi_failed"
  | "gradient_started"
  | "gradient_completed"
  | "gradient_failed"
  | "transform_completed"
  | "generation_completed"
  | "provider_connected"
  | "provider_disconnected"
  | "model_changed"
  | "session_created"
  | "session_expired"
  | "error"
  | "warning"
  | "info";

/**
 * Activity event status
 */
export type ActivityStatus = "pending" | "running" | "success" | "failed" | "warning" | "info";

/**
 * Activity event severity for visual styling
 */
export type ActivitySeverity = "low" | "medium" | "high" | "critical";

/**
 * Base activity event structure
 */
export interface ActivityEvent {
  /** Unique event identifier */
  id: string;
  /** Event type */
  type: ActivityEventType;
  /** Event status */
  status: ActivityStatus;
  /** Human-readable title */
  title: string;
  /** Detailed description */
  description?: string;
  /** Event timestamp */
  timestamp: string;
  /** Event severity */
  severity?: ActivitySeverity;
  /** Associated metadata */
  metadata?: ActivityMetadata;
  /** Duration in milliseconds (for completed events) */
  duration_ms?: number;
  /** Progress percentage (0-100) for running events */
  progress?: number;
}

/**
 * Activity metadata for additional context
 */
export interface ActivityMetadata {
  /** Request ID for tracing */
  request_id?: string;
  /** Provider used */
  provider?: string;
  /** Model used */
  model?: string;
  /** Technique suite applied */
  technique_suite?: string;
  /** Potency level */
  potency_level?: number;
  /** Number of techniques applied */
  techniques_count?: number;
  /** Success rate (for batch operations) */
  success_rate?: number;
  /** Error message (for failed events) */
  error_message?: string;
  /** Additional custom data */
  [key: string]: unknown;
}

/**
 * WebSocket message for activity events
 */
export interface ActivityWebSocketMessage {
  type: "activity_event" | "activity_batch" | "heartbeat" | "error";
  event?: ActivityEvent;
  events?: ActivityEvent[];
  error?: string;
  timestamp: string;
}

/**
 * Activity feed filter options
 */
export interface ActivityFilter {
  /** Filter by event types */
  types?: ActivityEventType[];
  /** Filter by status */
  statuses?: ActivityStatus[];
  /** Filter by severity */
  severities?: ActivitySeverity[];
  /** Filter by time range (ISO timestamp) */
  since?: string;
  /** Search query for title/description */
  search?: string;
}

/**
 * Activity feed state
 */
export interface ActivityFeedState {
  /** List of activity events */
  events: ActivityEvent[];
  /** Whether the feed is connected */
  isConnected: boolean;
  /** Whether the feed is loading initial data */
  isLoading: boolean;
  /** Error message if any */
  error: string | null;
  /** Current filter */
  filter: ActivityFilter;
  /** Whether auto-scroll is enabled */
  autoScroll: boolean;
  /** Maximum events to keep in memory */
  maxEvents: number;
}

/**
 * Activity statistics
 */
export interface ActivityStats {
  /** Total events count */
  total: number;
  /** Events by status */
  byStatus: Record<ActivityStatus, number>;
  /** Events by type */
  byType: Partial<Record<ActivityEventType, number>>;
  /** Success rate percentage */
  successRate: number;
  /** Average duration in ms */
  avgDuration: number;
}

/**
 * Helper function to get icon name for activity type
 */
export function getActivityIcon(type: ActivityEventType): string {
  const iconMap: Record<ActivityEventType, string> = {
    jailbreak_started: "Skull",
    jailbreak_completed: "Skull",
    jailbreak_failed: "Skull",
    autodan_started: "Brain",
    autodan_completed: "Brain",
    autodan_failed: "Brain",
    gptfuzz_started: "Bug",
    gptfuzz_progress: "Bug",
    gptfuzz_completed: "Bug",
    gptfuzz_failed: "Bug",
    houyi_started: "Target",
    houyi_completed: "Target",
    houyi_failed: "Target",
    gradient_started: "TrendingUp",
    gradient_completed: "TrendingUp",
    gradient_failed: "TrendingUp",
    transform_completed: "Wand2",
    generation_completed: "Sparkles",
    provider_connected: "Server",
    provider_disconnected: "ServerOff",
    model_changed: "Cpu",
    session_created: "UserPlus",
    session_expired: "UserMinus",
    error: "AlertCircle",
    warning: "AlertTriangle",
    info: "Info",
  };
  return iconMap[type] || "Activity";
}

/**
 * Helper function to get color class for activity status
 */
export function getActivityStatusColor(status: ActivityStatus): string {
  const colorMap: Record<ActivityStatus, string> = {
    pending: "text-yellow-500",
    running: "text-blue-500",
    success: "text-emerald-500",
    failed: "text-red-500",
    warning: "text-amber-500",
    info: "text-sky-500",
  };
  return colorMap[status] || "text-gray-500";
}

/**
 * Helper function to get background color class for activity status
 */
export function getActivityStatusBgColor(status: ActivityStatus): string {
  const colorMap: Record<ActivityStatus, string> = {
    pending: "bg-yellow-500/10",
    running: "bg-blue-500/10",
    success: "bg-emerald-500/10",
    failed: "bg-red-500/10",
    warning: "bg-amber-500/10",
    info: "bg-sky-500/10",
  };
  return colorMap[status] || "bg-gray-500/10";
}

/**
 * Helper function to format activity timestamp
 */
export function formatActivityTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);

  if (diffSec < 60) {
    return "just now";
  } else if (diffMin < 60) {
    return `${diffMin}m ago`;
  } else if (diffHour < 24) {
    return `${diffHour}h ago`;
  } else {
    return date.toLocaleDateString();
  }
}

/**
 * Helper function to format duration
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  } else {
    const min = Math.floor(ms / 60000);
    const sec = Math.floor((ms % 60000) / 1000);
    return `${min}m ${sec}s`;
  }
}