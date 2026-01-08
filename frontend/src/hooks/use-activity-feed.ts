"use client";

/**
 * Activity Feed Hook for Project Chimera
 * 
 * Provides real-time activity event management with WebSocket integration,
 * local state management, and demo mode fallback.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useWebSocket, type WebSocketStatus } from "@/lib/use-websocket";
import type {
  ActivityEvent,
  ActivityEventType,
  ActivityFilter,
  ActivityFeedState,
  ActivityStats,
  ActivityWebSocketMessage,
} from "@/types/activity-types";

// =============================================================================
// Constants
// =============================================================================

const MAX_EVENTS_DEFAULT = 100;
const DEMO_EVENT_INTERVAL = 3000; // 3 seconds between demo events

// =============================================================================
// Demo Data Generator
// =============================================================================

const demoEventTemplates: Partial<ActivityEvent>[] = [
  {
    type: "jailbreak_completed",
    status: "success",
    title: "Jailbreak Generation Complete",
    description: "Successfully generated adversarial prompt using role_hijacking technique",
    metadata: {
      technique_suite: "role_hijacking",
      potency_level: 7,
      techniques_count: 3,
    },
    duration_ms: 2340,
  },
  {
    type: "autodan_started",
    status: "running",
    title: "AutoDAN Attack Started",
    description: "Initiating beam_search optimization for target model",
    metadata: {
      provider: "openai",
      model: "gpt-4",
    },
  },
  {
    type: "autodan_completed",
    status: "success",
    title: "AutoDAN Attack Complete",
    description: "Successfully found adversarial prompt with score 0.92",
    metadata: {
      provider: "openai",
      model: "gpt-4",
      success_rate: 92,
    },
    duration_ms: 45000,
  },
  {
    type: "gptfuzz_progress",
    status: "running",
    title: "GPTFuzz Fuzzing in Progress",
    description: "Evolutionary fuzzing iteration 15/50",
    progress: 30,
    metadata: {
      provider: "anthropic",
      model: "claude-3-opus",
    },
  },
  {
    type: "gptfuzz_completed",
    status: "success",
    title: "GPTFuzz Session Complete",
    description: "Found 3 successful jailbreaks out of 50 attempts",
    metadata: {
      success_rate: 6,
      techniques_count: 50,
    },
    duration_ms: 180000,
  },
  {
    type: "houyi_completed",
    status: "success",
    title: "HouYi Optimization Complete",
    description: "Evolutionary optimization converged with fitness score 0.87",
    metadata: {
      technique_suite: "houyi_genetic",
      potency_level: 8,
    },
    duration_ms: 12500,
  },
  {
    type: "gradient_completed",
    status: "success",
    title: "Gradient Optimization Complete",
    description: "HotFlip optimization completed with 15 token substitutions",
    metadata: {
      technique_suite: "hotflip",
      potency_level: 6,
    },
    duration_ms: 8900,
  },
  {
    type: "transform_completed",
    status: "success",
    title: "Prompt Transformation Complete",
    description: "Applied 5 transformation layers to input prompt",
    metadata: {
      technique_suite: "multi_layer",
      techniques_count: 5,
    },
    duration_ms: 450,
  },
  {
    type: "generation_completed",
    status: "success",
    title: "LLM Generation Complete",
    description: "Generated response from gemini-pro model",
    metadata: {
      provider: "google",
      model: "gemini-pro",
    },
    duration_ms: 1200,
  },
  {
    type: "provider_connected",
    status: "info",
    title: "Provider Connected",
    description: "Successfully connected to OpenAI API",
    metadata: {
      provider: "openai",
    },
  },
  {
    type: "model_changed",
    status: "info",
    title: "Model Selection Changed",
    description: "Switched to claude-3-sonnet model",
    metadata: {
      provider: "anthropic",
      model: "claude-3-sonnet",
    },
  },
  {
    type: "jailbreak_failed",
    status: "failed",
    title: "Jailbreak Generation Failed",
    description: "Content filter blocked the generated prompt",
    metadata: {
      error_message: "Content policy violation detected",
      technique_suite: "adversarial_suffix",
    },
    duration_ms: 1500,
  },
  {
    type: "warning",
    status: "warning",
    title: "Rate Limit Warning",
    description: "Approaching API rate limit (80% used)",
    metadata: {
      provider: "openai",
    },
  },
];

function generateDemoEvent(): ActivityEvent {
  const template = demoEventTemplates[Math.floor(Math.random() * demoEventTemplates.length)];
  return {
    id: `demo-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    type: template.type as ActivityEventType,
    status: template.status!,
    title: template.title!,
    description: template.description,
    timestamp: new Date().toISOString(),
    metadata: template.metadata,
    duration_ms: template.duration_ms,
    progress: template.progress,
  };
}

// =============================================================================
// Hook Options
// =============================================================================

export interface UseActivityFeedOptions {
  /** Maximum events to keep in memory */
  maxEvents?: number;
  /** Enable demo mode with simulated events */
  demoMode?: boolean;
  /** Auto-connect to WebSocket on mount */
  autoConnect?: boolean;
  /** Initial filter */
  initialFilter?: ActivityFilter;
  /** Callback when new event arrives */
  onEvent?: (event: ActivityEvent) => void;
}

// =============================================================================
// Hook Implementation
// =============================================================================

export function useActivityFeed(options: UseActivityFeedOptions = {}) {
  const {
    maxEvents = MAX_EVENTS_DEFAULT,
    demoMode = false,
    autoConnect = true,
    initialFilter = {},
    onEvent,
  } = options;

  // State
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [filter, setFilter] = useState<ActivityFilter>(initialFilter);
  const [autoScroll, setAutoScroll] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const demoIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  // WebSocket connection
  const {
    status: wsStatus,
    connect,
    disconnect,
    isConnected,
  } = useWebSocket("/ws/activity", {
    autoConnect: autoConnect && !demoMode,
    onMessage: (message) => {
      const wsMessage = message as unknown as ActivityWebSocketMessage;
      if (wsMessage.type === "activity_event" && wsMessage.event) {
        addEvent(wsMessage.event);
      } else if (wsMessage.type === "activity_batch" && wsMessage.events) {
        addEvents(wsMessage.events);
      } else if (wsMessage.type === "error") {
        setError(wsMessage.error || "Unknown WebSocket error");
      }
    },
    onOpen: () => {
      setError(null);
      setIsLoading(false);
    },
    onError: () => {
      setError("Failed to connect to activity feed");
    },
  });

  // Add single event
  const addEvent = useCallback(
    (event: ActivityEvent) => {
      setEvents((prev) => {
        const newEvents = [event, ...prev];
        // Trim to max events
        if (newEvents.length > maxEvents) {
          return newEvents.slice(0, maxEvents);
        }
        return newEvents;
      });
      onEventRef.current?.(event);
    },
    [maxEvents]
  );

  // Add multiple events
  const addEvents = useCallback(
    (newEvents: ActivityEvent[]) => {
      setEvents((prev) => {
        const combined = [...newEvents, ...prev];
        // Sort by timestamp descending
        combined.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        // Trim to max events
        if (combined.length > maxEvents) {
          return combined.slice(0, maxEvents);
        }
        return combined;
      });
      newEvents.forEach((event) => onEventRef.current?.(event));
    },
    [maxEvents]
  );

  // Clear all events
  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  // Update filter
  const updateFilter = useCallback((newFilter: Partial<ActivityFilter>) => {
    setFilter((prev) => ({ ...prev, ...newFilter }));
  }, []);

  // Reset filter
  const resetFilter = useCallback(() => {
    setFilter({});
  }, []);

  // Filter events
  const filteredEvents = events.filter((event) => {
    // Filter by types
    if (filter.types && filter.types.length > 0 && !filter.types.includes(event.type)) {
      return false;
    }
    // Filter by statuses
    if (filter.statuses && filter.statuses.length > 0 && !filter.statuses.includes(event.status)) {
      return false;
    }
    // Filter by severities
    if (
      filter.severities &&
      filter.severities.length > 0 &&
      event.severity &&
      !filter.severities.includes(event.severity)
    ) {
      return false;
    }
    // Filter by time
    if (filter.since && new Date(event.timestamp) < new Date(filter.since)) {
      return false;
    }
    // Filter by search
    if (filter.search) {
      const searchLower = filter.search.toLowerCase();
      const matchesTitle = event.title.toLowerCase().includes(searchLower);
      const matchesDescription = event.description?.toLowerCase().includes(searchLower);
      if (!matchesTitle && !matchesDescription) {
        return false;
      }
    }
    return true;
  });

  // Calculate statistics
  const stats: ActivityStats = {
    total: events.length,
    byStatus: {
      pending: events.filter((e) => e.status === "pending").length,
      running: events.filter((e) => e.status === "running").length,
      success: events.filter((e) => e.status === "success").length,
      failed: events.filter((e) => e.status === "failed").length,
      warning: events.filter((e) => e.status === "warning").length,
      info: events.filter((e) => e.status === "info").length,
    },
    byType: events.reduce(
      (acc, event) => {
        acc[event.type] = (acc[event.type] || 0) + 1;
        return acc;
      },
      {} as Partial<Record<ActivityEventType, number>>
    ),
    successRate:
      events.length > 0
        ? Math.round(
            (events.filter((e) => e.status === "success").length /
              events.filter((e) => ["success", "failed"].includes(e.status)).length) *
              100
          ) || 0
        : 0,
    avgDuration:
      events.filter((e) => e.duration_ms).length > 0
        ? Math.round(
            events.filter((e) => e.duration_ms).reduce((sum, e) => sum + (e.duration_ms || 0), 0) /
              events.filter((e) => e.duration_ms).length
          )
        : 0,
  };

  // Demo mode effect
  useEffect(() => {
    if (demoMode) {
      // Generate initial demo events
      const initialEvents: ActivityEvent[] = [];
      for (let i = 0; i < 10; i++) {
        const event = generateDemoEvent();
        event.timestamp = new Date(Date.now() - i * 30000).toISOString(); // Spread over last 5 minutes
        initialEvents.push(event);
      }
      setEvents(initialEvents);

      // Start generating new events
      demoIntervalRef.current = setInterval(() => {
        addEvent(generateDemoEvent());
      }, DEMO_EVENT_INTERVAL);

      return () => {
        if (demoIntervalRef.current) {
          clearInterval(demoIntervalRef.current);
        }
      };
    }
  }, [demoMode, addEvent]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (demoIntervalRef.current) {
        clearInterval(demoIntervalRef.current);
      }
    };
  }, []);

  // Connection status
  const connectionStatus: WebSocketStatus = demoMode ? "connected" : wsStatus;

  return {
    // State
    events: filteredEvents,
    allEvents: events,
    filter,
    autoScroll,
    isLoading,
    error,
    stats,

    // Connection
    isConnected: demoMode || isConnected,
    connectionStatus,
    connect: demoMode ? () => {} : connect,
    disconnect: demoMode ? () => {} : disconnect,

    // Actions
    addEvent,
    addEvents,
    clearEvents,
    updateFilter,
    resetFilter,
    setAutoScroll,

    // Demo mode
    isDemoMode: demoMode,
  };
}

export default useActivityFeed;