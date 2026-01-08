"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/**
 * Optimized React Query Provider with performance-tuned defaults
 *
 * Performance optimizations (PERF-005):
 * - Increased staleTime to 60s to reduce unnecessary refetches
 * - Added gcTime (garbage collection) to manage memory efficiently
 * - Optimized retry logic with exponential backoff
 * - Configured structural sharing for efficient re-renders
 */
export default function QueryProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Smart retry logic with exponential backoff (PERF-006)
            retry: (failureCount, error) => {
              // Don't retry on 4xx client errors
              if (error instanceof Error) {
                const message = error.message.toLowerCase();
                if (message.includes("400") || message.includes("401") ||
                  message.includes("403") || message.includes("404") ||
                  message.includes("422")) {
                  return false;
                }
              }
              // Retry up to 2 times for server/network errors
              return failureCount < 2;
            },
            // Exponential backoff for retries
            retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),

            // Aggressive caching for stability (Strategy Target: 5 mins)
            staleTime: 5 * 60 * 1000,

            // Garbage collection time (Strategy Target: 30 mins)
            gcTime: 30 * 60 * 1000,

            // Don't refetch on window focus in development
            refetchOnWindowFocus: process.env.NODE_ENV === "production",

            // Disable refetch on reconnect to prevent thundering herd
            refetchOnReconnect: "always",

            // Handle network errors gracefully
            networkMode: "offlineFirst",

            // Enable structural sharing for efficient re-renders (PERF-009)
            structuralSharing: true,

            // Placeholder data while loading (improves perceived performance)
            placeholderData: (previousData: unknown) => previousData,
          },
          mutations: {
            // Don't retry mutations by default
            retry: false,
            networkMode: "offlineFirst",
            // Optimistic updates should be handled per-mutation
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}