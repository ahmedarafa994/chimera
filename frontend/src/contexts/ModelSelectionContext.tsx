"use client";

/**
 * Model Selection Context
 * 
 * This is a backward-compatible wrapper that delegates to the UnifiedModelProvider.
 * Existing components using useModelSelection() will continue to work.
 * 
 * For new code, prefer using useUnifiedModel() directly from unified-model-provider.
 */

import React, { createContext, useContext, ReactNode } from "react";
import {
  useUnifiedModel,
  UnifiedModelProvider,
  type ModelSelection,
} from "@/providers/unified-model-provider";

interface ModelSelectionContextType {
  selection: ModelSelection;
  setSelection: (provider: string, model: string) => Promise<void>;
  clearSelection: () => Promise<void>;
  isLoading: boolean;
}

const ModelSelectionContext = createContext<ModelSelectionContextType | undefined>(undefined);

/**
 * ModelSelectionProvider - Backward compatible provider
 * 
 * Note: This provider is now a thin wrapper. The actual state management
 * is handled by UnifiedModelProvider in the root layout.
 * 
 * If UnifiedModelProvider is already in the tree (which it should be),
 * this provider just passes through to it.
 */
export function ModelSelectionProvider({ children }: { children: ReactNode }) {
  // This provider is kept for backward compatibility but doesn't add its own state.
  // The UnifiedModelProvider in layout.tsx handles all state management.
  return <>{children}</>;
}

/**
 * useModelSelection - Backward compatible hook
 * 
 * Delegates to useUnifiedModel() from the unified provider.
 */
export function useModelSelection(): ModelSelectionContextType {
  // Try to use the unified provider
  try {
    const unified = useUnifiedModel();
    return {
      selection: unified.selection,
      setSelection: unified.setSelection,
      clearSelection: unified.clearSelection,
      isLoading: unified.isLoading,
    };
  } catch {
    // If unified provider is not available, throw a helpful error
    throw new Error(
      "useModelSelection must be used within a UnifiedModelProvider. " +
      "Make sure UnifiedModelProvider is in your component tree (usually in layout.tsx)."
    );
  }
}

// Re-export types for convenience
export type { ModelSelection };