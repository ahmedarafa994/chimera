/**
 * Model Selector Components
 * 
 * Exports all model selector variants for different use cases:
 * - OptimizedModelSelector: Performance-optimized with selective subscriptions (RECOMMENDED)
 * - UnifiedModelSelector: Production-ready with WebSocket sync, error handling, rate limiting
 * - ModelSelector: Original implementation with session management
 * - EnhancedModelSelector: Two-panel layout with provider health status
 * - ModelDropdown: Compact dropdown for header/toolbar use
 */

// Performance-optimized selector (recommended for production)
export { OptimizedModelSelector } from "./OptimizedModelSelector";

// Other variants
export { UnifiedModelSelector } from "./UnifiedModelSelector";
export { ModelSelector } from "./ModelSelector";
export { EnhancedModelSelector } from "./EnhancedModelSelector";
export { ModelDropdown } from "./ModelDropdown";

// Default export is the optimized selector for best performance
export { OptimizedModelSelector as default } from "./OptimizedModelSelector";

// Re-export hooks from optimized store
export {
  useOptimizedModelSelection,
  useModelSelectionValue,
  useProvidersList,
  useModelsList,
  useModelSelectionLoading,
  useModelSelectionActions,
} from "@/lib/stores/optimized-model-selection-store";