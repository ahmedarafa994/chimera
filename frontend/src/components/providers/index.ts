/**
 * Provider Components Index
 *
 * Exports all provider-related components for easy importing.
 *
 * @module components/providers
 */

// Main selector components
export {
  CascadingProviderModelSelector,
  CompactProviderModelSelector,
} from "./CascadingProviderModelSelector";

export { UnifiedProviderSelector } from "./UnifiedProviderSelector";
export { ProviderSettingsPage } from "./ProviderSettingsPage";

// Status and badge components
export {
  ProviderStatusBadge,
  ProviderStatusIndicator,
  getStatusFromHealthScore,
} from "./ProviderStatusBadge";

export {
  ModelCapabilityBadges,
  CapabilityBadge,
  CompactCapabilityIcons,
  hasCapability,
  hasAnyCapability,
  hasAllCapabilities,
} from "./ModelCapabilityBadges";

// Re-export types for convenience
export type {
  CascadingProviderModelSelectorProps,
  ProviderStatusBadgeProps,
  ModelCapabilityBadgesProps,
} from "@/types/unified-providers";
