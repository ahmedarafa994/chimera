/**
 * API Key Management Components
 *
 * Secure UI components for managing LLM provider API keys with encryption at rest.
 *
 * Features:
 * - Tabbed provider view for organizing keys by provider
 * - Add/edit API keys with masked input and real-time test connection
 * - Key status indicators (active, inactive, expired, rate limited, revoked)
 * - Role-based key designation (primary, backup, fallback) for failover
 * - Usage statistics and success rate tracking
 *
 * @example
 * ```tsx
 * import { ApiKeyManager, ApiKeyForm, ApiKeyList } from "@/components/api-keys";
 *
 * // Full manager with all features
 * <ApiKeyManager
 *   keys={keys}
 *   providers={providerSummaries}
 *   onCreateKey={handleCreate}
 *   onDeleteKey={handleDelete}
 *   onTestKey={handleTest}
 *   ...
 * />
 *
 * // Just the list component
 * <ApiKeyList
 *   keys={keys}
 *   onEdit={handleEdit}
 *   onDelete={handleDelete}
 * />
 *
 * // Just the form component
 * <ApiKeyForm
 *   isOpen={isOpen}
 *   onClose={handleClose}
 *   onSave={handleSave}
 * />
 * ```
 */

// Main manager component with tabbed provider view
export { ApiKeyManager } from "./ApiKeyManager";
export type { ApiKeyManagerProps, ProviderKeySummary } from "./ApiKeyManager";

// Form component for add/edit with masked input
export { ApiKeyForm } from "./ApiKeyForm";
export type {
  ApiKeyFormProps,
  ApiKeyFormData,
  ApiKeyRole,
  ApiKeyStatus,
  ProviderId,
} from "./ApiKeyForm";

// List component with key cards
export { ApiKeyList } from "./ApiKeyList";
export type { ApiKeyListProps, ApiKeyItem } from "./ApiKeyList";
