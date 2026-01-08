/**
 * API Migration Module
 * 
 * Provides backward compatibility and migration utilities.
 * 
 * @module lib/api/migration
 */

export {
  // Legacy API compatibility
  createLegacyAPI,
  createLegacyChimeraAPI,
  api,
  chimeraApi,
  type LegacyAPI,
  type LegacyChimeraAPI,
  
  // Service compatibility
  LegacyBaseService,
  
  // Hook compatibility
  createFetchFn,
  createMutationFn,
  
  // Error compatibility
  toLegacyError,
  isErrorType,
  
  // Configuration compatibility
  getLegacyConfig,
  setLegacyConfig,
  type LegacyConfig,
  
  // Endpoint mapping
  LEGACY_ENDPOINT_MAP,
  resolveLegacyEndpoint,
} from './compat';