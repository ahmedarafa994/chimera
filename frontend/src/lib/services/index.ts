/**
 * Services Index for Project Chimera Frontend
 *
 * Central export point for all API services
 */

// Admin Service
export { adminService, default as adminServiceDefault } from "./admin-service";
export * from "./admin-service";

// AutoAdv Service - export specific items to avoid duplicate 'setError'
export { autoAdvService, default as autoAdvServiceDefault } from "./autoadv-service";
export {
  // Types
  type AutoAdvStartRequest,
  type AutoAdvStartResponse,
  type AutoAdvProgress,
  type AutoAdvResult,
  type AutoAdvIterationResult,
  type AutoAdvWebSocketMessage,
  type AutoAdvRequest,
  type AutoAdvResponse,
  type AutoAdvState,
  // API Methods
  startAutoAdvAttack,
  getAutoAdvStatus,
  getAutoAdvResults,
  cancelAutoAdvSession,
  // WebSocket
  createAutoAdvWebSocket,
  getWebSocketUrl,
  // Convenience
  executeAutoAdvAttack,
  // Legacy methods
  startAutoAdv,
  generateAutoAdv,
  // State helpers - renamed to avoid conflict
  createInitialAutoAdvState,
  setSessionStarted,
  setWsConnected,
  setProgress as setAutoAdvProgress,
  setResult as setAutoAdvResult,
  setError as setAutoAdvError,
  resetAutoAdvState,
} from "./autoadv-service";

// AutoDAN Lifelong Service
export { autodanLifelongService, default as autodanLifelongServiceDefault } from "./autodan-lifelong-service";
export * from "./autodan-lifelong-service";

// Chat Service
export { chatService, default as chatServiceDefault } from "./chat-service";
export * from "./chat-service";

// Evasion Service
export { evasionService, default as evasionServiceDefault } from "./evasion-service";
export * from "./evasion-service";

// Provider Management Service
export { providerManagementService, default as providerManagementServiceDefault } from "./provider-management-service";
export * from "./provider-management-service";

// Provider Service
export { providerService, default as providerServiceDefault } from "./provider-service";
export * from "./provider-service";

// Session Service
export { sessionService, default as sessionServiceDefault } from "./session-service";
export * from "./session-service";
