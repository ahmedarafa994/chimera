/**
 * Hooks Index
 * Export all API-related React hooks
 */

export {
  useApi,
  useMutation,
  useQuery,
  usePaginatedQuery,
  useOptimisticMutation,
} from './use-api';

export type {
  UseApiState,
  UseApiOptions,
  UseApiResult,
} from './use-api';

export {
  useWebSocket,
  useStream,
  useModelSelectionSync,
} from './use-websocket';

export type {
  UseWebSocketOptions,
  UseWebSocketResult,
  UseStreamOptions,
  UseStreamResult,
} from './use-websocket';
