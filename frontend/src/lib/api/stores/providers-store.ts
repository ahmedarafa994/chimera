/**
 * Providers Store
 * Zustand store for provider state management
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { Provider, ProviderModel } from '../types';
import { providersApi } from '../services';
import { logger } from '../logger';

// ============================================================================
// Types
// ============================================================================

interface ProvidersState {
  // State
  providers: Provider[];
  currentProvider: string | null;
  currentModel: string | null;
  models: Record<string, ProviderModel[]>;
  isLoading: boolean;
  error: string | null;
  lastFetched: number | null;

  // Actions
  fetchProviders: () => Promise<void>;
  fetchModels: (providerId: string) => Promise<void>;
  setCurrentProvider: (providerId: string, modelId?: string) => Promise<void>;
  setCurrentModel: (modelId: string) => void;
  refreshProviders: () => Promise<void>;
  clearError: () => void;
}

// ============================================================================
// Store
// ============================================================================

export const useProvidersStore = create<ProvidersState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        providers: [],
        currentProvider: null,
        currentModel: null,
        models: {},
        isLoading: false,
        error: null,
        lastFetched: null,

        // Fetch all providers
        fetchProviders: async () => {
          const { lastFetched } = get();
          const now = Date.now();

          // Cache for 5 minutes
          if (lastFetched && now - lastFetched < 5 * 60 * 1000) {
            return;
          }

          set({ isLoading: true, error: null });

          try {
            const response = await providersApi.getAvailableProviders();
            set({
              providers: response.data.providers,
              currentProvider: response.data.default_provider,
              isLoading: false,
              lastFetched: now,
            });

            logger.logInfo('Providers fetched', { count: response.data.providers.length });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to fetch providers';
            set({ error: message, isLoading: false });
            logger.logError('Failed to fetch providers', error);
          }
        },

        // Fetch models for provider
        fetchModels: async (providerId: string) => {
          const { models } = get();

          if (models[providerId]) {
            return;
          }

          set({ isLoading: true, error: null });

          try {
            const response = await providersApi.getProviderModels(providerId);
            set({
              models: { ...models, [providerId]: response.data.models },
              isLoading: false,
            });

            logger.logInfo('Models fetched', { provider: providerId, count: response.data.models.length });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to fetch models';
            set({ error: message, isLoading: false });
            logger.logError('Failed to fetch models', error);
          }
        },

        // Set current provider
        setCurrentProvider: async (providerId: string, modelId?: string) => {
          set({ isLoading: true, error: null });

          try {
            await providersApi.setDefaultProvider({ provider_id: providerId, model_id: modelId });
            set({
              currentProvider: providerId,
              currentModel: modelId || null,
              isLoading: false,
            });

            // Fetch models for new provider
            const { fetchModels } = get();
            await fetchModels(providerId);

            logger.logInfo('Provider changed', { provider: providerId, model: modelId });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to set provider';
            set({ error: message, isLoading: false });
            logger.logError('Failed to set provider', error);
          }
        },

        // Set current model
        setCurrentModel: (modelId: string) => {
          set({ currentModel: modelId });
          logger.logInfo('Model changed', { model: modelId });
        },

        // Force refresh providers
        refreshProviders: async () => {
          set({ lastFetched: null });
          const { fetchProviders } = get();
          await fetchProviders();
        },

        // Clear error
        clearError: () => {
          set({ error: null });
        },
      }),
      {
        name: 'chimera-providers-storage',
        partialize: (state) => ({
          currentProvider: state.currentProvider,
          currentModel: state.currentModel,
        }),
      }
    ),
    { name: 'ProvidersStore' }
  )
);