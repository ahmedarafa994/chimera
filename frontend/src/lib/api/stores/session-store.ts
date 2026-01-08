/**
 * Session Store
 * Zustand store for session and message state management
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { Session, Message } from '../types';
import { sessionsApi } from '../services';
import { logger } from '../logger';

// ============================================================================
// Types
// ============================================================================

interface SessionState {
  // State
  currentSession: Session | null;
  sessions: Session[];
  messages: Message[];
  isLoading: boolean;
  isSending: boolean;
  error: string | null;
  hasMore: boolean;
  page: number;

  // Actions
  fetchCurrentSession: () => Promise<void>;
  createSession: (modelId?: string, provider?: string) => Promise<Session>;
  switchSession: (sessionId: string) => Promise<void>;
  fetchMessages: (sessionId: string, loadMore?: boolean) => Promise<void>;
  sendMessage: (content: string, stream?: boolean) => Promise<Message | null>;
  addMessage: (message: Message) => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  clearMessages: () => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  clearError: () => void;
}

// ============================================================================
// Store
// ============================================================================

export const useSessionStore = create<SessionState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        currentSession: null,
        sessions: [],
        messages: [],
        isLoading: false,
        isSending: false,
        error: null,
        hasMore: true,
        page: 1,

        // Fetch current session
        fetchCurrentSession: async () => {
          set({ isLoading: true, error: null });

          try {
            const response = await sessionsApi.getCurrentSession();
            set({
              currentSession: response.data,
              isLoading: false,
            });

            // Fetch messages for session
            const { fetchMessages } = get();
            await fetchMessages(response.data.session_id);

            logger.logInfo('Session fetched', { sessionId: response.data.session_id });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to fetch session';
            set({ error: message, isLoading: false });
            logger.logError('Failed to fetch session', error);
          }
        },

        // Create new session
        createSession: async (modelId?: string, provider?: string) => {
          set({ isLoading: true, error: null });

          try {
            const response = await sessionsApi.createSession({
              model_id: modelId,
              provider,
            });

            set({
              currentSession: response.data,
              messages: [],
              isLoading: false,
              hasMore: true,
              page: 1,
            });

            logger.logInfo('Session created', { sessionId: response.data.session_id });
            return response.data;
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to create session';
            set({ error: message, isLoading: false });
            logger.logError('Failed to create session', error);
            throw error;
          }
        },

        // Switch to different session
        switchSession: async (sessionId: string) => {
          set({ isLoading: true, error: null, messages: [], page: 1, hasMore: true });

          try {
            const response = await sessionsApi.getSession(sessionId);
            set({
              currentSession: response.data,
              isLoading: false,
            });

            // Fetch messages for session
            const { fetchMessages } = get();
            await fetchMessages(sessionId);

            logger.logInfo('Session switched', { sessionId });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to switch session';
            set({ error: message, isLoading: false });
            logger.logError('Failed to switch session', error);
          }
        },

        // Fetch messages
        fetchMessages: async (sessionId: string, loadMore = false) => {
          const { page, messages, hasMore } = get();

          if (loadMore && !hasMore) return;

          const currentPage = loadMore ? page + 1 : 1;
          set({ isLoading: true, error: null });

          try {
            const response = await sessionsApi.getMessages(sessionId, currentPage);
            const newMessages = loadMore
              ? [...messages, ...response.data.items]
              : response.data.items;

            set({
              messages: newMessages,
              hasMore: currentPage < response.data.total_pages,
              page: currentPage,
              isLoading: false,
            });

            logger.logInfo('Messages fetched', { sessionId, count: response.data.items.length });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to fetch messages';
            set({ error: message, isLoading: false });
            logger.logError('Failed to fetch messages', error);
          }
        },

        // Send message
        sendMessage: async (content: string, stream = false) => {
          const { currentSession, messages } = get();

          if (!currentSession) {
            set({ error: 'No active session' });
            return null;
          }

          // Optimistic update - add user message
          const userMessage: Message = {
            id: `temp-${Date.now()}`,
            session_id: currentSession.session_id,
            role: 'user',
            content,
            timestamp: new Date().toISOString(),
          };

          set({
            messages: [...messages, userMessage],
            isSending: true,
            error: null,
          });

          try {
            const response = await sessionsApi.sendMessage(currentSession.session_id, {
              content,
              stream,
            });

            // Replace temp message with real one and add response
            set((state) => ({
              messages: state.messages.map((m) =>
                m.id === userMessage.id ? { ...m, id: response.data.id } : m
              ),
              isSending: false,
            }));

            logger.logInfo('Message sent', { sessionId: currentSession.session_id });
            return response.data;
          } catch (error) {
            // Rollback optimistic update
            set((state) => ({
              messages: state.messages.filter((m) => m.id !== userMessage.id),
              isSending: false,
              error: error instanceof Error ? error.message : 'Failed to send message',
            }));

            logger.logError('Failed to send message', error);
            return null;
          }
        },

        // Add message (for streaming)
        addMessage: (message: Message) => {
          set((state) => ({
            messages: [...state.messages, message],
          }));
        },

        // Update message (for streaming updates)
        updateMessage: (messageId: string, updates: Partial<Message>) => {
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === messageId ? { ...m, ...updates } : m
            ),
          }));
        },

        // Clear messages
        clearMessages: async () => {
          const { currentSession } = get();

          if (!currentSession) return;

          set({ isLoading: true, error: null });

          try {
            await sessionsApi.clearMessages(currentSession.session_id);
            set({ messages: [], isLoading: false, page: 1, hasMore: true });

            logger.logInfo('Messages cleared', { sessionId: currentSession.session_id });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to clear messages';
            set({ error: message, isLoading: false });
            logger.logError('Failed to clear messages', error);
          }
        },

        // Delete session
        deleteSession: async (sessionId: string) => {
          set({ isLoading: true, error: null });

          try {
            await sessionsApi.deleteSession(sessionId);

            // If deleted session is current, clear state
            const { currentSession } = get();
            if (currentSession?.session_id === sessionId) {
              set({
                currentSession: null,
                messages: [],
                isLoading: false,
              });
            } else {
              set({ isLoading: false });
            }

            logger.logInfo('Session deleted', { sessionId });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to delete session';
            set({ error: message, isLoading: false });
            logger.logError('Failed to delete session', error);
          }
        },

        // Clear error
        clearError: () => {
          set({ error: null });
        },
      }),
      {
        name: 'chimera-session-storage',
        partialize: (state) => ({
          currentSession: state.currentSession,
        }),
      }
    ),
    { name: 'SessionStore' }
  )
);