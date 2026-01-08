"use client";

/**
 * Session Service
 *
 * Manages user sessions, model selection, and session persistence.
 * Uses the new API Client for backend communication.
 *
 * @module lib/services/session-service
 */

import { apiClient, ENDPOINTS } from '../api/core';
import { handleApiError } from '../errors';
import {
  CreateSessionResponseSchema,
  UpdateModelRequestSchema,
  CreateSessionRequestSchema
} from '../transforms/schemas';

export interface SessionInfo {
  sessionId: string;
  provider: string | null;
  model: string | null;
  createdAt: string;
  lastActivity: string;
}

const SESSION_ID_KEY = 'chimera_session_id';
const SESSION_DATA_KEY = 'chimera_session_data';

class SessionService {
  private currentSession: SessionInfo | null = null;
  private listeners: Set<(session: SessionInfo | null) => void> = new Set();
  private initialized = false;

  /**
   * Initialize the session
   * Tries to restore from localStorage or create a new one on the backend
   */
  async initialize(): Promise<SessionInfo | null> {
    if (this.initialized) return this.currentSession;

    try {
      // 1. Check for existing session ID in localStorage
      let sessionId = typeof window !== 'undefined' ? localStorage.getItem(SESSION_ID_KEY) : null;
      if (sessionId) sessionId = sessionId.replace(/^"|"$/g, '');

      // 2. Fetch session from backend
      // Note: apiClient automatically adds X-Session-ID if it's in localStorage
      const sessionData = await apiClient.get<any>(ENDPOINTS.SESSION, {
        responseSchema: CreateSessionResponseSchema
      });

      if (sessionData && sessionData.session_id) {
        this.currentSession = {
          sessionId: sessionData.session_id,
          provider: sessionData.provider,
          model: sessionData.model,
          createdAt: sessionData.created_at || new Date().toISOString(),
          lastActivity: sessionData.last_activity || new Date().toISOString(),
        };
      } else {
        // Create new session if none exists
        const newSession = await apiClient.post<any>(ENDPOINTS.SESSION, {}, {
          responseSchema: CreateSessionResponseSchema
        });
        this.currentSession = {
          sessionId: newSession.session_id,
          provider: newSession.provider,
          model: newSession.model,
          createdAt: newSession.created_at || new Date().toISOString(),
          lastActivity: newSession.last_activity || new Date().toISOString(),
        };
      }

      // 3. Persist and notify
      if (this.currentSession) {
        this.persist(this.currentSession);
        this.initialized = true;
        this.notify();
      }

      return this.currentSession;
    } catch (error) {
      handleApiError(error, { context: 'SessionService.initialize' });
      return null;
    }
  }

  /**
   * Change the current model/provider for the session
   */
  async setModel(provider: string, model: string): Promise<boolean> {
    try {
      await apiClient.put(ENDPOINTS.SESSION_MODEL, { provider, model }, {
        requestSchema: UpdateModelRequestSchema
      });

      if (this.currentSession) {
        this.currentSession = {
          ...this.currentSession,
          provider,
          model,
          lastActivity: new Date().toISOString(),
        };
        this.persist(this.currentSession);
        this.notify();
      }
      return true;
    } catch (error) {
      handleApiError(error, { context: 'SessionService.setModel' });
      return false;
    }
  }

  /**
   * Refresh session data from backend
   */
  async refresh(): Promise<SessionInfo | null> {
    try {
      const sessionData = await apiClient.get<any>(ENDPOINTS.SESSION);
      if (sessionData) {
        this.currentSession = {
          sessionId: sessionData.session_id,
          provider: sessionData.provider,
          model: sessionData.model,
          createdAt: sessionData.created_at || this.currentSession?.createdAt || new Date().toISOString(),
          lastActivity: sessionData.last_activity || new Date().toISOString(),
        };
        this.persist(this.currentSession);
        this.notify();
      }
      return this.currentSession;
    } catch (error) {
      handleApiError(error, { context: 'SessionService.refresh' });
      return this.currentSession;
    }
  }

  /**
   * Get the current session
   */
  getSession(): SessionInfo | null {
    return this.currentSession;
  }

  /**
   * Clear the current session (logout/reset)
   */
  async clear(): Promise<void> {
    try {
      await apiClient.delete(ENDPOINTS.SESSION);
    } catch (e) {
      // Ignore errors during cleanup
    } finally {
      this.currentSession = null;
      if (typeof window !== 'undefined') {
        localStorage.removeItem(SESSION_ID_KEY);
        localStorage.removeItem(SESSION_DATA_KEY);
      }
      this.notify();
    }
  }

  /**
   * Subscribe to session changes
   */
  subscribe(listener: (session: SessionInfo | null) => void): () => void {
    this.listeners.add(listener);
    listener(this.currentSession);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    this.listeners.forEach(l => l(this.currentSession));
  }

  private persist(session: SessionInfo): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem(SESSION_ID_KEY, session.sessionId);
    localStorage.setItem(SESSION_DATA_KEY, JSON.stringify(session));
  }
}

export const sessionService = new SessionService();
export default sessionService;
