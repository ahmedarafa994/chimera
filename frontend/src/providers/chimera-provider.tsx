"use client";

/**
 * Chimera Provider
 *
 * Unified React Context for managing global state:
 * - Session state (current provider/model)
 * - Available providers and their health
 * - Circuit breaker status
 *
 * Integrates directly with sessionService and providerService.
 */

import React, {
    createContext,
    useContext,
    useEffect,
    useState,
    useCallback,
    useMemo,
    type ReactNode,
} from "react";
import { Provider } from "../lib/api/core/types";
import { providerService } from "../lib/services/provider-service";
import { sessionService, SessionInfo } from "../lib/services/session-service";
import {
    circuitBreakerRegistry,
    CircuitBreakerStats,
} from "../lib/resilience/circuit-breaker";

// ============================================================================
// Types
// ============================================================================

export interface ChimeraContextValue {
    // Session
    session: SessionInfo | null;
    sessionLoading: boolean;

    // Providers
    providers: Provider[];
    providersLoading: boolean;
    activeProvider: Provider | null;

    // Resilience
    circuitBreakers: CircuitBreakerStats[];
    hasOpenCircuits: boolean;

    // Combined Status
    isReady: boolean;
    error: string | null;

    // Actions
    refresh: () => Promise<void>;
    setModel: (provider: string, model: string) => Promise<boolean>;
    resetCircuits: () => void;
}

// ============================================================================
// Context
// ============================================================================

const ChimeraContext = createContext<ChimeraContextValue | null>(null);

// ============================================================================
// Provider Component
// ============================================================================

export interface ChimeraProviderProps {
    children: ReactNode;
}

export function ChimeraProvider({ children }: ChimeraProviderProps) {
    const [session, setSession] = useState<SessionInfo | null>(null);
    const [providers, setProviders] = useState<Provider[]>([]);
    const [sessionLoading, setSessionLoading] = useState(true);
    const [providersLoading, setProvidersLoading] = useState(true);
    const [circuitBreakers, setCircuitBreakers] = useState<CircuitBreakerStats[]>([]);
    const [error, setError] = useState<string | null>(null);

    // Initialize and subscribe
    useEffect(() => {
        // 1. Session Sub
        const unsubSession = sessionService.subscribe((s) => {
            setSession(s);
            setSessionLoading(false);
        });

        // 2. Providers Sub
        const unsubProviders = providerService.subscribe((p) => {
            setProviders(p);
            setProvidersLoading(false);
        });

        // 3. Circuit Breaker Sub
        const unsubCircuits = circuitBreakerRegistry.addListener(() => {
            setCircuitBreakers(circuitBreakerRegistry.getAllStats());
        });

        // Initial load
        const init = async () => {
            try {
                await Promise.all([
                    sessionService.initialize(),
                    providerService.refresh()
                ]);
            } catch (err) {
                setError(err instanceof Error ? err.message : "Initialization failed");
            }
        };

        init();

        return () => {
            unsubSession();
            unsubProviders();
            unsubCircuits();
        };
    }, []);

    // Actions
    const refresh = useCallback(async () => {
        setProvidersLoading(true);
        await providerService.refresh();
        setProvidersLoading(false);
    }, []);

    const setModel = useCallback(async (providerId: string, modelId: string) => {
        const success = await sessionService.setModel(providerId, modelId);
        if (success) {
            await providerService.setActive(providerId, modelId);
        }
        return success;
    }, []);

    const resetCircuits = useCallback(() => {
        circuitBreakerRegistry.resetAll();
    }, []);

    // Derived state
    const activeProvider = useMemo(() => {
        if (!session?.provider) return null;
        return providers.find(p => p.id === session.provider) || null;
    }, [session, providers]);

    const hasOpenCircuits = useMemo(
        () => circuitBreakers.some(cb => cb.state === "OPEN"),
        [circuitBreakers]
    );

    const isReady = !sessionLoading && !providersLoading;

    const value = useMemo<ChimeraContextValue>(
        () => ({
            session,
            sessionLoading,
            providers,
            providersLoading,
            activeProvider,
            circuitBreakers,
            hasOpenCircuits,
            isReady,
            error,
            refresh,
            setModel,
            resetCircuits,
        }),
        [
            session,
            sessionLoading,
            providers,
            providersLoading,
            activeProvider,
            circuitBreakers,
            hasOpenCircuits,
            isReady,
            error,
            refresh,
            setModel,
            resetCircuits
        ]
    );

    return (
        <ChimeraContext.Provider value={value}>
            {children}
        </ChimeraContext.Provider>
    );
}

// ============================================================================
// Hooks
// ============================================================================

export function useChimera() {
    const context = useContext(ChimeraContext);
    if (!context) {
        throw new Error("useChimera must be used within a ChimeraProvider");
    }
    return context;
}

export function useSession() {
    const { session, sessionLoading, setModel } = useChimera();
    return { session, loading: sessionLoading, setModel };
}

export function useProviders() {
    const { providers, providersLoading, activeProvider, refresh } = useChimera();
    return { providers, loading: providersLoading, activeProvider, refresh };
}

export function useResilience() {
    const { circuitBreakers, hasOpenCircuits, resetCircuits } = useChimera();
    return { breakers: circuitBreakers, hasOpen: hasOpenCircuits, reset: resetCircuits };
}
