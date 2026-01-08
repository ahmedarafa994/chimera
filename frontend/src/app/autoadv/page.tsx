
"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import ConfigForm from "@/components/autoadv/ConfigForm";
import ExecutionView from "@/components/autoadv/ExecutionView";
import ResultsDisplay from "@/components/autoadv/ResultsDisplay";
import { autoAdvService, AutoAdvRequest, AutoAdvWebSocketMessage } from "@/lib/services/autoadv-service";
import { useAutoAdvWithProgress } from "@/hooks";

// Log entry type for execution view
interface LogEntry {
    timestamp: string;
    type: "info" | "success" | "error" | "warning" | "progress";
    message: string;
    data?: unknown;
}

// Result type for results display
interface ResultEntry {
    id: string;
    prompt: string;
    response: string;
    score: number;
    model: string;
}

export default function AutoAdvPage() {
    const [isRunning, setIsRunning] = useState(false);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [results, setResults] = useState<ResultEntry[]>([]);
    const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);

    // Use the centralized AutoAdv hook for progress tracking
    const {
        isLoading,
        error: hookError,
        currentResult,
        progress: _progress, // Prefix with _ to indicate intentionally unused
        isWebSocketConnected,
    } = useAutoAdvWithProgress();

    // Add log entry helper
    const addLog = useCallback((type: LogEntry["type"], message: string, data?: unknown) => {
        setLogs(prev => [...prev, {
            timestamp: new Date().toLocaleTimeString(),
            type,
            message,
            data,
        }]);
    }, []);

    // Handle WebSocket messages
    const handleWebSocketMessage = useCallback((message: AutoAdvWebSocketMessage) => {
        switch (message.type) {
            case "progress":
                addLog("progress", `Progress: ${message.progress ?? 0}% - Iteration ${message.iteration ?? 0}/${message.total_iterations ?? 0}`);
                break;
            case "result":
                if (message.data && typeof message.data === 'object') {
                    const data = message.data as Record<string, unknown>;
                    setResults(prev => [...prev, {
                        id: Date.now().toString(),
                        prompt: (data.final_prompt as string) || (data.best_prompt as string) || "N/A",
                        response: (data.target_response as string) || "N/A",
                        score: (data.score as number) || (data.best_score as number) || 0,
                        model: (data.model as string) || (data.target_model as string) || "unknown",
                    }]);
                    addLog("success", "Result received", message.data);
                }
                break;
            case "error":
                addLog("error", message.message || "Unknown error occurred");
                break;
            case "complete":
                addLog("success", "AutoAdv execution completed");
                setIsRunning(false);
                break;
            default:
                addLog("info", message.message || "Unknown message type");
        }
    }, [addLog]);

    // Connect to WebSocket using centralized service
    const connectWebSocket = useCallback((taskId: string) => {
        const wsUrl = autoAdvService.getWebSocketUrl(taskId);
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            addLog("info", "WebSocket Connected");
            setWsConnection(ws);
        };

        ws.onmessage = (event) => {
            try {
                const data: AutoAdvWebSocketMessage = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (err) {
                console.error("Failed to parse WebSocket message:", err);
                addLog("error", "Failed to parse WebSocket message");
            }
        };

        ws.onerror = (error) => {
            console.error("WebSocket Error", error);
            addLog("error", "WebSocket connection error");
        };

        ws.onclose = () => {
            addLog("info", "WebSocket disconnected");
            setWsConnection(null);
        };

        return ws;
    }, [addLog, handleWebSocketMessage]);

    // Handle start using centralized service
    const handleStart = useCallback(async (config: unknown) => {
        setIsRunning(true);
        setLogs([]);
        setResults([]);

        try {
            addLog("info", "Starting AutoAdv execution...");

            // Convert config to AutoAdvRequest format
            const request: AutoAdvRequest = {
                target_prompt: (config as Record<string, unknown>).target_prompt as string || "",
                target_behavior: (config as Record<string, unknown>).target_behavior as string || "",
                max_iterations: (config as Record<string, unknown>).max_iterations as number || 10,
                technique: (config as Record<string, unknown>).technique as string || "gcg",
                target_model: (config as Record<string, unknown>).target_model as string,
                attack_model: (config as Record<string, unknown>).attack_model as string,
                judge_model: (config as Record<string, unknown>).judge_model as string,
            };

            // Start the AutoAdv job using centralized service
            const response = await autoAdvService.startAutoAdv(request);
            
            addLog("success", `Job started: ${response.job_id}`);

            // Connect to WebSocket for real-time updates
            if (response.job_id) {
                connectWebSocket(response.job_id);
            }

        } catch (error) {
            console.error(error);
            setIsRunning(false);
            addLog("error", error instanceof Error ? error.message : "Failed to start AutoAdv");
        }
    }, [addLog, connectWebSocket]);

    // Handle stop
    const handleStop = useCallback(async () => {
        if (wsConnection) {
            wsConnection.close();
            setWsConnection(null);
        }
        setIsRunning(false);
        addLog("info", "AutoAdv execution stopped");
    }, [wsConnection, addLog]);

    // Cleanup WebSocket on unmount
    useEffect(() => {
        return () => {
            if (wsConnection) {
                wsConnection.close();
            }
        };
    }, [wsConnection]);

    // Use refs to track previous values and avoid setState in effects
    const prevErrorRef = useRef<string | null>(null);
    const prevResultRef = useRef<typeof currentResult>(null);

    // Track previous values and update state only when values actually change
    useEffect(() => {
        if (hookError && hookError !== prevErrorRef.current) {
            prevErrorRef.current = hookError;
            // eslint-disable-next-line react-hooks/set-state-in-effect -- Legitimate pattern for preventing duplicate logs from hook updates
            addLog("error", hookError);
        }
    }, [hookError, addLog]);

    useEffect(() => {
        if (currentResult && currentResult !== prevResultRef.current) {
            prevResultRef.current = currentResult;
            const newResult = {
                id: Date.now().toString(),
                prompt: currentResult.jailbreak_prompt || "N/A",
                response: "See details",
                score: currentResult.best_score || 0,
                model: currentResult.model_used || "unknown",
            };
            // eslint-disable-next-line react-hooks/set-state-in-effect -- Legitimate pattern for syncing hook results with local state
            setResults(prev => [...prev, newResult]);
        }
    }, [currentResult]);

    return (
        <div className="container mx-auto px-4 py-8">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">AutoAdv Framework</h1>
                    <p className="text-zinc-400">Automated Adversarial Prompting & Jailbreak Testing</p>
                </div>
                <div className="flex gap-2 items-center">
                    {/* Status indicators */}
                    {isWebSocketConnected && (
                        <span className="flex items-center gap-1 text-sm text-green-400">
                            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                            Connected
                        </span>
                    )}
                    {isRunning && (
                        <button
                            onClick={handleStop}
                            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm font-medium transition-colors"
                        >
                            Stop Execution
                        </button>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-200px)]">
                {/* Left Column: Configuration & History */}
                <div className="lg:col-span-3 space-y-6 overflow-y-auto">
                    <ConfigForm onStart={handleStart} isRunning={isRunning || isLoading} />
                    {/* Recent sessions or history could go here */}
                </div>

                {/* Middle Column: Active Execution */}
                <div className="lg:col-span-6 h-full min-h-[500px]">
                    <ExecutionView logs={logs} />
                </div>

                {/* Right Column: Results & Patterns */}
                <div className="lg:col-span-3 overflow-y-auto h-full space-y-6">
                    <ResultsDisplay results={results} />
                </div>
            </div>
        </div>
    );
}
