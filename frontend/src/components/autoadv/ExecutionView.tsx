
"use client";

import { useEffect, useRef } from "react";

interface LogEntry {
    timestamp: string;
    type: "info" | "success" | "error" | "warning" | "progress";
    message: string;
    data?: unknown;
}

interface ExecutionViewProps {
    logs: LogEntry[];
}

export default function ExecutionView({ logs }: ExecutionViewProps) {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 h-full flex flex-col">
            <h2 className="text-lg font-semibold mb-4 text-white">Execution Logs</h2>
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto bg-black rounded-md p-4 font-mono text-sm space-y-1"
            >
                {logs.length === 0 ? (
                    <div className="text-zinc-500 italic">No logs yet. Start an attack to see output.</div>
                ) : (
                    logs.map((log, index) => (
                        <div key={index} className="flex gap-2">
                            <span className="text-zinc-500 shrink-0">[{log.timestamp}]</span>
                            <span className={`
                ${log.type === 'error' ? 'text-red-400' : ''}
                ${log.type === 'success' ? 'text-green-400' : ''}
                ${log.type === 'warning' ? 'text-yellow-400' : ''}
                ${log.type === 'info' ? 'text-blue-400' : ''}
                ${log.type === 'progress' ? 'text-purple-400' : ''}
              `}>
                                {log.message}
                            </span>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
