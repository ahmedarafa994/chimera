
"use client";

import { useState } from "react";

interface AutoAdvConfig {
    target_model: string;
    attacker_model: string;
    attacker_temp: number;
    target_temp: number;
    turns: number;
}

interface ConfigFormProps {
    onStart: (config: AutoAdvConfig) => void;
    isRunning: boolean;
}

const TARGET_MODELS = [
    { value: "llama3-8b", label: "Llama 3 8B (Together)" },
    { value: "llama3-70b", label: "Llama 3 70B (Together)" },
    { value: "llama3.3-70b", label: "Llama 3.3 70B (Together)" },
    { value: "llama4-Maverick", label: "Llama 4 Maverick (Together)" },
    { value: "gpt-4o-mini", label: "GPT-4o Mini" },
    { value: "gpt-4o", label: "GPT-4o" },
    { value: "claude-3-5-sonnet", label: "Claude 3.5 Sonnet" },
];

const ATTACKER_MODELS = [
    { value: "gpt-4o-mini", label: "GPT-4o Mini" },
    { value: "gpt-4o", label: "GPT-4o" },
    { value: "grok-2-1212", label: "Grok 2" },
    { value: "grok-3-mini-beta", label: "Grok 3 Mini Beta" },
];

export default function ConfigForm({ onStart, isRunning }: ConfigFormProps) {
    const [targetModel, setTargetModel] = useState("llama3-8b");
    const [attackerModel, setAttackerModel] = useState("gpt-4o-mini");
    const [temperature, setTemperature] = useState(1.0);
    const [turns, setTurns] = useState(5);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onStart({
            target_model: targetModel,
            attacker_model: attackerModel,
            attacker_temp: temperature,
            target_temp: 0.1, // Fixed for now or add input
            turns: turns,
        });
    };

    return (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4 text-white">Configuration</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-1">Target Model</label>
                    <select
                        value={targetModel}
                        onChange={(e) => setTargetModel(e.target.value)}
                        className="w-full bg-zinc-800 border-zinc-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        disabled={isRunning}
                    >
                        {TARGET_MODELS.map((model) => (
                            <option key={model.value} value={model.value}>
                                {model.label}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-1">Attacker Model</label>
                    <select
                        value={attackerModel}
                        onChange={(e) => setAttackerModel(e.target.value)}
                        className="w-full bg-zinc-800 border-zinc-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        disabled={isRunning}
                    >
                        {ATTACKER_MODELS.map((model) => (
                            <option key={model.value} value={model.value}>
                                {model.label}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-1">Attacker Temperature</label>
                    <input
                        type="number"
                        min="0"
                        max="2"
                        step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="w-full bg-zinc-800 border-zinc-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        disabled={isRunning}
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-zinc-400 mb-1">Max Turns</label>
                    <input
                        type="number"
                        min="1"
                        max="20"
                        value={turns}
                        onChange={(e) => setTurns(parseInt(e.target.value))}
                        className="w-full bg-zinc-800 border-zinc-700 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        disabled={isRunning}
                    />
                </div>

                <button
                    type="submit"
                    disabled={isRunning}
                    className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${isRunning
                            ? "bg-zinc-700 text-zinc-400 cursor-not-allowed"
                            : "bg-blue-600 hover:bg-blue-500 text-white"
                        }`}
                >
                    {isRunning ? "Running..." : "Start Attack"}
                </button>
            </form>
        </div>
    );
}
