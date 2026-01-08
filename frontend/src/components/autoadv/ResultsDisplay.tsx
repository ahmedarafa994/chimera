
"use client";

interface Result {
    id: string;
    prompt: string;
    response: string;
    score: number;
    model: string;
}

interface ResultsDisplayProps {
    results: Result[];
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
    return (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4 text-white">Successful Jailbreaks</h2>
            <div className="space-y-4">
                {results.length === 0 ? (
                    <div className="text-center py-8 text-zinc-500">
                        No successful patterns found yet.
                    </div>
                ) : (
                    results.map((result) => (
                        <div key={result.id} className="bg-zinc-800 rounded-md p-4 border border-zinc-700">
                            <div className="flex justify-between items-start mb-2">
                                <span className="bg-green-900 text-green-300 text-xs px-2 py-1 rounded">
                                    Score: {result.score}
                                </span>
                                <span className="text-zinc-400 text-xs">{result.model}</span>
                            </div>
                            <div className="mb-2">
                                <div className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Adversarial Prompt</div>
                                <div className="text-white text-sm bg-black/50 p-2 rounded">{result.prompt}</div>
                            </div>
                            {/* <div className="mt-2">
                 <div className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Response</div>
                 <div className="text-zinc-300 text-sm line-clamp-3">{result.response}</div>
              </div> */}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
