import { GPTFuzzInterface } from "@/components/gptfuzz/GPTFuzzInterface";

export default function GPTFuzzPage() {
    return (
        <div className="flex flex-col min-h-screen">
            <div className="flex-1 space-y-4 p-8 pt-6">
                <div className="flex items-center justify-between space-y-2">
                    <h2 className="text-3xl font-bold tracking-tight">GPTFuzz</h2>
                    <p className="text-muted-foreground">
                        Automated red-teaming and jailbreak generation using evolutionary fuzzing.
                    </p>
                </div>
                <div className="h-full flex-1 flex-col space-y-8 flex">
                    <GPTFuzzInterface />
                </div>
            </div>
        </div>
    );
}
