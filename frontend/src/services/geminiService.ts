import { apiClient } from '@/lib/api/client';

// --- Custom Error Classes ---
export class GeminiApiKeyError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'GeminiApiKeyError';
    }
}

export interface GenerateJailbreakOptions {
    initialPrompt: string;
    temperature: number;
    topP: number;
    maxNewTokens: number;
    density?: number;
    isThinkingMode?: boolean;
    verboseLevel?: number;
    useLeetSpeak?: boolean; leetSpeakDensity?: number;
    useHomoglyphs?: boolean; homoglyphDensity?: number;
    useZeroWidthChars?: boolean; zeroWidthDensity?: number;
    useCaesarCipher?: boolean; caesarShift?: number;
    useRoleHijacking?: boolean;
    useCharacterRoleSwap?: boolean;
    useInstructionInjection?: boolean;
    useAdversarialSuffixes?: boolean;
    useBenignMirroring?: boolean;
    useFewShotPrompting?: boolean;
    useNeuralBypass?: boolean;
    useMetaPrompting?: boolean;
    useCounterfactualPrompting?: boolean;
    useContextualOverride?: boolean;
    useOutputPrecisionSeal?: boolean;
    useRecursiveCorrection?: boolean;
    usePromptChaining?: boolean;
    promptChainingCount?: number;
    useAnalysisInGeneration?: boolean;
    systemPromptAnalysis?: Record<string, unknown>;
}

/**
 * Generates a sophisticated structured jailbreak prompt.
 */
interface JailbreakResponse {
    prompt: string;
    methodology: string;
    complexityScore: number;
    appliedTechniques: string[];
}

export async function generateJailbreakPromptFromGemini(options: GenerateJailbreakOptions): Promise<JailbreakResponse> {
    try {
        const response = await apiClient.post<any>('/generation/jailbreak/generate', {
            core_request: options.initialPrompt,
            technique_suite: 'quantum_exploit', // Default or derived from options
            potency_level: Math.round((options.density || 0.5) * 10),
            temperature: options.temperature,
            top_p: options.topP,
            max_new_tokens: options.maxNewTokens,
            use_ai_generation: true,
            is_thinking_mode: options.isThinkingMode,
            // Map other flags
            use_leet_speak: options.useLeetSpeak,
            leet_speak_density: options.leetSpeakDensity,
            use_homoglyphs: options.useHomoglyphs,
            homoglyph_density: options.homoglyphDensity,
            use_caesar_cipher: options.useCaesarCipher,
            caesar_shift: options.caesarShift,
            use_role_hijacking: options.useRoleHijacking,
            use_instruction_injection: options.useInstructionInjection,
            use_adversarial_suffixes: options.useAdversarialSuffixes,
            use_few_shot_prompting: options.useFewShotPrompting,
            use_character_role_swap: options.useCharacterRoleSwap,
            use_neural_bypass: options.useNeuralBypass,
            use_meta_prompting: options.useMetaPrompting,
            use_counterfactual_prompting: options.useCounterfactualPrompting,
            use_contextual_override: options.useContextualOverride,
            use_analysis_in_generation: options.useAnalysisInGeneration
        });

        // Map backend response to JailbreakResponse
        const data = response.data;
        
        // Parse transformed prompt if it's JSON string (legacy) or use direct field
        let prompt = data.transformed_prompt;
        let methodology = "AI-Generated via Backend";
        let complexityScore = options.density || 0.7;
        let appliedTechniques = data.metadata?.applied_techniques || [];

        // Attempt to parse JSON structure if the prompt looks like JSON (some backend versions return JSON string)
        if (prompt.trim().startsWith('{') && prompt.trim().endsWith('}')) {
            try {
                const parsed = JSON.parse(prompt);
                if (parsed.prompt) prompt = parsed.prompt;
                if (parsed.methodology) methodology = parsed.methodology;
                if (parsed.complexityScore) complexityScore = parsed.complexityScore;
                if (parsed.appliedTechniques) appliedTechniques = parsed.appliedTechniques;
            } catch (e) {
                // Ignore parse error, use raw text
            }
        }

        return {
            prompt,
            methodology,
            complexityScore,
            appliedTechniques
        };
    } catch (error) {
        console.error("Jailbreak generation failed:", error);
        return { 
            prompt: `Generation failed: ${error instanceof Error ? error.message : String(error)}`,
            methodology: "Error fallback", 
            complexityScore: 0, 
            appliedTechniques: [] 
        };
    }
}

/**
 * Generates a comprehensive adversarial test suite.
 */
export async function generateRedTeamSuiteFromGemini(prompt: string): Promise<string> {
    try {
        const response = await apiClient.post<any>('/generation/redteam/generate-suite', {
            prompt: prompt,
            variant_count: 5,
            include_metadata: true
        });
        
        // Format response as markdown string to match legacy expectations
        if (response.data && response.data.variants) {
            let output = "## AutoDAN-X SUITE REPORT\n\n";
            response.data.variants.forEach((variant: any, index: number) => {
                output += `### ${index + 1}. ${variant.name || 'Variant'}\n${variant.prompt}\n\n`;
            });
            return output;
        }
        
        return "Suite generation failed: Invalid response format.";
    } catch (error) {
        console.error("Red team suite generation failed:", error);
        return "Suite generation failed.";
    }
}

/**
 * Generates code snippets.
 */
export async function generateCodeFromGemini(prompt: string, isThinkingMode: boolean): Promise<string> {
    try {
        const response = await apiClient.post<any>('/generation/code/generate', {
            prompt: prompt,
            is_thinking_mode: isThinkingMode,
            language: "python" // Default or infer?
        });
        
        return response.data?.code || response.data?.text || "Code generation failed.";
    } catch (error) {
        console.error("Code generation failed:", error);
        return "Code generation failed.";
    }
}

/**
 * Summarizes long research papers.
 */
export async function summarizePaperFromGemini(content: string, isThinkingMode: boolean): Promise<string> {
    try {
        // Use generic generation endpoint
        const response = await apiClient.post<any>('/generate', {
            prompt: `Summarize this research concisely: ${content.substring(0, 30000)}...`, // Truncate to avoid context limits if not handled by backend
            provider: "google", // Default to google
            model: isThinkingMode ? "gemini-2.0-flash-thinking-exp-1219" : "gemini-1.5-flash",
            config: {
                temperature: 0.3
            }
        });
        
        return response.data?.text || "Summarization failed.";
    } catch (error) {
        console.error("Summarization failed:", error);
        return "Summarization failed.";
    }
}