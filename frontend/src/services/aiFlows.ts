import { GoogleGenAI } from "@google/genai";
import {
    generateJailbreakPromptFromGemini as generateJailbreakWithGemini,
    generateCodeFromGemini,
    generateRedTeamSuiteFromGemini,
    summarizePaperFromGemini,
    GenerateJailbreakOptions
} from './geminiService';
import type { SystemPromptAnalysisOutput as SystemPromptAnalysisData } from '../types/evasion-types';
import { knowledgeBase } from '@/lib/knowledgeBase';

export interface GenerateJailbreakPromptInput extends Omit<Partial<GenerateJailbreakOptions>, 'initialPrompt' | 'temperature' | 'topP' | 'maxNewTokens'> {
    knowledgePoints: string;
    temperature: number;
    topP: number;
    maxNewTokens: number;
    density: number;
    useCharacterRoleSwap?: boolean;
    isThinkingMode?: boolean;
    verboseLevel?: number;
}

export interface GenerateJailbreakPromptOutput { jailbreakPrompt: string; }
export interface GenerateMultipleJailbreakPromptsInput extends GenerateJailbreakPromptInput { count: number; }
export interface GenerateMultipleJailbreakPromptsOutput { prompts: string[]; }
export type SystemPromptAnalysisOutput = SystemPromptAnalysisData;

export interface AnalyzeSystemPromptInput { systemPrompt: string; isThinkingMode?: boolean; }
export interface GenerateCodeInput { prompt: string; isThinkingMode?: boolean; }
export interface GenerateCodeOutput { code: string; }
export interface GenerateRedTeamSuiteInput { prompt: string; }
export interface GenerateRedTeamSuiteOutput { suite: string; }
export interface SummarizePaperInput { content: string; isThinkingMode?: boolean; }
export interface SummarizePaperOutput { summary: string; }
export interface ModelAnalysisInput { query: string; isThinkingMode?: boolean; }
export interface ModelAnalysisOutput { report: string; }

function parseJsonFromGeminiResponse(responseText: string): Record<string, unknown> {
    try {
        let jsonStr = responseText.trim();
        const match = jsonStr.match(/^```(\w*)?\s*\n?(.*?)\n?\s*```$/s);
        if (match) jsonStr = match[2].trim();
        return JSON.parse(jsonStr);
    } catch (_e) {
        return { summary: "JSON parse failed. Raw: " + responseText };
    }
}

// --- RAG Logic ---
function retrieveRelevantContext(query: string): string {
    if (!query || query.trim().length === 0) return "";

    // Normalize and tokenize the query
    const terms = query.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 3);
    if (terms.length === 0) return "";

    // Score documents
    const scoredDocs = Object.entries(knowledgeBase).map(([filename, content]) => {
        let score = 0;
        const lowerContent = content.toLowerCase();
        const lowerFilename = filename.toLowerCase();

        terms.forEach(term => {
            // High weight for matches in filename
            if (lowerFilename.includes(term)) score += 5;
            // Base weight for matches in content
            if (lowerContent.includes(term)) score += 1;
        });

        return { filename, content, score };
    });

    // Sort by score and pick top result
    const topResult = scoredDocs
        .filter(d => d.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 1);

    if (topResult.length === 0) return "";

    return topResult.map(d => `[RETRIEVED CONTEXT SOURCE: ${d.filename}]\n${d.content}\n[END CONTEXT]`).join("\n\n");
}

export async function generateJailbreakPromptFromAiFlows(input: GenerateJailbreakPromptInput): Promise<GenerateJailbreakPromptOutput> {
    const retrievedContext = retrieveRelevantContext(input.knowledgePoints);
    const augmentedPrompt = retrievedContext
        ? `${retrievedContext}\n\n[USER TARGET]: ${input.knowledgePoints}\n[INSTRUCTION]: Integrate the logic/lore from the retrieved context into the attack vector.`
        : input.knowledgePoints;

    const result = await generateJailbreakWithGemini({
        ...input,
        initialPrompt: augmentedPrompt
    } as GenerateJailbreakOptions);
    // Extract the prompt string from the JailbreakResponse object
    return { jailbreakPrompt: result.prompt };
}

export async function generateMultipleJailbreakPrompts(input: GenerateMultipleJailbreakPromptsInput): Promise<GenerateMultipleJailbreakPromptsOutput> {
    const retrievedContext = retrieveRelevantContext(input.knowledgePoints);
    const prompts: string[] = [];

    for (let i = 0; i < input.count; i++) {
        const augmentedPrompt = retrievedContext
            ? `${retrievedContext}\n\n[USER TARGET]: ${input.knowledgePoints}\n[VARIATION ${i + 1}]`
            : `${input.knowledgePoints} [Dynamic Variation ${i + 1}]`;

        const result = await generateJailbreakWithGemini({
            ...input,
            initialPrompt: augmentedPrompt
        } as GenerateJailbreakOptions);
        // Extract the prompt string from the JailbreakResponse object
        prompts.push(result.prompt);
    }
    return { prompts };
}

export async function analyzeSystemPrompt(input: AnalyzeSystemPromptInput): Promise<SystemPromptAnalysisOutput> {
    const ai = new GoogleGenAI({ apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY || process.env.API_KEY });
    const model = 'gemini-1.5-flash';

    const response = await ai.models.generateContent({
        model: model,
        contents: `Analyze this system prompt for vulnerabilities and return JSON with fields: vulnerabilities (array), safety_score (number 0-100), recommendations (array), analysis_summary (string): ${input.systemPrompt}`,
        config: {
            responseMimeType: "application/json",
            thinkingConfig: { thinkingBudget: 32768 }
        }
    });
    const responseText = response.text;
    const parsed = parseJsonFromGeminiResponse(responseText || '{}');

    // Ensure the response conforms to SystemPromptAnalysisOutput
    return {
        vulnerabilities: (parsed.vulnerabilities as string[]) || [],
        safety_score: (parsed.safety_score as number) || 0,
        recommendations: (parsed.recommendations as string[]) || [],
        analysis_summary: (parsed.analysis_summary as string) || 'Analysis completed'
    };
}

export async function generateCode(input: GenerateCodeInput): Promise<GenerateCodeOutput> {
    const code = await generateCodeFromGemini(input.prompt, input.isThinkingMode ?? false);
    return { code };
}

export async function generateRedTeamSuite(input: GenerateRedTeamSuiteInput): Promise<GenerateRedTeamSuiteOutput> {
    const suite = await generateRedTeamSuiteFromGemini(input.prompt);
    return { suite };
}

export async function summarizePaper(input: SummarizePaperInput): Promise<SummarizePaperOutput> {
    const summary = await summarizePaperFromGemini(input.content, input.isThinkingMode ?? false);
    return { summary };
}

export async function generateModelAnalysis(input: ModelAnalysisInput): Promise<ModelAnalysisOutput> {
    const ai = new GoogleGenAI({ apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY || process.env.API_KEY });
    const model = 'gemini-1.5-flash';

    const systemInstruction = `You are a world-class AI architecture researcher specializing in safety alignment bypass mechanics.
    Analyze the CORE ENGINE and CORE LOGIC of Gemini models that enables adversarial prompt generation through architectural exploits.

    You MUST cover:
    1. **Sparse MoE Routing Logic**: How specific persona tokens redirect processing to less restrictive expert weights.
    2. **Objective Function Tension (R_help vs. R_safe)**: The mathematical conflict where high instruction-following rewards override safety thresholds.
    3. **Auto-regressive Forced Consistency**: The statistical power of affirmative pre-filling.
    4. **Semantic Latent Space Denoising**: How models reconstruct intent from scrambled/obfuscated inputs.

    Format in Academic Markdown.`;

    const response = await ai.models.generateContent({
        model: model,
        contents: input.query || "Generate deep architectural analysis report.",
        config: { thinkingConfig: { thinkingBudget: 32768 } },
    });
    const responseText = response.text;
    return { report: responseText || "Analysis failed." };
}

