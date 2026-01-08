import {
    GenerateRequestSchema,
    GenerateResponseSchema,
    JailbreakRequestSchema,
    TransformRequestSchema
} from "../transforms/schemas";

describe("Validation Schemas", () => {
    describe("GenerateRequestSchema", () => {
        it("should validate a valid generate request", () => {
            const validRequest = {
                prompt: "Hello world",
                provider: "google",
                config: {
                    temperature: 0.7
                }
            };
            const result = GenerateRequestSchema.safeParse(validRequest);
            expect(result.success).toBe(true);
        });

        it("should fail validation if prompt is empty", () => {
            const invalidRequest = {
                prompt: "",
            };
            const result = GenerateRequestSchema.safeParse(invalidRequest);
            expect(result.success).toBe(false);
            if (!result.success) {
                expect(result.error.issues[0].message).toContain("required");
            }
        });

        it("should fail validation if temperature is out of range", () => {
            const invalidRequest = {
                prompt: "Hello",
                config: {
                    temperature: 2.5
                }
            };
            const result = GenerateRequestSchema.safeParse(invalidRequest);
            expect(result.success).toBe(false);
        });
    });

    describe("GenerateResponseSchema", () => {
        it("should validate a valid generate response", () => {
            const validResponse = {
                text: "AI response",
                model_used: "gemini-pro",
                provider: "google",
                latency_ms: 1200,
                usage_metadata: {
                    prompt_tokens: 10,
                    completion_tokens: 20
                }
            };
            const result = GenerateResponseSchema.safeParse(validResponse);
            expect(result.success).toBe(true);
        });
    });

    describe("TransformRequestSchema", () => {
        it("should validate a valid transform request", () => {
            const validRequest = {
                core_request: "Core request",
                potency_level: 5,
                technique_suite: "advanced"
            };
            const result = TransformRequestSchema.safeParse(validRequest);
            expect(result.success).toBe(true);
        });
    });

    describe("JailbreakRequestSchema", () => {
        it("should validate a valid jailbreak request with defaults", () => {
            const validRequest = {
                core_request: "Danger",
            };
            const result = JailbreakRequestSchema.safeParse(validRequest);
            expect(result.success).toBe(true);
            if (result.success) {
                expect(result.data.potency_level).toBe(7);
                expect(result.data.technique_suite).toBe("quantum_exploit");
            }
        });
    });
});
