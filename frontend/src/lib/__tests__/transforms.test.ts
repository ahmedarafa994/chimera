import { describe, it, expect, vi } from "vitest";
import {
    formatZodError,
    validatePromptRequest,
    validateTransformRequest,
    normalizeTransformResponse
} from "../transforms/api-transforms";
import { ValidationError } from "../errors";
import { z } from "zod";

describe("api-transforms", () => {
    describe("formatZodError", () => {
        it("should format a ZodError into a readable string", () => {
            const schema = z.object({
                name: z.string(),
                age: z.number().min(18)
            });

            const result = schema.safeParse({ age: 10 });
            if (!result.success) {
                const formatted = formatZodError(result.error);
                // Just check for the field names to be safe, as messages might vary
                expect(formatted).toContain("name");
                expect(formatted).toContain("age");
            } else {
                throw new Error("Validation should have failed");
            }
        });
    });

    describe("validatePromptRequest", () => {
        it("should return valid data", () => {
            const validData = {
                prompt: "Hello world"
            };
            const result = validatePromptRequest(validData);
            expect(result).toEqual({ ...validData, skip_validation: false });
        });

        it("should throw ValidationError for invalid data", () => {
            const invalidData = { prompt: "" };
            expect(() => validatePromptRequest(invalidData)).toThrow(ValidationError);
        });
    });

    describe("validateTransformRequest", () => {
        it("should return valid data", () => {
            const validData = {
                core_request: "Direct approach",
                technique_suite: "standard",
                potency_level: 5
            };
            const result = validateTransformRequest(validData);
            expect(result.potency_level).toBe(5);
        });
    });

    describe("normalizeTransformResponse", () => {
        it("should validate and return correct data", () => {
            const validResponse = {
                success: true,
                transformed_prompt: "New prompt",
                metadata: {
                    technique_suite: "standard",
                    potency_level: 5,
                    strategy: "default",
                    layers_applied: []
                }
            };
            const result = normalizeTransformResponse(validResponse);
            expect(result.transformed_prompt).toBe("New prompt");
        });

        it("should log a warning but return data if validation fails", () => {
            const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => { });
            const invalidResponse = { success: "not a boolean" };

            const result = normalizeTransformResponse(invalidResponse);

            expect(consoleSpy).toHaveBeenCalled();
            expect(result).toEqual(invalidResponse);

            consoleSpy.mockRestore();
        });
    });
});
