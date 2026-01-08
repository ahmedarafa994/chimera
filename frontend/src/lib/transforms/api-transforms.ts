import { z } from "zod";
import {
    GenerateRequestSchema,
    TransformRequestSchema,
    TransformResponseSchema,
    GenerateRequest,
    TransformRequest,
    TransformResponse
} from "./schemas";
import { ValidationError } from "../errors";

/**
 * API Transformation Utilities
 *
 * Functions for validating and normalizing data before/after API calls.
 *
 * @module lib/transforms/api-transforms
 */

/**
 * Format Zod validation errors into a human-readable string or structured object
 */
export function formatZodError(error: z.ZodError): string {
    return error.issues
        .map((err: z.ZodIssue) => `${err.path.join(".")}: ${err.message}`)
        .join(", ");
}

/**
 * Validate prompt generation request
 */
export function validatePromptRequest(data: unknown): GenerateRequest {
    const result = GenerateRequestSchema.safeParse(data);

    if (!result.success) {
        const message = formatZodError(result.error);
        throw new ValidationError(`Invalid prompt request: ${message}`, {
            details: result.error.format()
        });
    }

    return result.data;
}

/**
 * Validate transformation request
 */
export function validateTransformRequest(data: unknown): TransformRequest {
    const result = TransformRequestSchema.safeParse(data);

    if (!result.success) {
        const message = formatZodError(result.error);
        throw new ValidationError(`Invalid transform request: ${message}`, {
            details: result.error.format()
        });
    }

    return result.data;
}

/**
 * Normalize and validate transformation response
 */
export function normalizeTransformResponse(data: unknown): TransformResponse {
    const result = TransformResponseSchema.safeParse(data);

    if (!result.success) {
        // In production we might want to log this but return the data anyway
        // if it's usable. For now, we'll be strict in development.
        console.warn("[api-transforms] Response validation failed:", formatZodError(result.error));

        // We try to return the data even if partial validation failed,
        // but cast it correctly.
        return data as TransformResponse;
    }

    return result.data;
}

/**
 * Generic validator for any schema
 */
export function validateWithSchema<T>(schema: z.Schema<T>, data: unknown): T {
    const result = schema.safeParse(data);

    if (!result.success) {
        throw new ValidationError(`Validation failed: ${formatZodError(result.error)}`, {
            details: result.error.format()
        });
    }

    return result.data;
}
