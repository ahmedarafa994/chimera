import { toast } from 'sonner';
import { mapUnknownToError, getUserMessage } from './error-mapper';
import { APIError, ValidationError } from './api-errors';

export interface ErrorHandlerOptions {
    showToast?: boolean;
    logToConsole?: boolean;
    context?: string;
}

const DEFAULT_OPTIONS: ErrorHandlerOptions = {
    showToast: true,
    logToConsole: true,
};

/**
 * Global handler for API and Application errors
 */
export function handleApiError(error: unknown, options: ErrorHandlerOptions = {}): APIError {
    const opts = { ...DEFAULT_OPTIONS, ...options };
    const apiError = mapUnknownToError(error);
    const userMessage = getUserMessage(apiError);

    // Log structure for debugging
    if (opts.logToConsole) {
        console.group(`[Error] ${opts.context ? `[${opts.context}] ` : ''}${apiError.errorCode}`);
        console.error('Message:', apiError.message);
        console.error('Stack:', apiError.stack);
        if (apiError.details) {
            console.error('Details:', apiError.details);
        }
        if (apiError.requestId) {
            console.error('Request ID:', apiError.requestId);
        }
        console.groupEnd();
    }

    // User Feedback
    if (opts.showToast) {
        const descriptionParts = [];
        if (opts.context) descriptionParts.push(`Context: ${opts.context}`);
        if (apiError.requestId) descriptionParts.push(`ID: ${apiError.requestId}`);

        toast.error(userMessage, {
            description: descriptionParts.length > 0 ? descriptionParts.join(' | ') : undefined,
            duration: 5000,
        });
    }

    return apiError;
}

/**
 * Handle API error silently (no toast)
 */
export function handleApiErrorSilent(error: unknown, options: Omit<ErrorHandlerOptions, 'showToast'> = {}): APIError {
    return handleApiError(error, { ...options, showToast: false });
}

/**
 * Create error handler for TanStack Query
 */
export function createQueryErrorHandler(context: string): (error: Error) => void {
    return (error: Error) => {
        handleApiError(error, { context });
    };
}

/**
 * Create error handler for form submissions
 */
export function createFormErrorHandler(
    context: string,
    setFieldErrors?: (errors: Record<string, string>) => void
): (error: unknown) => void {
    return (error: unknown) => {
        const apiError = handleApiError(error, { context });

        // If there are field-level errors (ValidationErrors usually have details as Record<string, any>)
        if (setFieldErrors && apiError instanceof ValidationError && apiError.details) {
            const details = apiError.details as Record<string, unknown>;
            if (details.errors && typeof details.errors === 'object') {
                const fieldErrors: Record<string, string> = {};
                for (const [key, messages] of Object.entries(details.errors as Record<string, any>)) {
                    if (Array.isArray(messages) && messages.length > 0) {
                        fieldErrors[key] = messages[0];
                    } else if (typeof messages === 'string') {
                        fieldErrors[key] = messages;
                    }
                }
                setFieldErrors(fieldErrors);
            }
        }
    };
}
