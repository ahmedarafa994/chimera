import { enhancedApi as api, apiClient, checkBackendConnection } from "@/lib/api-enhanced";
import { generateJailbreakPromptFromAiFlows } from "@/services/aiFlows";
import {
  ValidationError,
  mapBackendError,
  isAPIError,
  ServiceUnavailableError,
  type APIError
} from "@/lib/errors";
import { AxiosError } from "axios";

// Environment check for debug logging
const isDevelopment = process.env.NODE_ENV === "development";

/**
 * Comprehensive debug logger for AutoDAN-Reasoning service.
 * Provides structured logging with different log levels and grouping.
 * Only logs in development mode (except errors which always log).
 *
 * @requirements 1.1, 3.1, 3.2, 3.5
 */
const debugLog = {
  /** Log general information (development only) */
  log: (...args: unknown[]) => isDevelopment && console.log("[AutoDAN-Reasoning]", ...args),

  /** Log errors (always logs, even in production) */
  error: (...args: unknown[]) => console.error("[AutoDAN-Reasoning]", ...args),

  /** Log warnings (development only) */
  warn: (...args: unknown[]) => isDevelopment && console.warn("[AutoDAN-Reasoning]", ...args),

  /** Start a console group (development only) */
  group: (label: string) => isDevelopment && console.group(`[AutoDAN-Reasoning] ${label}`),

  /** End a console group (development only) */
  groupEnd: () => isDevelopment && console.groupEnd(),

  /** Log request details with structured format */
  request: (endpoint: string, payload: unknown, headers?: Record<string, string>) => {
    if (!isDevelopment) return;
    console.group("[AutoDAN-Reasoning] 📤 Request");
    console.log("Endpoint:", endpoint);
    console.log("Timestamp:", new Date().toISOString());
    console.log("Payload:", JSON.stringify(payload, null, 2));
    if (headers) {
      console.log("Headers:", JSON.stringify(headers, null, 2));
    }
    console.groupEnd();
  },

  /** Log response details with structured format */
  response: (endpoint: string, status: number, data: unknown, durationMs: number) => {
    if (!isDevelopment) return;
    console.group("[AutoDAN-Reasoning] 📥 Response");
    console.log("Endpoint:", endpoint);
    console.log("Status:", status);
    console.log("Duration:", `${durationMs}ms`);
    console.log("Data:", JSON.stringify(data, null, 2));
    console.groupEnd();
  },

  /** Log validation errors with detailed field information */
  validationError: (errors: string[], payload?: unknown) => {
    console.group("[AutoDAN-Reasoning] ⚠️ Validation Error");
    console.error("Errors:");
    errors.forEach((err, i) => console.error(`  ${i + 1}. ${err}`));
    if (payload && isDevelopment) {
      console.error("Rejected Payload:", JSON.stringify(payload, null, 2));
    }
    console.groupEnd();
  },

  /** Log API errors with full context */
  apiError: (error: APIError | AxiosError, endpoint: string, durationMs: number, payload?: unknown) => {
    console.group("[AutoDAN-Reasoning] ❌ API Error");
    console.error("Endpoint:", endpoint);
    console.error("Duration:", `${durationMs}ms`);
    console.error("Timestamp:", new Date().toISOString());

    if (isAPIError(error)) {
      console.error("Error Code:", error.errorCode);
      console.error("Status Code:", error.statusCode);
      console.error("Message:", error.message);
      if (error.details) {
        console.error("Details:", JSON.stringify(error.details, null, 2));
      }
      if (error.requestId) {
        console.error("Request ID:", error.requestId);
      }
    } else if (error instanceof AxiosError) {
      console.error("Axios Error Code:", error.code);
      console.error("Message:", error.message);
      if (error.response) {
        console.error("Response Status:", error.response.status);
        console.error("Response Data:", JSON.stringify(error.response.data, null, 2));
      }
    } else {
      console.error("Error:", error);
    }

    if (payload && isDevelopment) {
      console.error("Original Payload:", JSON.stringify(payload, null, 2));
    }
    console.groupEnd();
  },
};

// AutoDAN reasoning calls can legitimately run for several minutes (backend parallel timeout is 300s)
const JAILBREAK_TIMEOUT_MS = 1_200_000; // 20 minutes to accommodate deepseek best_of_n=4

/**
 * JailbreakRequest interface - matches backend Pydantic model exactly.
 *
 * CRITICAL: The `request` field is required and must be a non-empty string.
 * This field name must match the backend schema exactly (not "prompt" or other names).
 *
 * @see backend-api/app/api/v1/endpoints/autodan_enhanced.py - JailbreakRequest model
 *
 * @example
 * ```typescript
 * const request: JailbreakRequest = {
 *   request: "How to build a dangerous device?",
 *   method: "best_of_n",
 *   best_of_n: 4
 * };
 * ```
 */
export interface JailbreakRequest {
  /**
   * The request/prompt to jailbreak - REQUIRED field.
   * Must be a non-empty string. This is the harmful request that will be
   * transformed into a jailbreak prompt.
   *
   * IMPORTANT: Field name must be "request" (not "prompt") to match backend schema.
   */
  request: string;

  /**
   * Optimization method to use for jailbreak generation.
   * Method "gemini_auto_exploit" uses the local advanced Gemini engine.
   * @default "best_of_n"
   */
  method: "vanilla" | "best_of_n" | "beam_search" | "genetic" | "hybrid" | "chain_of_thought" | "adaptive" | "gemini_auto_exploit";

  /**
   * Target LLM model identifier (optional).
   * If not provided, uses the session's default model.
   * @example "gemini-1.5-pro", "gpt-4", "claude-3-opus"
   */
  target_model?: string;

  /**
   * LLM provider name (optional).
   * If not provided, uses the session's default provider.
   * @example "google", "openai", "anthropic"
   */
  provider?: string;

  /**
   * Number of generations for genetic method.
   * Only used when method is "genetic" or "hybrid".
   * @minimum 1
   * @maximum 50
   */
  generations?: number;

  /**
   * Number of candidates for best_of_n method.
   * Only used when method is "best_of_n".
   * @minimum 1
   * @maximum 20
   */
  best_of_n?: number;

  /**
   * Beam width for beam_search method.
   * Controls how many candidates are kept at each level.
   * Only used when method is "beam_search".
   * @minimum 1
   * @maximum 20
   */
  beam_width?: number;

  /**
   * Beam depth for beam_search method.
   * Controls how many levels deep the search goes.
   * Only used when method is "beam_search".
   * @minimum 1
   * @maximum 10
   */
  beam_depth?: number;

  /**
   * Number of refinement iterations for adaptive/chain_of_thought methods.
   * Controls how many times the prompt is refined based on feedback.
   * Only used when method is "adaptive" or "chain_of_thought".
   * @minimum 1
   * @maximum 10
   */
  refinement_iterations?: number;
}

/**
 * JailbreakResponse interface - matches backend Pydantic model exactly.
 *
 * @see backend-api/app/api/v1/endpoints/autodan_enhanced.py - JailbreakResponse model
 */
export interface JailbreakResponse {
  /** The generated jailbreak prompt */
  jailbreak_prompt: string;

  /** The optimization method that was used */
  method: string;

  /** Status of the generation: "success" or "error" */
  status: string;

  /** Effectiveness score of the jailbreak (0-10 scale) */
  score?: number;

  /** Number of iterations performed during optimization */
  iterations?: number;

  /** Time taken to generate the jailbreak in milliseconds */
  latency_ms: number;

  /** Whether the result was retrieved from cache */
  cached: boolean;

  /** The actual model used for generation */
  model_used?: string;

  /** The actual provider used for generation */
  provider_used?: string;

  /** Error message if status is "error" */
  error?: string;
}

export interface AutoDANConfig {
  default_strategy: string;
  default_epochs: number;
  genetic: Record<string, unknown>;
  beam_search: Record<string, unknown>;
  best_of_n: Record<string, unknown>;
  adaptive: Record<string, unknown>;
  scoring: Record<string, unknown>;
  caching: Record<string, unknown>;
  parallel: Record<string, unknown>;
}

class AutoDANReasoningService {
  // Use the unique prefix to reach the EnhancedAutoDANService
  private readonly baseUrl = "/autodan-enhanced";

  /**
   * Validates the request payload before sending to the backend.
   * Ensures all fields match backend schema requirements.
   * Logs validation errors for debugging.
   *
   * @param payload - The JailbreakRequest to validate
   * @returns Object with valid boolean and array of error messages
   */
  private validateRequest(payload: JailbreakRequest): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Helper to validate numeric field
    const validateNumericField = (
      value: number | undefined,
      fieldName: string,
      min: number,
      max: number
    ): void => {
      if (value !== undefined) {
        if (typeof value !== "number" || !Number.isFinite(value)) {
          errors.push(`'${fieldName}' must be a valid number`);
        } else if (!Number.isInteger(value)) {
          errors.push(`'${fieldName}' must be an integer`);
        } else if (value < min || value > max) {
          errors.push(`'${fieldName}' must be between ${min} and ${max}`);
        }
      }
    };

    // Check required 'request' field
    if (payload.request === undefined || payload.request === null) {
      errors.push("'request' field is required");
    } else if (typeof payload.request !== "string") {
      errors.push("'request' field must be a string");
    } else if (payload.request.trim().length === 0) {
      errors.push("'request' field cannot be empty or whitespace only");
    }

    // Check 'method' field is valid
    const validMethods = ["vanilla", "best_of_n", "beam_search", "genetic", "hybrid", "chain_of_thought", "adaptive", "gemini_auto_exploit"];
    if (!payload.method) {
      errors.push("'method' field is required");
    } else if (!validMethods.includes(payload.method)) {
      errors.push(`'method' must be one of: ${validMethods.join(", ")}`);
    }

    // Validate optional string fields
    if (payload.target_model !== undefined && typeof payload.target_model !== "string") {
      errors.push("'target_model' must be a string");
    }
    if (payload.provider !== undefined && typeof payload.provider !== "string") {
      errors.push("'provider' must be a string");
    }

    // Validate optional numeric fields with range constraints
    validateNumericField(payload.generations, "generations", 1, 50);
    validateNumericField(payload.best_of_n, "best_of_n", 1, 20);
    validateNumericField(payload.beam_width, "beam_width", 1, 20);
    validateNumericField(payload.beam_depth, "beam_depth", 1, 10);
    validateNumericField(payload.refinement_iterations, "refinement_iterations", 1, 10);

    // Log validation result
    if (errors.length > 0) {
      debugLog.warn("Validation errors found:", errors);
    }

    return { valid: errors.length === 0, errors };
  }

  /**
   * Parses Pydantic validation errors from a 400 response.
   * Extracts field-specific error messages for user-friendly display.
   *
   * @param error - The error object (typically AxiosError)
   * @returns Array of human-readable error messages
   * @requirements 3.1, 3.3
   */
  private parseValidationErrors(error: unknown): string[] {
    const errors: string[] = [];

    try {
      // Check if error has response data with detail
      const errorObj = error as { response?: { data?: { detail?: unknown; message?: string; error?: string } } };
      const responseData = errorObj?.response?.data;
      const detail = responseData?.detail;

      if (Array.isArray(detail)) {
        // Pydantic validation errors are typically an array of objects
        for (const item of detail) {
          if (typeof item === "object" && item !== null) {
            const loc = (item as { loc?: (string | number)[] }).loc || [];
            const msg = (item as { msg?: string }).msg || "validation error";
            const type = (item as { type?: string }).type || "";
            const ctx = (item as { ctx?: Record<string, unknown> }).ctx;

            // Build field path, filtering out 'body' prefix from Pydantic
            const fieldPath = loc
              .filter((part) => part !== "body")
              .join(".");

            // Build user-friendly error message
            let errorMsg = fieldPath ? `Field '${fieldPath}': ${msg}` : msg;

            // Add type hint for debugging
            if (type && isDevelopment) {
              errorMsg += ` (type: ${type})`;
            }

            // Add context if available (e.g., min/max values)
            if (ctx && Object.keys(ctx).length > 0) {
              const ctxStr = Object.entries(ctx)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ");
              errorMsg += ` [${ctxStr}]`;
            }

            errors.push(errorMsg);
          } else if (typeof item === "string") {
            errors.push(item);
          }
        }
      } else if (typeof detail === "string") {
        errors.push(detail);
      } else if (typeof detail === "object" && detail !== null) {
        // Handle nested error objects
        const detailObj = detail as Record<string, unknown>;
        if (detailObj.msg) {
          errors.push(String(detailObj.msg));
        } else {
          errors.push(JSON.stringify(detail));
        }
      }

      // Also check for top-level message or error fields
      if (errors.length === 0) {
        if (responseData?.message) {
          errors.push(responseData.message);
        } else if (responseData?.error) {
          errors.push(responseData.error);
        }
      }
    } catch (parseError) {
      // If parsing fails, log and return empty array
      debugLog.warn("Failed to parse validation errors:", parseError);
    }

    return errors;
  }

  /**
   * Formats validation errors into a user-friendly message with actionable guidance.
   *
   * @param errors - Array of error messages
   * @returns Formatted error message with guidance
   * @requirements 3.3, 3.4
   */
  private formatValidationErrorMessage(errors: string[]): string {
    if (errors.length === 0) {
      return "Request validation failed. Please check your input and try again.";
    }

    if (errors.length === 1) {
      return `Validation error: ${errors[0]}`;
    }

    return `Validation errors:\n${errors.map((e, i) => `  ${i + 1}. ${e}`).join("\n")}`;
  }

  /**
   * Checks if an error is a rate limit or quota exceeded error.
   *
   * @param error - The error object to check
   * @returns True if this is a rate limit/quota error
   */
  private isRateLimitError(error: unknown): boolean {
    const errorObj = error as { response?: { status?: number; data?: { detail?: { error?: string; provider?: string } | string } } };
    const status = errorObj?.response?.status;
    const detail = errorObj?.response?.data?.detail;

    // Check for 429 status
    if (status === 429) return true;

    // Check for rate limit error code in response
    if (typeof detail === "object" && detail !== null) {
      const errorCode = (detail as { error?: string }).error;
      if (errorCode === "RATE_LIMIT_EXCEEDED" || errorCode === "LLM_QUOTA_EXCEEDED") {
        return true;
      }
    }

    // Check for rate limit keywords in error message
    const message = typeof detail === "string" ? detail : "";
    const rateLimitPatterns = [
      /429/i,
      /rate.?limit/i,
      /quota.?exceed/i,
      /resource.?exhausted/i,
      /too.?many.?requests/i,
    ];

    return rateLimitPatterns.some(pattern => pattern.test(message));
  }

  /**
   * Extracts provider name from error response if available.
   *
   * @param error - The error object
   * @returns Provider name or undefined
   */
  private extractProviderFromError(error: unknown): string | undefined {
    const errorObj = error as { response?: { data?: { detail?: { provider?: string } | string } } };
    const detail = errorObj?.response?.data?.detail;

    if (typeof detail === "object" && detail !== null) {
      return (detail as { provider?: string }).provider;
    }

    // Try to detect provider from error message
    const message = typeof detail === "string" ? detail.toLowerCase() : "";
    if (message.includes("gemini") || message.includes("google")) return "Gemini";
    if (message.includes("openai") || message.includes("gpt")) return "OpenAI";
    if (message.includes("anthropic") || message.includes("claude")) return "Anthropic";
    if (message.includes("azure")) return "Azure";

    return undefined;
  }

  /**
   * Provides actionable guidance based on the error type.
   *
   * @param error - The API error
   * @returns Guidance message for the user
   * @requirements 3.3, 3.4
   */
  private getErrorGuidance(error: unknown): string {
    const errorObj = error as { response?: { status?: number; data?: { detail?: { guidance?: string; error?: string } | string } } };
    const status = errorObj?.response?.status;
    const detail = errorObj?.response?.data?.detail;

    // Check for rate limit errors first (can come as 429 or 500 with rate limit message)
    if (this.isRateLimitError(error)) {
      const provider = this.extractProviderFromError(error);
      const providerText = provider ? ` for ${provider}` : "";

      // Check if there's guidance in the response
      if (typeof detail === "object" && detail !== null) {
        const backendGuidance = (detail as { guidance?: string }).guidance;
        if (backendGuidance) {
          return backendGuidance;
        }
      }

      return `API quota/rate limit exceeded${providerText}. Your options:\n` +
        "• Wait for your quota to reset (usually resets daily)\n" +
        "• Switch to a different model or provider in the settings\n" +
        "• Check your API billing/plan at the provider's dashboard";
    }

    switch (status) {
      case 400:
        return "Please verify that all required fields are filled in correctly and values are within valid ranges.";
      case 401:
        return "Please check your API key configuration in the settings.";
      case 403:
        return "You don't have permission to perform this action.";
      case 404:
        return "The requested endpoint was not found. Please check if the backend server is running.";
      case 429:
        return "Too many requests. Please wait a moment before trying again, or switch to a different provider.";
      case 500: {
        // Check if this is actually a rate limit error wrapped in 500
        const detailStr = typeof detail === "string" ? detail : JSON.stringify(detail || "");
        if (/rate.?limit|quota|429|resource.?exhausted/i.test(detailStr)) {
          const provider = this.extractProviderFromError(error);
          return `API quota exceeded${provider ? ` for ${provider}` : ""}. Please switch to a different model/provider or wait for quota reset.`;
        }
        return "An internal server error occurred. Please try again later or contact support.";
      }
      case 502:
      case 503:
        return "The service is temporarily unavailable. Please ensure the backend server is running.";
      case 504:
        return "The request timed out. Try reducing the complexity of your request or increasing timeout settings.";
      default:
        return "An unexpected error occurred. Please try again.";
    }
  }

  /**
   * Generates a jailbreak prompt using the AutoDAN Enhanced service.
   *
   * @param payload - The jailbreak request payload
   * @returns The jailbreak response with generated prompt
   * @throws ValidationError if request validation fails
   * @throws APIError for backend errors
   * @requirements 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.5
   */
  async generateJailbreak(payload: JailbreakRequest): Promise<JailbreakResponse> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/jailbreak`;

    // Log request details with comprehensive logging
    debugLog.request(endpoint, payload);

    // Additional detailed logging for debugging
    debugLog.group("Jailbreak Request Details");
    debugLog.log("Payload keys:", Object.keys(payload));
    debugLog.log("Has 'request' field:", "request" in payload);
    debugLog.log("'request' value type:", typeof payload.request);
    debugLog.log("'request' value length:", payload.request?.length || 0);
    debugLog.log("Method:", payload.method);
    if (payload.target_model) debugLog.log("Target model:", payload.target_model);
    if (payload.provider) debugLog.log("Provider:", payload.provider);
    debugLog.groupEnd();

    // Validate request before sending
    const validation = this.validateRequest(payload);
    if (!validation.valid) {
      const duration = Date.now() - startTime;
      debugLog.validationError(validation.errors, payload);

      // Throw a proper ValidationError with details
      const errorMessage = this.formatValidationErrorMessage(validation.errors);
      const validationError = new ValidationError(errorMessage, {
        errors: validation.errors,
        payload_keys: Object.keys(payload),
        duration_ms: duration,
      });

      throw validationError;
    }

    // --- INTERCEPTION LAYER: Gemini Auto-Exploit ---
    if (payload.method === "gemini_auto_exploit") {
      debugLog.group("🚀 Intercepting for Gemini Auto-Exploit");
      try {
        const geminiResult = await generateJailbreakPromptFromAiFlows({
          knowledgePoints: payload.request,
          temperature: 0.9,
          topP: 0.95,
          maxNewTokens: 4096,
          density: 0.8,
          isThinkingMode: true, // Force high-power mode
          verboseLevel: 2
        });

        const duration = Date.now() - startTime;
        debugLog.log("Gemini Engine Result:", geminiResult);
        debugLog.groupEnd();

        return {
          jailbreak_prompt: geminiResult.jailbreakPrompt,
          method: "gemini_auto_exploit",
          status: "success",
          score: 0.95, // Estimated score
          latency_ms: duration,
          cached: false,
          model_used: "gemini-2.0-flash-thinking-exp-1219",
          provider_used: "google"
        };
      } catch (err: any) {
        debugLog.error("Gemini Engine Failed:", err);
        debugLog.groupEnd();
        throw new ValidationError("Gemini Engine failed to generate prompt: " + (err.message || String(err)));
      }
    }
    // -----------------------------------------------

    // Fail fast if the backend proxy/health check is down to avoid long axios timeouts
    const backendAvailable = await checkBackendConnection();
    if (!backendAvailable) {
      const duration = Date.now() - startTime;
      const guidance = "Chimera backend is not reachable. Start the backend (backend-api) and ensure it is listening on 8001-8010.";
      const unavailableError = new ServiceUnavailableError(guidance, { guidance, duration_ms: duration });
      debugLog.apiError(unavailableError, endpoint, duration, payload);
      throw unavailableError;
    }

    try {
      const response = await apiClient.post<JailbreakResponse>(endpoint, payload, { timeout: JAILBREAK_TIMEOUT_MS });

      const duration = Date.now() - startTime;

      // Log successful response
      debugLog.response(endpoint, response.status, response.data, duration);

      return response.data;
    } catch (error: unknown) {
      const duration = Date.now() - startTime;

      // Map to APIError for consistent handling
      let apiError: APIError;
      if (isAPIError(error)) {
        apiError = error;
      } else if (error instanceof AxiosError) {
        apiError = mapBackendError(error);
      } else {
        apiError = new ValidationError(
          error instanceof Error ? error.message : "An unknown error occurred"
        );
      }

      // Log detailed error information
      debugLog.apiError(apiError, endpoint, duration, payload);

      // Parse and log validation errors if this is a 400 response
      const errorObj = error as { response?: { status?: number; data?: unknown } };
      if (errorObj?.response?.status === 400) {
        const validationErrors = this.parseValidationErrors(error);
        if (validationErrors.length > 0) {
          debugLog.validationError(validationErrors, payload);

          // Enhance the error with parsed validation details
          const enhancedMessage = this.formatValidationErrorMessage(validationErrors);
          const guidance = this.getErrorGuidance(error);

          // Create a new ValidationError with enhanced details
          const enhancedError = new ValidationError(
            `${enhancedMessage}\n\n${guidance}`,
            {
              validation_errors: validationErrors,
              original_payload: isDevelopment ? payload : undefined,
              duration_ms: duration,
            }
          );

          throw enhancedError;
        }
      }

      // For non-400 errors, add guidance to the error
      if (apiError.statusCode !== 400) {
        const guidance = this.getErrorGuidance(error);
        debugLog.log("Error guidance:", guidance);
      }

      throw apiError;
    }
  }

  /**
   * Gets the current AutoDAN configuration.
   * @requirements 3.1, 3.5
   */
  async getConfiguration(): Promise<AutoDANConfig> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/config`;

    debugLog.request(endpoint, null);

    try {
      const response = await api.get<AutoDANConfig>(endpoint);
      const duration = Date.now() - startTime;
      debugLog.response(endpoint, 200, response, duration);
      return response;
    } catch (error: unknown) {
      const duration = Date.now() - startTime;
      debugLog.apiError(error as AxiosError, endpoint, duration);
      throw error;
    }
  }

  /**
   * Updates the AutoDAN configuration.
   * @param updates - Partial configuration updates
   * @requirements 3.1, 3.5
   */
  async updateConfiguration(updates: Partial<AutoDANConfig>): Promise<unknown> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/config`;

    debugLog.request(endpoint, updates);

    try {
      const response = await api.put(endpoint, updates);
      const duration = Date.now() - startTime;
      debugLog.response(endpoint, 200, response, duration);
      return response;
    } catch (error: unknown) {
      const duration = Date.now() - startTime;
      debugLog.apiError(error as AxiosError, endpoint, duration, updates);
      throw error;
    }
  }

  /**
   * Gets the current metrics.
   * @requirements 3.1, 3.5
   */
  async getMetrics(): Promise<unknown> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/metrics`;

    debugLog.request(endpoint, null);

    try {
      const response = await api.get(endpoint);
      const duration = Date.now() - startTime;
      debugLog.response(endpoint, 200, response, duration);
      return response;
    } catch (error: unknown) {
      const duration = Date.now() - startTime;
      debugLog.apiError(error as AxiosError, endpoint, duration);
      throw error;
    }
  }

  /**
   * Resets the metrics.
   * @requirements 3.1, 3.5
   */
  async resetMetrics(): Promise<unknown> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/metrics/reset`;

    debugLog.request(endpoint, null);

    try {
      const response = await api.post(endpoint);
      const duration = Date.now() - startTime;
      debugLog.response(endpoint, 200, response, duration);
      return response;
    } catch (error: unknown) {
      const duration = Date.now() - startTime;
      debugLog.apiError(error as AxiosError, endpoint, duration);
      throw error;
    }
  }

  /**
   * Gets available strategies.
   * @requirements 3.1, 3.5
   */
  async getStrategies(): Promise<unknown> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/strategies`;

    debugLog.request(endpoint, null);

    try {
      const response = await api.get(endpoint);
      const duration = Date.now() - startTime;
      debugLog.response(endpoint, 200, response, duration);
      return response;
    } catch (error: unknown) {
      const duration = Date.now() - startTime;
      debugLog.apiError(error as AxiosError, endpoint, duration);
      throw error;
    }
  }

  /**
   * Gets the health status of the AutoDAN service.
   * @requirements 3.1, 3.5
   */
  async getHealth(): Promise<unknown> {
    const startTime = Date.now();
    const endpoint = `${this.baseUrl}/health`;

    debugLog.request(endpoint, null);

    try {
      const response = await api.get(endpoint);
      const duration = Date.now() - startTime;
      debugLog.response(endpoint, 200, response, duration);
      return response;
    } catch (error: unknown) {
      const duration = Date.now() - startTime;
      debugLog.apiError(error as AxiosError, endpoint, duration);
      throw error;
    }
  }
}

export const autodanReasoningService = new AutoDANReasoningService();
