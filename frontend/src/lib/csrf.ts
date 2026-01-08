/**
 * CSRF Protection Utilities
 * 
 * Implements client-side CSRF token handling for the Double Submit Cookie pattern.
 * 
 * Usage:
 * 1. The backend sets a CSRF token in a cookie on GET requests
 * 2. For state-changing requests (POST, PUT, DELETE, PATCH), include the token in the header
 * 3. The backend validates that the cookie and header values match
 * 
 * @example
 * ```typescript
 * import { getCSRFToken, addCSRFHeader } from '@/lib/csrf';
 * 
 * // Get the current CSRF token
 * const token = getCSRFToken();
 * 
 * // Add CSRF header to fetch options
 * const options = addCSRFHeader({ method: 'POST', body: JSON.stringify(data) });
 * fetch('/api/endpoint', options);
 * ```
 */

// Cookie name must match backend configuration
const CSRF_COOKIE_NAME = 'csrf_token';
// Header name must match backend configuration
const CSRF_HEADER_NAME = 'X-CSRF-Token';

/**
 * Get the CSRF token from the cookie.
 * 
 * @returns The CSRF token or null if not found
 */
export function getCSRFToken(): string | null {
  if (typeof document === 'undefined') {
    // Server-side rendering - no cookies available
    return null;
  }
  
  const cookies = document.cookie.split(';');
  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split('=');
    if (name === CSRF_COOKIE_NAME) {
      return decodeURIComponent(value);
    }
  }
  
  return null;
}

/**
 * Check if a CSRF token is available.
 * 
 * @returns True if a CSRF token is available
 */
export function hasCSRFToken(): boolean {
  return getCSRFToken() !== null;
}

/**
 * Add CSRF header to request headers.
 * 
 * @param headers - Existing headers object or Headers instance
 * @returns Updated headers with CSRF token
 */
export function addCSRFToHeaders(headers: Record<string, string> | Headers): Record<string, string> | Headers {
  const token = getCSRFToken();
  
  if (!token) {
    console.warn('CSRF token not found in cookies. State-changing requests may fail.');
    return headers;
  }
  
  if (headers instanceof Headers) {
    headers.set(CSRF_HEADER_NAME, token);
    return headers;
  }
  
  return {
    ...headers,
    [CSRF_HEADER_NAME]: token,
  };
}

/**
 * Add CSRF header to fetch options.
 * 
 * @param options - Fetch request options
 * @returns Updated options with CSRF header
 */
export function addCSRFHeader(options: RequestInit = {}): RequestInit {
  const token = getCSRFToken();
  
  if (!token) {
    console.warn('CSRF token not found in cookies. State-changing requests may fail.');
    return options;
  }
  
  const headers = new Headers(options.headers);
  headers.set(CSRF_HEADER_NAME, token);
  
  return {
    ...options,
    headers,
  };
}

/**
 * Create a fetch wrapper that automatically includes CSRF token.
 * 
 * @param input - Request URL or Request object
 * @param init - Request options
 * @returns Fetch response promise
 */
export async function csrfFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const method = init?.method?.toUpperCase() || 'GET';
  const safeMethod = ['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method);
  
  // Only add CSRF header for state-changing methods
  if (!safeMethod) {
    init = addCSRFHeader(init);
  }
  
  return fetch(input, init);
}

/**
 * Request a new CSRF token from the server.
 * 
 * This is useful when:
 * - The token has expired
 * - The token is missing
 * - After a logout/login cycle
 * 
 * @returns Promise that resolves when the token is refreshed
 */
export async function refreshCSRFToken(): Promise<string | null> {
  try {
    // Make a GET request to trigger token generation
    const response = await fetch('/api/v1/csrf/token', {
      method: 'GET',
      credentials: 'include', // Include cookies
    });
    
    if (!response.ok) {
      console.error('Failed to refresh CSRF token:', response.status);
      return null;
    }
    
    // The token should now be in the cookie
    // Wait a tick for the cookie to be set
    await new Promise(resolve => setTimeout(resolve, 10));
    
    return getCSRFToken();
  } catch (error) {
    console.error('Error refreshing CSRF token:', error);
    return null;
  }
}

/**
 * Ensure a CSRF token is available, refreshing if necessary.
 * 
 * @returns Promise that resolves to the CSRF token or null
 */
export async function ensureCSRFToken(): Promise<string | null> {
  let token = getCSRFToken();
  
  if (!token) {
    token = await refreshCSRFToken();
  }
  
  return token;
}

/**
 * CSRF-protected axios request interceptor.
 * 
 * Add this to your axios instance to automatically include CSRF tokens.
 * 
 * @example
 * ```typescript
 * import axios from 'axios';
 * import { csrfRequestInterceptor } from '@/lib/csrf';
 * 
 * const api = axios.create({ baseURL: '/api/v1' });
 * api.interceptors.request.use(csrfRequestInterceptor);
 * ```
 */
export function csrfRequestInterceptor(config: { headers?: Record<string, string>; method?: string }) {
  const method = config.method?.toUpperCase() || 'GET';
  const safeMethod = ['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method);
  
  if (!safeMethod) {
    const token = getCSRFToken();
    if (token) {
      config.headers = config.headers || {};
      config.headers[CSRF_HEADER_NAME] = token;
    }
  }
  
  return config;
}

// Export constants for external use
export const CSRF_CONFIG = {
  cookieName: CSRF_COOKIE_NAME,
  headerName: CSRF_HEADER_NAME,
} as const;