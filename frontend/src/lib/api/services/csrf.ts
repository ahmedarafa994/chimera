/**
 * CSRF Token Service
 * 
 * Provides utilities for CSRF token management in the frontend.
 * SEC-003: Frontend CSRF token handling implementation.
 */

import { getApiConfig, getActiveApiUrl } from '../core/config';

// CSRF token storage
let csrfToken: string | null = null;
let tokenExpiry: number = 0;

/**
 * Get CSRF token from cookie
 */
export function getCSRFTokenFromCookie(): string | null {
  if (typeof document === 'undefined') return null;
  
  const match = document.cookie.match(/csrf_token=([^;]+)/);
  return match ? match[1] : null;
}

/**
 * Fetch a new CSRF token from the server
 */
export async function fetchCSRFToken(): Promise<string> {
  const config = getApiConfig();
  const baseUrl = getActiveApiUrl();
  
  try {
    const response = await fetch(`${baseUrl}/csrf/token`, {
      method: 'GET',
      credentials: 'include', // Include cookies
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch CSRF token: ${response.status}`);
    }
    
    const data = await response.json();
    csrfToken = data.token;
    tokenExpiry = Date.now() + (data.expires_in_seconds * 1000);

    if (!csrfToken) {
      throw new Error('CSRF token not received from server');
    }

    return csrfToken;
  } catch (error) {
    console.error('Failed to fetch CSRF token:', error);
    throw error;
  }
}

/**
 * Get the current CSRF token, fetching a new one if needed
 */
export async function getCSRFToken(): Promise<string> {
  // Check if we have a valid cached token
  if (csrfToken && Date.now() < tokenExpiry - 60000) { // 1 minute buffer
    return csrfToken;
  }
  
  // Try to get from cookie first
  const cookieToken = getCSRFTokenFromCookie();
  if (cookieToken) {
    csrfToken = cookieToken;
    tokenExpiry = Date.now() + (24 * 60 * 60 * 1000); // Assume 24 hours
    return csrfToken;
  }
  
  // Fetch new token
  return fetchCSRFToken();
}

/**
 * Add CSRF token to request headers
 */
export async function addCSRFHeader(headers: HeadersInit = {}): Promise<HeadersInit> {
  try {
    const token = await getCSRFToken();
    return {
      ...headers,
      'X-CSRF-Token': token,
    };
  } catch (error) {
    console.warn('Could not add CSRF token to headers:', error);
    return headers;
  }
}

/**
 * Validate a CSRF token with the server
 */
export async function validateCSRFToken(token: string): Promise<boolean> {
  const baseUrl = getActiveApiUrl();
  
  try {
    const response = await fetch(`${baseUrl}/csrf/validate?token=${encodeURIComponent(token)}`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      return false;
    }
    
    const data = await response.json();
    return data.valid === true;
  } catch (error) {
    console.error('CSRF token validation failed:', error);
    return false;
  }
}

/**
 * Clear cached CSRF token
 */
export function clearCSRFToken(): void {
  csrfToken = null;
  tokenExpiry = 0;
}

/**
 * Hook for React components to use CSRF token
 */
export function useCSRFToken() {
  return {
    getToken: getCSRFToken,
    addHeader: addCSRFHeader,
    validate: validateCSRFToken,
    clear: clearCSRFToken,
  };
}