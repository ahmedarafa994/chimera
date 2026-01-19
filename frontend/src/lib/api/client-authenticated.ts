/**
 * IMMEDIATE FIX: Frontend Authentication Setup
 *
 * This is a quick fix for the "Authentication required" error.
 * Add this code to fix the authentication issue immediately.
 */

// Development API key that works with the backend
const DEV_API_KEY = 'dev-api-key-123456789';

// Set up authentication in localStorage (for immediate fix)
if (typeof window !== 'undefined') {
    localStorage.setItem('chimera_api_key', DEV_API_KEY);
    localStorage.setItem('chimera_auth_method', 'api_key');
}

// Patched API client with proper authentication
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

class AuthenticatedApiClient {
    private apiKey: string;
    private baseURL: string;
    private client: AxiosInstance;

    constructor() {
        this.apiKey = DEV_API_KEY;
        this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

        this.client = axios.create({
            baseURL: `${this.baseURL}/api/v1`,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-API-Key': this.apiKey
            }
        });

        // Add request interceptor to ensure API key is always included
        this.client.interceptors.request.use((config) => {
            config.headers['X-API-Key'] = this.apiKey;
            return config;
        });

        // Add response interceptor for better error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401) {
                    console.error('🔑 Authentication failed - API key may be invalid');
                    console.log('💡 Expected API key:', this.apiKey);
                }
                return Promise.reject(error);
            }
        );
    }

    // HTTP methods with automatic authentication
    async get<T = unknown>(url: string, config: AxiosRequestConfig = {}): Promise<AxiosResponse<T>> {
        return this.client.get<T>(url, config);
    }

    async post<T = unknown>(url: string, data?: unknown, config: AxiosRequestConfig = {}): Promise<AxiosResponse<T>> {
        return this.client.post<T>(url, data, config);
    }

    async put<T = unknown>(url: string, data?: unknown, config: AxiosRequestConfig = {}): Promise<AxiosResponse<T>> {
        return this.client.put<T>(url, data, config);
    }

    async delete<T = unknown>(url: string, config: AxiosRequestConfig = {}): Promise<AxiosResponse<T>> {
        return this.client.delete<T>(url, config);
    }

    async patch<T = unknown>(url: string, data?: unknown, config: AxiosRequestConfig = {}): Promise<AxiosResponse<T>> {
        return this.client.patch<T>(url, data, config);
    }

    // Alias for delete method (for compatibility)
    async del<T = unknown>(url: string, config: AxiosRequestConfig = {}): Promise<AxiosResponse<T>> {
        return this.delete<T>(url, config);
    }

    // Test authentication
    async testConnection(): Promise<boolean> {
        try {
            const response = await this.get('/health');
            console.log('✅ API authentication successful');
            return true;
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            console.error('❌ API authentication failed:', errorMessage);
            return false;
        }
    }
}

// Export the authenticated client
export const authenticatedApiClient = new AuthenticatedApiClient();

// For backward compatibility, also export as default
export default authenticatedApiClient;

// Quick test function you can call in browser console
export async function testAuthentication() {
    console.log('🧪 Testing API authentication...');

    const client = new AuthenticatedApiClient();
    const success = await client.testConnection();

    if (success) {
        console.log('✅ Authentication working! You can now use the API.');
    } else {
        console.log('❌ Authentication failed. Check backend server and API key.');
    }

    return success;
}