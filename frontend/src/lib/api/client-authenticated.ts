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
import axios from 'axios';

class AuthenticatedApiClient {
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
    async get(url, config = {}) {
        return this.client.get(url, config);
    }

    async post(url, data, config = {}) {
        return this.client.post(url, data, config);
    }

    async put(url, data, config = {}) {
        return this.client.put(url, data, config);
    }

    async delete(url, config = {}) {
        return this.client.delete(url, config);
    }

    async patch(url, data, config = {}) {
        return this.client.patch(url, data, config);
    }

    // Test authentication
    async testConnection() {
        try {
            const response = await this.get('/health');
            console.log('✅ API authentication successful');
            return true;
        } catch (error) {
            console.error('❌ API authentication failed:', error.message);
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