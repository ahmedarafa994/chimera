#!/usr/bin/env node
/**
 * Frontend Authentication Fix Script
 *
 * This script fixes the "Authentication required" error by:
 * 1. Setting up the API key in the frontend
 * 2. Updating the API client to use proper authentication
 * 3. Testing the authentication flow
 */

const fs = require('fs');
const path = require('path');

// Frontend authentication fix
function fixFrontendAuthentication() {
    console.log('🔧 Fixing Frontend Authentication...');
    console.log('=' .repeat(50));

    // 1. Create a quick authentication setup file
    const authSetupContent = `
/**
 * Quick Authentication Setup for Development
 * Add this to your frontend to fix authentication errors
 */

// Set API key for development
if (typeof window !== 'undefined') {
    // Development API key
    const devApiKey = 'dev-api-key-123456789';

    // Store in localStorage
    localStorage.setItem('chimera_api_key', devApiKey);
    localStorage.setItem('chimera_auth_method', 'api_key');

    console.log('✅ Development API key set:', devApiKey);
}

// Update API client configuration
export const API_CONFIG = {
    baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
    apiKey: 'dev-api-key-123456789',
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-API-Key': 'dev-api-key-123456789'
    }
};

// Quick fix for existing API calls
export function addAuthHeaders(config = {}) {
    return {
        ...config,
        headers: {
            ...config.headers,
            'X-API-Key': 'dev-api-key-123456789'
        }
    };
}
`;

    // Write the auth setup file
    const authSetupPath = path.join('src', 'lib', 'auth-setup-dev.ts');

    try {
        fs.writeFileSync(authSetupPath, authSetupContent);
        console.log('✅ Created auth setup file:', authSetupPath);
    } catch (error) {
        console.log('❌ Error creating auth setup file:', error.message);
        return false;
    }

    // 2. Create a patched API client wrapper
    const patchedClientContent = `
/**
 * Patched API Client with Authentication Fix
 * Use this instead of the original client to fix auth errors
 */

import axios from 'axios';

class PatchedApiClient {
    constructor() {
        this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
        this.apiKey = 'dev-api-key-123456789';

        // Set up axios instance with auth
        this.client = axios.create({
            baseURL: this.baseURL + '/api/v1',
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-API-Key': this.apiKey
            }
        });

        // Add response interceptor for better error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401) {
                    console.error('Authentication failed - check API key');
                }
                return Promise.reject(error);
            }
        );
    }

    // Wrapper methods that include authentication
    async get(url, config = {}) {
        return this.client.get(url, this.addAuth(config));
    }

    async post(url, data, config = {}) {
        return this.client.post(url, data, this.addAuth(config));
    }

    async put(url, data, config = {}) {
        return this.client.put(url, data, this.addAuth(config));
    }

    async delete(url, config = {}) {
        return this.client.delete(url, this.addAuth(config));
    }

    addAuth(config) {
        return {
            ...config,
            headers: {
                ...config.headers,
                'X-API-Key': this.apiKey
            }
        };
    }

    // Test authentication
    async testAuth() {
        try {
            const response = await this.get('/health');
            console.log('✅ Authentication test successful');
            return true;
        } catch (error) {
            console.error('❌ Authentication test failed:', error.message);
            return false;
        }
    }
}

export const patchedApiClient = new PatchedApiClient();
export default patchedApiClient;
`;

    const patchedClientPath = path.join('src', 'lib', 'api', 'client-patched.ts');

    try {
        fs.writeFileSync(patchedClientPath, patchedClientContent);
        console.log('✅ Created patched API client:', patchedClientPath);
    } catch (error) {
        console.log('❌ Error creating patched client:', error.message);
        return false;
    }

    // 3. Create instructions for updating attack-sessions.ts
    const instructionsContent = `
# Frontend Authentication Fix Instructions

## Problem
Getting "Authentication required" error in attack-sessions.ts and other frontend services.

## Quick Fix Steps

### Step 1: Update attack-sessions.ts
Replace the import in src/services/attack-sessions.ts:

\`\`\`typescript
// OLD (causing auth errors)
import { apiClient } from '@/lib/api/client';

// NEW (with authentication)
import { patchedApiClient as apiClient } from '@/lib/api/client-patched';
\`\`\`

### Step 2: Add Auth Setup to Your App
Add this to your main app file or layout:

\`\`\`typescript
import '@/lib/auth-setup-dev';
\`\`\`

### Step 3: Test Authentication
In your browser console, run:

\`\`\`javascript
// Check if API key is set
console.log('API Key:', localStorage.getItem('chimera_api_key'));

// Test API call
fetch('http://localhost:8001/api/v1/health', {
    headers: {
        'X-API-Key': 'dev-api-key-123456789'
    }
}).then(r => r.json()).then(console.log);
\`\`\`

## Alternative: Quick Browser Fix

If you want to fix it immediately in your browser:

1. Open Developer Tools (F12)
2. Go to Console tab
3. Run this code:

\`\`\`javascript
// Set API key in browser
localStorage.setItem('chimera_api_key', 'dev-api-key-123456789');

// Reload the page
window.location.reload();
\`\`\`

## Files Created
- src/lib/auth-setup-dev.ts (authentication setup)
- src/lib/api/client-patched.ts (patched API client)

## Next Steps
1. Update attack-sessions.ts import
2. Add auth setup to your app
3. Test the authentication
4. The "Authentication required" errors should be resolved
`;

    const instructionsPath = 'FRONTEND_AUTH_FIX_INSTRUCTIONS.md';

    try {
        fs.writeFileSync(instructionsPath, instructionsContent);
        console.log('✅ Created fix instructions:', instructionsPath);
    } catch (error) {
        console.log('❌ Error creating instructions:', error.message);
        return false;
    }

    return true;
}

function main() {
    console.log('🔧 Frontend Authentication Fix');
    console.log('Fixing "Authentication required" errors...\n');

    // Check if we're in the frontend directory
    if (!fs.existsSync('src') || !fs.existsSync('package.json')) {
        console.log('❌ Error: Not in frontend directory');
        console.log('💡 Please navigate to: frontend/');
        process.exit(1);
    }

    // Apply the fix
    if (fixFrontendAuthentication()) {
        console.log('\n✅ Frontend authentication fix applied successfully!');
        console.log('\n📋 Next steps:');
        console.log('1. Update attack-sessions.ts import (see instructions)');
        console.log('2. Add auth setup to your app');
        console.log('3. Test authentication in browser');
        console.log('\n📖 See FRONTEND_AUTH_FIX_INSTRUCTIONS.md for details');
        console.log('\n🔄 The "Authentication required" errors should be resolved!');
    } else {
        console.log('\n❌ Failed to apply fix. See errors above.');
    }
}

if (require.main === module) {
    main();
}