import { test as setup, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

/**
 * Authentication Setup for Playwright Spider Tests
 *
 * This setup file logs into the application and saves the authenticated
 * browser state for reuse across all tests (storageState).
 */

const FRONTEND_URL = 'http://localhost:3001';
const AUTH_FILE = path.join(__dirname, '..', '.auth', 'state.json');

// Test credentials - can be overridden via environment variables
const TEST_USERNAME = process.env.TEST_USERNAME || 'admin';
const TEST_PASSWORD = process.env.TEST_PASSWORD || 'admin';

setup('authenticate', async ({ page }) => {
    // Ensure .auth directory exists
    const authDir = path.dirname(AUTH_FILE);
    if (!fs.existsSync(authDir)) {
        fs.mkdirSync(authDir, { recursive: true });
    }

    console.log('[Auth Setup] Navigating to login page...');
    await page.goto(`${FRONTEND_URL}/login`, { waitUntil: 'networkidle' });

    // Wait for login form to be ready
    await page.waitForSelector('[id="login-username"]', { state: 'visible' });

    console.log(`[Auth Setup] Logging in as: ${TEST_USERNAME}`);

    // Fill in credentials
    await page.fill('[id="login-username"]', TEST_USERNAME);
    await page.fill('[id="login-password"]', TEST_PASSWORD);

    // Submit the form
    await page.click('button[type="submit"]');

    // Wait for successful redirect to dashboard or any authenticated page
    // The app redirects to /dashboard after successful login
    await page.waitForURL('**/dashboard**', { timeout: 30000 });

    console.log('[Auth Setup] Login successful, saving auth state...');

    // Verify we're authenticated by checking for dashboard elements
    // This ensures the login was actually successful
    await expect(page).toHaveURL(/dashboard/);

    // Save the storage state (cookies + localStorage)
    await page.context().storageState({ path: AUTH_FILE });

    console.log(`[Auth Setup] Auth state saved to: ${AUTH_FILE}`);
});
