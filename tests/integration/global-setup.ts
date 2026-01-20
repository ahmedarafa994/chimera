import { chromium, FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const FRONTEND_URL = 'http://localhost:3001';
const BACKEND_URL = 'http://localhost:8001';
const AUTH_STATE_PATH = path.join(__dirname, '.auth-state.json');

/**
 * Global Setup: Authenticates once and saves storage state for all tests.
 *
 * Test credentials are read from environment variables:
 * - SPIDER_TEST_USER (default: admin)
 * - SPIDER_TEST_PASS (default: admin123)
 */
async function globalSetup(config: FullConfig) {
    const username = process.env.SPIDER_TEST_USER || 'admin';
    const password = process.env.SPIDER_TEST_PASS || 'admin123';

    console.log(`[Spider Setup] Authenticating as ${username}...`);

    const browser = await chromium.launch();
    const context = await browser.newContext();
    const page = await context.newPage();

    try {
        // Navigate to login page
        await page.goto(`${FRONTEND_URL}/login`, { waitUntil: 'networkidle' });

        // Fill login form using actual IDs from LoginForm.tsx
        const usernameInput = page.locator('#login-username');
        const passwordInput = page.locator('#login-password');
        const submitButton = page.locator('button[type="submit"]:has-text("Sign In")');

        await usernameInput.fill(username);
        await passwordInput.fill(password);
        await submitButton.click();

        // Wait for successful navigation (dashboard or home)
        await page.waitForURL((url) => !url.pathname.includes('/login'), { timeout: 15000 });
        await page.waitForLoadState('networkidle');

        console.log('[Spider Setup] ✅ Login successful');

        // Save storage state for reuse in tests
        await context.storageState({ path: AUTH_STATE_PATH });
        console.log(`[Spider Setup] Saved auth state to ${AUTH_STATE_PATH}`);

    } catch (error) {
        console.error('[Spider Setup] ❌ Login failed:', error);
        // Save a screenshot for debugging
        await page.screenshot({ path: 'spider-login-failure.png' });
        throw error;
    } finally {
        await browser.close();
    }
}

export default globalSetup;
