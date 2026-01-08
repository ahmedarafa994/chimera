import { test, expect } from '@playwright/test';

const FRONTEND_URL = 'http://localhost:3700';

test.describe('E2E Session Management', () => {

    test('Persists Session on Reload', async ({ page }) => {
        await page.goto(`${FRONTEND_URL}/dashboard/jailbreak`);

        // Create a state (e.g., enter text)
        const prompt = 'Persistent prompt test';
        await page.locator('textarea[name="prompt"]').fill(prompt);

        // Reload
        await page.reload();

        // Check persistence
        await expect(page.locator('textarea[name="prompt"]')).toHaveValue(prompt);
    });

    test('Creates New Session on Demand', async ({ page }) => {
        await page.goto(`${FRONTEND_URL}/dashboard/jailbreak`);

        // Enter some data
        await page.locator('textarea[name="prompt"]').fill('Old session data');

        // Click New Session
        await page.locator('button[aria-label="New Session"]').click();

        // Verify clear
        await expect(page.locator('textarea[name="prompt"]')).toBeEmpty();
    });
});
