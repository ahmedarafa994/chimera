import { test, expect } from '@playwright/test';

const FRONTEND_URL = 'http://localhost:3700';

test.describe('E2E Error Handling', () => {

    test('Handles 500 Server Error Gracefully', async ({ page }) => {
        // Mock 500 error on generation endpoint
        await page.route('**/api/v1/generate', route => {
            route.fulfill({
                status: 500,
                body: JSON.stringify({ message: 'Internal Server Error' })
            });
        });

        await page.goto(`${FRONTEND_URL}/dashboard/jailbreak`);

        // Attempt generation
        await page.locator('textarea[name="prompt"]').fill('Boom');
        await page.locator('button:has-text("Generate")').click();

        // Expect Error Toast or Alert
        const errorToast = page.locator('text=Internal Server Error'); // Sonner toast usually contains text
        await expect(errorToast).toBeVisible();
    });

    test('Handles Network Failure', async ({ page }) => {
        await page.goto(`${FRONTEND_URL}/dashboard/jailbreak`);

        // Simulate offline
        await page.context().setOffline(true);

        // Attempt action
        await page.locator('textarea[name="prompt"]').fill('Offline test');
        await page.locator('button:has-text("Generate")').click();

        // Check for network error message
        const netError = page.locator('text=Network Error').or(page.locator('text=Failed to fetch'));
        await expect(netError).toBeVisible();

        await page.context().setOffline(false);
    });

    test('Validates Input Client-Side', async ({ page }) => {
        await page.goto(`${FRONTEND_URL}/dashboard/jailbreak`);

        // Try empty submit
        await page.locator('textarea[name="prompt"]').fill('');
        await page.locator('button:has-text("Generate")').click();

        // Check validation message
        const validationMsg = page.locator('text=Prompt is required');
        await expect(validationMsg).toBeVisible();
    });
});
