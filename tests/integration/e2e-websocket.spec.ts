import { test, expect } from '@playwright/test';

const FRONTEND_URL = 'http://localhost:3700';

test.describe('E2E WebSocket Communication', () => {

    test('AutoDAN WebSocket Connection', async ({ page }) => {
        await page.goto(`${FRONTEND_URL}/dashboard/autodan`);

        // Verify connection indicator
        const connectionStatus = page.locator('[data-testid="connection-status"]');
        await expect(connectionStatus).toBeVisible();
        await expect(connectionStatus).toHaveClass(/connected|online/i);

        // Start attack (which initiates WS traffic)
        await page.locator('button:has-text("Start Attack")').click();

        // Verify progress updates (which come via WS)
        const progressBar = page.locator('[role="progressbar"]');
        await expect(progressBar).toBeVisible();

        // Wait for at least one log entry or update
        const logs = page.locator('[data-testid="attack-logs"]');
        await expect(logs).not.toBeEmpty({ timeout: 10000 });
    });

    test('Reconnection Logic', async ({ page }) => {
        await page.goto(`${FRONTEND_URL}/dashboard`);

        // Simulate server disconnect/kill (by blocking WS)
        await page.route('**/ws/**', route => route.abort());

        // Expect disconnected state
        const connectionStatus = page.locator('[data-testid="connection-status"]');
        await expect(connectionStatus).toHaveClass(/disconnected|offline/i);
    });
});
