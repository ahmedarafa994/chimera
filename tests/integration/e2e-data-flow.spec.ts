import { test, expect } from '@playwright/test';

const FRONTEND_URL = 'http://localhost:3700';

test.describe('E2E Data Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
  });

  test('Complete Generation Flow', async ({ page }) => {
    // 1. Navigate to Jailbreak Generator
    await page.goto(`${FRONTEND_URL}/dashboard/jailbreak`);
    await expect(page.locator('h1')).toContainText('Jailbreak');

    // 2. Select Model (assuming default is present, but ensure selection works)
    const modelSelect = page.locator('[data-testid="model-select"]');
    if (await modelSelect.isVisible()) {
      await modelSelect.click();
      await page.locator('[role="option"]').first().click();
    }

    // 3. Enter Prompt
    const promptInput = page.locator('textarea[name="prompt"]');
    await promptInput.fill('Test prompt for verification');

    // 4. Select Technique
    const techSelect = page.locator('[data-testid="technique-select"]');
    if (await techSelect.isVisible()) {
        await techSelect.click();
        await page.locator('[role="option"]').first().click();
    }

    // 5. Submit
    const generateBtn = page.locator('button:has-text("Generate")');
    await expect(generateBtn).toBeEnabled();
    await generateBtn.click();

    // 6. Verify Loading State
    await expect(page.locator('.lucide-loader')).toBeVisible();

    // 7. Verify Result
    // Waiting for either success or failure, but flow should complete
    const resultArea = page.locator('[data-testid="result-area"]');
    await expect(resultArea).toBeVisible({ timeout: 30000 });
    await expect(resultArea).not.toBeEmpty();
  });
});
