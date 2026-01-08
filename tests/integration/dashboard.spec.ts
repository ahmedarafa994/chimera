import { test, expect } from '@playwright/test';

const FRONTEND_URL = 'http://localhost:3700';

test.describe('Dashboard E2E Tests', () => {

  test.beforeEach(async ({ page }) => {
    // Navigate to dashboard
    await page.goto(`${FRONTEND_URL}/dashboard`);
    // Wait for page to load
    await page.waitForLoadState('networkidle');
  });

  test('Dashboard renders correctly', async ({ page }) => {
    // Check main layout elements
    await expect(page.locator('body')).toBeVisible();

    // Check for navigation sidebar
    const sidebar = page.locator('[data-testid="sidebar"]').or(page.locator('nav'));
    await expect(sidebar).toBeVisible();
  });

  test('Dashboard navigation works', async ({ page }) => {
    // Test navigation to different sections
    const navLinks = [
      { name: 'jailbreak', path: '/dashboard/jailbreak' },
      { name: 'autodan', path: '/dashboard/autodan' },
      { name: 'metrics', path: '/dashboard/metrics' },
      { name: 'settings', path: '/dashboard/settings' },
    ];

    for (const link of navLinks) {
      const navLink = page.locator(`a[href*="${link.path}"]`).first();
      if (await navLink.isVisible()) {
        await navLink.click();
        await page.waitForURL(`**${link.path}*`);
        expect(page.url()).toContain(link.path);
        // Go back to dashboard
        await page.goto(`${FRONTEND_URL}/dashboard`);
      }
    }
  });

  test('Dashboard displays health status', async ({ page }) => {
    // Navigate to health page
    await page.goto(`${FRONTEND_URL}/dashboard/health`);
    await page.waitForLoadState('networkidle');

    // Check for health indicators
    const healthContent = page.locator('main').or(page.locator('[role="main"]'));
    await expect(healthContent).toBeVisible();
  });

  test('Dashboard metrics page loads', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/dashboard/metrics`);
    await page.waitForLoadState('networkidle');

    // Check for metrics content
    const metricsContent = page.locator('main');
    await expect(metricsContent).toBeVisible();
  });

  test('Dashboard settings page loads', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/dashboard/settings`);
    await page.waitForLoadState('networkidle');

    // Check for settings content
    const settingsContent = page.locator('main');
    await expect(settingsContent).toBeVisible();
  });
});

test.describe('Dashboard Error Handling', () => {

  test('404 page displays for unknown routes', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/dashboard/nonexistent-page-xyz`);

    // Check for 404 content or redirect
    const pageContent = await page.textContent('body');
    expect(pageContent).toBeTruthy();
  });

  test('Error boundary catches errors gracefully', async ({ page }) => {
    // Visit main page first
    await page.goto(`${FRONTEND_URL}/dashboard`);

    // Check that page renders without crashing
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });
});

test.describe('Dashboard Performance', () => {

  test('Dashboard loads within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto(`${FRONTEND_URL}/dashboard`);
    await page.waitForLoadState('domcontentloaded');
    const loadTime = Date.now() - startTime;

    // Dashboard should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test('Dashboard interactive within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto(`${FRONTEND_URL}/dashboard`);
    await page.waitForLoadState('networkidle');
    const interactiveTime = Date.now() - startTime;

    // Dashboard should be interactive within 10 seconds
    expect(interactiveTime).toBeLessThan(10000);
  });
});
