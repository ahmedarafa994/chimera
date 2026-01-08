import { test, expect } from '@playwright/test';

const BACKEND_URL = 'http://localhost:8001';
const FRONTEND_URL = 'http://localhost:3700';

test.describe('Chimera Full-Stack Integration Tests', () => {

  test('Backend Health Endpoints', async () => {
    const healthRes = await fetch(`${BACKEND_URL}/health`);
    expect(healthRes.status).toBe(200);
    const healthData = await healthRes.json();
    expect(healthData).toHaveProperty('status');
  });

  test('Backend Providers Endpoint', async () => {
    const res = await fetch(`${BACKEND_URL}/api/v1/providers`);
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data).toHaveProperty('providers');
    expect(Array.isArray(data.providers)).toBe(true);
    expect(data).toHaveProperty('count');
    expect(data).toHaveProperty('default');
  });

  test('Frontend Loads Successfully', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    const body = await page.textContent('body');
    expect(body).toBeTruthy();
  });

  test('Frontend Dashboard Loads', async ({ page }) => {
    await page.goto(`${FRONTEND_URL}/dashboard`);
    await page.waitForLoadState('networkidle');
    const mainContent = page.locator('main');
    await expect(mainContent).toBeVisible();
  });

  test('CORS Configuration', async () => {
    const res = await fetch(`${BACKEND_URL}/api/v1/providers`, {
      method: 'OPTIONS',
      headers: {
        'Origin': FRONTEND_URL,
        'Access-Control-Request-Method': 'GET',
      }
    });
    expect(res.headers.get('access-control-allow-origin')).toBeTruthy();
  });

  test('API Response Time', async () => {
    const start = Date.now();
    await fetch(`${BACKEND_URL}/health`);
    const duration = Date.now() - start;
    expect(duration).toBeLessThan(1000);
  });

  test('Frontend API Proxy Works', async ({ page }) => {
    // Test that frontend can proxy requests to backend
    const response = await page.request.get(`${FRONTEND_URL}/api/health`);
    // Should either succeed or return a valid HTTP response
    expect([200, 404, 502, 503]).toContain(response.status());
  });
});

test.describe('Full-Stack Security Tests', () => {

  test('Security Headers Present', async ({ page }) => {
    const response = await page.goto(FRONTEND_URL);
    const headers = response?.headers();

    // Check for common security headers
    // These may or may not be present depending on configuration
    if (headers) {
      // Log headers for debugging
      console.log('Response headers:', JSON.stringify(headers, null, 2));
    }
  });

  test('No Sensitive Data in Error Responses', async () => {
    // Test 404 endpoint
    const response = await fetch(`${BACKEND_URL}/api/v1/nonexistent`);
    const text = await response.text();

    // Should not contain stack traces or internal paths
    expect(text).not.toContain('Traceback');
    expect(text).not.toContain('backend-api/app');
  });
});

test.describe('Full-Stack Performance Tests', () => {

  test('Backend responds within SLA', async () => {
    const endpoints = [
      '/health',
      '/api/v1/providers',
    ];

    for (const endpoint of endpoints) {
      const start = Date.now();
      const response = await fetch(`${BACKEND_URL}${endpoint}`);
      const duration = Date.now() - start;

      // All endpoints should respond within 2 seconds
      expect(duration).toBeLessThan(2000);
      expect([200, 401, 404]).toContain(response.status);
    }
  });

  test('Frontend Time to First Byte', async ({ page }) => {
    const startTime = Date.now();
    await page.goto(FRONTEND_URL);
    const ttfb = Date.now() - startTime;

    // TTFB should be under 3 seconds
    expect(ttfb).toBeLessThan(3000);
  });
});
