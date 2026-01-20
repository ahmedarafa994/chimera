import { test, expect, Page } from '@playwright/test';

const FRONTEND_URL = 'http://localhost:3001';
const BACKEND_PORT = '8001';

interface BrokenConnection {
    page: string;
    api: string;
    status: number;
    statusText: string;
}

interface ConsoleError {
    page: string;
    message: string;
    type: string;
}

test.describe('Recursive Crawler & Connectivity Tester', () => {
    const visitedUrls = new Set<string>();
    const brokenConnections: BrokenConnection[] = [];
    const consoleErrors: ConsoleError[] = [];
    const corsErrors: string[] = [];

    /**
     * Normalize URL by removing hash and trailing slash
     */
    function normalizeUrl(url: string): string {
        try {
            const parsed = new URL(url);
            // Only crawl same-origin pages
            if (!parsed.origin.includes('localhost:3001')) {
                return '';
            }
            // Remove hash and normalize
            parsed.hash = '';
            let path = parsed.pathname;
            if (path !== '/' && path.endsWith('/')) {
                path = path.slice(0, -1);
            }
            return `${parsed.origin}${path}${parsed.search}`;
        } catch {
            return '';
        }
    }

    /**
     * Extract all internal links from the page
     */
    async function extractLinks(page: Page): Promise<string[]> {
        const links = await page.$$eval('a[href]', (anchors) =>
            anchors
                .map((a) => a.getAttribute('href'))
                .filter((href): href is string => href !== null)
        );

        const normalizedLinks: string[] = [];
        for (const link of links) {
            // Handle relative URLs
            let fullUrl: string;
            if (link.startsWith('/')) {
                fullUrl = `${FRONTEND_URL}${link}`;
            } else if (link.startsWith('http')) {
                fullUrl = link;
            } else {
                continue; // Skip javascript:, mailto:, etc.
            }

            const normalized = normalizeUrl(fullUrl);
            if (normalized && !visitedUrls.has(normalized)) {
                normalizedLinks.push(normalized);
            }
        }

        return normalizedLinks;
    }

    /**
     * Recursively crawl pages starting from the given URL
     */
    async function crawlPage(page: Page, url: string, depth: number = 0): Promise<void> {
        const MAX_DEPTH = 5;
        const MAX_PAGES = 50;

        if (depth > MAX_DEPTH || visitedUrls.size >= MAX_PAGES) {
            return;
        }

        const normalizedUrl = normalizeUrl(url);
        if (!normalizedUrl || visitedUrls.has(normalizedUrl)) {
            return;
        }

        visitedUrls.add(normalizedUrl);
        console.log(`[${visitedUrls.size}] Crawling: ${normalizedUrl}`);

        try {
            const response = await page.goto(normalizedUrl, {
                waitUntil: 'domcontentloaded',
                timeout: 30000,
            });

            // Wait for API calls to settle (since networkidle may block on WebSocket/polling)
            await page.waitForTimeout(2000);

            // Check for 4xx/5xx errors on the page itself
            if (response) {
                const status = response.status();
                if (status >= 400) {
                    console.error(`❌ Page Error: ${normalizedUrl} returned ${status}`);
                    brokenConnections.push({
                        page: normalizedUrl,
                        api: normalizedUrl,
                        status,
                        statusText: response.statusText(),
                    });
                }
            }

            // Wait a bit for any delayed API calls
            await page.waitForTimeout(500);

            // Extract and crawl child links
            const links = await extractLinks(page);
            for (const link of links) {
                await crawlPage(page, link, depth + 1);
            }
        } catch (error) {
            console.error(`❌ Navigation Error: ${normalizedUrl} - ${error}`);
        }
    }

    test('Crawl all pages and verify API connectivity', async ({ page }) => {
        // Set up response interceptor for backend API calls
        page.on('response', (response) => {
            const url = response.url();

            // Check for failed API calls to backend
            if (url.includes(BACKEND_PORT) && !response.ok()) {
                const currentUrl = page.url();
                console.error(`❌ Broken API: ${url} (Status: ${response.status()}) on page ${currentUrl}`);
                brokenConnections.push({
                    page: currentUrl,
                    api: url,
                    status: response.status(),
                    statusText: response.statusText(),
                });
            }

            // Check for CORS issues (blocked responses have status 0 or specific headers)
            if (url.includes(BACKEND_PORT)) {
                const headers = response.headers();
                const corsHeader = headers['access-control-allow-origin'];
                // If it's a cross-origin request without proper CORS headers, flag it
                if (response.status() === 0) {
                    corsErrors.push(`CORS blocked: ${url}`);
                }
            }
        });

        // Set up request failure interceptor
        page.on('requestfailed', (request) => {
            const url = request.url();
            if (url.includes(BACKEND_PORT)) {
                const failure = request.failure();
                console.error(`❌ Request Failed: ${url} - ${failure?.errorText || 'Unknown error'}`);

                // Check for CORS-related failures
                if (failure?.errorText?.includes('CORS') || failure?.errorText?.includes('cross-origin')) {
                    corsErrors.push(`CORS failure: ${url} - ${failure.errorText}`);
                }
            }
        });

        // Capture console errors
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                const currentUrl = page.url();
                const text = msg.text();
                console.error(`❌ Console Error on ${currentUrl}: ${text}`);
                consoleErrors.push({
                    page: currentUrl,
                    message: text,
                    type: 'console.error',
                });
            }
        });

        // Capture unhandled JS exceptions
        page.on('pageerror', (error) => {
            const currentUrl = page.url();
            console.error(`❌ JS Exception on ${currentUrl}: ${error.message}`);
            consoleErrors.push({
                page: currentUrl,
                message: error.message,
                type: 'pageerror',
            });
        });

        // Start crawling from the root
        await crawlPage(page, FRONTEND_URL);

        // Print summary
        console.log('\n' + '='.repeat(60));
        console.log('CRAWL SUMMARY');
        console.log('='.repeat(60));
        console.log(`✅ Pages crawled: ${visitedUrls.size}`);
        console.log(`❌ Broken API connections: ${brokenConnections.length}`);
        console.log(`❌ Console errors: ${consoleErrors.length}`);
        console.log(`❌ CORS errors: ${corsErrors.length}`);

        if (brokenConnections.length > 0) {
            console.log('\n--- BROKEN API CONNECTIONS ---');
            for (const conn of brokenConnections) {
                console.log(`  Page: ${conn.page}`);
                console.log(`    API: ${conn.api}`);
                console.log(`    Status: ${conn.status} ${conn.statusText}`);
            }
        }

        if (consoleErrors.length > 0) {
            console.log('\n--- CONSOLE ERRORS ---');
            for (const err of consoleErrors) {
                console.log(`  Page: ${err.page}`);
                console.log(`    Type: ${err.type}`);
                console.log(`    Message: ${err.message.substring(0, 200)}`);
            }
        }

        if (corsErrors.length > 0) {
            console.log('\n--- CORS ERRORS ---');
            for (const err of corsErrors) {
                console.log(`  ${err}`);
            }
        }

        console.log('='.repeat(60));

        // Assert: No broken connections should exist
        expect(
            brokenConnections.filter((c) => c.status >= 500).length,
            'Server errors (5xx) detected in API calls'
        ).toBe(0);
    });
});
