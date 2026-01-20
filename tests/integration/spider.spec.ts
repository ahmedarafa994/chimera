import { test, expect, Page } from '@playwright/test';

/**
 * Recursive E2E Testing "Spider"
 *
 * This test crawls the application recursively, starting from the root,
 * and verifies connectivity between the frontend (3001) and backend (8001).
 *
 * Features:
 * - Recursive link crawling with visited URL tracking
 * - Stateful authentication (uses storageState from auth.setup.ts)
 * - API call interception for backend requests
 * - CORS failure detection
 * - Console error capture
 * - Page error (JS exception) capture
 * - Summary table output
 */

// =============================================================================
// Configuration
// =============================================================================

const FRONTEND_URL = 'http://localhost:3001';
const BACKEND_PORT = '8001';
const MAX_DEPTH = 5;
const MAX_PAGES = 50;

// =============================================================================
// Types
// =============================================================================

interface ApiCallResult {
    url: string;
    status: number;
    statusText: string;
    ok: boolean;
    isCors: boolean;
    errorText?: string;
}

interface PageResult {
    url: string;
    loadStatus: 'OK' | 'FAILED';
    httpStatus: number;
    apiCalls: ApiCallResult[];
    consoleErrors: string[];
    pageErrors: string[];
}

interface CrawlContext {
    visitedUrls: Set<string>;
    pageResults: PageResult[];
    currentPageApiCalls: ApiCallResult[];
    currentPageConsoleErrors: string[];
    currentPagePageErrors: string[];
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Normalize URL by removing hash and trailing slash
 */
function normalizeUrl(url: string): string {
    try {
        const parsed = new URL(url);
        // Only crawl same-origin pages (frontend)
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
 * Check if URL is internal (same-origin) and not external
 */
function isInternalUrl(href: string, baseUrl: string): boolean {
    try {
        // Handle relative URLs
        const url = new URL(href, baseUrl);
        const base = new URL(baseUrl);

        // Must be same origin
        if (url.origin !== base.origin) {
            return false;
        }

        // Skip non-page URLs
        const skipPatterns = [
            'javascript:',
            'mailto:',
            'tel:',
            '#',
            'data:',
            'blob:',
        ];
        for (const pattern of skipPatterns) {
            if (href.startsWith(pattern)) {
                return false;
            }
        }

        // Skip external domains commonly linked
        const externalDomains = [
            'github.com',
            'twitter.com',
            'x.com',
            'linkedin.com',
            'facebook.com',
            'google.com',
            'googleapis.com',
        ];
        for (const domain of externalDomains) {
            if (url.hostname.includes(domain)) {
                return false;
            }
        }

        return true;
    } catch {
        return false;
    }
}

/**
 * Extract all internal links from the page
 */
async function extractLinks(page: Page, baseUrl: string): Promise<string[]> {
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
        } else if (link.startsWith('./') || link.startsWith('../')) {
            try {
                fullUrl = new URL(link, baseUrl).href;
            } catch {
                continue;
            }
        } else {
            continue;
        }

        if (isInternalUrl(fullUrl, FRONTEND_URL)) {
            const normalized = normalizeUrl(fullUrl);
            if (normalized) {
                normalizedLinks.push(normalized);
            }
        }
    }

    return [...new Set(normalizedLinks)]; // Dedupe
}

/**
 * Print summary table in console and return formatted string
 */
function printSummaryTable(results: PageResult[]): void {
    console.log('\n' + '═'.repeat(80));
    console.log('SPIDER CRAWL SUMMARY');
    console.log('═'.repeat(80));

    // Summary stats
    const totalPages = results.length;
    const failedPages = results.filter((r) => r.loadStatus === 'FAILED').length;
    const pagesWithApiErrors = results.filter((r) =>
        r.apiCalls.some((a) => !a.ok)
    ).length;
    const pagesWithConsoleErrors = results.filter(
        (r) => r.consoleErrors.length > 0
    ).length;
    const pagesWithPageErrors = results.filter(
        (r) => r.pageErrors.length > 0
    ).length;
    const corsErrors = results.reduce(
        (acc, r) => acc + r.apiCalls.filter((a) => a.isCors).length,
        0
    );

    console.log(`\n📊 STATISTICS:`);
    console.log(`  ✅ Pages Crawled: ${totalPages}`);
    console.log(
        `  ${failedPages > 0 ? '❌' : '✅'} Failed Pages: ${failedPages}`
    );
    console.log(
        `  ${pagesWithApiErrors > 0 ? '❌' : '✅'} Pages with API Errors: ${pagesWithApiErrors}`
    );
    console.log(
        `  ${pagesWithConsoleErrors > 0 ? '⚠️' : '✅'} Pages with Console Errors: ${pagesWithConsoleErrors}`
    );
    console.log(
        `  ${pagesWithPageErrors > 0 ? '❌' : '✅'} Pages with JS Exceptions: ${pagesWithPageErrors}`
    );
    console.log(`  ${corsErrors > 0 ? '❌' : '✅'} CORS Errors: ${corsErrors}`);

    // Detailed table
    console.log(`\n📋 DETAILED RESULTS:`);
    console.log('─'.repeat(80));
    console.log(
        '| Page URL'.padEnd(40) +
        '| Load'.padEnd(8) +
        '| API'.padEnd(6) +
        '| Errors |'
    );
    console.log('─'.repeat(80));

    for (const result of results) {
        const shortUrl =
            result.url.length > 35
                ? '...' + result.url.slice(-32)
                : result.url;
        const loadIcon = result.loadStatus === 'OK' ? '✅' : '❌';
        const apiIcon = result.apiCalls.every((a) => a.ok) ? '✅' : '❌';
        const errorCount =
            result.consoleErrors.length + result.pageErrors.length;
        const errorIcon = errorCount === 0 ? '✅' : `⚠️${errorCount}`;

        console.log(
            `| ${shortUrl.padEnd(38)} | ${loadIcon.padEnd(5)} | ${apiIcon.padEnd(3)} | ${errorIcon.padEnd(6)} |`
        );
    }
    console.log('─'.repeat(80));

    // Detailed errors if any
    const pagesWithErrors = results.filter(
        (r) =>
            r.loadStatus === 'FAILED' ||
            r.apiCalls.some((a) => !a.ok) ||
            r.consoleErrors.length > 0 ||
            r.pageErrors.length > 0
    );

    if (pagesWithErrors.length > 0) {
        console.log(`\n🔍 ERROR DETAILS:`);
        for (const result of pagesWithErrors) {
            console.log(`\n  📄 ${result.url}`);

            if (result.loadStatus === 'FAILED') {
                console.log(`     ❌ Page Load Failed (HTTP ${result.httpStatus})`);
            }

            for (const api of result.apiCalls.filter((a) => !a.ok)) {
                const icon = api.isCors ? '🚫 CORS' : '❌ API';
                console.log(
                    `     ${icon}: ${api.url} → ${api.status} ${api.statusText}`
                );
                if (api.errorText) {
                    console.log(`        Error: ${api.errorText}`);
                }
            }

            for (const error of result.consoleErrors.slice(0, 3)) {
                console.log(`     ⚠️ Console: ${error.substring(0, 100)}`);
            }
            if (result.consoleErrors.length > 3) {
                console.log(
                    `     ... and ${result.consoleErrors.length - 3} more console errors`
                );
            }

            for (const error of result.pageErrors.slice(0, 3)) {
                console.log(`     💥 JS Error: ${error.substring(0, 100)}`);
            }
            if (result.pageErrors.length > 3) {
                console.log(
                    `     ... and ${result.pageErrors.length - 3} more JS errors`
                );
            }
        }
    }

    console.log('\n' + '═'.repeat(80));
}

// =============================================================================
// Main Spider Test
// =============================================================================

test.describe('Recursive Spider & Connectivity Tester', () => {
    // Test credentials - can be overridden via environment variables
    const TEST_USERNAME = process.env.TEST_USERNAME || 'admin';
    const TEST_PASSWORD = process.env.TEST_PASSWORD || 'admin';

    test('Crawl all pages and verify API connectivity', async ({ page }) => {
        // ---------------------------------------------------------------------
        // Authenticate first (inline login)
        // ---------------------------------------------------------------------
        console.log('[Spider] Authenticating before crawl...');

        try {
            await page.goto(`${FRONTEND_URL}/login`, { waitUntil: 'networkidle', timeout: 15000 });

            // Check if we're already logged in (redirected to dashboard)
            if (page.url().includes('/dashboard')) {
                console.log('[Spider] Already authenticated!');
            } else {
                // Wait for login form
                const loginForm = await page.waitForSelector('[id="login-username"]', { state: 'visible', timeout: 5000 }).catch(() => null);

                if (loginForm) {
                    await page.fill('[id="login-username"]', TEST_USERNAME);
                    await page.fill('[id="login-password"]', TEST_PASSWORD);
                    await page.click('button[type="submit"]');

                    // Wait for redirect to dashboard
                    await page.waitForURL('**/dashboard**', { timeout: 15000 }).catch(() => {
                        console.log('[Spider] Login redirect timeout - proceeding anyway');
                    });
                    console.log('[Spider] Authentication successful!');
                } else {
                    console.log('[Spider] No login form found - proceeding unauthenticated');
                }
            }
        } catch (error) {
            console.log('[Spider] Auth skipped:', String(error).substring(0, 100));
        }

        // ---------------------------------------------------------------------
        // Initialize Crawl Context
        // ---------------------------------------------------------------------
        const ctx: CrawlContext = {
            visitedUrls: new Set(),
            pageResults: [],
            currentPageApiCalls: [],
            currentPageConsoleErrors: [],
            currentPagePageErrors: [],
        };

        // ---------------------------------------------------------------------
        // Setup Event Listeners
        // ---------------------------------------------------------------------


        // Intercept network responses for backend API calls
        page.on('response', (response) => {
            const url = response.url();

            // Check for backend API calls
            if (url.includes(`:${BACKEND_PORT}`) || url.includes('/api/')) {
                const isOk = response.ok();
                ctx.currentPageApiCalls.push({
                    url,
                    status: response.status(),
                    statusText: response.statusText(),
                    ok: isOk,
                    isCors: false,
                });

                if (!isOk) {
                    console.log(
                        `  ❌ API Error: ${url} → ${response.status()} ${response.statusText()}`
                    );
                }
            }
        });

        // Intercept failed requests (including CORS)
        page.on('requestfailed', (request) => {
            const url = request.url();
            const failure = request.failure();

            if (url.includes(`:${BACKEND_PORT}`) || url.includes('/api/')) {
                const isCors =
                    failure?.errorText?.toLowerCase().includes('cors') ||
                    failure?.errorText?.toLowerCase().includes('cross-origin') ||
                    false;

                ctx.currentPageApiCalls.push({
                    url,
                    status: 0,
                    statusText: 'FAILED',
                    ok: false,
                    isCors,
                    errorText: failure?.errorText,
                });

                console.log(
                    `  ❌ Request Failed: ${url} - ${failure?.errorText || 'Unknown'}`
                );
            }
        });

        // Capture console errors
        page.on('console', (msg) => {
            if (msg.type() === 'error') {
                const text = msg.text();
                ctx.currentPageConsoleErrors.push(text);
            }
        });

        // Capture unhandled JS exceptions
        page.on('pageerror', (error) => {
            ctx.currentPagePageErrors.push(error.message);
            console.log(`  💥 JS Exception: ${error.message.substring(0, 100)}`);
        });

        // ---------------------------------------------------------------------
        // Recursive Crawling Function
        // ---------------------------------------------------------------------

        async function crawlPage(url: string, depth: number = 0): Promise<void> {
            // Guard clauses
            if (depth > MAX_DEPTH) {
                console.log(`  ⏭️ Max depth reached, skipping: ${url}`);
                return;
            }
            if (ctx.visitedUrls.size >= MAX_PAGES) {
                console.log(`  ⏭️ Max pages reached, stopping crawl`);
                return;
            }

            const normalizedUrl = normalizeUrl(url);
            if (!normalizedUrl || ctx.visitedUrls.has(normalizedUrl)) {
                return;
            }

            ctx.visitedUrls.add(normalizedUrl);

            // Reset per-page tracking
            ctx.currentPageApiCalls = [];
            ctx.currentPageConsoleErrors = [];
            ctx.currentPagePageErrors = [];

            console.log(
                `\n[${ctx.visitedUrls.size}/${MAX_PAGES}] 🔍 Crawling (depth=${depth}): ${normalizedUrl}`
            );

            let loadStatus: 'OK' | 'FAILED' = 'OK';
            let httpStatus = 200;

            try {
                // Navigate with networkidle wait strategy
                const response = await page.goto(normalizedUrl, {
                    waitUntil: 'domcontentloaded',
                    timeout: 30000,
                });

                // Wait for API calls to settle (since networkidle may block on WebSocket/polling)
                await page.waitForTimeout(2000);

                if (response) {
                    httpStatus = response.status();
                    if (httpStatus >= 400) {
                        loadStatus = 'FAILED';
                        console.log(`  ❌ Page Error: HTTP ${httpStatus}`);
                    } else {
                        console.log(`  ✅ Page Loaded: HTTP ${httpStatus}`);
                    }
                }

                // Additional wait for any delayed API calls
                await page.waitForTimeout(500);

                // Record page result
                ctx.pageResults.push({
                    url: normalizedUrl,
                    loadStatus,
                    httpStatus,
                    apiCalls: [...ctx.currentPageApiCalls],
                    consoleErrors: [...ctx.currentPageConsoleErrors],
                    pageErrors: [...ctx.currentPagePageErrors],
                });

                // Extract and crawl child links
                const links = await extractLinks(page, normalizedUrl);
                console.log(`  📎 Found ${links.length} internal links`);

                for (const link of links) {
                    if (!ctx.visitedUrls.has(normalizeUrl(link))) {
                        await crawlPage(link, depth + 1);
                    }
                }
            } catch (error) {
                loadStatus = 'FAILED';
                console.log(`  ❌ Navigation Error: ${error}`);

                ctx.pageResults.push({
                    url: normalizedUrl,
                    loadStatus,
                    httpStatus: 0,
                    apiCalls: [...ctx.currentPageApiCalls],
                    consoleErrors: [...ctx.currentPageConsoleErrors],
                    pageErrors: [String(error)],
                });
            }
        }

        // ---------------------------------------------------------------------
        // Start Crawl
        // ---------------------------------------------------------------------

        console.log('\n🕷️ Starting Spider Crawl...');
        console.log(`   Frontend: ${FRONTEND_URL}`);
        console.log(`   Backend Port: ${BACKEND_PORT}`);
        console.log(`   Max Depth: ${MAX_DEPTH}`);
        console.log(`   Max Pages: ${MAX_PAGES}`);

        await crawlPage(FRONTEND_URL);

        // Print summary
        printSummaryTable(ctx.pageResults);

        // ---------------------------------------------------------------------
        // Assertions
        // ---------------------------------------------------------------------

        // Count critical failures
        const serverErrors = ctx.pageResults.reduce(
            (acc, r) => acc + r.apiCalls.filter((a) => a.status >= 500).length,
            0
        );
        const corsErrors = ctx.pageResults.reduce(
            (acc, r) => acc + r.apiCalls.filter((a) => a.isCors).length,
            0
        );
        const failedPages = ctx.pageResults.filter(
            (r) => r.loadStatus === 'FAILED'
        ).length;

        // Assert no server errors (5xx)
        expect(
            serverErrors,
            `Found ${serverErrors} server errors (5xx) in API calls`
        ).toBe(0);

        // Assert no CORS errors
        expect(corsErrors, `Found ${corsErrors} CORS errors`).toBe(0);

        // Assert majority of pages loaded (allow some 404s for dynamic routes)
        const successRate = (ctx.pageResults.length - failedPages) / ctx.pageResults.length;
        expect(
            successRate,
            `Only ${(successRate * 100).toFixed(1)}% of pages loaded successfully`
        ).toBeGreaterThan(0.8);
    });
});
