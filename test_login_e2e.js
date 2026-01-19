const puppeteer = require('puppeteer');

async function testChimeraLogin() {
    console.log('🚀 Starting Chimera Login End-to-End Test');
    console.log('=' .repeat(60));

    let browser;
    let success = false;

    try {
        // Launch browser
        console.log('1️⃣ Launching browser...');
        browser = await puppeteer.launch({
            headless: false, // Set to true for headless testing
            defaultViewport: { width: 1280, height: 720 },
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        const page = await browser.newPage();

        // Enable request interception to monitor API calls
        await page.setRequestInterception(true);
        let loginApiCalled = false;
        let loginApiResponse = null;

        page.on('request', (request) => {
            if (request.url().includes('/api/v1/auth/login')) {
                console.log('🔍 Login API call detected:', request.url());
                loginApiCalled = true;
            }
            request.continue();
        });

        page.on('response', async (response) => {
            if (response.url().includes('/api/v1/auth/login')) {
                console.log('📨 Login API response:', response.status());
                loginApiResponse = response.status();
                try {
                    const text = await response.text();
                    console.log('📄 Response preview:', text.substring(0, 200));
                } catch (e) {
                    console.log('❌ Could not read response body');
                }
            }
        });

        // Navigate to login page
        console.log('2️⃣ Navigating to login page...');
        await page.goto('http://localhost:3001/login', { waitUntil: 'networkidle2', timeout: 30000 });

        // Check if page loaded successfully
        const pageTitle = await page.title();
        console.log('📄 Page title:', pageTitle);

        // Wait for login form to be visible
        console.log('3️⃣ Waiting for login form...');
        await page.waitForSelector('input[type="text"], input[type="email"]', { timeout: 10000 });
        await page.waitForSelector('input[type="password"]', { timeout: 10000 });
        await page.waitForSelector('button[type="submit"]', { timeout: 10000 });

        console.log('✅ Login form elements found');

        // Fill in credentials
        console.log('4️⃣ Filling in admin credentials...');
        await page.type('input[type="text"], input[type="email"]', 'admin');
        await page.type('input[type="password"]', 'Admin123!@#');

        console.log('✅ Credentials entered');

        // Submit the form
        console.log('5️⃣ Submitting login form...');
        const submitButton = await page.$('button[type="submit"]');
        await submitButton.click();

        // Wait for either redirect or error message
        console.log('6️⃣ Waiting for login response...');

        try {
            // Wait for navigation or error message (with longer timeout for slow API)
            await Promise.race([
                page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 60000 }),
                page.waitForSelector('.error, [role="alert"], .alert-destructive', { timeout: 60000 }),
                page.waitForFunction(() => window.location.pathname !== '/login', { timeout: 60000 })
            ]);

            const currentUrl = page.url();
            console.log('🌐 Current URL after login attempt:', currentUrl);

            if (currentUrl.includes('/dashboard') || currentUrl.includes('/admin') || !currentUrl.includes('/login')) {
                console.log('✅ Login successful! Redirected to:', currentUrl);
                success = true;

                // Check for user info or admin elements
                try {
                    await page.waitForSelector('[data-testid="user-menu"], .user-menu, .dashboard', { timeout: 5000 });
                    console.log('✅ Dashboard elements loaded');
                } catch (e) {
                    console.log('⚠️ Dashboard elements not found, but redirect succeeded');
                }

            } else {
                // Check for error messages
                const errorElement = await page.$('.error, [role="alert"], .alert-destructive');
                if (errorElement) {
                    const errorText = await page.evaluate(el => el.textContent, errorElement);
                    console.log('❌ Login failed with error:', errorText);
                } else {
                    console.log('❌ Login failed - still on login page');
                }
            }

        } catch (timeoutError) {
            console.log('⏰ Login attempt timed out after 60 seconds');
            console.log('🔍 API call made:', loginApiCalled);
            console.log('📊 API response status:', loginApiResponse);

            // Check current state
            const currentUrl = page.url();
            console.log('🌐 Final URL:', currentUrl);

            // Check for any error messages
            const errorMessages = await page.$$eval('.error, [role="alert"], .alert-destructive',
                elements => elements.map(el => el.textContent));
            if (errorMessages.length > 0) {
                console.log('❌ Error messages found:', errorMessages);
            }
        }

        // Take a screenshot for debugging
        console.log('📸 Taking screenshot...');
        await page.screenshot({ path: 'login_test_result.png', fullPage: true });
        console.log('✅ Screenshot saved as login_test_result.png');

    } catch (error) {
        console.log('❌ Test failed with error:', error.message);
        console.log('🔧 Full error:', error);
    } finally {
        if (browser) {
            await browser.close();
        }
    }

    console.log('\n' + '=' .repeat(60));
    console.log('🏁 Test Summary:');
    console.log(`Status: ${success ? '✅ SUCCESS' : '❌ FAILED'}`);
    console.log('📋 Manual test instructions:');
    console.log('   1. Open http://localhost:3001/login');
    console.log('   2. Enter username: admin');
    console.log('   3. Enter password: Admin123!@#');
    console.log('   4. Click Sign In');
    console.log('=' .repeat(60));

    return success;
}

// Only run if puppeteer is available
async function main() {
    try {
        await testChimeraLogin();
    } catch (error) {
        if (error.message.includes('puppeteer')) {
            console.log('ℹ️ Puppeteer not available for automated testing');
            console.log('📋 Please test manually:');
            console.log('   1. Open http://localhost:3001/login in your browser');
            console.log('   2. Enter username: admin');
            console.log('   3. Enter password: Admin123!@#');
            console.log('   4. Click Sign In');
            console.log('   5. Verify you are redirected to the dashboard');
        } else {
            console.error('❌ Test failed:', error.message);
        }
    }
}

if (require.main === module) {
    main();
}

module.exports = { testChimeraLogin };