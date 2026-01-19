/**
 * Wait for Services Script
 * Polls backend and frontend until they're ready (with timeout)
 */

const http = require('http');

const MAX_WAIT_TIME = 60000; // 60 seconds
const POLL_INTERVAL = 1000; // 1 second

const SERVICES = [
  {
    name: 'Backend API',
    url: 'http://localhost:8001/health',
    priority: 1,
  },
  {
    name: 'Frontend',
    url: 'http://localhost:3001',
    priority: 2,
  },
];

const COLORS = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m',
  bold: '\x1b[1m',
};

/**
 * Check if service is ready
 * @param {string} url - Service URL
 * @returns {Promise<boolean>} - True if service is ready
 */
function isServiceReady(url) {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout: 3000 }, (res) => {
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

/**
 * Wait for a service to be ready
 * @param {Object} service - Service configuration
 * @returns {Promise<boolean>} - True if service became ready within timeout
 */
async function waitForService(service) {
  const startTime = Date.now();
  let dots = 0;
  
  process.stdout.write(`${COLORS.cyan}⏳ Waiting for ${service.name}${COLORS.reset}`);
  
  while (Date.now() - startTime < MAX_WAIT_TIME) {
    const ready = await isServiceReady(service.url);
    
    if (ready) {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      process.stdout.write(`\r${COLORS.green}✅ ${service.name} is ready${COLORS.reset} (${elapsed}s)          \n`);
      return true;
    }
    
    // Show progress dots
    dots = (dots + 1) % 4;
    const dotsStr = '.'.repeat(dots + 1).padEnd(4);
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    process.stdout.write(`\r${COLORS.cyan}⏳ Waiting for ${service.name}${dotsStr}${COLORS.reset} (${elapsed}s)`);
    
    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
  }
  
  process.stdout.write(`\r${COLORS.red}❌ ${service.name} did not respond within ${MAX_WAIT_TIME / 1000}s${COLORS.reset}          \n`);
  return false;
}

async function main() {
  console.log('\n' + COLORS.bold + '========================================');
  console.log('   CHIMERA SERVICE STARTUP MONITOR');
  console.log('========================================' + COLORS.reset + '\n');
  
  console.log(`${COLORS.yellow}Timeout: ${MAX_WAIT_TIME / 1000} seconds per service${COLORS.reset}\n`);
  
  // Sort services by priority
  const sortedServices = [...SERVICES].sort((a, b) => a.priority - b.priority);
  
  let allReady = true;
  
  for (const service of sortedServices) {
    const ready = await waitForService(service);
    if (!ready) {
      allReady = false;
      break; // Stop checking if a higher priority service fails
    }
  }
  
  console.log('\n----------------------------------------');
  
  if (allReady) {
    console.log(`\n${COLORS.green}${COLORS.bold}✅ All services are ready!${COLORS.reset}\n`);
    console.log('Service URLs:');
    console.log(`  ${COLORS.blue}Backend API:${COLORS.reset}  http://localhost:8001`);
    console.log(`  ${COLORS.blue}Frontend:${COLORS.reset}     http://localhost:3001`);
    console.log(`  ${COLORS.blue}Dashboard:${COLORS.reset}    http://localhost:3001/dashboard`);
    console.log(`  ${COLORS.blue}Jailbreak:${COLORS.reset}    http://localhost:3001/dashboard/jailbreak`);
    console.log(`  ${COLORS.blue}API Docs:${COLORS.reset}     http://localhost:8001/docs\n`);
    process.exit(0);
  } else {
    console.log(`\n${COLORS.red}${COLORS.bold}❌ Service startup failed.${COLORS.reset}`);
    console.log('\nCheck the following:');
    console.log('  1. Are both terminal windows showing the servers?');
    console.log('  2. Are there any error messages in the server terminals?');
    console.log('  3. Run: npm run check:ports to verify port availability');
    console.log('  4. Check CANNOT_GET_ROOT_TROUBLESHOOTING.md for detailed help\n');
    process.exit(1);
  }
}

main().catch(console.error);
