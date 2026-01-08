/**
 * Health Check Script
 * Checks the health status of backend and frontend services
 */

const http = require('http');

const SERVICES = [
  {
    name: 'Backend API',
    url: 'http://localhost:8001/health',
    expectedStatus: 200,
  },
  {
    name: 'Frontend',
    url: 'http://localhost:3000',
    expectedStatus: 200,
  },
];

const COLORS = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m',
};

/**
 * Check service health
 * @param {Object} service - Service configuration
 * @returns {Promise<Object>} - Health check result
 */
function checkService(service) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    
    const req = http.get(service.url, { timeout: 5000 }, (res) => {
      const responseTime = Date.now() - startTime;
      
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        resolve({
          name: service.name,
          url: service.url,
          status: res.statusCode === service.expectedStatus ? 'healthy' : 'unhealthy',
          statusCode: res.statusCode,
          responseTime,
          body: data.substring(0, 200),
        });
      });
    });
    
    req.on('error', (error) => {
      resolve({
        name: service.name,
        url: service.url,
        status: 'unreachable',
        statusCode: null,
        responseTime: null,
        error: error.message,
      });
    });
    
    req.on('timeout', () => {
      req.destroy();
      resolve({
        name: service.name,
        url: service.url,
        status: 'timeout',
        statusCode: null,
        responseTime: null,
        error: 'Request timed out after 5 seconds',
      });
    });
  });
}

/**
 * Format status with color
 * @param {string} status - Service status
 * @returns {string} - Colored status string
 */
function formatStatus(status) {
  switch (status) {
    case 'healthy':
      return `${COLORS.green}● HEALTHY${COLORS.reset}`;
    case 'unhealthy':
      return `${COLORS.yellow}● UNHEALTHY${COLORS.reset}`;
    case 'unreachable':
      return `${COLORS.red}● UNREACHABLE${COLORS.reset}`;
    case 'timeout':
      return `${COLORS.red}● TIMEOUT${COLORS.reset}`;
    default:
      return `${COLORS.yellow}● UNKNOWN${COLORS.reset}`;
  }
}

async function main() {
  console.log('\n' + COLORS.bold + '========================================');
  console.log('   CHIMERA SERVICE HEALTH CHECK');
  console.log('========================================' + COLORS.reset + '\n');
  
  let allHealthy = true;
  
  for (const service of SERVICES) {
    console.log(`${COLORS.blue}Checking ${service.name}...${COLORS.reset}`);
    const result = await checkService(service);
    
    console.log(`  Status: ${formatStatus(result.status)}`);
    console.log(`  URL: ${result.url}`);
    
    if (result.statusCode !== null) {
      console.log(`  HTTP Status: ${result.statusCode}`);
    }
    
    if (result.responseTime !== null) {
      console.log(`  Response Time: ${result.responseTime}ms`);
    }
    
    if (result.error) {
      console.log(`  ${COLORS.red}Error: ${result.error}${COLORS.reset}`);
    }
    
    if (result.status !== 'healthy') {
      allHealthy = false;
    }
    
    console.log('');
  }
  
  console.log('----------------------------------------');
  
  if (allHealthy) {
    console.log(`\n${COLORS.green}${COLORS.bold}✅ All services are healthy!${COLORS.reset}\n`);
    process.exit(0);
  } else {
    console.log(`\n${COLORS.red}${COLORS.bold}❌ Some services are not healthy.${COLORS.reset}`);
    console.log('\nTroubleshooting tips:');
    console.log('  1. Ensure backend is running: cd backend-api && py run.py');
    console.log('  2. Ensure frontend is running: cd frontend && npm run dev');
    console.log('  3. Check for port conflicts: npm run check:ports');
    console.log('  4. Review logs in the terminal windows\n');
    process.exit(1);
  }
}

main().catch(console.error);
