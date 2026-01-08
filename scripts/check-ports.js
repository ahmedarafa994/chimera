/**
 * Port Availability Check Script
 * Verifies that required ports (3000, 8001) are available before starting services
 */

const net = require('net');

const REQUIRED_PORTS = [
  { port: 3000, service: 'Frontend (Next.js)' },
  { port: 8001, service: 'Backend (FastAPI)' },
];

/**
 * Check if a port is available
 * @param {number} port - Port number to check
 * @returns {Promise<boolean>} - True if port is available, false otherwise
 */
function checkPort(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    
    server.once('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        resolve(false);
      } else {
        resolve(false);
      }
    });
    
    server.once('listening', () => {
      server.close();
      resolve(true);
    });
    
    server.listen(port, '127.0.0.1');
  });
}

/**
 * Get process using a specific port (Windows)
 * @param {number} port - Port number
 * @returns {Promise<string>} - Process info or empty string
 */
async function getProcessOnPort(port) {
  const { exec } = require('child_process');
  
  return new Promise((resolve) => {
    exec(`netstat -ano | findstr :${port}`, (error, stdout) => {
      if (error || !stdout) {
        resolve('');
        return;
      }
      
      const lines = stdout.trim().split('\n');
      const listening = lines.find(line => line.includes('LISTENING'));
      if (listening) {
        const parts = listening.trim().split(/\s+/);
        const pid = parts[parts.length - 1];
        
        exec(`tasklist /FI "PID eq ${pid}" /FO CSV /NH`, (err, taskOutput) => {
          if (err || !taskOutput) {
            resolve(`PID: ${pid}`);
            return;
          }
          const processName = taskOutput.split(',')[0]?.replace(/"/g, '') || 'Unknown';
          resolve(`${processName} (PID: ${pid})`);
        });
      } else {
        resolve('');
      }
    });
  });
}

async function main() {
  console.log('\n========================================');
  console.log('   CHIMERA PORT AVAILABILITY CHECK');
  console.log('========================================\n');
  
  let allAvailable = true;
  const results = [];
  
  for (const { port, service } of REQUIRED_PORTS) {
    const available = await checkPort(port);
    
    if (available) {
      results.push({
        port,
        service,
        status: '✅ Available',
        process: '',
      });
    } else {
      allAvailable = false;
      const process = await getProcessOnPort(port);
      results.push({
        port,
        service,
        status: '❌ In Use',
        process: process || 'Unknown process',
      });
    }
  }
  
  // Display results in table format
  console.log('Port    | Service               | Status        | Process');
  console.log('--------|----------------------|---------------|------------------');
  
  for (const { port, service, status, process } of results) {
    const portStr = port.toString().padEnd(7);
    const serviceStr = service.padEnd(21);
    const statusStr = status.padEnd(14);
    console.log(`${portStr} | ${serviceStr} | ${statusStr} | ${process}`);
  }
  
  console.log('');
  
  if (allAvailable) {
    console.log('✅ All required ports are available. Ready to start services!\n');
    process.exit(0);
  } else {
    console.log('❌ Some ports are in use. Please free them before starting services.');
    console.log('\nTo free a port, you can:');
    console.log('  1. Close the application using the port');
    console.log('  2. Run: taskkill /PID <pid> /F');
    console.log('  3. Restart your computer\n');
    process.exit(1);
  }
}

main().catch(console.error);
