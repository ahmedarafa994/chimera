const https = require('http');

const testLogin = async (username, password) => {
  const data = JSON.stringify({
    username: username,
    password: password
  });

  const options = {
    hostname: 'localhost',
    port: 8005,  // Updated to new backend port
    path: '/api/v1/auth/login',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length,
    },
    timeout: 30000
  };

  return new Promise((resolve, reject) => {
    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => {
        body += chunk;
      });

      res.on('end', () => {
        try {
          const result = JSON.parse(body);
          resolve({
            statusCode: res.statusCode,
            headers: res.headers,
            body: result
          });
        } catch (e) {
          resolve({
            statusCode: res.statusCode,
            headers: res.headers,
            body: body
          });
        }
      });
    });

    req.on('error', (err) => {
      reject(err);
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    req.write(data);
    req.end();
  });
};

const testBackendHealth = async () => {
  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'localhost',
      port: 8005,  // Updated to new backend port
      path: '/health',
      method: 'GET',
      timeout: 5000
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          resolve({ raw: body });
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Timeout'));
    });

    req.end();
  });
};

async function runTests() {
  console.log('🛡️ Chimera Login Test Suite');
  console.log('=' .repeat(50));

  // Test 1: Backend Health
  console.log('\n1️⃣ Testing Backend Health...');
  try {
    const health = await testBackendHealth();
    console.log('✅ Backend is responding');
    console.log('   Status:', health.status || 'Unknown');
    console.log('   Environment:', health.environment || 'Unknown');
  } catch (error) {
    console.log('❌ Backend health check failed:', error.message);
    return;
  }

  // Test 2: Admin Login
  console.log('\n2️⃣ Testing Admin Login...');
  try {
    const result = await testLogin('admin', 'Admin123!@#');

    if (result.statusCode === 200) {
      console.log('✅ Login successful!');
      console.log('   User ID:', result.body.user?.id || 'Unknown');
      console.log('   Username:', result.body.user?.username || 'Unknown');
      console.log('   Role:', result.body.user?.role || 'Unknown');
      console.log('   Token Type:', result.body.token_type);
      console.log('   Access Token:', (result.body.access_token || '').substring(0, 20) + '...');
    } else {
      console.log('❌ Login failed');
      console.log('   Status Code:', result.statusCode);
      console.log('   Response:', JSON.stringify(result.body, null, 2));
    }
  } catch (error) {
    console.log('❌ Login request failed:', error.message);
  }

  // Test 3: Invalid Login
  console.log('\n3️⃣ Testing Invalid Login...');
  try {
    const result = await testLogin('admin', 'wrongpassword');

    if (result.statusCode === 401) {
      console.log('✅ Invalid login properly rejected');
      console.log('   Response:', result.body.detail || result.body.message);
    } else {
      console.log('❌ Unexpected response for invalid login');
      console.log('   Status Code:', result.statusCode);
      console.log('   Response:', JSON.stringify(result.body, null, 2));
    }
  } catch (error) {
    console.log('❌ Invalid login test failed:', error.message);
  }

  console.log('\n' + '=' .repeat(50));
  console.log('🎯 Test suite completed!');
  console.log('\nNext steps:');
  console.log('• Open http://localhost:3001/login in your browser');
  console.log('• Use credentials: admin / Admin123!@#');
  console.log('• Or open test_login.html for interactive testing');
}

// Run the tests
runTests().catch(console.error);