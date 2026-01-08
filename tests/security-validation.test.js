// P1 Security Vulnerability Validation Tests
// These tests ensure critical security vulnerabilities are remediated

const fs = require('fs');
const path = require('path');

describe('P1 Critical Security Vulnerabilities', () => {

  describe('P1-VULN-001: API Key Exposure', () => {
    test('Project_Chimera .env should not contain exposed API keys', () => {
      const envPath = path.join(__dirname, '../Project_Chimera/.env');
      if (fs.existsSync(envPath)) {
        const envContent = fs.readFileSync(envPath, 'utf8');
        expect(envContent).not.toMatch(/AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4/);
        expect(envContent).not.toMatch(/AIzaSyBFQC_Cc2oLjsLqiP03bchz1GUEUSSyop8/);
      }
    });

    test('backend-api .env should not contain exposed API keys', () => {
      const envPath = path.join(__dirname, '../backend-api/.env');
      if (fs.existsSync(envPath)) {
        const envContent = fs.readFileSync(envPath, 'utf8');
        expect(envContent).not.toMatch(/AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4/);
        expect(envContent).not.toMatch(/AIzaSyBFQC_Cc2oLjsLqiP03bchz1GUEUSSyop8/);
      }
    });

    test('Git repository should not contain .env files', () => {
      const gitignorePath = path.join(__dirname, '../.gitignore');
      if (fs.existsSync(gitignorePath)) {
        const gitignoreContent = fs.readFileSync(gitignorePath, 'utf8');
        expect(gitignoreContent).toMatch(/\.env/);
      }
    });
  });

  describe('P1-VULN-002: Flask Security Configuration', () => {
    test('Flask app should have debug mode disabled', () => {
      const appPath = path.join(__dirname, '../Project_Chimera/app.py');
      if (fs.existsSync(appPath)) {
        const appContent = fs.readFileSync(appPath, 'utf8');
        expect(appContent).not.toMatch(/debug=True/);
        expect(appContent).toMatch(/debug=False/);
      }
    });

    test('Flask app should have authentication middleware', () => {
      const appPath = path.join(__dirname, '../Project_Chimera/app.py');
      if (fs.existsSync(appPath)) {
        const appContent = fs.readFileSync(appPath, 'utf8');
        expect(appContent).toMatch(/@app\.before_request|authentication|api_key/);
      }
    });

    test('Flask app should have rate limiting', () => {
      const appPath = path.join(__dirname, '../Project_Chimera/app.py');
      if (fs.existsSync(appPath)) {
        const appContent = fs.readFileSync(appPath, 'utf8');
        expect(appContent).toMatch(/rate_limit|limiter|throttle/);
      }
    });
  });

  describe('P1-VULN-003: Frontend API Key Security', () => {
    test('Frontend should not expose API keys in client code', () => {
      const apiPath = path.join(__dirname, '../chimera-ui/src/lib/api.ts');
      if (fs.existsSync(apiPath)) {
        const apiContent = fs.readFileSync(apiPath, 'utf8');
        expect(apiContent).not.toMatch(/AIzaSy/);
        expect(apiContent).not.toMatch(/NEXT_PUBLIC_.*API_KEY/);
      }
    });

    test('Environment variables should not contain sensitive keys with NEXT_PUBLIC_ prefix', () => {
      const envPath = path.join(__dirname, '../chimera-ui/.env.local');
      if (fs.existsSync(envPath)) {
        const envContent = fs.readFileSync(envPath, 'utf8');
        const lines = envContent.split('\n');
        const sensitiveKeys = lines.filter(line =>
          line.startsWith('NEXT_PUBLIC_') &&
          (line.includes('API_KEY') || line.includes('SECRET') || line.includes('TOKEN'))
        );
        expect(sensitiveKeys).toHaveLength(0);
      }
    });
  });

  describe('Security Headers and Configuration', () => {
    test('Flask app should implement security headers', () => {
      const appPath = path.join(__dirname, '../Project_Chimera/app.py');
      if (fs.existsSync(appPath)) {
        const appContent = fs.readFileSync(appPath, 'utf8');
        expect(appContent).toMatch(/X-Content-Type-Options|X-Frame-Options|CSP/);
      }
    });

    test('CORS should be properly configured with specific origins', () => {
      const appPath = path.join(__dirname, '../Project_Chimera/app.py');
      if (fs.existsSync(appPath)) {
        const appContent = fs.readFileSync(appPath, 'utf8');
        expect(appContent).not.toMatch(/CORS\(app\)/);
        expect(appContent).toMatch(/origins=\[|resources=\{.*origins/);
      }
    });
  });
});