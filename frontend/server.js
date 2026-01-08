/**
 * Custom Next.js Production Server
 *
 * This is an OPTIONAL custom server for production deployments.
 * It provides:
 * - Custom request logging
 * - Health check endpoint
 * - SPA fallback routing
 * - Graceful shutdown
 *
 * Usage:
 *   1. Build: npm run build
 *   2. Start: node server.js
 *
 * Note: For most deployments, the default `npm start` is sufficient.
 * Use this custom server only if you need the additional features.
 */

/* eslint-disable @typescript-eslint/no-require-imports */
/* eslint-disable no-console */
/* eslint-disable no-restricted-properties */

const { createServer } = require("http");
const { parse } = require("url");
const next = require("next");

const dev = process.env.NODE_ENV !== "production";
const hostname = process.env.HOSTNAME || "0.0.0.0";
const port = parseInt(process.env.PORT || "3000", 10);

// Initialize Next.js app
const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

// Request logging
const logRequest = (req, statusCode, duration) => {
  const timestamp = new Date().toISOString();
  const method = req.method;
  const url = req.url;
  const statusColor = statusCode >= 400 ? "\x1b[31m" : "\x1b[32m";
  const reset = "\x1b[0m";
  
  console.log(
    `${timestamp} | ${method.padEnd(6)} | ${statusColor}${statusCode}${reset} | ${duration}ms | ${url}`
  );
};

app.prepare().then(() => {
  const server = createServer(async (req, res) => {
    const startTime = Date.now();
    
    try {
      const parsedUrl = parse(req.url, true);
      const { pathname } = parsedUrl;
      
      // Custom health check endpoint
      if (pathname === "/health" || pathname === "/_health") {
        res.statusCode = 200;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({
          status: "healthy",
          timestamp: new Date().toISOString(),
          uptime: process.uptime(),
          environment: process.env.NODE_ENV || "development",
        }));
        logRequest(req, 200, Date.now() - startTime);
        return;
      }
      
      // Intercept response to log status
      const originalEnd = res.end;
      res.end = function(...args) {
        logRequest(req, res.statusCode, Date.now() - startTime);
        return originalEnd.apply(this, args);
      };
      
      await handle(req, res, parsedUrl);
      
    } catch (err) {
      console.error("Server error:", err);
      res.statusCode = 500;
      res.end("Internal Server Error");
      logRequest(req, 500, Date.now() - startTime);
    }
  });
  
  // CRITICAL: Handle WebSocket upgrade for HMR
  server.on('upgrade', (req, socket, head) => {
    // Forward WebSocket upgrade to Next.js handler
    handle(req, socket, head);
  });
  
  // Graceful shutdown
  const shutdown = (signal) => {
    console.log(`\n${signal} received. Shutting down gracefully...`);
    server.close(() => {
      console.log("Server closed.");
      process.exit(0);
    });
    
    // Force shutdown after 10 seconds
    setTimeout(() => {
      console.error("Forced shutdown after timeout.");
      process.exit(1);
    }, 10000);
  };
  
  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));
  
  server.listen(port, hostname, (err) => {
    if (err) throw err;
    console.log("");
    console.log("╔════════════════════════════════════════╗");
    console.log("║   Chimera Frontend Server Started      ║");
    console.log("╠════════════════════════════════════════╣");
    console.log(`║   Mode: ${(dev ? "Development" : "Production").padEnd(28)} ║`);
    console.log(`║   URL:  http://${hostname}:${port}`.padEnd(41) + " ║");
    console.log("╚════════════════════════════════════════╝");
    console.log("");
  });
});
