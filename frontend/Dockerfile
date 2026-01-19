# =============================================================================
# Frontend Development Dockerfile
# Optimized for development with hot-reloading
# =============================================================================
# 
# Usage:
#   docker build -t chimera-frontend-dev -f Dockerfile .
#   docker run -p 3000:3000 -v $(pwd):/app -v /app/node_modules chimera-frontend-dev
#
# For production builds, use Dockerfile.prod instead.
# =============================================================================

FROM node:18-alpine

# Set working directory
WORKDIR /app

# Set development environment
ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1

# Copy package files first for better caching
COPY package*.json ./

# Install all dependencies (including devDependencies)
RUN npm install

# Copy source files
COPY . .

# Expose the development server port
EXPOSE 3001

# Set environment variable for dev server binding
ENV HOSTNAME="0.0.0.0"

# Health check for development server
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3001 || exit 1

# Start the development server
CMD ["npm", "run", "dev"]