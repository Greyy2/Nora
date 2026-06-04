# Grey Frontend Dockerfile - Production Optimized
FROM node:20-alpine AS builder

# Metadata
LABEL maintainer="Grey Optimization Team"
LABEL version="1.0"
LABEL description="Grey React Frontend"

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY . .

# Build for production (optional, can use dev mode)
# RUN npm run build

# Development stage (hot reload)
FROM node:20-alpine

WORKDIR /app

# Copy from builder
# Copy from builder with ownership change
COPY --from=builder --chown=node:node /app/node_modules ./node_modules
COPY --from=builder --chown=node:node /app/package*.json ./
COPY --chown=node:node . .

USER node

# Expose port
EXPOSE 5720

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1:5720 || exit 1

# Start dev server with hot reload
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5720"]
