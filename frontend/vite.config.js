import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

/**
 * Frontend-Backend Connection Setup
 * 
 * Environment Detection:
 * - Local Dev: Backend running on http://localhost:8000
 * - Docker: Backend running on http://backend:8000 (service name in grey-network)
 * - Production: Use VITE_API_URL environment variable
 * 
 * Priority:
 * 1. VITE_API_URL env var (highest priority)
 * 2. Docker detection via NODE_ENV (if 'production' in Docker container)
 * 3. Default to localhost:8000 for local dev
 */

// Detect backend target based on environment
const getBackendTarget = () => {
    // If VITE_API_URL is explicitly set, use it (Docker Compose sets this)
    if (process.env.VITE_API_URL) {
        console.log(`[Vite] Using VITE_API_URL: ${process.env.VITE_API_URL}`);
        return process.env.VITE_API_URL;
    }
    
    // If running inside Docker container (NODE_ENV=production OR in docker-compose)
    // Docker network: backend service is available as 'backend:8000'
    const isDockerContainer = process.env.DOCKER_BACKEND === 'true' || 
                            process.env.NODE_ENV === 'production' &&
                            !process.env.LOCAL_DEV;
    
    if (isDockerContainer) {
        console.log('[Vite] Docker environment detected, using http://backend:8000');
        return 'http://backend:8000';
    }
    
    // Default to localhost for local development
    const localBackend = 'http://localhost:8000';
    console.log(`[Vite] Local dev environment, using ${localBackend}`);
    return localBackend;
};

const BACKEND_TARGET = getBackendTarget();

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
            '@client': path.resolve(__dirname, './src/pages/Client'),
        },
    },
    server: {
        host: '0.0.0.0',
        port: 5721,
        strictPort: true,
        allowedHosts: [
            'bt.noracapitalgroup.com',
            'localhost',
            '127.0.0.1'
        ],
        hmr: {
            clientPort: 443,
            protocol: 'wss'
        },
        proxy: {
            // WebSocket proxy for QuantaAlpha mining WS (must come before /api)
            '/api/ai/quanta/v2': {
                target: BACKEND_TARGET,
                changeOrigin: true,
                secure: false,
                ws: true,
                logLevel: 'debug',
            },
            '/api/ai/quanta': {
                target: BACKEND_TARGET,
                changeOrigin: true,
                secure: false,
                logLevel: 'debug',
            },
            '/api': {
                target: BACKEND_TARGET,
                changeOrigin: true,
                secure: false,
                logLevel: 'debug',
            },
            '/vinh': {
                // Static results mount point
                target: BACKEND_TARGET,
                changeOrigin: true,
                secure: false,
            }
        }
    }
})
