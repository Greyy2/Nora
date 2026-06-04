#!/bin/bash

# Grey Docker Management - Restart Services
# Usage: ./restart.sh

set -e

echo "🔄 Restarting Grey Backtester..."

cd "$(dirname "$0")"

# Stop services
docker compose down

# Wait a moment
sleep 2

# Start services
docker compose up -d

# Wait for services
sleep 5

echo ""
echo "✅ Grey Backtester restarted!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5720"
