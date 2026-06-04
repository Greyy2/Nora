#!/bin/bash

# Grey Docker Management - Stop Services
# Usage: ./stop.sh

set -e

echo "⏹️  Stopping Grey Backtester..."

cd "$(dirname "$0")"

docker compose down

echo "✅ All services stopped."
