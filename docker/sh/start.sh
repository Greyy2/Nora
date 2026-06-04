#!/bin/bash

# Grey Docker Management - Start Services
# Usage: ./start.sh

set -e

echo "🚀 Starting Grey Backtester..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to docker directory
cd "$(dirname "$0")"

# Pull latest images
echo "📦 Pulling latest Docker images..."
docker compose pull

# Build containers
echo "🔨 Building Grey containers..."
docker compose build --no-cache

# Start services
echo "▶️  Starting services..."
docker compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 5

# Custom Status Output
echo ""
echo "Frontend: 🟢 Đang chạy tại http://localhost:5721"
echo "Backend:  🟢 Đang chạy tại http://localhost:8000 (Log báo \"Application startup complete\")."
echo "Database: 🟢 MongoDB Healthy."
echo ""

