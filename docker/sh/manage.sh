#!/bin/bash
# Grey Docker Management Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Grey Optimization Engine - Docker Manager${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    if ! docker compose version > /dev/null 2>&1; then
        print_error "Docker Compose (v2) is not available (try upgrading Docker)"
        exit 1
    fi
    print_success "Docker and Docker Compose (v2) are available"
}

cd "$(dirname "$0")"

# Start services
start() {
    print_header
    print_info "Starting Grey services..."

    docker compose -f ../docker-compose.yml up -d
    
    print_success "Services started"
    print_info "Frontend: http://localhost:5720"
    print_info "Backend:  http://localhost:8000"
    print_info "MongoDB:  localhost:27020"
    
    echo ""
    print_info "View logs: docker compose -f ../docker-compose.yml logs -f"
}

# Stop services
stop() {
    print_info "Stopping Grey services..."
    docker compose -f ../docker-compose.yml down
    print_success "Services stopped"
}

# Restart services
restart() {
    print_info "Restarting Grey services..."
    docker compose -f ../docker-compose.yml restart
    print_success "Services restarted"
}

# View logs
logs() {
    SERVICE=${1:-backend}
    docker compose -f ../docker-compose.yml logs -f $SERVICE
}

# Show status
status() {
    print_header
    print_info "Service Status:"
    docker compose -f ../docker-compose.yml ps
    
    echo ""
    print_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Clean everything
clean() {
    print_info "Cleaning Grey Docker environment..."
    read -p "This will remove all containers, volumes, and cached data. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose -f ../docker-compose.yml down -v
        docker system prune -f
        print_success "Environment cleaned"
    else
        print_info "Clean cancelled"
    fi
}

# Rebuild
rebuild() {
    print_info "Rebuilding Grey services..."
    docker compose -f ../docker-compose.yml down
    docker compose -f ../docker-compose.yml build --no-cache
    docker compose -f ../docker-compose.yml up -d
    print_success "Services rebuilt and started"
}

# Shell access
shell() {
    SERVICE=${1:-backend}
    print_info "Opening shell in $SERVICE..."
    docker compose -f ../docker-compose.yml exec $SERVICE sh
}

# Health check
health() {
    print_header
    print_info "Health Check:"
    
    # Backend
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend is healthy"
    else
        print_error "Backend is not responding"
    fi
    
    # MongoDB
    if docker exec grey-mongodb mongosh --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
        print_success "MongoDB is healthy"
    else
        print_error "MongoDB is not responding"
    fi
    
    # Frontend
    if curl -sf http://localhost:5720 > /dev/null 2>&1; then
        print_success "Frontend is healthy"
    else
        print_error "Frontend is not responding"
    fi
}

# Show help
show_help() {
    print_header
    echo "Usage: ./manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start          Start all services"
    echo "  stop           Stop all services"
    echo "  restart        Restart all services"
    echo "  logs [service] View logs (default: backend)"
    echo "  status         Show service status and resource usage"
    echo "  health         Run health checks"
    echo "  shell [service] Open shell in service (default: backend)"
    echo "  rebuild        Rebuild and restart services"
    echo "  clean          Remove all containers and volumes"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./manage.sh start"
    echo "  ./manage.sh logs backend"
    echo "  ./manage.sh shell mongodb"
}

# Main
check_docker

case "${1:-help}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs $2
        ;;
    status)
        status
        ;;
    health)
        health
        ;;
    shell)
        shell $2
        ;;
    rebuild)
        rebuild
        ;;
    clean)
        clean
        ;;
    help|*)
        show_help
        ;;
esac
