#!/bin/bash

# ==============================================================
# Grey Backtester - Quick Start Script
# ==============================================================
# This script starts all necessary services for local development:
# 1. MongoDB
# 2. Backend (FastAPI)
# 3. Frontend (Vite React)
#
# Usage:
#   ./start.sh [--docker] [--help]
#
# Options:
#   --docker     Use Docker Compose instead of local installation
#   --help       Show this message
# ==============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_DOCKER=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --help|-h)
            head -20 "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Helper Functions
# ============================================================

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    fi
    return 0
}

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
        return $?
    fi

    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
        return $?
    fi

    print_error "Docker Compose is not available. Install Docker Compose v2 or docker-compose."
    return 1
}

wait_for_port() {
    local host=$1
    local port=$2
    local service=$3
    local timeout=30
    local elapsed=0

    print_info "Waiting for $service on $host:$port..."
    
    while [ $elapsed -lt $timeout ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            print_success "$service is ready!"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    print_error "$service failed to start within ${timeout}s"
    return 1
}

# ============================================================
# Docker Mode
# ============================================================

start_docker() {
    print_header "Starting Grey with Docker Compose"
    
    check_command docker || exit 1
    
    cd "$SCRIPT_DIR"
    
    print_info "Starting Docker containers..."
    compose -f docker/docker-compose.yml up -d || exit 1
    
    print_info "Waiting for services to be ready..."
    wait_for_port 127.0.0.1 27020 "MongoDB" || exit 1
    wait_for_port 127.0.0.1 8000 "Backend" || exit 1
    wait_for_port 127.0.0.1 5720 "Frontend" || exit 1
    if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        print_error "Backend health check failed"
        compose -f docker/docker-compose.yml logs --tail=80 backend || true
        exit 1
    fi
    if ! curl -fsS http://127.0.0.1:8000/health/db >/dev/null 2>&1; then
        print_error "Backend database health check failed"
        compose -f docker/docker-compose.yml logs --tail=80 backend || true
        exit 1
    fi
    
    print_header "✅ All services started successfully!"
    
    echo -e "Access the application:"
    echo -e "  ${GREEN}Frontend:${NC} http://localhost:5720/backtest"
    echo -e "  ${GREEN}Backend:${NC} http://localhost:8000"
    echo -e "  ${GREEN}MongoDB:${NC} mongodb://localhost:27020"
    echo -e "\n${YELLOW}Tip:${NC} Run 'docker compose -f docker/docker-compose.yml logs -f' to view logs"
    echo
}

# ============================================================
# Local Mode
# ============================================================

start_local() {
    print_header "Starting Grey Locally"
    
    # Check dependencies
    check_command python3 || exit 1
    check_command node || exit 1
    check_command npm || exit 1
    
    # Check if MongoDB is available (local or Docker)
    if ! check_command mongod && ! docker ps --format '{{.Names}}' | grep -q mongodb; then
        print_warning "MongoDB not found locally or in Docker"
        print_info "To start MongoDB in Docker:"
        echo -e "${YELLOW}docker run -d --name grey-mongodb -p 27020:27017 -v /data/grey/mongodb:/data/db mongo:6.0${NC}"
        exit 1
    fi
    
    # Setup Python environment
    print_info "Setting up Python environment..."
    cd "$SCRIPT_DIR"
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    source venv/bin/activate
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip install -q -r backend/requirements.txt 2>/dev/null || {
        print_error "Failed to install Python dependencies"
        exit 1
    }
    print_success "Python dependencies installed"
    
    # Setup Node dependencies
    print_info "Installing Node dependencies..."
    cd frontend
    npm install --silent 2>/dev/null || {
        print_error "Failed to install Node dependencies"
        exit 1
    }
    print_success "Node dependencies installed"
    cd "$SCRIPT_DIR"
    
    # Create .env.local if not exists
    if [ ! -f "frontend/.env.local" ]; then
        print_info "Creating frontend/.env.local..."
        cat > frontend/.env.local << 'EOF'
LOCAL_DEV=true
VITE_DEBUG=true
EOF
        print_success "frontend/.env.local created"
    fi
    
    # Print connection info
    print_header "✅ Ready to start services!"
    
    echo -e "Open ${GREEN}3 terminal windows${NC} and run:\n"
    
    echo -e "${YELLOW}Terminal 1 (MongoDB):${NC}"
    echo -e "  If using Docker: ${BLUE}docker run -d --name grey-mongodb -p 27020:27017 mongo:6.0${NC}"
    echo -e "  If local: ${BLUE}mongod${NC}\n"
    
    echo -e "${YELLOW}Terminal 2 (Backend):${NC}"
    echo -e "  ${BLUE}cd $SCRIPT_DIR/backend${NC}"
    echo -e "  ${BLUE}source ../venv/bin/activate${NC}"
    echo -e "  ${BLUE}export DEBUG=false${NC}"
    echo -e "  ${BLUE}export MONGO_URI=mongodb://localhost:27020/vinh${NC}"
    echo -e "  ${BLUE}export MONGO_DB=vinh${NC}"
    echo -e "  ${BLUE}uvicorn main:app --reload --host 0.0.0.0 --port 8000${NC}\n"
    
    echo -e "${YELLOW}Terminal 3 (Frontend):${NC}"
    echo -e "  ${BLUE}cd $SCRIPT_DIR/frontend${NC}"
    echo -e "  ${BLUE}npm run dev${NC}\n"
    
    echo -e "Then access: ${GREEN}http://localhost:5721/backtest${NC}\n"
    
    # Offer to use tmux
    if check_command tmux; then
        read -p "Do you want to use tmux to start all services? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            start_with_tmux
        fi
    fi
}

start_with_tmux() {
    print_header "Starting with tmux"
    
    local session_name="grey-dev"
    
    # Kill existing session if it exists
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # Create new session
    tmux new-session -d -s "$session_name" -x 250 -y 50
    
    # Create windows
    tmux new-window -t "$session_name" -n "mongo"
    tmux new-window -t "$session_name" -n "backend"
    tmux new-window -t "$session_name" -n "frontend"
    
    print_info "Starting MongoDB..."
    tmux send-keys -t "$session_name:mongo" "echo '🗄️  Starting MongoDB...'; mongod 2>/dev/null || docker run -it --rm --name grey-mongodb -p 27020:27017 mongo:6.0" Enter
    
    print_info "Starting Backend..."
    cd backend
    tmux send-keys -t "$session_name:backend" "cd '$SCRIPT_DIR/backend' && source ../venv/bin/activate && export DEBUG=false MONGO_URI=mongodb://localhost:27020/vinh MONGO_DB=vinh && echo '🚀 Starting Backend FastAPI...' && uvicorn main:app --reload --host 0.0.0.0 --port 8000" Enter
    
    print_info "Starting Frontend..."
    cd "$SCRIPT_DIR/frontend"
    tmux send-keys -t "$session_name:frontend" "cd '$SCRIPT_DIR/frontend' && echo '⚛️  Starting Frontend Vite...' && npm run dev" Enter
    
    sleep 2
    print_success "tmux session '$session_name' created with 3 windows"
    print_info "Attaching to session..."
    tmux attach-session -t "$session_name"
}

# ============================================================
# Main
# ============================================================

print_header "🚀 Grey Backtester - Quick Start"

if [ "$USE_DOCKER" = true ]; then
    start_docker
else
    start_local
fi
