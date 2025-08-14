#!/bin/bash

# Production entrypoint script for Reflexion Agent
# Handles initialization, configuration, and graceful startup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# Configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
WORKERS="${WORKERS:-4}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-100}"
ENABLE_QUANTUM_ENHANCEMENT="${ENABLE_QUANTUM_ENHANCEMENT:-true}"
ENABLE_AUTONOMOUS_SCALING="${ENABLE_AUTONOMOUS_SCALING:-true}"
ENABLE_ADVANCED_MONITORING="${ENABLE_ADVANCED_MONITORING:-true}"
SECURITY_STRICT_MODE="${SECURITY_STRICT_MODE:-true}"
RATE_LIMIT_REQUESTS_PER_HOUR="${RATE_LIMIT_REQUESTS_PER_HOUR:-1000}"
OPENTELEMETRY_ENABLED="${OPENTELEMETRY_ENABLED:-false}"

# Health check URL
HEALTH_CHECK_URL="http://localhost:8001/health"

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received termination signal, initiating graceful shutdown..."
    
    # Send termination signal to main process if it exists
    if [[ -n "${MAIN_PID:-}" ]]; then
        log_info "Stopping main application process (PID: $MAIN_PID)"
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown with timeout
        local timeout=30
        local count=0
        
        while kill -0 "$MAIN_PID" 2>/dev/null && [[ $count -lt $timeout ]]; do
            sleep 1
            ((count++))
        done
        
        # Force kill if still running
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            log_warn "Force killing main process after ${timeout}s timeout"
            kill -KILL "$MAIN_PID" 2>/dev/null || true
        fi
    fi
    
    log_info "Graceful shutdown completed"
    exit 0
}

# Trap signals
trap cleanup SIGTERM SIGINT SIGQUIT

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check Python version
    local python_version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log_debug "Python version: $python_version"
    
    # Check available memory
    local available_memory
    if command -v free >/dev/null 2>&1; then
        available_memory=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
        log_debug "Memory usage: $available_memory"
    fi
    
    # Check disk space
    local disk_usage
    disk_usage=$(df -h /app | awk 'NR==2 {print $5}')
    log_debug "Disk usage: $disk_usage"
    
    # Check required directories
    local required_dirs=("/app/logs" "/app/cache" "/app/models" "/app/tmp")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Set proper permissions
    chmod 755 /app/logs /app/cache /app/models /app/tmp
    
    log_info "Pre-flight checks completed successfully"
}

# Initialize configuration
initialize_config() {
    log_info "Initializing configuration for environment: $ENVIRONMENT"
    
    # Create runtime configuration
    cat > /app/runtime_config.json << EOF
{
    "environment": "$ENVIRONMENT",
    "log_level": "$LOG_LEVEL",
    "workers": $WORKERS,
    "max_concurrent_requests": $MAX_CONCURRENT_REQUESTS,
    "quantum_enhancement": $ENABLE_QUANTUM_ENHANCEMENT,
    "autonomous_scaling": $ENABLE_AUTONOMOUS_SCALING,
    "advanced_monitoring": $ENABLE_ADVANCED_MONITORING,
    "security_strict_mode": $SECURITY_STRICT_MODE,
    "rate_limit_requests_per_hour": $RATE_LIMIT_REQUESTS_PER_HOUR,
    "opentelemetry_enabled": $OPENTELEMETRY_ENABLED,
    "startup_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    log_debug "Runtime configuration written to /app/runtime_config.json"
}

# Wait for dependencies
wait_for_dependencies() {
    log_info "Waiting for dependencies to be ready..."
    
    # Wait for Redis if configured
    if [[ -n "${REDIS_HOST:-}" ]]; then
        local redis_host="${REDIS_HOST:-redis}"
        local redis_port="${REDIS_PORT:-6379}"
        
        log_info "Waiting for Redis at $redis_host:$redis_port"
        timeout 60 bash -c "until nc -z $redis_host $redis_port; do sleep 1; done" || {
            log_warn "Redis connection timeout - continuing without Redis"
        }
    fi
    
    # Wait for PostgreSQL if configured
    if [[ -n "${POSTGRES_HOST:-}" ]]; then
        local postgres_host="${POSTGRES_HOST:-postgres}"
        local postgres_port="${POSTGRES_PORT:-5432}"
        
        log_info "Waiting for PostgreSQL at $postgres_host:$postgres_port"
        timeout 60 bash -c "until nc -z $postgres_host $postgres_port; do sleep 1; done" || {
            log_warn "PostgreSQL connection timeout - continuing without database"
        }
    fi
    
    log_info "Dependency checks completed"
}

# Health check function
run_health_check() {
    log_debug "Running health check..."
    
    if command -v curl >/dev/null 2>&1; then
        if curl -f -s "$HEALTH_CHECK_URL" > /dev/null; then
            log_debug "Health check passed"
            return 0
        else
            log_debug "Health check failed"
            return 1
        fi
    else
        log_debug "curl not available, skipping health check"
        return 0
    fi
}

# Start health check server in background
start_health_server() {
    log_info "Starting health check server on port 8001"
    
    cat > /app/health_server.py << 'EOF'
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "uptime_seconds": int(time.time() - start_time),
                "version": "1.0.0"
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress access logs

start_time = time.time()
server = HTTPServer(('0.0.0.0', 8001), HealthHandler)
server.serve_forever()
EOF
    
    python /app/health_server.py &
    local health_server_pid=$!
    
    # Wait for health server to start
    sleep 2
    
    if kill -0 "$health_server_pid" 2>/dev/null; then
        log_info "Health check server started successfully (PID: $health_server_pid)"
    else
        log_warn "Health check server failed to start"
    fi
}

# Main startup sequence
main() {
    log_info "Starting Reflexion Agent - Production Environment"
    log_info "Version: $(cat /app/src/reflexion/__init__.py | grep __version__ | cut -d'"' -f2 2>/dev/null || echo 'unknown')"
    
    # Run initialization steps
    preflight_checks
    initialize_config
    wait_for_dependencies
    start_health_server
    
    log_info "Initialization completed, starting main application..."
    
    # Export configuration as environment variables
    export REFLEXION_CONFIG_FILE="/app/runtime_config.json"
    export REFLEXION_LOG_LEVEL="$LOG_LEVEL"
    export REFLEXION_ENVIRONMENT="$ENVIRONMENT"
    
    # Start the main application
    if [[ "$#" -eq 0 ]]; then
        # Default command
        log_info "Starting with default configuration: $WORKERS workers"
        exec python -m src.reflexion.api.main \
            --host 0.0.0.0 \
            --port 8000 \
            --workers "$WORKERS" &
    else
        # Custom command
        log_info "Starting with custom command: $*"
        exec "$@" &
    fi
    
    MAIN_PID=$!
    log_info "Main application started (PID: $MAIN_PID)"
    
    # Wait for the main process
    wait $MAIN_PID
    
    local exit_code=$?
    log_info "Main application exited with code: $exit_code"
    exit $exit_code
}

# Run main function with all arguments
main "$@"