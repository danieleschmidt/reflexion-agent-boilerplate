#!/bin/bash

# Reflexion Agent Boilerplate - Deployment Script
# Automated deployment for different environments

set -e

# Configuration
PROJECT_NAME="reflexion-agent"
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-development}
COMPOSE_PROFILE=${COMPOSE_PROFILE:-full}
BACKUP_BEFORE_DEPLOY=${BACKUP_BEFORE_DEPLOY:-true}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-300}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Help function
show_help() {
    cat << EOF
Reflexion Agent Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy          Deploy the application
    stop            Stop the application
    restart         Restart the application
    status          Show deployment status
    logs            Show application logs
    backup          Create backup
    restore         Restore from backup
    health          Check application health

Options:
    -h, --help                  Show this help message
    -e, --env ENVIRONMENT       Deployment environment (development, staging, production)
    -p, --profile PROFILE       Docker Compose profile (dev, full, monitoring)
    --no-backup                 Skip backup before deployment
    --wait SECONDS              Health check timeout [default: 300]
    --force                     Force deployment without confirmations

Examples:
    $0 deploy                           # Deploy with default settings
    $0 -e production deploy             # Production deployment
    $0 -p monitoring deploy             # Deploy with monitoring
    $0 restart                          # Restart services
    $0 logs reflexion-app               # Show app logs
EOF
}

# Parse command line arguments
COMMAND=""
FORCE_DEPLOY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        -p|--profile)
            COMPOSE_PROFILE="$2"
            shift 2
            ;;
        --no-backup)
            BACKUP_BEFORE_DEPLOY=false
            shift
            ;;
        --wait)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        deploy|stop|restart|status|logs|backup|restore|health)
            COMMAND="$1"
            shift
            break
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
done

# Validate environment
case $DEPLOYMENT_ENV in
    development|staging|production)
        ;;
    *)
        log_error "Invalid environment: $DEPLOYMENT_ENV. Must be one of: development, staging, production"
        ;;
esac

# Load environment-specific configuration
if [ -f ".env.${DEPLOYMENT_ENV}" ]; then
    log_info "Loading environment configuration: .env.${DEPLOYMENT_ENV}"
    source ".env.${DEPLOYMENT_ENV}"
elif [ -f ".env" ]; then
    log_info "Loading default environment configuration: .env"
    source ".env"
else
    log_warning "No environment configuration found"
fi

# Deployment functions
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check if Docker is running
    if ! docker version &> /dev/null; then
        log_error "Docker is not running or not installed"
    fi
    
    # Check if docker-compose is available
    if ! docker-compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
    fi
    
    # Check if required files exist
    local required_files=("docker-compose.yml" "Dockerfile")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file missing: $file"
        fi
    done
    
    # Check environment variables
    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD")
        for var in "${required_vars[@]}"; do
            if [ -z "${!var}" ]; then
                log_warning "Environment variable $var is not set"
            fi
        done
    fi
    
    log_success "Prerequisites check completed"
}

create_backup() {
    if [ "$BACKUP_BEFORE_DEPLOY" = false ]; then
        log_info "Skipping backup as requested"
        return
    fi
    
    log_info "Creating backup before deployment..."
    
    local backup_dir="backups/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database if running
    if docker-compose ps postgres | grep -q "Up"; then
        log_info "Backing up PostgreSQL database..."
        docker-compose exec -T postgres pg_dump -U reflexion reflexion > "$backup_dir/postgres_backup.sql"
    fi
    
    # Backup Redis if running
    if docker-compose ps redis | grep -q "Up"; then
        log_info "Backing up Redis data..."
        docker-compose exec -T redis redis-cli BGSAVE
        docker cp reflexion-redis:/data/dump.rdb "$backup_dir/redis_backup.rdb"
    fi
    
    # Backup application data
    if [ -d "data" ]; then
        log_info "Backing up application data..."
        tar -czf "$backup_dir/app_data.tar.gz" data/
    fi
    
    # Backup configuration
    tar -czf "$backup_dir/config.tar.gz" .env* docker-compose.yml
    
    log_success "Backup created in: $backup_dir"
    echo "$backup_dir" > .last_backup
}

deploy_application() {
    log_info "Starting deployment to $DEPLOYMENT_ENV environment..."
    
    # Confirmation for production
    if [ "$DEPLOYMENT_ENV" = "production" ] && [ "$FORCE_DEPLOY" = false ]; then
        echo -n "Are you sure you want to deploy to PRODUCTION? (yes/no): "
        read -r confirmation
        if [ "$confirmation" != "yes" ]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    check_prerequisites
    create_backup
    
    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose --profile "$COMPOSE_PROFILE" pull
    
    # Build application image
    log_info "Building application image..."
    docker-compose build reflexion-app
    
    # Deploy services
    log_info "Deploying services with profile: $COMPOSE_PROFILE"
    docker-compose --profile "$COMPOSE_PROFILE" up -d
    
    # Wait for services to be healthy
    wait_for_health
    
    log_success "Deployment completed successfully!"
}

stop_application() {
    log_info "Stopping application..."
    docker-compose --profile "$COMPOSE_PROFILE" down
    log_success "Application stopped"
}

restart_application() {
    log_info "Restarting application..."
    docker-compose --profile "$COMPOSE_PROFILE" restart
    wait_for_health
    log_success "Application restarted"
}

show_status() {
    log_info "Application status:"
    docker-compose ps
    
    echo ""
    log_info "Service health:"
    docker-compose --profile "$COMPOSE_PROFILE" exec reflexion-app reflexion --version 2>/dev/null || echo "Reflexion app: NOT READY"
    
    if docker-compose ps postgres | grep -q "Up"; then
        docker-compose exec postgres pg_isready -U reflexion || echo "PostgreSQL: NOT READY"
    fi
    
    if docker-compose ps redis | grep -q "Up"; then
        docker-compose exec redis redis-cli ping || echo "Redis: NOT READY"
    fi
}

show_logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        log_info "Showing logs for service: $service"
        docker-compose logs -f "$service"
    else
        log_info "Showing logs for all services"
        docker-compose logs -f
    fi
}

wait_for_health() {
    log_info "Waiting for services to be healthy (timeout: ${HEALTH_CHECK_TIMEOUT}s)..."
    
    local elapsed=0
    local interval=5
    
    while [ $elapsed -lt $HEALTH_CHECK_TIMEOUT ]; do
        local healthy=true
        
        # Check if main app is responding
        if ! docker-compose exec -T reflexion-app reflexion --version &> /dev/null; then
            healthy=false
        fi
        
        # Check database if running
        if docker-compose ps postgres | grep -q "Up"; then
            if ! docker-compose exec -T postgres pg_isready -U reflexion &> /dev/null; then
                healthy=false
            fi
        fi
        
        # Check Redis if running
        if docker-compose ps redis | grep -q "Up"; then
            if ! docker-compose exec -T redis redis-cli ping &> /dev/null; then
                healthy=false
            fi
        fi
        
        if [ "$healthy" = true ]; then
            log_success "All services are healthy!"
            return 0
        fi
        
        echo -n "."
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    
    echo ""
    log_error "Health check timeout after ${HEALTH_CHECK_TIMEOUT} seconds"
}

check_health() {
    log_info "Performing health check..."
    
    local exit_code=0
    
    # Check main application
    if docker-compose exec -T reflexion-app reflexion --version &> /dev/null; then
        log_success "Reflexion app: HEALTHY"
    else
        log_error "Reflexion app: UNHEALTHY"
        exit_code=1
    fi
    
    # Check database
    if docker-compose ps postgres | grep -q "Up"; then
        if docker-compose exec -T postgres pg_isready -U reflexion &> /dev/null; then
            log_success "PostgreSQL: HEALTHY"
        else
            log_error "PostgreSQL: UNHEALTHY"
            exit_code=1
        fi
    fi
    
    # Check Redis
    if docker-compose ps redis | grep -q "Up"; then
        if docker-compose exec -T redis redis-cli ping &> /dev/null; then
            log_success "Redis: HEALTHY"
        else
            log_error "Redis: UNHEALTHY"
            exit_code=1
        fi
    fi
    
    exit $exit_code
}

# Execute command
case $COMMAND in
    deploy)
        deploy_application
        ;;
    stop)
        stop_application
        ;;
    restart)
        restart_application
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$@"
        ;;
    backup)
        BACKUP_BEFORE_DEPLOY=true
        create_backup
        ;;
    health)
        check_health
        ;;
    *)
        log_error "Unknown command: $COMMAND. Use -h for help."
        ;;
esac