#!/bin/bash
set -e

# Production deployment script for Reflexion Agent Framework
# This script handles the complete production deployment pipeline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-docker-compose}"
ENVIRONMENT="${ENVIRONMENT:-production}"

echo "🚀 Reflexion Agent Framework - Production Deployment"
echo "=================================================="
echo "Deployment Mode: $DEPLOYMENT_MODE"
echo "Environment: $ENVIRONMENT"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root in production
    if [[ $ENVIRONMENT == "production" && $EUID -eq 0 ]]; then
        warn "Running as root in production is not recommended"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
    fi
    
    # Check Docker Compose
    if [[ $DEPLOYMENT_MODE == "docker-compose" ]] && ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is required but not installed"
    fi
    
    # Check Kubernetes tools
    if [[ $DEPLOYMENT_MODE == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            error "kubectl is required for Kubernetes deployment"
        fi
        
        if ! command -v helm &> /dev/null; then
            warn "Helm is recommended for Kubernetes deployment"
        fi
    fi
    
    # Check Python for local development
    if [[ $DEPLOYMENT_MODE == "local" ]]; then
        if ! command -v python3 &> /dev/null; then
            error "Python 3 is required"
        fi
        
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$python_version >= 3.9" | bc -l) -ne 1 ]]; then
            error "Python 3.9+ is required, found $python_version"
        fi
    fi
    
    info "Prerequisites check completed"
}

# Validate environment variables
validate_environment() {
    log "Validating environment configuration..."
    
    required_vars=()
    
    if [[ $DEPLOYMENT_MODE == "docker-compose" || $DEPLOYMENT_MODE == "kubernetes" ]]; then
        required_vars+=(
            "DB_PASSWORD"
            "REDIS_PASSWORD"
        )
        
        if [[ $DEPLOYMENT_MODE == "kubernetes" ]]; then
            required_vars+=(
                "KUBE_NAMESPACE"
                "DOCKER_REGISTRY"
            )
        fi
    fi
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
    fi
    
    # Set default values
    export DB_PASSWORD="${DB_PASSWORD:-reflexion_secure_password}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-redis_secure_password}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin_secure_password}"
    export KUBE_NAMESPACE="${KUBE_NAMESPACE:-reflexion-production}"
    export DOCKER_REGISTRY="${DOCKER_REGISTRY:-reflexion}"
    
    info "Environment validation completed"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build -f docker/Dockerfile.production -t "${DOCKER_REGISTRY}/reflexion-agent:latest" .
    docker build -f docker/Dockerfile.production -t "${DOCKER_REGISTRY}/reflexion-agent:$(date +%Y%m%d-%H%M%S)" .
    
    info "Docker images built successfully"
}

# Deploy using Docker Compose
deploy_docker_compose() {
    log "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT/docker"
    
    # Create environment file
    cat > .env << EOF
DB_PASSWORD=$DB_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
GRAFANA_PASSWORD=$GRAFANA_PASSWORD
COMPOSE_PROJECT_NAME=reflexion-production
EOF
    
    # Stop existing services
    docker-compose -f docker-compose.production.yml down --volumes --remove-orphans || true
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be healthy
    info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    docker-compose -f docker-compose.production.yml ps
    
    info "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT/kubernetes"
    
    # Create namespace
    kubectl create namespace "$KUBE_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic reflexion-secrets \
        --from-literal=database-url="postgresql://reflexion:$DB_PASSWORD@postgres:5432/reflexion" \
        --from-literal=redis-url="redis://:$REDIS_PASSWORD@redis:6379/0" \
        --namespace="$KUBE_NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ConfigMap
    kubectl apply -f configmap.yaml -n "$KUBE_NAMESPACE"
    
    # Apply PersistentVolumes
    kubectl apply -f pvc.yaml -n "$KUBE_NAMESPACE"
    
    # Apply Services
    kubectl apply -f service.yaml -n "$KUBE_NAMESPACE"
    
    # Apply Deployments
    kubectl apply -f deployment.yaml -n "$KUBE_NAMESPACE"
    
    # Apply Ingress
    kubectl apply -f ingress.yaml -n "$KUBE_NAMESPACE"
    
    # Wait for rollout
    kubectl rollout status deployment/reflexion-api -n "$KUBE_NAMESPACE" --timeout=300s
    kubectl rollout status deployment/reflexion-worker -n "$KUBE_NAMESPACE" --timeout=300s
    
    info "Kubernetes deployment completed"
}

# Deploy locally for development
deploy_local() {
    log "Setting up local development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -e ".[dev]"
    
    # Start local services (Redis, PostgreSQL) with Docker
    docker-compose -f docker-compose.dev.yml up -d postgres redis
    
    # Wait for services
    sleep 10
    
    # Run database migrations (if any)
    # python scripts/migrate.py
    
    info "Local development environment ready"
    info "To start the application: source venv/bin/activate && python examples/basic_usage.py"
}

# Run health checks
run_health_checks() {
    log "Running post-deployment health checks..."
    
    case $DEPLOYMENT_MODE in
        "docker-compose")
            # Check Docker Compose services
            cd "$PROJECT_ROOT/docker"
            if docker-compose -f docker-compose.production.yml ps | grep -q "unhealthy\|Exit"; then
                error "Some services are unhealthy"
            fi
            
            # Run health check script
            docker-compose -f docker-compose.production.yml exec reflexion-api python scripts/healthcheck.py || warn "Health check script failed"
            ;;
            
        "kubernetes")
            # Check pod status
            kubectl get pods -n "$KUBE_NAMESPACE"
            
            # Check if all pods are ready
            kubectl wait --for=condition=ready pod -l app=reflexion-api -n "$KUBE_NAMESPACE" --timeout=300s
            kubectl wait --for=condition=ready pod -l app=reflexion-worker -n "$KUBE_NAMESPACE" --timeout=300s
            ;;
            
        "local")
            # Check if local services are running
            if ! pgrep -f "reflexion" > /dev/null; then
                warn "No reflexion processes found running locally"
            fi
            ;;
    esac
    
    info "Health checks completed"
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    report_file="$PROJECT_ROOT/deployment-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
Reflexion Agent Framework - Deployment Report
=============================================
Deployment Date: $(date)
Deployment Mode: $DEPLOYMENT_MODE
Environment: $ENVIRONMENT
Project Root: $PROJECT_ROOT

System Information:
- OS: $(uname -s)
- Architecture: $(uname -m)
- Docker Version: $(docker --version)
EOF

    case $DEPLOYMENT_MODE in
        "docker-compose")
            echo "" >> "$report_file"
            echo "Docker Compose Services:" >> "$report_file"
            cd "$PROJECT_ROOT/docker"
            docker-compose -f docker-compose.production.yml ps >> "$report_file" 2>/dev/null || echo "Unable to get service status" >> "$report_file"
            ;;
            
        "kubernetes")
            echo "" >> "$report_file"
            echo "Kubernetes Resources:" >> "$report_file"
            kubectl get all -n "$KUBE_NAMESPACE" >> "$report_file" 2>/dev/null || echo "Unable to get Kubernetes resources" >> "$report_file"
            ;;
    esac
    
    echo "" >> "$report_file"
    echo "Deployment completed successfully!" >> "$report_file"
    
    info "Deployment report saved to: $report_file"
}

# Cleanup function for graceful shutdown
cleanup() {
    warn "Deployment interrupted - cleaning up..."
    
    case $DEPLOYMENT_MODE in
        "docker-compose")
            cd "$PROJECT_ROOT/docker"
            docker-compose -f docker-compose.production.yml down --remove-orphans
            ;;
        "kubernetes")
            kubectl delete namespace "$KUBE_NAMESPACE" --ignore-not-found=true
            ;;
    esac
    
    exit 1
}

# Main deployment flow
main() {
    # Set trap for cleanup
    trap cleanup SIGINT SIGTERM
    
    log "Starting Reflexion Agent Framework deployment..."
    
    check_prerequisites
    validate_environment
    
    case $DEPLOYMENT_MODE in
        "docker-compose")
            build_images
            deploy_docker_compose
            ;;
        "kubernetes")
            build_images
            deploy_kubernetes
            ;;
        "local")
            deploy_local
            ;;
        *)
            error "Unknown deployment mode: $DEPLOYMENT_MODE"
            ;;
    esac
    
    run_health_checks
    generate_report
    
    log "🎉 Deployment completed successfully!"
    
    # Print access information
    case $DEPLOYMENT_MODE in
        "docker-compose")
            echo ""
            info "Service URLs:"
            info "- Reflexion API: http://localhost:8000"
            info "- Grafana Dashboard: http://localhost:3000 (admin/$(echo $GRAFANA_PASSWORD))"
            info "- Prometheus: http://localhost:9090"
            ;;
        "kubernetes")
            echo ""
            info "Kubernetes deployment completed in namespace: $KUBE_NAMESPACE"
            info "Use 'kubectl get services -n $KUBE_NAMESPACE' to get service endpoints"
            ;;
        "local")
            echo ""
            info "Local development environment is ready"
            info "Activate with: source venv/bin/activate"
            ;;
    esac
}

# Help function
show_help() {
    echo "Reflexion Agent Framework - Production Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -m, --mode MODE         Deployment mode: docker-compose, kubernetes, local (default: docker-compose)"
    echo "  -e, --environment ENV   Environment: production, staging, development (default: production)"
    echo ""
    echo "Environment Variables:"
    echo "  DB_PASSWORD            Database password (required for docker-compose/kubernetes)"
    echo "  REDIS_PASSWORD         Redis password (required for docker-compose/kubernetes)"
    echo "  GRAFANA_PASSWORD       Grafana admin password (optional)"
    echo "  KUBE_NAMESPACE         Kubernetes namespace (default: reflexion-production)"
    echo "  DOCKER_REGISTRY        Docker registry for images (default: reflexion)"
    echo ""
    echo "Examples:"
    echo "  $0 -m docker-compose -e production"
    echo "  $0 -m kubernetes -e production"
    echo "  $0 -m local -e development"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--mode)
            DEPLOYMENT_MODE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main deployment
main