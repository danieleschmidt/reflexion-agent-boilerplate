#!/bin/bash

# Reflexion Agent Boilerplate - Build Script
# Standardized build process for all environments

set -e

# Configuration
PROJECT_NAME="reflexion-agent"
VERSION=${VERSION:-$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])" 2>/dev/null || echo "dev")}
BUILD_TARGET=${BUILD_TARGET:-production}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-""}
PUSH_IMAGE=${PUSH_IMAGE:-false}

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
Reflexion Agent Build Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -t, --target TARGET     Build target (production, development, test) [default: production]
    -v, --version VERSION   Version tag for the image [default: auto-detected]
    -r, --registry REGISTRY Docker registry URL
    -p, --push              Push image to registry after build
    -c, --clean             Clean build (remove cache)
    --no-cache              Build without using Docker cache
    --lint                  Run linting before build
    --test                  Run tests before build
    --security              Run security scan after build

Examples:
    $0                                          # Basic production build
    $0 -t development -v latest                 # Development build
    $0 -t production -r my-registry.com -p      # Build and push to registry
    $0 --lint --test --security                 # Full quality pipeline
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGE=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --lint)
            RUN_LINT=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --security)
            RUN_SECURITY=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
done

# Validate build target
case $BUILD_TARGET in
    production|development|test)
        ;;
    *)
        log_error "Invalid build target: $BUILD_TARGET. Must be one of: production, development, test"
        ;;
esac

# Build configuration
IMAGE_NAME="${PROJECT_NAME}:${VERSION}"
if [ -n "$DOCKER_REGISTRY" ]; then
    FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${IMAGE_NAME}"
else
    FULL_IMAGE_NAME="$IMAGE_NAME"
fi

log_info "Starting build process..."
log_info "Target: $BUILD_TARGET"
log_info "Version: $VERSION"
log_info "Image: $FULL_IMAGE_NAME"

# Create necessary directories
mkdir -p {data,logs,temp,dist}

# Pre-build quality checks
if [ "$RUN_LINT" = true ]; then
    log_info "Running linting..."
    if command -v ruff &> /dev/null; then
        ruff check src/ tests/ || log_error "Linting failed"
        ruff format --check src/ tests/ || log_error "Code formatting check failed"
    else
        log_warning "Ruff not found, skipping linting"
    fi
    
    if command -v mypy &> /dev/null; then
        mypy src/ || log_error "Type checking failed"
    else
        log_warning "MyPy not found, skipping type checking"
    fi
fi

if [ "$RUN_TESTS" = true ]; then
    log_info "Running tests..."
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short || log_error "Tests failed"
    else
        log_warning "Pytest not found, skipping tests"
    fi
fi

# Clean build artifacts if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Cleaning build artifacts..."
    rm -rf dist/ build/ *.egg-info/
    docker system prune -f
fi

# Build Docker image
log_info "Building Docker image..."

# Prepare Docker build args
DOCKER_BUILD_ARGS="--target $BUILD_TARGET"
DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg VERSION=$VERSION"
DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"

if [ "$NO_CACHE" = true ]; then
    DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --no-cache"
fi

# Build the image
docker build $DOCKER_BUILD_ARGS -t "$FULL_IMAGE_NAME" . || log_error "Docker build failed"

# Tag additional versions
if [ "$BUILD_TARGET" = "production" ]; then
    docker tag "$FULL_IMAGE_NAME" "${PROJECT_NAME}:latest"
    if [ -n "$DOCKER_REGISTRY" ]; then
        docker tag "$FULL_IMAGE_NAME" "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest"
    fi
fi

log_success "Docker image built successfully: $FULL_IMAGE_NAME"

# Post-build security scan
if [ "$RUN_SECURITY" = true ]; then
    log_info "Running security scan..."
    
    # Check if Docker Scout is available
    if docker scout version &> /dev/null; then
        docker scout cves "$FULL_IMAGE_NAME" || log_warning "Security scan found issues"
    elif command -v trivy &> /dev/null; then
        trivy image "$FULL_IMAGE_NAME" || log_warning "Trivy scan found issues"
    else
        log_warning "No security scanner available (Docker Scout or Trivy)"
    fi
fi

# Test the built image
log_info "Testing built image..."
docker run --rm "$FULL_IMAGE_NAME" reflexion --version || log_warning "Image test failed"

# Push to registry if requested
if [ "$PUSH_IMAGE" = true ]; then
    if [ -z "$DOCKER_REGISTRY" ]; then
        log_error "Registry not specified, cannot push image"
    fi
    
    log_info "Pushing image to registry..."
    docker push "$FULL_IMAGE_NAME" || log_error "Failed to push image"
    
    if [ "$BUILD_TARGET" = "production" ]; then
        docker push "${DOCKER_REGISTRY}/${PROJECT_NAME}:latest" || log_error "Failed to push latest tag"
    fi
    
    log_success "Image pushed successfully to registry"
fi

# Generate build report
BUILD_REPORT="build-report-$(date +%Y%m%d-%H%M%S).json"
cat > "$BUILD_REPORT" << EOF
{
  "build_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "version": "$VERSION",
  "target": "$BUILD_TARGET",
  "image_name": "$FULL_IMAGE_NAME",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "build_success": true,
  "quality_checks": {
    "linting": $([ "$RUN_LINT" = true ] && echo "true" || echo "false"),
    "testing": $([ "$RUN_TESTS" = true ] && echo "true" || echo "false"),
    "security": $([ "$RUN_SECURITY" = true ] && echo "true" || echo "false")
  },
  "image_pushed": $([ "$PUSH_IMAGE" = true ] && echo "true" || echo "false")
}
EOF

log_success "Build completed successfully!"
log_info "Build report saved to: $BUILD_REPORT"
log_info "Image available: $FULL_IMAGE_NAME"

# Display next steps
echo ""
echo "Next steps:"
echo "  • Run the application: docker run -it $FULL_IMAGE_NAME"
echo "  • Start development: docker-compose --profile dev up"
echo "  • Full deployment: docker-compose --profile full up -d"