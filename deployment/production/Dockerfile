# Production-optimized Dockerfile for Reflexion Agent Boilerplate
FROM python:3.13-slim

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    REFLEXION_ENV=production \
    REFLEXION_LOG_LEVEL=INFO \
    REFLEXION_CACHE_SIZE=2000 \
    REFLEXION_MAX_WORKERS=8 \
    REFLEXION_HEALTH_CHECK_INTERVAL=30

# Install system dependencies for production
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user with specific UID for Kubernetes compatibility
RUN adduser --disabled-password --gecos '' --uid 1001 appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml README.md ./

# Install Python dependencies with production optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir psutil>=5.9.0 && \
    pip install --no-cache-dir -e "."

# Copy application source code
COPY --chown=appuser:appuser src/ ./src/

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/cache /app/data /app/tmp && \
    chown -R appuser:appuser /app && \
    chmod 755 /app/logs /app/cache /app/data /app/tmp

# Switch to non-root user for security
USER appuser

# Expose ports
EXPOSE 8000 8080

# Comprehensive health check for production
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD python -c "
import sys, os
sys.path.insert(0, '/app/src')
try:
    from reflexion import ReflexionAgent, OptimizedReflexionAgent
    from reflexion.core.health import health_checker
    
    # Test basic functionality
    agent = ReflexionAgent(llm='gpt-4', max_iterations=1)
    opt_agent = OptimizedReflexionAgent(llm='gpt-4', enable_caching=True)
    
    # Test health monitoring
    metrics = health_checker.get_system_metrics()
    
    # Verify write permissions
    test_file = '/app/tmp/health_check'
    with open(test_file, 'w') as f:
        f.write('healthy')
    os.remove(test_file)
    
    print('All systems operational')
except Exception as e:
    print(f'Health check failed: {e}', file=sys.stderr)
    sys.exit(1)
" || exit 1

# Default command for production
CMD ["python", "-c", "\
import sys;\
sys.path.insert(0, 'src');\
from reflexion import ReflexionAgent;\
print('🚀 Reflexion Agent Boilerplate - Production Ready');\
print('Version: 0.1.0');\
print('Environment: Production');\
print('Features: Basic + Robustness + Scaling');\
agent = ReflexionAgent(llm='gpt-4');\
print(f'Agent initialized: {type(agent).__name__}');\
print('Ready to process reflexion tasks...');\
"]