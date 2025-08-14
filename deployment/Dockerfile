# Multi-stage production Dockerfile for Reflexion Agent
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY README.md ./

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -e ".[dev,all]"

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R appuser:appuser /app
USER appuser

# Expose development ports
EXPOSE 8000 8001

CMD ["python", "-m", "reflexion.api.main", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Testing stage
FROM development as testing

# Switch back to root for test setup
USER root

# Install additional test dependencies
RUN pip install pytest-xdist pytest-cov pytest-benchmark

# Run tests
RUN python -m pytest tests/ -v --cov=src/reflexion --cov-report=xml --cov-report=term-missing

# Production stage
FROM base as production

# Install only production dependencies
RUN pip install --no-deps /app

# Copy only necessary source files
COPY src/ /app/src/
COPY deployment/entrypoint.sh /app/entrypoint.sh
COPY deployment/healthcheck.py /app/healthcheck.py

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/models /app/tmp

# Make scripts executable
RUN chmod +x /app/entrypoint.sh

# Security hardening
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py

# Expose production ports
EXPOSE 8000 8001

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "reflexion.api.main", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
