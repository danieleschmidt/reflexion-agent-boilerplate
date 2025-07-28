# Multi-stage build for Reflexion Agent Boilerplate
ARG PYTHON_VERSION=3.11

# Build stage
FROM python:${PYTHON_VERSION}-slim as builder

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir build

# Copy source code
COPY . .

# Build the package
RUN python -m build

# Production stage
FROM python:${PYTHON_VERSION}-slim as production

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy built package from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir *.whl && \
    rm *.whl

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import reflexion; print('OK')" || exit 1

# Expose port for web interface (if applicable)
EXPOSE 8000

# Default command
CMD ["python", "-m", "reflexion.cli", "--help"]

# Development stage
FROM production as development

# Switch back to root for dev tools installation
USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black isort flake8 mypy

# Copy source code for development
COPY --chown=appuser:appuser . /app/src/

# Install in development mode
WORKDIR /app/src
RUN pip install -e ".[dev]"

# Switch back to non-root user
USER appuser

WORKDIR /app/src

# Development command
CMD ["bash"]