# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM base as production


# Create non-root user for security
RUN groupadd -r reflexion && useradd -r -g reflexion reflexion

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY examples/ /app/examples/
COPY README.md /app/
COPY pyproject.toml /app/

# Install the package
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.reflexion.core.security import health_checker; \
                   import sys; \
                   result = health_checker.run_health_checks(); \
                   sys.exit(0 if result['overall_status'] == 'healthy' else 1)"

# Security: Switch to non-root user
USER reflexion

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.reflexion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
