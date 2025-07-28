FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for development
RUN pip install --upgrade pip setuptools wheel

# Install pre-commit and other dev tools
RUN pip install pre-commit black flake8 isort pylint pytest pytest-cov

# Set up working directory
WORKDIR /workspaces/reflexion-agent-boilerplate

# Copy requirements first for better caching
COPY requirements*.txt ./
RUN pip install -r requirements.txt || echo "No requirements.txt found"

# Install the package in development mode
COPY . .
RUN pip install -e ".[dev]" || echo "Package installation skipped"

# Set up git configuration for devcontainer
RUN git config --global --add safe.directory /workspaces/reflexion-agent-boilerplate