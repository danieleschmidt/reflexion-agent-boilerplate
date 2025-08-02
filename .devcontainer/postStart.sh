#!/bin/bash

# Reflexion Agent Boilerplate - Post Start Script
# This script runs every time the dev container starts

set -e

echo "🔄 Starting Reflexion Agent development environment..."

# Ensure proper permissions on data directories
chmod -R 755 /workspace/{data,logs} 2>/dev/null || true

# Start essential services
echo "🚀 Starting essential development services..."

# Check if docker-compose services are running, start if needed
if [ -f "/workspace/docker-compose.yml" ]; then
    cd /workspace
    echo "📊 Starting monitoring services..."
    docker-compose up -d prometheus grafana 2>/dev/null || echo "⚠️ Could not start monitoring services"
fi

# Display environment status
echo "📋 Environment Status:"
echo "  🐍 Python: $(python --version)"
echo "  📦 Pip: $(pip --version | cut -d' ' -f1-2)"
echo "  🔧 Environment: ${ENVIRONMENT:-development}"
echo "  🐛 Debug: ${DEBUG:-true}"

# Quick health check
echo "🔍 Health Check:"
if [ -f "/workspace/.env" ]; then
    echo "  ✅ .env file exists"
else
    echo "  ⚠️  .env file missing - copy from .env.example"
fi

if [ -d "/workspace/data" ]; then
    echo "  ✅ Data directory ready"
else
    mkdir -p /workspace/data
    echo "  📁 Created data directory"
fi

echo ""
echo "🎯 Ready for development!"
echo "💡 Run './dev_start.sh' for available commands"