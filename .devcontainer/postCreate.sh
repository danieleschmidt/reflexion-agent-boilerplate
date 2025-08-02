#!/bin/bash

# Reflexion Agent Boilerplate - Post Create Setup Script
# This script runs after the dev container is created

set -e

echo "🚀 Setting up Reflexion Agent development environment..."

# Create necessary directories
mkdir -p /workspace/{data,logs,debug,temp}
mkdir -p /workspace/data/{memory,cache,backups}
mkdir -p /workspace/logs/{app,error,debug}

# Set proper permissions
chmod -R 755 /workspace/{data,logs,debug,temp}

# Install Python dependencies
echo "📦 Installing Python dependencies..."
cd /workspace
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Setup Git configuration for commits
echo "🔐 Configuring Git settings..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf false
git config --global core.eol lf

# Create .env file from template if it doesn't exist
if [ ! -f "/workspace/.env" ]; then
    echo "📝 Creating .env file from template..."
    cp /workspace/.env.example /workspace/.env
    echo "⚠️  Please update .env with your actual configuration values"
fi

# Setup database directory
echo "🗄️  Setting up local database directory..."
mkdir -p /workspace/data/database
chmod 755 /workspace/data/database

# Install additional development tools
echo "🛠️  Installing additional development tools..."
pip install --user jupyterlab-git jupyterlab-lsp python-lsp-server[all]

# Setup Jupyter Lab configuration
echo "📓 Configuring Jupyter Lab..."
mkdir -p /home/vscode/.jupyter
cat > /home/vscode/.jupyter/jupyter_lab_config.py << 'EOF'
# Jupyter Lab configuration for Reflexion development
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8001
c.ServerApp.open_browser = False
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.token = ''
c.ServerApp.password = ''
c.LabApp.default_url = '/lab'
EOF

# Setup shell aliases and functions
echo "⚡ Setting up shell aliases..."
cat >> /home/vscode/.zshrc << 'EOF'

# Reflexion Agent aliases
alias reflexion-test="pytest tests/ -v"
alias reflexion-lint="ruff check src/ tests/"
alias reflexion-format="black src/ tests/ && isort src/ tests/"
alias reflexion-type="mypy src/"
alias reflexion-security="bandit -r src/"
alias reflexion-coverage="pytest tests/ --cov=src --cov-report=html"
alias reflexion-docs="cd docs && python -m http.server 8080"
alias reflexion-logs="tail -f logs/reflexion.log"
alias reflexion-clean="find . -type d -name '__pycache__' -exec rm -rf {} + && find . -name '*.pyc' -delete"

# Docker aliases
alias dc="docker-compose"
alias dcup="docker-compose up -d"
alias dcdown="docker-compose down"
alias dclogs="docker-compose logs -f"

# Git aliases
alias gs="git status"
alias ga="git add"
alias gc="git commit"
alias gp="git push"
alias gl="git log --oneline -10"
alias gd="git diff"

# Development functions
reflexion-dev() {
    echo "🚀 Starting Reflexion development servers..."
    docker-compose up -d postgres redis
    cd /workspace
    python -m reflexion.cli serve --dev --reload
}

reflexion-monitor() {
    echo "📊 Starting monitoring stack..."
    docker-compose up -d prometheus grafana
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000 (admin/admin)"
}

reflexion-profile() {
    echo "🔍 Running performance profiling..."
    python -m cProfile -s cumulative -m pytest tests/performance/ | head -20
}

EOF

# Setup environment variables for development
echo "🌍 Setting up development environment variables..."
cat >> /home/vscode/.zshrc << 'EOF'

# Development environment variables
export PYTHONPATH="/workspace/src:$PYTHONPATH"
export ENVIRONMENT="development"
export DEBUG="true"
export LOG_LEVEL="DEBUG"
export REFLEXION_DEV_MODE="true"

EOF

# Source the updated shell configuration
source /home/vscode/.zshrc

# Install VS Code extensions that weren't caught by devcontainer.json
echo "🔌 Installing additional VS Code extensions..."
code --install-extension ms-python.debugpy || true
code --install-extension ms-toolsai.jupyter-keymap || true
code --install-extension ms-toolsai.jupyter-renderers || true

# Setup directory structure for different development scenarios
echo "📁 Creating development directory structure..."
mkdir -p /workspace/{examples,experiments,benchmarks/results}
mkdir -p /workspace/examples/{basic,advanced,integrations}
mkdir -p /workspace/experiments/{memory,evaluation,frameworks}

# Create example configuration files
echo "📋 Creating example configuration files..."
cat > /workspace/examples/basic_config.yaml << 'EOF'
# Basic Reflexion Agent Configuration Example
llm:
  provider: openai
  model: gpt-4
  temperature: 0.1

reflexion:
  max_iterations: 3
  reflection_type: structured
  success_threshold: 0.8

memory:
  type: episodic
  capacity: 100
  consolidation_threshold: 0.8

evaluation:
  timeout: 60
  cache_results: true
EOF

# Setup development database
echo "🗃️  Initializing development database..."
if command -v sqlite3 &> /dev/null; then
    sqlite3 /workspace/data/database/reflexion_dev.db << 'EOF'
CREATE TABLE IF NOT EXISTS development_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    category TEXT,
    content TEXT,
    tags TEXT
);

INSERT INTO development_notes (category, content, tags) VALUES 
('setup', 'Development environment initialized', 'setup,devcontainer'),
('config', 'Remember to update .env with your API keys', 'configuration,api');
EOF
fi

# Setup testing infrastructure
echo "🧪 Setting up testing infrastructure..."
mkdir -p /workspace/tests/{fixtures,mocks,data}
echo "test_*.py" > /workspace/tests/fixtures/.gitkeep
echo "*.json" > /workspace/tests/data/.gitkeep

# Create development startup script
echo "📄 Creating development startup script..."
cat > /workspace/dev_start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Reflexion Agent development environment..."
echo "🔧 Environment: $(echo $ENVIRONMENT)"
echo "📊 Debug mode: $(echo $DEBUG)"
echo ""
echo "Available commands:"
echo "  reflexion-test     - Run tests"
echo "  reflexion-lint     - Run linting"
echo "  reflexion-format   - Format code"
echo "  reflexion-dev      - Start development server"
echo "  reflexion-monitor  - Start monitoring"
echo ""
echo "🌐 Available ports:"
echo "  8000 - API Server"
echo "  8001 - Jupyter Lab"
echo "  8080 - Documentation"
echo "  9090 - Prometheus"
echo "  3000 - Grafana"
echo ""
echo "📝 Don't forget to configure your .env file!"
EOF

chmod +x /workspace/dev_start.sh

# Final setup steps
echo "🎯 Finalizing setup..."
cd /workspace

# Verify installation
echo "✅ Verifying installation..."
python -c "import reflexion; print(f'Reflexion package loaded successfully from: {reflexion.__file__ if hasattr(reflexion, \"__file__\") else \"built-in\"}')" || echo "⚠️  Reflexion package not found - this is expected if not yet implemented"

# Display completion message
echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "🚀 To get started:"
echo "  1. Update .env with your API keys"
echo "  2. Run: source dev_start.sh"
echo "  3. Start developing with: reflexion-dev"
echo ""
echo "📚 Documentation available at: /workspace/docs/"
echo "🧪 Run tests with: reflexion-test"
echo "🔍 Monitor with: reflexion-monitor"
echo ""
echo "Happy coding! 🎯"