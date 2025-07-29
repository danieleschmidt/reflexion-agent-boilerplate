# Makefile for Reflexion Agent Boilerplate

.PHONY: help install dev-install test test-unit test-integration test-cov clean lint format typecheck pre-commit benchmark docker-build docker-run setup-dev

# Default target
help: ## Show this help message
	@echo "Reflexion Agent Boilerplate - Available commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install package in production mode
	pip install -e .

dev-install: ## Install package in development mode with all dependencies
	pip install -e ".[dev]"
	pre-commit install

# Testing
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Code Quality
lint: ## Run linting with flake8
	flake8 src/ tests/ benchmarks/

format: ## Format code with black and sort imports
	black src/ tests/ benchmarks/
	isort src/ tests/ benchmarks/

typecheck: ## Run type checking with mypy
	mypy src/

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

# Quality assurance
quality: lint typecheck test ## Run all quality checks

# Performance
benchmark: ## Run performance benchmarks
	PYTHONPATH=src python benchmarks/run_all.py

# Docker
docker-build: ## Build Docker image
	docker build -t reflexion-agent:latest .

docker-build-dev: ## Build development Docker image
	docker build --target builder -t reflexion-agent:dev .

docker-run: ## Run Docker container
	docker run --rm -it reflexion-agent:latest

docker-dev: ## Run development environment
	docker-compose --profile dev up -d

docker-full: ## Run full environment with monitoring
	docker-compose --profile full --profile monitoring up -d

docker-down: ## Stop Docker environment
	docker-compose down -v

# Development setup
setup-dev: dev-install ## Complete development environment setup
	@echo "Development environment setup complete!"
	@echo "You can now run:"
	@echo "  make test      - Run tests"
	@echo "  make quality   - Run quality checks"  
	@echo "  make benchmark - Run benchmarks"

# Cleanup
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-docker: ## Clean Docker resources
	docker system prune -f
	docker volume prune -f

# Documentation
docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

# Release preparation
release-check: quality test benchmark ## Run all checks before release
	@echo "Release checks completed successfully!"

# Environment info
info: ## Show environment information
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Project structure:"
	@find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" | head -10