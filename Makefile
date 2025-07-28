# Makefile for Reflexion Agent Boilerplate

# Variables
PYTHON := python3
PIP := pip
PACKAGE_NAME := reflexion-agent-boilerplate
TEST_PATH := tests/
SOURCE_PATH := reflexion/

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development setup
.PHONY: install
install: ## Install package in development mode
	$(PIP) install -e ".[dev]"

.PHONY: install-prod
install-prod: ## Install package for production
	$(PIP) install .

.PHONY: setup-dev
setup-dev: install ## Setup development environment
	pre-commit install
	@echo "Development environment setup complete!"

# Code quality
.PHONY: format
format: ## Format code with black and isort
	black $(SOURCE_PATH) $(TEST_PATH)
	isort $(SOURCE_PATH) $(TEST_PATH)

.PHONY: format-check
format-check: ## Check code formatting
	black --check $(SOURCE_PATH) $(TEST_PATH)
	isort --check-only $(SOURCE_PATH) $(TEST_PATH)

.PHONY: lint
lint: ## Run linting with flake8 and pylint
	flake8 $(SOURCE_PATH) $(TEST_PATH)
	pylint $(SOURCE_PATH) $(TEST_PATH)

.PHONY: typecheck
typecheck: ## Run type checking with mypy
	mypy $(SOURCE_PATH)

.PHONY: security
security: ## Run security checks with bandit and safety
	bandit -r $(SOURCE_PATH)
	safety check

.PHONY: quality
quality: format-check lint typecheck security ## Run all code quality checks

# Testing
.PHONY: test
test: ## Run all tests
	pytest $(TEST_PATH) -v --cov=$(SOURCE_PATH) --cov-report=html --cov-report=term-missing

.PHONY: test-unit
test-unit: ## Run unit tests
	pytest $(TEST_PATH)unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests
	pytest $(TEST_PATH)integration/ -v

.PHONY: test-benchmark
test-benchmark: ## Run performance benchmarks
	pytest $(TEST_PATH)benchmark/ -v --benchmark-only

.PHONY: test-fast
test-fast: ## Run fast tests (exclude slow tests)
	pytest $(TEST_PATH) -v -m "not slow"

.PHONY: test-coverage
test-coverage: ## Generate test coverage report
	pytest $(TEST_PATH) --cov=$(SOURCE_PATH) --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Building and packaging
.PHONY: build
build: clean ## Build package
	$(PYTHON) -m build

.PHONY: dist
dist: build ## Create distribution packages
	@echo "Distribution packages created in dist/"

.PHONY: upload-test
upload-test: dist ## Upload to test PyPI
	twine upload --repository testpypi dist/*

.PHONY: upload
upload: dist ## Upload to PyPI
	twine upload dist/*

# Documentation
.PHONY: docs
docs: ## Build documentation
	sphinx-build -b html docs/ docs/_build/html

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	sphinx-autobuild docs/ docs/_build/html --host 0.0.0.0 --port 8000

.PHONY: docs-clean
docs-clean: ## Clean documentation build
	rm -rf docs/_build/

# Development utilities
.PHONY: clean
clean: ## Clean build artifacts and cache
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .tox/

.PHONY: reset
reset: clean ## Reset development environment
	rm -rf venv/ .venv/
	$(PIP) cache purge

.PHONY: deps-update
deps-update: ## Update dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev]"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Docker
.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t $(PACKAGE_NAME) .

.PHONY: docker-run
docker-run: ## Run Docker container
	docker run -it --rm $(PACKAGE_NAME)

.PHONY: docker-test
docker-test: ## Run tests in Docker
	docker run --rm $(PACKAGE_NAME) make test

# Database (for development)
.PHONY: db-init
db-init: ## Initialize development database
	$(PYTHON) -c "from reflexion.database import init_db; init_db()"

.PHONY: db-reset
db-reset: ## Reset development database
	$(PYTHON) -c "from reflexion.database import reset_db; reset_db()"

# Profiling and debugging
.PHONY: profile
profile: ## Run performance profiling
	$(PYTHON) -m cProfile -o profile.stats scripts/profile_reflexion.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

.PHONY: debug
debug: ## Run with debug logging
	DEBUG=true $(PYTHON) -m reflexion.cli

# Release management
.PHONY: version
version: ## Show current version
	$(PYTHON) -c "from reflexion import __version__; print(__version__)"

.PHONY: changelog
changelog: ## Generate changelog
	conventional-changelog -p conventionalcommits -i CHANGELOG.md -s

# CI/CD helpers
.PHONY: ci-install
ci-install: ## Install dependencies for CI
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: ci-test
ci-test: ## Run CI test suite
	pytest $(TEST_PATH) -v --cov=$(SOURCE_PATH) --cov-report=xml --cov-report=term-missing

.PHONY: ci-quality
ci-quality: ## Run CI quality checks
	black --check $(SOURCE_PATH) $(TEST_PATH)
	isort --check-only $(SOURCE_PATH) $(TEST_PATH)
	flake8 $(SOURCE_PATH) $(TEST_PATH)
	mypy $(SOURCE_PATH)
	bandit -r $(SOURCE_PATH)

# Monitoring
.PHONY: metrics
metrics: ## Generate project metrics
	@echo "Lines of code:"
	@find $(SOURCE_PATH) -name "*.py" -exec wc -l {} + | tail -1
	@echo "Test coverage:"
	@pytest $(TEST_PATH) --cov=$(SOURCE_PATH) --cov-report=term-missing | grep "TOTAL"