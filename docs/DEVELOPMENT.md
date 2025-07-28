# Development Guide
## Reflexion Agent Boilerplate

### Getting Started

#### Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for containerized development)
- PostgreSQL (for local development)
- Redis (for caching and session storage)

#### Development Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/reflexion-agent-boilerplate.git
   cd reflexion-agent-boilerplate
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start development services**
   ```bash
   docker-compose up -d postgres redis
   ```

#### Alternative: DevContainer Setup

For a consistent development environment:

1. **Open in VS Code**
   - Install the "Remote - Containers" extension
   - Open the project folder
   - Click "Reopen in Container" when prompted

2. **Or use GitHub Codespaces**
   - Create a new Codespace from the repository
   - All dependencies will be pre-installed

### Project Structure

```
reflexion-agent-boilerplate/
├── reflexion/                 # Main package
│   ├── core/                 # Core reflexion logic
│   ├── adapters/             # Framework adapters
│   ├── memory/               # Memory systems
│   ├── evaluators/           # Task evaluators
│   ├── utils/                # Utility functions
│   └── cli/                  # Command-line interface
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── benchmark/            # Performance tests
├── docs/                     # Documentation
├── monitoring/               # Monitoring configs
├── scripts/                  # Utility scripts
└── .github/                  # GitHub workflows
```

### Development Workflow

#### 1. Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

#### 2. Development Loop

1. **Write tests first** (TDD approach recommended)
   ```bash
   # Create test file
   touch tests/unit/test_your_feature.py
   
   # Write failing tests
   # Implement feature
   # Make tests pass
   ```

2. **Run tests frequently**
   ```bash
   make test-unit          # Run unit tests
   make test-integration   # Run integration tests
   make test               # Run all tests
   ```

3. **Check code quality**
   ```bash
   make format             # Format code
   make lint               # Check linting
   make typecheck          # Type checking
   make quality            # All quality checks
   ```

#### 3. Committing Changes

```bash
# Stage changes
git add .

# Commit with conventional commit format
git commit -m "feat: add new reflexion strategy"
# or
git commit -m "fix: resolve memory leak in episodic storage"
# or
git commit -m "docs: update API documentation"
```

#### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create PR through GitHub UI or CLI
gh pr create --title "Add new reflexion strategy" --body "Description of changes"
```

### Code Style Guidelines

#### Python Code Style

- **Formatting**: Use Black (configured in pyproject.toml)
- **Import Sorting**: Use isort with Black profile
- **Line Length**: 88 characters (Black default)
- **Type Hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs

#### Example Code Structure

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ReflectionResult:
    """Results from a reflexion iteration.
    
    Args:
        success: Whether the reflection was successful
        insights: List of insights generated
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata
    """
    success: bool
    insights: List[str]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class ReflexionAgent:
    """Main reflexion agent implementation."""
    
    def __init__(self, llm: str, max_iterations: int = 3) -> None:
        """Initialize the reflexion agent.
        
        Args:
            llm: LLM model identifier
            max_iterations: Maximum reflection iterations
        """
        self.llm = llm
        self.max_iterations = max_iterations
    
    def reflect(self, task: str, output: str) -> ReflectionResult:
        """Perform reflection on task output.
        
        Args:
            task: The original task description
            output: The generated output to reflect on
            
        Returns:
            ReflectionResult containing insights and metadata
            
        Raises:
            ReflexionError: If reflection fails
        """
        # Implementation here
        pass
```

### Testing Guidelines

#### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

from reflexion.core import ReflexionAgent


class TestReflexionAgent:
    """Test cases for ReflexionAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = ReflexionAgent(llm="gpt-4")
        assert agent.llm == "gpt-4"
        assert agent.max_iterations == 3
    
    @pytest.mark.asyncio
    async def test_async_reflection(self, mock_llm):
        """Test asynchronous reflection."""
        # Test implementation
        pass
    
    @pytest.mark.integration
    def test_with_real_llm(self):
        """Integration test with real LLM."""
        # Only runs in integration test suite
        pass
```

#### Test Categories

- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Tests that verify component interactions
- **Benchmark Tests**: Performance and load testing
- **End-to-End Tests**: Full workflow testing

#### Running Tests

```bash
# Run specific test categories
pytest tests/unit/ -v                    # Unit tests only
pytest tests/integration/ -v             # Integration tests only
pytest tests/benchmark/ -v               # Benchmark tests only

# Run with coverage
pytest --cov=reflexion --cov-report=html

# Run specific test
pytest tests/unit/test_core.py::TestReflexionAgent::test_initialization -v

# Run tests matching pattern
pytest -k "test_reflection" -v
```

### Debugging

#### Using Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint() function (Python 3.7+)
breakpoint()
```

#### Logging

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Debug information")
    logger.info("General information")
    logger.warning("Warning message")
    logger.error("Error occurred")
```

#### Environment Variables for Debugging

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Enable additional debugging features
export REFLEXION_DEBUG=true
export PYTHONPATH=.
```

### Performance Optimization

#### Profiling

```bash
# Profile application
python -m cProfile -o profile.stats scripts/profile_reflexion.py

# Analyze results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Use make target
make profile
```

#### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler scripts/memory_test.py
```

#### Benchmarking

```bash
# Run benchmark suite
make test-benchmark

# Compare performance
pytest tests/benchmark/ --benchmark-compare
```

### Framework-Specific Development

#### Adding New Framework Adapter

1. **Create adapter module**
   ```bash
   touch reflexion/adapters/your_framework.py
   ```

2. **Implement adapter interface**
   ```python
   from reflexion.adapters.base import BaseAdapter
   
   class YourFrameworkAdapter(BaseAdapter):
       def __init__(self, agent, **kwargs):
           super().__init__(agent, **kwargs)
       
       def enhance_agent(self):
           # Framework-specific enhancement logic
           pass
   ```

3. **Add tests**
   ```bash
   touch tests/integration/test_your_framework_integration.py
   ```

4. **Update documentation**

### Database Development

#### Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

#### Testing with Database

```python
@pytest.fixture
def db_session():
    """Database session for testing."""
    # Setup test database
    # Yield session
    # Cleanup
```

### API Development

#### Adding New Endpoints

1. **Define endpoint**
   ```python
   from fastapi import APIRouter
   
   router = APIRouter()
   
   @router.post("/reflect")
   async def reflect_endpoint(request: ReflectionRequest):
       # Implementation
       pass
   ```

2. **Add validation**
   ```python
   from pydantic import BaseModel, Field
   
   class ReflectionRequest(BaseModel):
       task: str = Field(..., description="Task description")
       output: str = Field(..., description="Output to reflect on")
   ```

3. **Add tests**
   ```python
   def test_reflect_endpoint(client):
       response = client.post("/reflect", json={
           "task": "test task",
           "output": "test output"
       })
       assert response.status_code == 200
   ```

### Documentation

#### API Documentation

- **OpenAPI**: Automatically generated from FastAPI
- **Access**: Available at `/docs` when running the application

#### Code Documentation

- **Docstrings**: Google-style docstrings for all public APIs
- **Type Hints**: Complete type annotations
- **Examples**: Include usage examples in docstrings

#### Building Documentation

```bash
make docs              # Build documentation
make docs-serve        # Serve documentation locally
```

### Troubleshooting

#### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   
   # Check PYTHONPATH
   export PYTHONPATH=.
   ```

2. **Test Failures**
   ```bash
   # Clear pytest cache
   pytest --cache-clear
   
   # Run with verbose output
   pytest -v -s
   ```

3. **Pre-commit Hooks Failing**
   ```bash
   # Run hooks manually
   pre-commit run --all-files
   
   # Update hooks
   pre-commit autoupdate
   ```

4. **Docker Issues**
   ```bash
   # Rebuild containers
   docker-compose down
   docker-compose up --build
   
   # Clean Docker state
   docker system prune -a
   ```

### Getting Help

- **Documentation**: Check the docs/ directory
- **Issues**: Search GitHub issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Request review from maintainers

### Contributing Guidelines

1. **Read CONTRIBUTING.md** for detailed contribution guidelines
2. **Follow code style** guidelines
3. **Write tests** for all new features
4. **Update documentation** as needed
5. **Use conventional commits** for commit messages