# Contributing to Reflexion Agent Boilerplate

We love your input! We want to make contributing to Reflexion Agent Boilerplate as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Request Process

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team at conduct@your-org.com.

## Bug Reports

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/your-org/reflexion-agent-boilerplate/issues/new).

### Great Bug Reports Include:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature has already been requested
2. Provide a clear and detailed explanation of the feature
3. Explain why this feature would be useful
4. Include examples of how the feature would be used

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, but recommended)

### Setting Up Your Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/reflexion-agent-boilerplate.git
   cd reflexion-agent-boilerplate
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to ensure everything works**
   ```bash
   make test
   ```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use Black for code formatting
- Use isort for import sorting
- Use type hints for all public functions
- Use Google-style docstrings

### Code Quality Tools

We use several tools to maintain code quality:

```bash
make format     # Format code with Black and isort
make lint       # Lint with flake8 and pylint
make typecheck  # Type checking with mypy
make quality    # Run all quality checks
```

### Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```
feat(core): add hierarchical reflection support
fix(memory): resolve memory leak in episodic storage
docs(api): update reflexion agent documentation
test(integration): add autogen adapter tests
```

## Testing

### Test Structure

- `tests/unit/`: Fast, isolated unit tests
- `tests/integration/`: Integration tests with external dependencies
- `tests/benchmark/`: Performance and load tests

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

from reflexion.core import ReflexionAgent


class TestReflexionAgent:
    """Test cases for ReflexionAgent."""
    
    def test_initialization(self):
        """Test agent initialization with default parameters."""
        agent = ReflexionAgent(llm="gpt-4")
        assert agent.llm == "gpt-4"
        assert agent.max_iterations == 3
    
    def test_reflection_success(self, mock_llm):
        """Test successful reflection process."""
        # Test implementation
        pass
    
    @pytest.mark.integration
    def test_with_real_api(self):
        """Integration test with real API (marked for CI)."""
        # Only runs in integration test suite
        pass
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-benchmark

# Run tests with coverage
pytest --cov=reflexion --cov-report=html

# Run specific test file
pytest tests/unit/test_core.py -v
```

## Documentation

### Code Documentation

- Use Google-style docstrings for all public functions and classes
- Include type hints for all function parameters and return values
- Provide usage examples in docstrings for complex functions

```python
def reflect(self, task: str, output: str, context: Optional[Dict[str, Any]] = None) -> ReflectionResult:
    """Perform reflection on task output.
    
    Args:
        task: The original task description
        output: The generated output to reflect on
        context: Optional context information for reflection
        
    Returns:
        ReflectionResult containing insights and confidence score
        
    Raises:
        ReflexionError: If reflection process fails
        ValueError: If task or output is empty
        
    Example:
        >>> agent = ReflexionAgent(llm="gpt-4")
        >>> result = agent.reflect("Write code", "def hello(): print('hi')")
        >>> print(result.insights)
        ['Function could use better naming', 'Missing docstring']
    """
```

### API Documentation

- API documentation is automatically generated from docstrings
- Access via `/docs` endpoint when running the application
- Keep OpenAPI schemas up to date

### User Documentation

- Update relevant documentation in `docs/` directory
- Include practical examples and use cases
- Update `README.md` if adding user-facing features

## Framework Adapters

### Adding a New Framework Adapter

1. **Create the adapter module**
   ```bash
   touch reflexion/adapters/your_framework.py
   ```

2. **Implement the adapter**
   ```python
   from reflexion.adapters.base import BaseAdapter
   
   class YourFrameworkAdapter(BaseAdapter):
       """Adapter for Your Framework integration."""
       
       def __init__(self, agent, **kwargs):
           super().__init__(agent, **kwargs)
           # Framework-specific initialization
       
       def enhance_agent(self):
           """Add reflexion capabilities to the framework agent."""
           # Implementation here
   ```

3. **Add comprehensive tests**
   ```bash
   touch tests/integration/test_your_framework_integration.py
   ```

4. **Update documentation**
   - Add usage examples to README
   - Create framework-specific guide in `docs/guides/`
   - Update API documentation

5. **Add optional dependency**
   ```toml
   # In pyproject.toml
   [project.optional-dependencies]
   your_framework = ["your-framework>=1.0.0"]
   ```

## Performance Considerations

### Benchmarking

- Add benchmark tests for new performance-critical features
- Use `pytest-benchmark` for consistent measurements
- Include both micro-benchmarks and integration benchmarks

```python
def test_reflection_performance(benchmark):
    """Benchmark reflection processing time."""
    agent = ReflexionAgent(llm="mock")
    
    def run_reflection():
        return agent.reflect("test task", "test output")
    
    result = benchmark(run_reflection)
    assert result is not None
```

### Memory Usage

- Be mindful of memory usage in long-running processes
- Use memory profiling tools for memory-intensive features
- Implement proper cleanup in finally blocks or context managers

## Security

### Security Guidelines

- Never commit secrets or API keys
- Validate all user inputs
- Use parameterized queries for database operations
- Follow the principle of least privilege
- Keep dependencies updated

### Security Testing

- Run security scans: `make security`
- Test with various input types including edge cases
- Consider security implications of new features

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Ensure all tests pass
4. Update documentation if needed
5. Create release PR
6. Tag release after merge

## Community

### Getting Help

- **Documentation**: Check `docs/` directory
- **GitHub Issues**: Search existing issues
- **GitHub Discussions**: For questions and ideas
- **Discord**: [Join our Discord server](https://discord.gg/your-org)

### Maintainer Response Times

- **Bug reports**: Within 48 hours
- **Feature requests**: Within 1 week
- **Pull requests**: Within 1 week
- **Security issues**: Within 24 hours

### Recognition

We believe in recognizing contributors:

- Contributors are listed in release notes
- Significant contributors may be invited as maintainers
- We highlight interesting contributions in our blog/newsletter

## FAQ

### How do I add support for a new LLM provider?

1. Create a new provider class inheriting from `BaseLLMProvider`
2. Implement required methods (`generate`, `embed`, etc.)
3. Add configuration options
4. Write comprehensive tests
5. Update documentation

### Can I contribute documentation only?

Absolutely! Documentation contributions are highly valued:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Translate documentation to other languages

### How do I propose architectural changes?

For significant architectural changes:

1. Open a GitHub Discussion first
2. Describe the problem and proposed solution
3. Get feedback from maintainers and community
4. Create a detailed RFC (Request for Comments)
5. Implement after approval

### What if my PR is rejected?

Don't be discouraged! Common reasons for rejection:

- Doesn't align with project goals
- Needs more discussion
- Requires additional tests or documentation
- Code quality issues

We'll provide feedback and guidance for improvement.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache 2.0 License).

## Questions?

If you have questions about contributing, please:

1. Check this document first
2. Search existing GitHub issues and discussions
3. Open a new discussion or issue
4. Contact the maintainers directly

Thank you for contributing to Reflexion Agent Boilerplate! 🎉