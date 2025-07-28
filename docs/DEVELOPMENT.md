# Development Guide

## Setup

```bash
# Clone repository
git clone https://github.com/your-org/reflexion-agent-boilerplate
cd reflexion-agent-boilerplate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Project Structure

```
reflexion/
├── core/          # Core reflexion engine
├── adapters/      # Framework integrations
├── memory/        # Memory systems
├── evaluators/    # Task evaluation
└── templates/     # Domain-specific templates
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=reflexion tests/

# Run benchmarks
python benchmarks/run_all.py
```

## Code Quality

- Format: `black reflexion/`
- Lint: `flake8 reflexion/`
- Type check: `mypy reflexion/`

## Documentation

- Docstrings: [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- API docs: Auto-generated with Sphinx
- Examples: Include in docstrings and `examples/` directory