# Manual Setup Requirements

## Repository Configuration

### Branch Protection Rules
Configure in `Settings > Branches > Add rule`:
- Branch name pattern: `main`
- Require pull request reviews before merging
- Require status checks to pass before merging
- Include administrators

### Repository Secrets
Add in `Settings > Secrets and variables > Actions`:
- `PYPI_API_TOKEN`: PyPI publishing token
- `CODECOV_TOKEN`: Codecov integration token

### Repository Settings
- Enable Issues and Projects
- Set repository topics: `python`, `ai`, `agents`, `reflexion`
- Configure homepage URL and description

## External Service Setup

### Codecov Integration
1. Visit [codecov.io](https://codecov.io/)
2. Connect GitHub repository
3. Copy integration token to repository secrets

### PyPI Package Publishing  
1. Create account at [pypi.org](https://pypi.org/)
2. Generate API token in account settings
3. Add token to repository secrets

## GitHub Actions Workflows

**Note**: The following workflow files require manual creation due to security restrictions:

### Required Files
- `.github/workflows/ci.yml` - Comprehensive CI pipeline
- `.github/workflows/security.yml` - Advanced security scanning
- `.github/workflows/release.yml` - Automated release process

### Complete Workflow Code
The full workflow implementations are documented in the deployment guide (`docs/DEPLOYMENT.md`) under the "GitHub Actions" section. These include:

1. **CI Workflow**: Multi-Python version testing, coverage, security checks, Docker builds
2. **Security Workflow**: CodeQL analysis, Semgrep scanning, dependency checks, container scanning  
3. **Release Workflow**: Automated PyPI publishing with SBOM generation

### Setup Instructions
1. Create `.github/workflows/` directory
2. Copy workflow YAML from `docs/DEPLOYMENT.md` 
3. Commit and push to enable automated CI/CD
4. Configure repository secrets as documented above

### Validation
After setup, run `make ci-check` to validate all workflows locally.

## Development Environment Setup

### Pre-commit Hooks Installation
```bash
pip install pre-commit
pre-commit install
```

### IDE Configuration
- Configure Python interpreter to project virtual environment
- Enable format-on-save with Black formatter
- Set up type checking with MyPy