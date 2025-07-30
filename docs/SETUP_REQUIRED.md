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
- `.github/workflows/ci.yml` - Continuous integration
- `.github/workflows/quality.yml` - Code quality checks  
- `.github/workflows/security.yml` - Security scanning
- `.github/workflows/release.yml` - Automated releases

### Workflow Setup Instructions
1. Copy templates from `docs/workflows/` to `.github/workflows/`
2. Review and adjust Python versions and dependencies
3. Configure repository secrets as documented above
4. Enable Actions in repository settings

### Additional Automation
- **Dependabot**: Automated dependency updates configured in `.github/dependabot.yml`
- **Pre-commit**: Enhanced hooks configured in `.pre-commit-config.yaml`
- **Issue Templates**: Feature requests and bug reports in `.github/ISSUE_TEMPLATE/`
- **PR Template**: Standardized pull request template in `.github/pull_request_template.md`

### Workflow Templates
Reference templates available in:
- `docs/workflows/ci.yml.template` - Production-ready CI configuration
- `docs/workflows/security.yml.template` - Comprehensive security checks
- `docs/workflows/release.yml.template` - Automated semantic releases
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)

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