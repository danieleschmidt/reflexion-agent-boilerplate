# Upgrade Guide

This document provides guidance for upgrading the Reflexion Agent Boilerplate to newer versions.

## Version Compatibility

### Supported Python Versions
- **Current**: Python 3.9 - 3.12
- **Deprecated**: Python 3.8 (EOL support)
- **Planned**: Python 3.13 support in v0.2.0

### Framework Compatibility Matrix

| Framework | v0.1.x | v0.2.x | v0.3.x |
|-----------|--------|--------|--------|
| AutoGen   | ✅ 0.2+ | ✅ 0.3+ | ✅ 0.4+ |
| CrewAI    | ✅ 0.1+ | ✅ 0.2+ | ✅ 0.3+ |
| LangChain | ✅ 0.1+ | ✅ 0.2+ | ✅ 0.3+ |
| Claude-Flow | ✅ 0.1+ | ✅ 0.2+ | ✅ 0.3+ |

## Pre-Upgrade Checklist

### 1. Backup Your Data
```bash
# Backup memory store
cp -r ~/.reflexion/memory ~/.reflexion/memory.backup.$(date +%Y%m%d)

# Backup configuration
cp config.yml config.yml.backup.$(date +%Y%m%d)

# Export existing metrics
curl http://localhost:9090/api/v1/query_range?query=reflexions_total > metrics_backup.json
```

### 2. Version Assessment
```bash
# Check current version
reflexion --version

# Check compatibility
python -c "
import reflexion
print(f'Current version: {reflexion.__version__}')
print(f'Python version: {reflexion.python_version_info}')
print(f'Compatible frameworks: {reflexion.supported_frameworks}')
"
```

### 3. Test Environment Validation
```bash
# Run full test suite before upgrade
make test

# Run compatibility checks
python scripts/compatibility_check.py

# Backup working environment
pip freeze > requirements_working.txt
```

## Upgrade Procedures

### Minor Version Updates (0.1.x → 0.1.y)

Minor updates typically include:
- Bug fixes
- Performance improvements
- New optional features
- Security patches

```bash
# Standard upgrade
pip install --upgrade reflexion-agent-boilerplate

# Verify installation
reflexion --version
reflexion validate-config

# Run post-upgrade tests
make test-integration
```

### Major Version Updates (0.1.x → 0.2.x)

Major updates may include:
- Breaking API changes
- New required dependencies
- Configuration format changes
- Memory format updates

```bash
# Review breaking changes first
curl -s https://api.github.com/repos/your-org/reflexion-agent-boilerplate/releases/latest | jq '.body'

# Upgrade with dependency resolution
pip install --upgrade reflexion-agent-boilerplate --upgrade-strategy eager

# Run migration scripts
reflexion migrate --from-version 0.1.x --to-version 0.2.x

# Update configuration
reflexion config upgrade
```

## Breaking Changes by Version

### v0.2.0 (Planned)
- **Memory Store Format**: New JSON-based format
- **Configuration**: YAML replaces TOML for config files
- **API Changes**: `ReflexionAgent.run()` signature updated

#### Migration Steps for v0.2.0:
```python
# Old API (v0.1.x)
agent = ReflexionAgent(llm="gpt-4", max_iterations=3)
result = agent.run(task="example", criteria="passes tests")

# New API (v0.2.x)  
agent = ReflexionAgent(
    llm_config={"model": "gpt-4", "temperature": 0.7},
    reflexion_config={"max_iterations": 3, "strategy": "adaptive"}
)
result = agent.execute(
    task=Task(description="example", success_criteria="passes tests")
)
```

### v0.3.0 (Future)
- **Async-First API**: All methods become async
- **Plugin System**: New plugin architecture
- **Enhanced Memory**: Vector-based memory storage

## Configuration Migration

### Config File Updates

#### v0.1.x → v0.2.x
```bash
# Automatic migration
reflexion config migrate --input config.toml --output config.yml

# Manual migration example:
# OLD (config.toml)
[reflexion]
max_iterations = 3
success_threshold = 0.8

[llm]
model = "gpt-4"
temperature = 0.7

# NEW (config.yml) 
reflexion:
  max_iterations: 3
  success_threshold: 0.8
  strategy: adaptive

llm:
  model: gpt-4
  temperature: 0.7
  timeout: 30
```

### Memory Store Migration
```bash
# Migrate memory format
reflexion memory migrate \
  --from ~/.reflexion/memory \
  --to ~/.reflexion/memory_v2 \
  --format json

# Verify migration
reflexion memory validate --path ~/.reflexion/memory_v2
```

## Testing After Upgrade

### Regression Testing
```bash
# Core functionality
make test

# Framework integrations
pytest tests/adapters/ -v

# Performance regression
python benchmarks/regression_test.py --baseline v0.1.0

# Memory compatibility
python tests/memory/migration_test.py
```

### Validation Checklist
- [ ] All tests pass
- [ ] Configuration loads correctly
- [ ] Memory store accessible
- [ ] Framework adapters work
- [ ] Performance within acceptable range
- [ ] Monitoring/metrics still functional

## Rollback Procedures

### Quick Rollback
```bash
# Reinstall previous version
pip install reflexion-agent-boilerplate==0.1.0

# Restore configuration
cp config.yml.backup.20240730 config.yml

# Restore memory store
rm -rf ~/.reflexion/memory
mv ~/.reflexion/memory.backup.20240730 ~/.reflexion/memory

# Verify rollback
reflexion --version
make test-fast
```

### Complete Environment Restoration
```bash
# Restore entire environment
pip uninstall reflexion-agent-boilerplate
pip install -r requirements_working.txt

# Restore all data
tar -xzf backup_full_20240730.tar.gz

# Validate restoration
python scripts/validate_environment.py
```

## Common Upgrade Issues

### Dependency Conflicts
```bash
# Resolve dependency conflicts
pip install --upgrade-strategy eager reflexion-agent-boilerplate

# Check for conflicts
pip check

# Manual resolution if needed
pip install --force-reinstall reflexion-agent-boilerplate
```

### Memory Format Incompatibility
```bash
# Convert old memory format
python scripts/convert_memory.py \
  --input ~/.reflexion/memory \
  --output ~/.reflexion/memory_new \
  --target-version 0.2.0

# Validate conversion
python scripts/validate_memory.py ~/.reflexion/memory_new
```

### Configuration Validation Errors
```bash
# Check configuration syntax
reflexion config validate

# Fix common issues
reflexion config fix --dry-run
reflexion config fix --apply

# Reset to defaults if needed
reflexion config reset --backup
```

## Post-Upgrade Optimization

### Performance Tuning
```bash
# Re-run performance optimization
reflexion optimize --profile

# Update monitoring baselines
python scripts/update_baselines.py

# Rebuild caches
reflexion cache rebuild
```

### Feature Updates
```bash
# Enable new features
reflexion features enable adaptive_learning
reflexion features enable enhanced_memory

# Configure new settings
reflexion config set reflexion.strategy adaptive
reflexion config set memory.vector_store true
```

## Support and Troubleshooting

### Getting Help
- **Documentation**: https://docs.your-org.com/reflexion/upgrade
- **Issues**: https://github.com/your-org/reflexion-agent-boilerplate/issues
- **Discussions**: https://github.com/your-org/reflexion-agent-boilerplate/discussions
- **Discord**: https://discord.gg/your-org

### Debug Information
```bash
# Generate debug report
reflexion debug-info > debug_report.txt

# Include in issue reports:
# - Current version
# - Python version
# - Operating system
# - Full error traceback
# - Configuration (sanitized)
```

### Emergency Contacts
For critical production issues:
- **Email**: reflexion-support@your-org.com
- **Priority Support**: Available for enterprise customers