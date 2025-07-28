# Pull Request

## Description

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

<!-- Mark the relevant option with an "x" -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Configuration change
- [ ] 🧪 Test improvement
- [ ] ♻️ Code refactoring
- [ ] ⚡ Performance improvement
- [ ] 🔒 Security improvement

## Related Issue

<!-- Link to the issue this PR addresses -->
Fixes #(issue_number)

## Changes Made

<!-- Provide a detailed list of changes made -->

- Change 1
- Change 2
- Change 3

## Testing

<!-- Describe the tests you ran and how to reproduce them -->

### Test Environment
- Python version: 
- Operating System: 
- Framework (if applicable): 

### Test Cases
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

### Test Commands
```bash
# Commands used to test the changes
make test
python -m pytest tests/specific_test.py -v
```

## Screenshots/Examples

<!-- If applicable, add screenshots or code examples -->

```python
# Example usage of new feature
agent = ReflexionAgent(...)
result = agent.new_method(...)
```

## Checklist

<!-- Mark completed items with an "x" -->

### Code Quality
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My code passes all existing tests
- [ ] I have added tests that prove my fix is effective or that my feature works

### Documentation
- [ ] I have updated the documentation accordingly
- [ ] I have updated the CHANGELOG.md file
- [ ] I have added docstrings to new functions/classes
- [ ] I have updated the README if needed

### Dependencies
- [ ] I have not introduced any new dependencies
- [ ] OR I have justified any new dependencies in the description
- [ ] I have updated requirements files if needed
- [ ] I have tested with both minimum and latest dependency versions

### Compatibility
- [ ] My changes are backward compatible
- [ ] OR I have documented breaking changes in the description
- [ ] I have tested with multiple Python versions (if applicable)
- [ ] I have tested with multiple frameworks (if applicable)

### Security
- [ ] I have not introduced any security vulnerabilities
- [ ] I have not exposed any sensitive information
- [ ] I have validated all user inputs (if applicable)
- [ ] I have run security scans (`make security`)

## Performance Impact

<!-- Describe any performance implications -->

- [ ] No performance impact
- [ ] Positive performance impact
- [ ] Negative performance impact (justified below)

<!-- If negative impact, please justify -->

## Breaking Changes

<!-- List any breaking changes and migration instructions -->

- None

<!-- OR provide migration guide -->

## Additional Notes

<!-- Any additional information for reviewers -->

## Reviewer Notes

<!-- For maintainers - remove if not applicable -->

- [ ] Code review completed
- [ ] Architecture review completed
- [ ] Security review completed
- [ ] Performance review completed
- [ ] Documentation review completed

---

**Thank you for contributing to Reflexion Agent Boilerplate!** 🎉