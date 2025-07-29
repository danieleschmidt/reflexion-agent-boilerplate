# Software Bill of Materials (SBOM)

## Overview

This document describes the Software Bill of Materials (SBOM) generation and management process for the Reflexion Agent Boilerplate project, ensuring supply chain transparency and security.

## SBOM Generation

### Automated Generation

SBOM generation is integrated into the CI/CD pipeline and can be triggered manually:

```bash
# Generate SBOM for current dependencies
pip install cyclonedx-bom
cyclonedx-bom requirements -o sbom.json

# Generate SBOM with additional metadata
cyclonedx-bom requirements \
  --output-format json \
  --output-file sbom-detailed.json \
  --include-license \
  --include-vulnerability-data
```

### SBOM Formats

The project supports multiple SBOM formats:

- **SPDX**: Industry standard format
- **CycloneDX**: Security-focused format with vulnerability data
- **SWID**: Software identification standard

### Generation Schedule

- **On every release**: Comprehensive SBOM with all dependencies
- **On dependency updates**: Incremental SBOM showing changes
- **Weekly**: Automated vulnerability assessment of SBOM
- **On-demand**: For security audits and compliance reviews

## SBOM Content

### Core Components

#### Direct Dependencies

```json
{
  "component": {
    "type": "library",
    "name": "pytest",
    "version": "7.4.0",
    "purl": "pkg:pypi/pytest@7.4.0",
    "licenses": [{"license": {"name": "MIT"}}],
    "hashes": [{"alg": "SHA-256", "content": "..."}]
  }
}
```

#### Transitive Dependencies

All transitive dependencies are included with:
- Version pinning information
- License compatibility analysis
- Vulnerability status
- Supply chain risk assessment

#### Build Tools

- Python interpreter version
- Build system dependencies (setuptools, wheel)
- Development tools (black, flake8, mypy)
- Testing frameworks (pytest, coverage)

### Metadata

- **Creation timestamp**: When SBOM was generated
- **Creator**: Tool and version used for generation
- **Supplier**: Organization responsible for the component
- **License**: Component licensing information
- **Copyright**: Copyright holders and dates

## Vulnerability Management

### Vulnerability Scanning

```yaml
# GitHub Actions workflow snippet
- name: Generate SBOM and scan for vulnerabilities
  run: |
    cyclonedx-bom requirements -o sbom.json
    grype sbom:sbom.json -o table
    grype sbom:sbom.json -o json --file vulnerability-report.json
```

### Vulnerability Database Sources

- **National Vulnerability Database (NVD)**
- **GitHub Security Advisories**
- **PyPI Security Advisories** 
- **OSV.dev Database**
- **Snyk Vulnerability Database**

### Response Process

1. **Detection**: Automated daily vulnerability scans
2. **Assessment**: Risk evaluation within 24 hours
3. **Prioritization**: Based on CVSS score and exploitability
4. **Remediation**: Update dependencies or implement mitigations
5. **Verification**: Confirm vulnerability resolution
6. **Documentation**: Update SBOM and security documentation

## License Compliance

### License Analysis

```bash
# Generate license report
pip-licenses --format=json --output-file=licenses.json
pip-licenses --format=markdown --output-file=LICENSES.md

# Check license compatibility
pip-licenses --format=csv | grep -E "(GPL|AGPL|SSPL)"
```

### Approved Licenses

**Permissive Licenses** (✅ Approved):
- MIT License
- Apache License 2.0
- BSD License (2-clause, 3-clause)
- ISC License

**Copyleft Licenses** (⚠️ Review Required):
- GNU Lesser General Public License (LGPL)
- Eclipse Public License (EPL)
- Mozilla Public License (MPL)

**Restricted Licenses** (❌ Not Permitted):
- GNU General Public License (GPL)
- GNU Affero General Public License (AGPL)
- Server Side Public License (SSPL)

### License Compatibility Matrix

| Project License | Compatible Dependencies | Incompatible |
|----------------|-------------------------|--------------|
| Apache-2.0 | MIT, BSD, ISC, Apache | GPL, AGPL |
| MIT | MIT, BSD, ISC | GPL, AGPL |
| BSD | MIT, BSD, ISC | GPL, AGPL |

## Supply Chain Security

### Package Verification

```bash
# Verify package integrity
pip install --require-hashes -r requirements-lock.txt

# Check package signatures (when available)
pip install package-name --trusted-host pypi.org --verify-signatures
```

### Repository Security

- **Package Repository**: PyPI (primary), with fallback to TestPyPI
- **Mirror Security**: Verified mirrors only
- **Transport Security**: HTTPS/TLS 1.3 required
- **Integrity Checks**: SHA-256 hashes for all packages

### Build Reproducibility

```dockerfile
# Dockerfile with pinned base images
FROM python:3.11.6-slim@sha256:abc123...

# Pinned system packages
RUN apt-get update && apt-get install -y \
    curl=7.88.1-10+deb12u4 \
    && rm -rf /var/lib/apt/lists/*
```

## SBOM Distribution

### Storage Locations

- **GitHub Releases**: Attached to each release
- **Container Registry**: Embedded in Docker images
- **Artifact Repository**: Centralized SBOM storage
- **Documentation Site**: Public SBOM access

### Access Control

- **Public**: Basic SBOM information
- **Partner**: Detailed dependency information
- **Internal**: Complete SBOM with vulnerability data
- **Security Team**: Full supply chain analysis

### Format Standards

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:12345678-1234-5678-9abc-123456789012",
  "version": 1,
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "tools": [
      {
        "vendor": "CycloneDX",
        "name": "cyclonedx-bom",
        "version": "3.11.0"
      }
    ]
  }
}
```

## Integration Points

### CI/CD Integration

```yaml
# .github/workflows/sbom.yml
name: SBOM Generation
on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Generate SBOM
        run: |
          pip install cyclonedx-bom
          cyclonedx-bom requirements -o sbom.json
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom.json
```

### Dependency Management

```bash
# requirements-lock.txt with hashes
pytest==7.4.0 \
    --hash=sha256:abc123... \
    --hash=sha256:def456...
black==23.3.0 \
    --hash=sha256:123abc... \
    --hash=sha256:456def...
```

### Container Integration

```dockerfile
# Multi-stage build with SBOM generation
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install cyclonedx-bom && \
    pip install -r requirements.txt && \
    cyclonedx-bom -r requirements.txt -o /sbom.json

FROM python:3.11-slim as production
COPY --from=builder /sbom.json /app/sbom.json
LABEL org.opencontainers.artifact.description="Reflexion Agent SBOM"
```

## Monitoring and Alerting

### Automated Monitoring

- **New Vulnerabilities**: Daily scans for new CVEs
- **License Changes**: Alert on license policy violations
- **Dependency Updates**: Notification of available updates
- **Supply Chain Attacks**: Monitoring for compromised packages

### Metrics and KPIs

- **Vulnerability Response Time**: Target < 72 hours for critical
- **SBOM Freshness**: Updated within 24 hours of changes
- **License Compliance**: 100% compliant dependencies
- **Supply Chain Risk**: Risk score trending over time

## Contact and Resources

- **SBOM Team**: sbom@your-org.com
- **Security Team**: security@your-org.com
- **Compliance Team**: compliance@your-org.com

### Useful Tools

- [CycloneDX](https://cyclonedx.org/) - SBOM generation
- [Grype](https://github.com/anchore/grype) - Vulnerability scanning
- [Syft](https://github.com/anchore/syft) - SBOM generation for containers
- [SPDX Tools](https://spdx.dev/tools/) - SPDX format tools

### Standards and References

- [NIST SP 800-161](https://csrc.nist.gov/publications/detail/sp/800-161/rev-1/final) - Supply Chain Risk Management
- [CISA SBOM Guide](https://www.cisa.gov/sbom) - SBOM implementation guidance  
- [OpenSSF Supply Chain Security](https://openssf.org/) - Best practices
- [SLSA Framework](https://slsa.dev/) - Supply chain integrity