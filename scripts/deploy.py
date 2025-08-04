#!/usr/bin/env python3
"""Deployment script for Reflexion Agent Boilerplate."""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflexion.deployment import (
    create_config, apply_regional_config, save_deployment_files,
    DeploymentEnvironment, Region
)


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy Reflexion Agent Boilerplate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/deploy.py --env production --region us-east-1
  python scripts/deploy.py --env staging --region eu-west-1 --output ./staging-deploy
  python scripts/deploy.py --env development --local
        """
    )
    
    parser.add_argument(
        "--env", "--environment",
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment (default: development)"
    )
    
    parser.add_argument(
        "--region",
        choices=["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"],
        default="us-east-1",
        help="Deployment region (default: us-east-1)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="./deployment",
        help="Output directory for deployment files (default: ./deployment)"
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Generate files for local development only"
    )
    
    parser.add_argument(
        "--min-instances",
        type=int,
        help="Minimum number of instances"
    )
    
    parser.add_argument(
        "--max-instances", 
        type=int,
        help="Maximum number of instances"
    )
    
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Application version (default: 1.0.0)"
    )
    
    parser.add_argument(
        "--skip-k8s",
        action="store_true",
        help="Skip Kubernetes manifest generation"
    )
    
    args = parser.parse_args()
    
    print("🚀 Reflexion Agent Deployment")
    print("=" * 40)
    print(f"Environment: {args.env}")
    print(f"Region: {args.region}")
    print(f"Output: {args.output}")
    print(f"Version: {args.version}")
    print()
    
    try:
        # Create base configuration
        config_overrides = {
            "version": args.version
        }
        
        if args.min_instances:
            config_overrides["scaling__min_instances"] = args.min_instances
        
        if args.max_instances:
            config_overrides["scaling__max_instances"] = args.max_instances
        
        config = create_config(
            environment=args.env,
            region=args.region,
            **config_overrides
        )
        
        # Apply regional configurations
        config = apply_regional_config(config)
        
        # Local development adjustments
        if args.local:
            config.scaling.min_instances = 1
            config.scaling.max_instances = 2
            config.monitoring.log_level = "DEBUG"
            config.debug = True
            print("🏠 Configured for local development")
        
        # Generate deployment files
        print("📝 Generating deployment files...")
        
        output_path = save_deployment_files(config, args.output)
        
        # Generate additional files for production
        if config.environment == DeploymentEnvironment.PRODUCTION:
            generate_production_files(output_path, config)
        
        print()
        print("✅ Deployment files generated successfully!")
        print()
        print("📋 Next Steps:")
        
        if args.local:
            print("  1. Review the generated docker-compose.yml")
            print("  2. Run: docker-compose up -d")
            print("  3. Access the service at http://localhost:8000")
        
        elif config.environment == DeploymentEnvironment.PRODUCTION:
            print("  1. Review all generated files in the deployment directory")
            print("  2. Update secrets in k8s/secret.yaml with actual values")
            print("  3. Build and push the Docker image:")
            print(f"     docker build -t reflexion-agent:{config.version} .")
            print(f"     docker push reflexion-agent:{config.version}")
            print("  4. Apply Kubernetes manifests:")
            print("     kubectl apply -f deployment/k8s/")
            print("  5. Monitor the deployment:")
            print("     kubectl get pods -n reflexion-production")
        
        else:
            print("  1. Review the generated files")
            print("  2. Build the Docker image if needed")
            print("  3. Deploy using docker-compose or Kubernetes")
        
        print()
        print(f"📁 Files location: {output_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return 1
    
    return 0


def generate_production_files(output_path: Path, config):
    """Generate additional production files."""
    
    # Generate production README
    readme_content = f"""# Reflexion Agent Production Deployment

## Overview

This directory contains production deployment files for the Reflexion Agent service.

**Environment**: {config.environment.value}
**Region**: {config.region.value}  
**Version**: {config.version}

## Files

- `Dockerfile` - Container image definition
- `docker-compose.yml` - Multi-service local deployment
- `k8s/` - Kubernetes manifests for production deployment
- `config.json` - Deployment configuration reference

## Prerequisites

### For Docker Deployment
- Docker Engine 20.10+
- Docker Compose 2.0+

### For Kubernetes Deployment
- Kubernetes cluster 1.21+
- kubectl configured
- Helm 3.0+ (optional)

## Quick Start

### Local Development
```bash
docker-compose up -d
```

### Production Deployment
1. Build and push image:
```bash
docker build -t reflexion-agent:{config.version} .
docker push your-registry/reflexion-agent:{config.version}
```

2. Update secrets in k8s/secret.yaml

3. Deploy to Kubernetes:
```bash
kubectl apply -f k8s/
```

## Configuration

### Environment Variables
- `REFLEXION_ENV`: {config.environment.value}
- `REFLEXION_REGION`: {config.region.value}
- `REFLEXION_LOG_LEVEL`: {config.monitoring.log_level}

### Secrets Required
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `JWT_SECRET`: JWT signing secret

## Monitoring

- **Metrics**: Prometheus endpoint at `/metrics`
- **Health**: Health check at `/health`
- **Logs**: Structured JSON logs to stdout

## Scaling

- **Min instances**: {config.scaling.min_instances}
- **Max instances**: {config.scaling.max_instances}
- **CPU target**: {config.scaling.target_cpu_utilization}%
- **Memory target**: {config.scaling.target_memory_utilization}%

## Security

- TLS encryption in transit
- Non-root container user
- Resource limits enforced
- Rate limiting: {config.security.api_rate_limit} req/min

## Compliance

- **GDPR**: {'Enabled' if config.compliance.gdpr_enabled else 'Disabled'}
- **CCPA**: {'Enabled' if config.compliance.ccpa_enabled else 'Disabled'}
- **PDPA**: {'Enabled' if config.compliance.pdpa_enabled else 'Disabled'}
- **Data retention**: {config.compliance.data_retention_days} days

## Support

For issues and questions, please refer to the main repository documentation.
"""
    
    (output_path / "README.md").write_text(readme_content)
    
    # Generate deployment checklist
    checklist_content = """# Production Deployment Checklist

## Pre-Deployment
- [ ] Review all configuration files
- [ ] Update secrets with production values
- [ ] Build and test Docker image locally
- [ ] Verify network connectivity to external services
- [ ] Confirm resource quotas and limits

## Security
- [ ] Secrets are stored securely (not in plain text)
- [ ] TLS certificates are valid
- [ ] Network policies are configured
- [ ] RBAC permissions are minimal
- [ ] Container security scanning passed

## Monitoring
- [ ] Prometheus scraping configured
- [ ] Grafana dashboards imported
- [ ] Alert rules configured
- [ ] Log aggregation working
- [ ] Health checks responding

## Deployment
- [ ] Namespace created
- [ ] ConfigMaps applied
- [ ] Secrets applied
- [ ] Deployments applied
- [ ] Services accessible
- [ ] Ingress routing works

## Post-Deployment
- [ ] All pods are running
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs flowing to aggregation
- [ ] Auto-scaling working
- [ ] Load testing completed

## Rollback Plan
- [ ] Previous version image available
- [ ] Rollback procedure documented
- [ ] Database migration rollback plan
- [ ] Monitoring for rollback triggers
"""
    
    (output_path / "DEPLOYMENT_CHECKLIST.md").write_text(checklist_content)
    
    print("📋 Generated production documentation")


if __name__ == "__main__":
    sys.exit(main())