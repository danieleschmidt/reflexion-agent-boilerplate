# Reflexion Agent Production Deployment

## Overview

This directory contains production deployment files for the Reflexion Agent service.

**Environment**: production
**Region**: us-east-1  
**Version**: 1.0.0

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
docker build -t reflexion-agent:1.0.0 .
docker push your-registry/reflexion-agent:1.0.0
```

2. Update secrets in k8s/secret.yaml

3. Deploy to Kubernetes:
```bash
kubectl apply -f k8s/
```

## Configuration

### Environment Variables
- `REFLEXION_ENV`: production
- `REFLEXION_REGION`: us-east-1
- `REFLEXION_LOG_LEVEL`: INFO

### Secrets Required
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `JWT_SECRET`: JWT signing secret

## Monitoring

- **Metrics**: Prometheus endpoint at `/metrics`
- **Health**: Health check at `/health`
- **Logs**: Structured JSON logs to stdout

## Scaling

- **Min instances**: 2
- **Max instances**: 10
- **CPU target**: 70.0%
- **Memory target**: 80.0%

## Security

- TLS encryption in transit
- Non-root container user
- Resource limits enforced
- Rate limiting: 1000 req/min

## Compliance

- **GDPR**: Disabled
- **CCPA**: Enabled
- **PDPA**: Enabled
- **Data retention**: 365 days

## Support

For issues and questions, please refer to the main repository documentation.
