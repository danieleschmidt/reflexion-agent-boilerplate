# Deployment Guide

This document covers deployment strategies and configurations for the Reflexion Agent Boilerplate.

## Deployment Strategies

### 1. Container Deployment

#### Basic Docker Deployment
```bash
# Build image
docker build -t reflexion-agent:latest .

# Run container
docker run -d \
  --name reflexion-agent \
  -p 8000:8000 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  reflexion-agent:latest
```

#### Docker Compose (Recommended)
```yaml
version: '3.8'
services:
  reflexion-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: reflexion
      POSTGRES_USER: reflexion
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### 2. Kubernetes Deployment

#### Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reflexion-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reflexion-agent
  template:
    metadata:
      labels:
        app: reflexion-agent
    spec:
      containers:
      - name: reflexion-agent
        image: reflexion-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: reflexion-agent-service
spec:
  selector:
    app: reflexion-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 3. Cloud Deployments

#### AWS ECS
```json
{
  "family": "reflexion-agent",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "reflexion-agent",
      "image": "your-account.dkr.ecr.region.amazonaws.com/reflexion-agent:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:reflexion/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/reflexion-agent",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: reflexion-agent
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/PROJECT_ID/reflexion-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        resources:
          limits:
            cpu: 1000m
            memory: 512Mi
```

## Environment Configuration

### Production Environment Variables
```bash
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false

# API Configuration
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/reflexion
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
CORS_ORIGINS=https://your-frontend.com

# Monitoring
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_METRICS=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
```

### Health Checks
```python
from fastapi import FastAPI
from reflexion.health import HealthChecker

app = FastAPI()
health_checker = HealthChecker()

@app.get("/health")
async def health_check():
    return await health_checker.check_all()

@app.get("/ready")
async def readiness_check():
    return await health_checker.check_ready()
```

## Scaling Considerations

### Horizontal Scaling
- Use stateless design
- Implement proper session management
- Use external storage for memory/cache
- Consider message queues for async processing

### Vertical Scaling
- Monitor memory usage for episodic memory
- Scale CPU for compute-intensive reflections
- Optimize database queries and indexing

### Auto-scaling Metrics
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reflexion-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reflexion-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Security

### Container Security
```dockerfile
# Use non-root user
FROM python:3.11-slim
RUN groupadd -r reflexion && useradd -r -g reflexion reflexion
USER reflexion

# Remove unnecessary packages
RUN apt-get autoremove -y && apt-get clean
```

### Network Security
- Use TLS/SSL for all external communications
- Implement proper firewall rules
- Use VPC/private networks where possible
- Enable network policies in Kubernetes

### Secret Management
- Use external secret managers (AWS Secrets Manager, Azure Key Vault)
- Never store secrets in container images
- Rotate secrets regularly
- Use service mesh for internal communication

## Monitoring and Logging

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "reflection_completed",
    task_id="task-123",
    iterations=3,
    success=True,
    duration_ms=1250
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

reflection_counter = Counter('reflexion_total', 'Total reflections')
reflection_duration = Histogram('reflexion_duration_seconds', 'Reflection duration')

@reflection_duration.time()
def perform_reflection():
    reflection_counter.inc()
    # Reflection logic here
```

## Disaster Recovery

### Backup Strategy
- Database backups with point-in-time recovery
- Configuration backups
- Container image versioning
- Memory state snapshots

### Recovery Procedures
1. Assess failure scope and impact
2. Restore from latest backup
3. Verify data integrity
4. Gradually restore traffic
5. Monitor for cascading failures

### Business Continuity
- Multi-region deployments
- Circuit breakers for external dependencies
- Graceful degradation strategies
- Incident response procedures