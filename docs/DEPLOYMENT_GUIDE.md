# Deployment Guide

## Overview

This guide covers deploying the Reflexion Agent Boilerplate across different environments, from development to enterprise production deployments.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.11+
- 2 GB RAM
- 1 GB disk space
- Docker 20.10+ (for containerized deployments)

**Recommended Production:**
- Python 3.13
- 8 GB RAM
- 10 GB disk space
- 4+ CPU cores
- Docker 24.0+
- Kubernetes 1.25+

### Dependencies

**Core Dependencies:**
```bash
pip install -e .
```

**Optional Dependencies:**
```bash
# For AutoGen integration
pip install autogen

# For CrewAI integration  
pip install crewai

# For LangChain integration
pip install langchain

# For enhanced monitoring
pip install psutil
```

## Environment Configurations

### Development Environment

**Configuration:**
```python
config = {
    'llm': 'gpt-4',
    'max_iterations': 2,
    'success_threshold': 0.7,
    'enable_caching': True,
    'enable_parallel_execution': False,
    'max_concurrent_tasks': 2,
    'cache_size': 500,
    'memory_path': './dev_memory.json'
}
```

**Environment Variables:**
```bash
export REFLEXION_ENV=development
export REFLEXION_LOG_LEVEL=DEBUG
export REFLEXION_CACHE_SIZE=500
export REFLEXION_MAX_WORKERS=2
```

### Staging Environment

**Configuration:**
```python
config = {
    'llm': 'gpt-4',
    'max_iterations': 3,
    'success_threshold': 0.8,
    'enable_caching': True,
    'enable_parallel_execution': True,
    'max_concurrent_tasks': 4,
    'cache_size': 1000,
    'memory_path': './staging_memory.json'
}
```

**Environment Variables:**
```bash
export REFLEXION_ENV=staging
export REFLEXION_LOG_LEVEL=INFO
export REFLEXION_CACHE_SIZE=1000
export REFLEXION_MAX_WORKERS=4
export REFLEXION_HEALTH_CHECK_INTERVAL=30
```

### Production Environment

**Configuration:**
```python
config = {
    'llm': 'gpt-4',
    'max_iterations': 3,
    'success_threshold': 0.8,
    'enable_caching': True,
    'enable_parallel_execution': True,
    'max_concurrent_tasks': 8,
    'cache_size': 2000,
    'memory_path': '/app/data/production_memory.json'
}
```

**Environment Variables:**
```bash
export REFLEXION_ENV=production
export REFLEXION_LOG_LEVEL=INFO
export REFLEXION_CACHE_SIZE=2000
export REFLEXION_MAX_WORKERS=8
export REFLEXION_HEALTH_CHECK_INTERVAL=30
export REFLEXION_CIRCUIT_BREAKER_THRESHOLD=5
export REFLEXION_RATE_LIMIT_RPM=100
```

## Docker Deployment

### Building the Image

**Development Image:**
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install -e .

COPY src/ ./src/
EXPOSE 8000

CMD ["python", "-c", "from reflexion import ReflexionAgent; print('Development ready')"]
```

**Production Image:**
```bash
# Use the provided production Dockerfile
docker build -f deployment/production/Dockerfile -t reflexion-agent:production .
```

### Running Containers

**Development:**
```bash
docker run -d \
  --name reflexion-dev \
  -e REFLEXION_ENV=development \
  -e REFLEXION_LOG_LEVEL=DEBUG \
  -v $(pwd)/logs:/app/logs \
  -p 8000:8000 \
  reflexion-agent:dev
```

**Production:**
```bash
docker run -d \
  --name reflexion-prod \
  -e REFLEXION_ENV=production \
  -e REFLEXION_LOG_LEVEL=INFO \
  -e REFLEXION_CACHE_SIZE=2000 \
  -e REFLEXION_MAX_WORKERS=8 \
  -v /opt/reflexion/data:/app/data \
  -v /opt/reflexion/logs:/app/logs \
  -p 8000:8000 \
  --memory=4g \
  --cpus=2.0 \
  --health-cmd="python -c 'from reflexion.core.health import health_checker; print(health_checker.is_healthy())'" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  reflexion-agent:production
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  reflexion-agent:
    build:
      context: .
      dockerfile: deployment/production/Dockerfile
    environment:
      - REFLEXION_ENV=production
      - REFLEXION_LOG_LEVEL=INFO
      - REFLEXION_CACHE_SIZE=2000
      - REFLEXION_MAX_WORKERS=8
      - REFLEXION_HEALTH_CHECK_INTERVAL=30
    volumes:
      - reflexion-data:/app/data
      - reflexion-logs:/app/logs
      - reflexion-cache:/app/cache
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "python", "-c", "from reflexion.core.health import health_checker; exit(0 if health_checker.is_healthy() else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

volumes:
  reflexion-data:
  reflexion-logs:
  reflexion-cache:
  redis-data:
```

**Running with Compose:**
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f reflexion-agent

# Scale the service
docker-compose up -d --scale reflexion-agent=3

# Health check
docker-compose ps
```

## Kubernetes Deployment

### Namespace and ConfigMap

**namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: reflexion
```

**configmap.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: reflexion-config
  namespace: reflexion
data:
  REFLEXION_ENV: "production"
  REFLEXION_LOG_LEVEL: "INFO"
  REFLEXION_CACHE_SIZE: "2000"
  REFLEXION_MAX_WORKERS: "8"
  REFLEXION_HEALTH_CHECK_INTERVAL: "30"
```

### Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reflexion-agent
  namespace: reflexion
  labels:
    app: reflexion-agent
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
        image: reflexion-agent:production
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: reflexion-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: reflexion-data
          mountPath: /app/data
        - name: reflexion-logs
          mountPath: /app/logs
        - name: reflexion-cache
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      volumes:
      - name: reflexion-data
        persistentVolumeClaim:
          claimName: reflexion-data-pvc
      - name: reflexion-logs
        persistentVolumeClaim:
          claimName: reflexion-logs-pvc
      - name: reflexion-cache
        emptyDir:
          sizeLimit: 1Gi
```

### Persistent Storage

**pvc.yaml:**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: reflexion-data-pvc
  namespace: reflexion
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: reflexion-logs-pvc
  namespace: reflexion
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: standard
  resources:
    requests:
      storage: 5Gi
```

### Service and Ingress

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: reflexion-agent-service
  namespace: reflexion
spec:
  selector:
    app: reflexion-agent
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
```

**ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: reflexion-agent-ingress
  namespace: reflexion
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - reflexion.yourdomain.com
    secretName: reflexion-tls
  rules:
  - host: reflexion.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: reflexion-agent-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

**hpa.yaml:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reflexion-agent-hpa
  namespace: reflexion
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reflexion-agent
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Deployment Commands

```bash
# Apply all Kubernetes manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Verify deployment
kubectl get pods -n reflexion
kubectl get services -n reflexion
kubectl get ingress -n reflexion

# View logs
kubectl logs -f deployment/reflexion-agent -n reflexion

# Scale deployment
kubectl scale deployment reflexion-agent --replicas=5 -n reflexion

# Rolling update
kubectl set image deployment/reflexion-agent reflexion-agent=reflexion-agent:v2.0 -n reflexion
kubectl rollout status deployment/reflexion-agent -n reflexion
```

## Cloud Provider Specific Deployments

### AWS ECS/Fargate

**Task Definition:**
```json
{
  "family": "reflexion-agent",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/reflexionTaskRole",
  "containerDefinitions": [
    {
      "name": "reflexion-agent",
      "image": "your-ecr-repo/reflexion-agent:production",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "REFLEXION_ENV", "value": "production"},
        {"name": "REFLEXION_LOG_LEVEL", "value": "INFO"},
        {"name": "REFLEXION_CACHE_SIZE", "value": "2000"}
      ],
      "mountPoints": [
        {
          "sourceVolume": "reflexion-data",
          "containerPath": "/app/data"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c 'from reflexion.core.health import health_checker; exit(0 if health_checker.is_healthy() else 1)'"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/reflexion-agent",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "reflexion-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "rootDirectory": "/reflexion-data"
      }
    }
  ]
}
```

### Google Cloud Run

**service.yaml:**
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: reflexion-agent
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/max-instances: "10"
        run.googleapis.com/min-instances: "1"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/your-project/reflexion-agent:production
        env:
        - name: REFLEXION_ENV
          value: "production"
        - name: REFLEXION_LOG_LEVEL
          value: "INFO"
        - name: REFLEXION_CACHE_SIZE
          value: "2000"
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          timeoutSeconds: 10
          failureThreshold: 3
          periodSeconds: 30
```

### Azure Container Instances

**deployment-template.json:**
```json
{
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerName": {
      "type": "string",
      "defaultValue": "reflexion-agent"
    }
  },
  "variables": {},
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-09-01",
      "name": "[parameters('containerName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "[parameters('containerName')]",
            "properties": {
              "image": "your-acr-repo.azurecr.io/reflexion-agent:production",
              "ports": [
                {
                  "port": 8000,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "REFLEXION_ENV",
                  "value": "production"
                },
                {
                  "name": "REFLEXION_LOG_LEVEL",
                  "value": "INFO"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 2,
                  "memoryInGB": 4
                }
              },
              "volumeMounts": [
                {
                  "name": "reflexion-data",
                  "mountPath": "/app/data"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ]
        },
        "volumes": [
          {
            "name": "reflexion-data",
            "azureFile": {
              "shareName": "reflexion-data",
              "storageAccountName": "your-storage-account",
              "storageAccountKey": "your-storage-key"
            }
          }
        ]
      }
    }
  ]
}
```

## Monitoring and Observability

### Health Endpoints

The application exposes several health check endpoints:

```python
# Health check endpoint
GET /health
Response: {"status": "healthy", "timestamp": "...", "uptime": "..."}

# Readiness check
GET /ready  
Response: {"ready": true, "checks": {...}}

# Metrics endpoint
GET /metrics
Response: {"performance": {...}, "system": {...}}
```

### Logging Configuration

**Production Logging:**
```python
import logging
from pythonjsonlogger import jsonlogger

# Configure structured logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

### Monitoring Integration

**Prometheus Metrics:**
```python
# Add to your application
from prometheus_client import Counter, Histogram, Gauge

TASK_COUNTER = Counter('reflexion_tasks_total', 'Total tasks processed')
TASK_DURATION = Histogram('reflexion_task_duration_seconds', 'Task duration')
CACHE_HIT_RATE = Gauge('reflexion_cache_hit_rate', 'Cache hit rate')
```

### Alert Rules

**alerts.yaml:**
```yaml
groups:
- name: reflexion-agent
  rules:
  - alert: ReflexionAgentDown
    expr: up{job="reflexion-agent"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Reflexion agent is down"
      
  - alert: HighErrorRate
    expr: rate(reflexion_tasks_failed_total[5m]) / rate(reflexion_tasks_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: LowCacheHitRate
    expr: reflexion_cache_hit_rate < 0.3
    for: 5m
    labels:
      severity: info
    annotations:
      summary: "Cache hit rate is low"
```

## Security Considerations

### Container Security

**Security Hardening:**
```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' --uid 1001 appuser
USER appuser

# Read-only filesystem where possible
RUN chmod -R 755 /app && chown -R appuser:appuser /app

# Minimal base image
FROM python:3.13-slim
```

### Network Security

**Network Policies (Kubernetes):**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: reflexion-agent-netpol
  namespace: reflexion
spec:
  podSelector:
    matchLabels:
      app: reflexion-agent
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Secret Management

**Kubernetes Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: reflexion-secrets
  namespace: reflexion
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  database-url: <base64-encoded-url>
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check container memory usage
   docker stats reflexion-agent
   
   # Kubernetes resource usage
   kubectl top pod -n reflexion
   ```

2. **Performance Issues**
   ```bash
   # Check cache hit rates
   curl http://localhost:8000/metrics | grep cache_hit_rate
   
   # Review application logs
   kubectl logs -f deployment/reflexion-agent -n reflexion
   ```

3. **Health Check Failures**
   ```bash
   # Manual health check
   curl http://localhost:8000/health
   
   # Check dependencies
   docker exec reflexion-agent python -c "from reflexion.core.health import health_checker; print(health_checker.get_health_status())"
   ```

### Debug Commands

```bash
# Enter container for debugging
kubectl exec -it deployment/reflexion-agent -n reflexion -- /bin/bash

# View application configuration
docker exec reflexion-agent env | grep REFLEXION

# Test application functionality
docker exec reflexion-agent python -c "from reflexion import ReflexionAgent; agent = ReflexionAgent('gpt-4'); print('OK')"
```

## Backup and Recovery

### Data Backup

```bash
# Backup memory data
kubectl cp reflexion/reflexion-agent-pod:/app/data/production_memory.json ./backup/memory-$(date +%Y%m%d).json

# Backup logs
kubectl cp reflexion/reflexion-agent-pod:/app/logs ./backup/logs-$(date +%Y%m%d)/
```

### Disaster Recovery

```bash
# Restore from backup
kubectl cp ./backup/memory-20240101.json reflexion/reflexion-agent-pod:/app/data/production_memory.json

# Restart services
kubectl rollout restart deployment/reflexion-agent -n reflexion
```

## Performance Tuning

### JVM-like Tuning for Python

```bash
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
```

### Resource Optimization

```yaml
# Kubernetes resource tuning
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

---

*This deployment guide provides comprehensive instructions for deploying the Reflexion Agent Boilerplate across various environments and cloud platforms, ensuring optimal performance, security, and reliability.*