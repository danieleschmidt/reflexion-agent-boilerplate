"""Docker deployment utilities for reflexion agents."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import DeploymentConfig, DeploymentEnvironment


def generate_dockerfile(config: DeploymentConfig) -> str:
    """Generate Dockerfile for the reflexion service."""
    
    # Base image selection based on environment
    if config.environment == DeploymentEnvironment.PRODUCTION:
        base_image = "python:3.11-slim"
        user_setup = """
# Create non-root user for security
RUN groupadd -r reflexion && useradd -r -g reflexion reflexion
"""
    else:
        base_image = "python:3.11"
        user_setup = ""
    
    dockerfile = f"""# Multi-stage build for production optimization
FROM {base_image} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM base as production

{user_setup}
# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY examples/ /app/examples/
COPY README.md /app/
COPY pyproject.toml /app/

# Install the package
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "from src.reflexion.core.security import health_checker; \\
                   import sys; \\
                   result = health_checker.run_health_checks(); \\
                   sys.exit(0 if result['overall_status'] == 'healthy' else 1)"

# Security: Switch to non-root user
{"USER reflexion" if config.environment == DeploymentEnvironment.PRODUCTION else ""}

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.reflexion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    return dockerfile


def generate_docker_compose(config: DeploymentConfig) -> str:
    """Generate docker-compose.yml for the reflexion service."""
    
    compose = {
        "version": "3.8",
        "services": {
            "reflexion-api": {
                "build": {
                    "context": ".",
                    "target": "production" if config.environment == DeploymentEnvironment.PRODUCTION else "base"
                },
                "ports": ["8000:8000"],
                "environment": {
                    "REFLEXION_ENV": config.environment.value,
                    "REFLEXION_REGION": config.region.value,
                    "REFLEXION_LOG_LEVEL": config.monitoring.log_level,
                    "PYTHONPATH": "/app"
                },
                "volumes": [
                    "./logs:/app/logs",
                    "./data:/app/data"
                ],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "40s"
                },
                "depends_on": ["redis", "postgres"]
            },
            
            "redis": {
                "image": "redis:7-alpine",
                "ports": ["6379:6379"],
                "volumes": ["redis_data:/data"],
                "restart": "unless-stopped",
                "command": "redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru"
            },
            
            "postgres": {
                "image": "postgres:15-alpine",
                "environment": {
                    "POSTGRES_DB": "reflexion",
                    "POSTGRES_USER": "reflexion",
                    "POSTGRES_PASSWORD": "reflexion_pass"
                },
                "ports": ["5432:5432"],
                "volumes": [
                    "postgres_data:/var/lib/postgresql/data",
                    "./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql"
                ],
                "restart": "unless-stopped"
            },
            
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": ["9090:9090"],
                "volumes": [
                    "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                    "prometheus_data:/prometheus"
                ],
                "command": [
                    "--config.file=/etc/prometheus/prometheus.yml",
                    "--storage.tsdb.path=/prometheus",
                    "--web.console.libraries=/etc/prometheus/console_libraries",
                    "--web.console.templates=/etc/prometheus/consoles",
                    "--web.enable-lifecycle"
                ],
                "restart": "unless-stopped"
            },
            
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": ["3000:3000"],
                "environment": {
                    "GF_SECURITY_ADMIN_PASSWORD": "admin"
                },
                "volumes": [
                    "grafana_data:/var/lib/grafana",
                    "./monitoring/dashboards:/var/lib/grafana/dashboards"
                ],
                "restart": "unless-stopped"
            }
        },
        
        "volumes": {
            "redis_data": {},
            "postgres_data": {},
            "prometheus_data": {},
            "grafana_data": {}
        },
        
        "networks": {
            "default": {
                "driver": "bridge"
            }
        }
    }
    
    # Add environment-specific configurations
    if config.environment == DeploymentEnvironment.PRODUCTION:
        # Production security enhancements
        compose["services"]["reflexion-api"]["security_opt"] = ["no-new-privileges:true"]
        compose["services"]["reflexion-api"]["read_only"] = True
        compose["services"]["reflexion-api"]["tmpfs"] = ["/tmp", "/app/logs"]
        
        # Resource limits
        compose["services"]["reflexion-api"]["deploy"] = {
            "resources": {
                "limits": {
                    "cpus": "2.0",
                    "memory": "2G"
                },
                "reservations": {
                    "cpus": "0.5",
                    "memory": "512M"
                }
            }
        }
    
    return json.dumps(compose, indent=2)


def generate_kubernetes_manifests(config: DeploymentConfig) -> Dict[str, str]:
    """Generate Kubernetes manifests for the reflexion service."""
    
    manifests = {}
    
    # Namespace
    manifests["namespace.yaml"] = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: reflexion-{config.environment.value}
  labels:
    app: reflexion
    environment: {config.environment.value}
    region: {config.region.value}
"""
    
    # ConfigMap
    manifests["configmap.yaml"] = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: reflexion-config
  namespace: reflexion-{config.environment.value}
data:
  REFLEXION_ENV: "{config.environment.value}"
  REFLEXION_REGION: "{config.region.value}"
  REFLEXION_LOG_LEVEL: "{config.monitoring.log_level}"
  PYTHONPATH: "/app"
"""
    
    # Secret
    manifests["secret.yaml"] = f"""
apiVersion: v1
kind: Secret
metadata:
  name: reflexion-secrets
  namespace: reflexion-{config.environment.value}
type: Opaque
data:
  # Base64 encoded secrets (replace with actual values)
  openai-api-key: ""
  anthropic-api-key: ""
  jwt-secret: ""
"""
    
    # Deployment
    manifests["deployment.yaml"] = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reflexion-api
  namespace: reflexion-{config.environment.value}
  labels:
    app: reflexion-api
    version: {config.version}
spec:
  replicas: {config.scaling.min_instances}
  selector:
    matchLabels:
      app: reflexion-api
  template:
    metadata:
      labels:
        app: reflexion-api
        version: {config.version}
    spec:
      containers:
      - name: reflexion-api
        image: reflexion-agent:{config.version}
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: reflexion-config
        - secretRef:
            name: reflexion-secrets
        resources:
          limits:
            cpu: 2000m
            memory: 2Gi
          requests:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
"""
    
    # Service
    manifests["service.yaml"] = f"""
apiVersion: v1
kind: Service
metadata:
  name: reflexion-api-service
  namespace: reflexion-{config.environment.value}
  labels:
    app: reflexion-api
spec:
  selector:
    app: reflexion-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
"""
    
    # HorizontalPodAutoscaler
    manifests["hpa.yaml"] = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reflexion-api-hpa
  namespace: reflexion-{config.environment.value}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reflexion-api
  minReplicas: {config.scaling.min_instances}
  maxReplicas: {config.scaling.max_instances}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {int(config.scaling.target_cpu_utilization)}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {int(config.scaling.target_memory_utilization)}
"""
    
    # Ingress (if production)
    if config.environment == DeploymentEnvironment.PRODUCTION:
        manifests["ingress.yaml"] = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: reflexion-api-ingress
  namespace: reflexion-{config.environment.value}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "{config.security.api_rate_limit}"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.reflexion.your-domain.com
    secretName: reflexion-tls
  rules:
  - host: api.reflexion.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: reflexion-api-service
            port:
              number: 80
"""
    
    return manifests


def save_deployment_files(config: DeploymentConfig, output_dir: str = "./deployment"):
    """Save all deployment files to the specified directory."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate and save Dockerfile
    dockerfile_content = generate_dockerfile(config)
    (output_path / "Dockerfile").write_text(dockerfile_content)
    
    # Generate and save docker-compose.yml
    compose_content = generate_docker_compose(config)
    (output_path / "docker-compose.yml").write_text(compose_content)
    
    # Generate and save Kubernetes manifests
    k8s_dir = output_path / "k8s"
    k8s_dir.mkdir(exist_ok=True)
    
    manifests = generate_kubernetes_manifests(config)
    for filename, content in manifests.items():
        (k8s_dir / filename).write_text(content)
    
    # Save configuration as JSON for reference
    config_dict = {
        "environment": config.environment.value,
        "region": config.region.value,
        "version": config.version,
        "debug": config.debug,
        "scaling": {
            "min_instances": config.scaling.min_instances,
            "max_instances": config.scaling.max_instances,
            "target_cpu_utilization": config.scaling.target_cpu_utilization,
            "target_memory_utilization": config.scaling.target_memory_utilization
        },
        "monitoring": {
            "log_level": config.monitoring.log_level,
            "enable_metrics": config.monitoring.enable_metrics,
            "enable_tracing": config.monitoring.enable_tracing
        }
    }
    
    (output_path / "config.json").write_text(json.dumps(config_dict, indent=2))
    
    print(f"✅ Deployment files generated in {output_path}")
    print(f"📁 Files created:")
    print(f"  - Dockerfile")
    print(f"  - docker-compose.yml")
    print(f"  - k8s/ (Kubernetes manifests)")
    print(f"  - config.json")
    
    return output_path