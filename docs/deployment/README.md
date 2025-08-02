# Deployment Guide

This guide covers deploying the Reflexion Agent Boilerplate in various environments.

## 🚀 Quick Start

### Local Development
```bash
# Start development environment
docker-compose --profile dev up -d

# Or use the deployment script
./scripts/deploy.sh -e development -p dev deploy
```

### Production Deployment
```bash
# Build and deploy
./scripts/build.sh -t production --lint --test --security
./scripts/deploy.sh -e production deploy
```

## 🏗️ Build Process

### Using the Build Script

The build script provides a standardized way to build Docker images:

```bash
# Basic production build
./scripts/build.sh

# Development build with version
./scripts/build.sh -t development -v 1.0.0

# Full quality pipeline
./scripts/build.sh --lint --test --security

# Build and push to registry
./scripts/build.sh -r my-registry.com -p
```

#### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --target` | Build target (production, development, test) | production |
| `-v, --version` | Version tag for the image | auto-detected |
| `-r, --registry` | Docker registry URL | none |
| `-p, --push` | Push image to registry after build | false |
| `--lint` | Run linting before build | false |
| `--test` | Run tests before build | false |
| `--security` | Run security scan after build | false |

### Manual Docker Build

```bash
# Production build
docker build --target production -t reflexion-agent:latest .

# Development build
docker build --target builder -t reflexion-agent:dev .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t reflexion-agent:latest .
```

## 🚀 Deployment Process

### Using the Deployment Script

```bash
# Deploy to development
./scripts/deploy.sh -e development deploy

# Deploy to production with monitoring
./scripts/deploy.sh -e production -p monitoring deploy

# Check deployment status
./scripts/deploy.sh status

# View logs
./scripts/deploy.sh logs reflexion-app
```

#### Deployment Commands

| Command | Description |
|---------|-------------|
| `deploy` | Deploy the application |
| `stop` | Stop the application |
| `restart` | Restart the application |
| `status` | Show deployment status |
| `logs` | Show application logs |
| `backup` | Create backup |
| `health` | Check application health |

### Manual Docker Compose

```bash
# Development environment
docker-compose --profile dev up -d

# Full production environment
docker-compose --profile full up -d

# With monitoring
docker-compose --profile full --profile monitoring up -d
```

## 🏗️ Environment Configuration

### Environment Files

Create environment-specific configuration files:

```bash
# Development
.env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Staging
.env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Production
.env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
```

### Required Environment Variables

#### Core Application
```env
# LLM Configuration
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/reflexion
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
```

#### Production Additional
```env
# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/cert.pem
SSL_KEY_PATH=/etc/ssl/private/key.pem

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENABLED=true
METRICS_ENDPOINT=/metrics

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=300
```

## 🌐 Docker Compose Profiles

### Available Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| `dev` | App (dev), Redis, PostgreSQL | Development |
| `full` | App (prod), Redis, PostgreSQL, Nginx | Production |
| `monitoring` | Prometheus, Grafana | Monitoring |

### Profile Combinations

```bash
# Development with monitoring
docker-compose --profile dev --profile monitoring up -d

# Full production stack
docker-compose --profile full --profile monitoring up -d
```

## 🔧 Service Configuration

### Application Service

```yaml
reflexion-app:
  image: reflexion-agent:latest
  restart: unless-stopped
  environment:
    - PYTHONPATH=/app/src
    - LOG_LEVEL=INFO
  volumes:
    - ./data:/app/data
    - ./logs:/app/logs
  healthcheck:
    test: ["CMD", "reflexion", "--version"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Database Services

```yaml
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: reflexion
    POSTGRES_USER: reflexion
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
  volumes:
    - postgres-data:/var/lib/postgresql/data
    - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql

redis:
  image: redis:7-alpine
  command: redis-server --appendonly yes
  volumes:
    - redis-data:/data
```

## 🔒 Security Considerations

### Production Security

1. **Use secrets management**:
   ```bash
   # Docker Swarm secrets
   docker secret create postgres_password /path/to/password
   
   # External secrets (AWS Secrets Manager, etc.)
   export POSTGRES_PASSWORD=$(aws secretsmanager get-secret-value --secret-id prod/postgres --query SecretString --output text)
   ```

2. **Network security**:
   ```yaml
   networks:
     reflexion-network:
       driver: bridge
       ipam:
         config:
           - subnet: 172.20.0.0/16
   ```

3. **Container security**:
   - Use non-root users
   - Read-only root filesystems where possible
   - Security scanning with Docker Scout or Trivy

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://reflexion-app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'reflexion-app'
    static_configs:
      - targets: ['reflexion-app:8000']
    metrics_path: /metrics
```

### Grafana Dashboards

Pre-configured dashboards are available in `monitoring/dashboards/`:
- `reflexion-dashboard.json` - Application metrics
- `performance-dashboard.json` - Performance metrics

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and Deploy
        run: |
          ./scripts/build.sh --lint --test --security
          ./scripts/deploy.sh -e production deploy
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - ./scripts/build.sh -t production

deploy:
  stage: deploy
  script:
    - ./scripts/deploy.sh -e production deploy
  only:
    - main
```

## 🗄️ Backup and Recovery

### Automated Backups

```bash
# Create backup
./scripts/deploy.sh backup

# Schedule backups with cron
0 2 * * * /path/to/scripts/deploy.sh backup
```

### Manual Backup

```bash
# Database backup
docker-compose exec postgres pg_dump -U reflexion reflexion > backup.sql

# Redis backup
docker-compose exec redis redis-cli BGSAVE
docker cp reflexion-redis:/data/dump.rdb ./redis-backup.rdb

# Application data backup
tar -czf data-backup.tar.gz data/
```

### Recovery

```bash
# Restore database
docker-compose exec -T postgres psql -U reflexion reflexion < backup.sql

# Restore Redis
docker cp ./redis-backup.rdb reflexion-redis:/data/dump.rdb
docker-compose restart redis

# Restore application data
tar -xzf data-backup.tar.gz
```

## 🚨 Troubleshooting

### Common Issues

1. **Container won't start**:
   ```bash
   # Check logs
   docker-compose logs reflexion-app
   
   # Check container status
   docker-compose ps
   ```

2. **Database connection issues**:
   ```bash
   # Test database connection
   docker-compose exec postgres pg_isready -U reflexion
   
   # Check network connectivity
   docker-compose exec reflexion-app ping postgres
   ```

3. **Permission issues**:
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 data/ logs/
   ```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
reflexion-app:
  deploy:
    replicas: 3
    resources:
      limits:
        cpus: '1'
        memory: 1G
```

### Load Balancing

```nginx
# nginx.conf
upstream reflexion_backend {
    server reflexion-app-1:8000;
    server reflexion-app-2:8000;
    server reflexion-app-3:8000;
}

server {
    location / {
        proxy_pass http://reflexion_backend;
    }
}
```

## 🔧 Performance Tuning

### Application Tuning

```env
# Worker configuration
MAX_WORKERS=4
WORKER_TIMEOUT=300
WORKER_CLASS=uvicorn.workers.UvicornWorker

# Database tuning
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

### Resource Limits

```yaml
services:
  reflexion-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

---

For more detailed information, see:
- [Build Scripts Documentation](./build-scripts.md)
- [Security Guide](../security.md)
- [Monitoring Guide](../monitoring/monitoring-guide.md)