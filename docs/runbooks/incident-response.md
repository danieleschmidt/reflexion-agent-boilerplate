# Incident Response Runbook

This runbook provides step-by-step procedures for responding to common incidents in the Reflexion Agent system.

## 🚨 Incident Classification

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P0 - Critical | System completely down | 15 minutes | Total service outage, data loss |
| P1 - High | Major functionality impacted | 1 hour | Core features unavailable, security breach |
| P2 - Medium | Partial functionality impacted | 4 hours | Performance degradation, non-critical features down |
| P3 - Low | Minor issues | 24 hours | UI glitches, documentation errors |

## 🔍 Initial Response Procedure

### 1. Immediate Assessment (First 5 minutes)

```bash
# Quick health check
./scripts/deploy.sh health

# Check service status
docker-compose ps

# Check recent logs
docker-compose logs --tail=100 reflexion-app
```

### 2. Escalation Criteria

- **Escalate to P0**: System completely inaccessible or data corruption detected
- **Escalate to P1**: Multiple users affected, security concerns, or core functionality down
- **Keep as P2+**: Single user affected, performance issues, or cosmetic problems

### 3. Communication

- **Internal**: Update incident tracking system (e.g., PagerDuty, Opsgenie)
- **External**: Notify users via status page if P0/P1 incident
- **Stakeholders**: Inform management for P0/P1 incidents

## 🔧 Common Incident Types

### Application Not Responding

#### Symptoms
- HTTP 500/502/503 errors
- Timeouts on API requests
- Health check failures

#### Investigation Steps
```bash
# 1. Check container status
docker-compose ps

# 2. Check application logs
docker-compose logs reflexion-app --tail=200

# 3. Check resource usage
docker stats

# 4. Check system resources
df -h
free -m
```

#### Resolution Steps
```bash
# 1. Try graceful restart
docker-compose restart reflexion-app

# 2. If restart fails, check for stuck processes
docker-compose exec reflexion-app ps aux

# 3. Force container recreation
docker-compose up -d --force-recreate reflexion-app

# 4. Check for data corruption
docker-compose exec reflexion-app python -c "from src.reflexion import health_check; health_check()"
```

### Database Connection Issues

#### Symptoms
- "Connection refused" errors
- Slow database queries
- Connection pool exhaustion

#### Investigation Steps
```bash
# 1. Check PostgreSQL status
docker-compose exec postgres pg_isready -U reflexion

# 2. Check active connections
docker-compose exec postgres psql -U reflexion -c "SELECT count(*) FROM pg_stat_activity;"

# 3. Check database logs
docker-compose logs postgres --tail=100

# 4. Check disk space
docker-compose exec postgres df -h
```

#### Resolution Steps
```bash
# 1. Restart PostgreSQL if needed
docker-compose restart postgres

# 2. Clear connection pool if stuck
docker-compose restart reflexion-app

# 3. Check for long-running queries
docker-compose exec postgres psql -U reflexion -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"

# 4. Kill long-running queries if necessary
docker-compose exec postgres psql -U reflexion -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes';"
```

### High Memory Usage

#### Symptoms
- Application OOM kills
- Slow response times
- System memory warnings

#### Investigation Steps
```bash
# 1. Check memory usage by container
docker stats --no-stream

# 2. Check host memory
free -m
cat /proc/meminfo

# 3. Check for memory leaks in application
docker-compose exec reflexion-app python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### Resolution Steps
```bash
# 1. Restart application to clear memory
docker-compose restart reflexion-app

# 2. Increase memory limits if needed
# Edit docker-compose.yml to increase memory limits

# 3. Scale horizontally if possible
docker-compose up -d --scale reflexion-app=2

# 4. Enable memory profiling for analysis
# Set DEBUG_MEMORY=true in environment
```

### Redis Connection Issues

#### Symptoms
- Cache miss errors
- Session data loss
- "Connection refused" to Redis

#### Investigation Steps
```bash
# 1. Check Redis status
docker-compose exec redis redis-cli ping

# 2. Check Redis info
docker-compose exec redis redis-cli info

# 3. Check Redis logs
docker-compose logs redis --tail=100

# 4. Check memory usage
docker-compose exec redis redis-cli info memory
```

#### Resolution Steps
```bash
# 1. Restart Redis
docker-compose restart redis

# 2. Clear Redis cache if corrupted
docker-compose exec redis redis-cli FLUSHALL

# 3. Check Redis configuration
docker-compose exec redis cat /etc/redis/redis.conf
```

### SSL/TLS Certificate Issues

#### Symptoms
- Browser security warnings
- API clients failing with SSL errors
- Certificate expiration warnings

#### Investigation Steps
```bash
# 1. Check certificate expiration
openssl x509 -in /path/to/cert.pem -text -noout | grep "Not After"

# 2. Check certificate chain
openssl s509 -in /path/to/cert.pem -text -noout

# 3. Test SSL connection
openssl s_client -connect localhost:443 -servername your-domain.com
```

#### Resolution Steps
```bash
# 1. Renew certificate (Let's Encrypt example)
certbot renew --dry-run

# 2. Update certificate files
cp /etc/letsencrypt/live/your-domain/fullchain.pem /path/to/ssl/
cp /etc/letsencrypt/live/your-domain/privkey.pem /path/to/ssl/

# 3. Restart nginx
docker-compose restart nginx
```

## 📊 Monitoring and Alerting

### Key Metrics to Monitor

#### Application Metrics
- Response time (95th percentile < 500ms)
- Error rate (< 1%)
- Request rate
- Active users

#### Infrastructure Metrics
- CPU usage (< 80%)
- Memory usage (< 85%)
- Disk usage (< 90%)
- Network I/O

#### Business Metrics
- Successful reflexion cycles
- Agent performance improvements
- User satisfaction scores

### Alert Thresholds

```yaml
# Prometheus alerting rules
groups:
  - name: reflexion-alerts
    rules:
      - alert: ApplicationDown
        expr: up{job="reflexion-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Reflexion application is down"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemFree_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
```

## 🔄 Recovery Procedures

### Data Recovery

#### Database Recovery
```bash
# 1. Stop application
docker-compose stop reflexion-app

# 2. Restore from backup
docker-compose exec -T postgres psql -U reflexion -c "DROP DATABASE IF EXISTS reflexion;"
docker-compose exec -T postgres psql -U reflexion -c "CREATE DATABASE reflexion;"
docker-compose exec -T postgres psql -U reflexion reflexion < /path/to/backup.sql

# 3. Start application
docker-compose start reflexion-app
```

#### File System Recovery
```bash
# 1. Stop services
docker-compose down

# 2. Restore from backup
tar -xzf /path/to/backup.tar.gz -C ./

# 3. Fix permissions
sudo chown -R 1000:1000 data/ logs/

# 4. Start services
docker-compose up -d
```

### Configuration Recovery

#### Rollback Deployment
```bash
# 1. Stop current deployment
docker-compose down

# 2. Checkout previous version
git checkout HEAD~1

# 3. Rebuild and deploy
./scripts/build.sh
./scripts/deploy.sh deploy

# 4. Verify rollback
./scripts/deploy.sh health
```

## 📈 Post-Incident Activities

### 1. Document the Incident

Create an incident report with:
- Timeline of events
- Root cause analysis
- Impact assessment
- Actions taken
- Lessons learned

### 2. Update Monitoring

Based on the incident:
- Add new alerts if gaps were identified
- Adjust alert thresholds
- Improve monitoring coverage

### 3. Preventive Measures

- Update runbooks with new procedures
- Improve system resilience
- Conduct team training
- Review and update backup procedures

### 4. Follow-up Actions

- Schedule post-mortem meeting
- Assign action items with owners
- Track improvement implementations
- Update documentation

## 🛠️ Tools and Resources

### Monitoring Tools
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Logs**: Docker Compose logs and application logs

### Communication Tools
- **Status Page**: Communicate with users
- **Slack/Teams**: Internal team communication
- **PagerDuty**: Incident management and escalation

### Diagnostic Commands

```bash
# System health overview
./scripts/deploy.sh status

# Application performance
curl -s http://localhost:8000/metrics | grep -E "(response_time|error_rate)"

# Database performance
docker-compose exec postgres psql -U reflexion -c "SELECT * FROM pg_stat_database WHERE datname='reflexion';"

# Memory analysis
docker-compose exec reflexion-app python -m memory_profiler your_script.py

# Network connectivity
docker-compose exec reflexion-app netstat -tulpn
```

## 📞 Escalation Contacts

### On-Call Rotation
- **Primary**: [Contact information]
- **Secondary**: [Contact information]
- **Manager**: [Contact information]

### External Vendors
- **Cloud Provider**: [Support contact]
- **Third-party Services**: [Support contacts]

---

**Remember**: Stay calm, follow the procedures, document everything, and don't hesitate to escalate when needed.