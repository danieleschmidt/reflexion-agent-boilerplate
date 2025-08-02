# Maintenance Runbook

This runbook covers routine maintenance procedures for the Reflexion Agent system.

## 📋 Maintenance Schedule

### Daily Tasks (Automated)

- **Log Rotation**: Automatic log cleanup and rotation
- **Backup Verification**: Verify daily backups completed successfully
- **Health Checks**: Automated health monitoring
- **Security Scans**: Daily vulnerability checks

### Weekly Tasks

- **Dependency Updates**: Check for security updates
- **Performance Review**: Analyze performance metrics
- **Backup Testing**: Test backup restoration
- **Storage Cleanup**: Clean up temporary files and old data

### Monthly Tasks

- **Security Audit**: Comprehensive security review
- **Capacity Planning**: Review resource usage trends
- **Documentation Update**: Update runbooks and procedures
- **DR Testing**: Test disaster recovery procedures

### Quarterly Tasks

- **Architecture Review**: Evaluate system architecture
- **Performance Optimization**: Implement performance improvements
- **Training Updates**: Update team training materials
- **Vendor Review**: Review third-party service agreements

## 🔄 Routine Maintenance Procedures

### System Updates

#### 1. Application Updates

```bash
# 1. Check current version
docker-compose exec reflexion-app reflexion --version

# 2. Pull latest code
git fetch origin
git checkout main
git pull origin main

# 3. Build and test new version
./scripts/build.sh --lint --test

# 4. Deploy with backup
./scripts/deploy.sh backup
./scripts/deploy.sh deploy

# 5. Verify deployment
./scripts/deploy.sh health
```

#### 2. Dependency Updates

```bash
# 1. Update Python dependencies
pip install --upgrade pip
pip list --outdated

# 2. Update specific packages
pip install --upgrade package_name

# 3. Update requirements
pip freeze > requirements.txt

# 4. Rebuild containers
docker-compose build --no-cache

# 5. Test updated dependencies
pytest tests/ -v
```

#### 3. Container Image Updates

```bash
# 1. Check for base image updates
docker pull python:3.11-slim
docker pull postgres:15-alpine
docker pull redis:7-alpine

# 2. Rebuild with updated base images
docker-compose build --pull --no-cache

# 3. Update production deployment
./scripts/deploy.sh deploy
```

### Database Maintenance

#### 1. PostgreSQL Maintenance

```bash
# Analyze database performance
docker-compose exec postgres psql -U reflexion -c "
ANALYZE;
SELECT schemaname, tablename, n_dead_tup, n_live_tup, 
       round(n_dead_tup * 100.0 / (n_live_tup + n_dead_tup), 2) AS dead_pct
FROM pg_stat_user_tables 
WHERE n_dead_tup > 0 
ORDER BY dead_pct DESC;
"

# Vacuum dead tuples
docker-compose exec postgres psql -U reflexion -c "VACUUM ANALYZE;"

# Reindex if needed
docker-compose exec postgres psql -U reflexion -c "REINDEX DATABASE reflexion;"

# Check database size
docker-compose exec postgres psql -U reflexion -c "
SELECT pg_size_pretty(pg_database_size('reflexion')) as db_size;
"
```

#### 2. Redis Maintenance

```bash
# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Analyze key distribution
docker-compose exec redis redis-cli --bigkeys

# Clean up expired keys
docker-compose exec redis redis-cli eval "
for i, name in ipairs(redis.call('KEYS', ARGV[1])) do
    redis.call('DEL', name)
end
" 0 "*expired*"

# Check Redis persistence
docker-compose exec redis redis-cli lastsave
```

### Log Management

#### 1. Log Rotation

```bash
# Check log sizes
du -sh logs/*

# Rotate logs manually if needed
logrotate -f /etc/logrotate.d/reflexion

# Archive old logs
tar -czf logs/archive/logs-$(date +%Y%m%d).tar.gz logs/*.log.1
rm logs/*.log.1
```

#### 2. Log Analysis

```bash
# Check error patterns
grep -E "(ERROR|CRITICAL)" logs/reflexion.log | tail -20

# Analyze response times
awk '/response_time/ {print $NF}' logs/access.log | sort -n | tail -10

# Check memory usage patterns
grep "memory_usage" logs/reflexion.log | awk '{print $3}' | sort -n
```

### Backup Procedures

#### 1. Create Backup

```bash
# Full system backup
./scripts/deploy.sh backup

# Manual database backup
docker-compose exec postgres pg_dump -U reflexion reflexion > backup-$(date +%Y%m%d).sql

# Manual file backup
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/
```

#### 2. Test Backup Restoration

```bash
# Test database restoration
docker-compose exec -T postgres psql -U reflexion -c "CREATE DATABASE test_restore;"
docker-compose exec -T postgres psql -U reflexion test_restore < backup-$(date +%Y%m%d).sql

# Verify restored data
docker-compose exec postgres psql -U reflexion test_restore -c "SELECT count(*) FROM your_table;"

# Cleanup test database
docker-compose exec postgres psql -U reflexion -c "DROP DATABASE test_restore;"
```

### Security Maintenance

#### 1. Security Updates

```bash
# Update system packages (if using custom base images)
docker run --rm -it your-base-image apt-get update && apt-get upgrade -y

# Scan for vulnerabilities
docker scout cves reflexion-agent:latest

# Check for Python security updates
pip install --upgrade pip-audit
pip-audit
```

#### 2. Certificate Management

```bash
# Check certificate expiration
openssl x509 -in ssl/cert.pem -text -noout | grep "Not After"

# Renew Let's Encrypt certificates
certbot renew --dry-run

# Update certificates
cp /etc/letsencrypt/live/yourdomain/fullchain.pem ssl/
cp /etc/letsencrypt/live/yourdomain/privkey.pem ssl/
docker-compose restart nginx
```

#### 3. Access Review

```bash
# Review active sessions
docker-compose exec postgres psql -U reflexion -c "SELECT * FROM pg_stat_activity;"

# Check authentication logs
grep "authentication" logs/auth.log | tail -20

# Review API access patterns
awk '{print $1}' logs/access.log | sort | uniq -c | sort -nr | head -10
```

### Performance Optimization

#### 1. Database Optimization

```bash
# Analyze slow queries
docker-compose exec postgres psql -U reflexion -c "
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"

# Check index usage
docker-compose exec postgres psql -U reflexion -c "
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
"

# Analyze table statistics
docker-compose exec postgres psql -U reflexion -c "
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
"
```

#### 2. Application Performance

```bash
# Check memory usage
docker stats --no-stream

# Profile application performance
docker-compose exec reflexion-app python -m cProfile -s cumulative your_script.py

# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

### Monitoring and Alerting

#### 1. Review Monitoring Setup

```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Verify Grafana dashboards
curl -s -H "Authorization: Bearer $GRAFANA_API_KEY" http://localhost:3000/api/dashboards/home

# Test alert rules
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.state != "inactive")'
```

#### 2. Update Monitoring Configuration

```bash
# Update Prometheus configuration
vim monitoring/prometheus.yml

# Reload Prometheus configuration
curl -X POST http://localhost:9090/-/reload

# Update Grafana dashboards
# Import updated dashboard JSON files through Grafana UI
```

## 🧹 Cleanup Procedures

### Storage Cleanup

```bash
# Clean up Docker system
docker system prune -af

# Clean up old images
docker image prune -af

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Clean up old backups
find backups/ -name "*.tar.gz" -mtime +90 -delete

# Clean up temporary files
find data/temp/ -type f -mtime +7 -delete
```

### Database Cleanup

```bash
# Clean up old sessions
docker-compose exec postgres psql -U reflexion -c "
DELETE FROM sessions WHERE created_at < NOW() - INTERVAL '30 days';
"

# Clean up old logs table
docker-compose exec postgres psql -U reflexion -c "
DELETE FROM application_logs WHERE timestamp < NOW() - INTERVAL '90 days';
"

# Clean up orphaned records
docker-compose exec postgres psql -U reflexion -c "
DELETE FROM memory_episodes WHERE agent_id NOT IN (SELECT id FROM agents);
"
```

## 📊 Performance Monitoring

### Key Performance Indicators

#### System Metrics
- CPU utilization (< 80%)
- Memory usage (< 85%)
- Disk I/O (< 80% of capacity)
- Network latency (< 100ms)

#### Application Metrics
- Response time (95th percentile < 500ms)
- Throughput (requests per second)
- Error rate (< 1%)
- Active connections

#### Business Metrics
- Reflexion success rate
- User satisfaction
- Agent improvement rate
- System availability (99.9%)

### Monitoring Dashboards

```bash
# View system metrics
curl -s "http://localhost:9090/api/v1/query?query=up" | jq

# View application metrics
curl -s "http://localhost:8000/metrics" | grep -E "(http_requests|response_time)"

# Generate performance report
python scripts/generate_performance_report.py --period=weekly
```

## 🔧 Troubleshooting Common Issues

### Performance Degradation

1. **Check resource usage**:
   ```bash
   docker stats --no-stream
   htop
   ```

2. **Analyze slow queries**:
   ```bash
   docker-compose exec postgres psql -U reflexion -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 5;"
   ```

3. **Review application logs**:
   ```bash
   grep -E "(slow|timeout|error)" logs/reflexion.log | tail -20
   ```

### Memory Leaks

1. **Monitor memory growth**:
   ```bash
   while true; do docker stats --no-stream | grep reflexion-app; sleep 60; done
   ```

2. **Profile memory usage**:
   ```bash
   docker-compose exec reflexion-app python -m memory_profiler your_script.py
   ```

3. **Restart services if needed**:
   ```bash
   docker-compose restart reflexion-app
   ```

## 📅 Maintenance Calendar

### Daily (Automated)
- [ ] Health checks
- [ ] Log rotation
- [ ] Backup verification
- [ ] Security scans

### Weekly
- [ ] Dependency update check
- [ ] Performance review
- [ ] Backup testing
- [ ] Storage cleanup

### Monthly
- [ ] Security audit
- [ ] Capacity planning review
- [ ] Documentation updates
- [ ] DR testing

### Quarterly
- [ ] Architecture review
- [ ] Performance optimization
- [ ] Training material updates
- [ ] Vendor review

## 📝 Maintenance Log Template

```
Date: [YYYY-MM-DD]
Maintainer: [Name]
Type: [Daily/Weekly/Monthly/Quarterly]

Tasks Completed:
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

Issues Found:
- Issue description and resolution

Performance Metrics:
- CPU: XX%
- Memory: XX%
- Disk: XX%
- Response Time: XXXms

Next Actions:
- Action item 1
- Action item 2

Notes:
[Any additional observations or recommendations]
```

---

**Remember**: Regular maintenance prevents major issues. Always test changes in a non-production environment first, and maintain detailed logs of all maintenance activities.