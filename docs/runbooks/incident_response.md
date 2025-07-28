# Incident Response Runbook
## Reflexion Agent Boilerplate

### Incident Classification

#### Severity Levels

**P0 - Critical**
- Complete service outage
- Data loss or corruption
- Security breach
- Response time: Immediate (< 15 minutes)

**P1 - High**
- Major feature unavailable
- Significant performance degradation
- API errors affecting multiple users
- Response time: < 1 hour

**P2 - Medium**
- Minor feature issues
- Performance issues affecting some users
- Non-critical API errors
- Response time: < 4 hours

**P3 - Low**
- Cosmetic issues
- Documentation problems
- Enhancement requests
- Response time: < 24 hours

### Incident Response Process

#### 1. Detection and Alert

**Automated Alerts**
- Prometheus alerts via Alertmanager
- Log-based alerts from monitoring systems
- Health check failures
- User-reported issues

**Manual Detection**
- User reports
- Internal testing
- Monitoring dashboard review

#### 2. Initial Response (0-15 minutes)

1. **Acknowledge the incident**
   ```bash
   # Update incident status
   echo "Incident acknowledged by $(whoami) at $(date)" >> incident.log
   ```

2. **Assess severity**
   - Check monitoring dashboards
   - Review error logs
   - Determine impact scope

3. **Create incident channel**
   ```bash
   # Create Slack channel: #incident-YYYY-MM-DD-HHMMSS
   # Or use your preferred communication tool
   ```

4. **Notify stakeholders** (for P0/P1)
   - Engineering team
   - Product managers
   - Customer support

#### 3. Investigation and Diagnosis

**Health Check Commands**
```bash
# Check service status
python scripts/health_check.py --detailed --format json

# Check system resources
docker stats
df -h
free -h

# Check application logs
docker-compose logs reflexion --tail=100

# Check database connectivity
python -c "from reflexion.database import check_connection; print(check_connection())"

# Check Redis connectivity
redis-cli ping
```

**Log Analysis**
```bash
# Search for errors in the last hour
grep -i error /var/log/reflexion/*.log | tail -20

# Monitor real-time logs
tail -f /var/log/reflexion/app.log

# Check for specific patterns
grep -E "(timeout|connection|failed)" /var/log/reflexion/*.log
```

**Performance Analysis**
```bash
# Check response times
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/health

# Monitor database queries
# (Database-specific monitoring commands)

# Check memory usage
ps aux | grep reflexion
```

#### 4. Mitigation and Recovery

**Immediate Mitigation Steps**

1. **Scale up resources** (if performance issue)
   ```bash
   # Increase container resources
   docker-compose up --scale reflexion=3
   
   # Or update resource limits
   docker update --memory=2g --cpus=2 reflexion_container
   ```

2. **Restart services** (if needed)
   ```bash
   # Restart application
   docker-compose restart reflexion
   
   # Restart database (last resort)
   docker-compose restart postgres
   ```

3. **Enable circuit breaker** (if external dependency issue)
   ```bash
   # Set environment variable to disable failing service
   export DISABLE_EXTERNAL_SERVICE=true
   docker-compose restart reflexion
   ```

4. **Rollback deployment** (if recent deployment caused issue)
   ```bash
   # Rollback to previous version
   git checkout previous-stable-tag
   docker-compose up --build -d
   ```

**Database Issues**

```bash
# Check database connections
SELECT count(*) FROM pg_stat_activity;

# Check long-running queries
SELECT pid, query, query_start, state 
FROM pg_stat_activity 
WHERE state = 'active' 
AND query_start < now() - interval '5 minutes';

# Kill problematic queries (if safe)
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE ...;

# Check disk space
df -h /var/lib/postgresql/data/

# Restart database (if needed)
docker-compose restart postgres
```

**Memory Issues**

```bash
# Check memory usage
free -h
docker stats

# Find memory-consuming processes
ps aux --sort=-%mem | head -10

# Restart high-memory containers
docker restart container_name

# Clear cache (if safe)
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

#### 5. Communication

**Internal Communication**
- Update incident channel every 30 minutes
- Use clear, factual language
- Include next steps and ETA

**External Communication** (for customer-facing issues)
- Status page updates
- Customer notification emails
- Social media (if applicable)

**Communication Template**
```
## Incident Update - [TIMESTAMP]

**Status**: [INVESTIGATING/IDENTIFIED/MONITORING/RESOLVED]

**Impact**: [Description of impact on users/systems]

**Root Cause**: [What we know so far]

**Mitigation**: [What we're doing to resolve]

**Next Update**: [When next update will be provided]
```

#### 6. Resolution and Monitoring

1. **Verify fix**
   ```bash
   # Run health checks
   python scripts/health_check.py --detailed
   
   # Check key metrics
   curl http://localhost:8000/metrics | grep reflexion_
   
   # Verify user-reported functionality
   # (Run specific test cases)
   ```

2. **Monitor for 30+ minutes**
   - Watch error rates
   - Monitor performance metrics
   - Check for new alerts

3. **Communicate resolution**
   - Update incident channel
   - Notify stakeholders
   - Update status page

### Common Incident Scenarios

#### High Error Rate

**Symptoms**
- Increased 5xx error responses
- Prometheus alert: `HighErrorRate`
- User complaints about failures

**Investigation Steps**
```bash
# Check recent deployments
git log --oneline -10

# Analyze error patterns
grep -E "500|error" /var/log/reflexion/*.log | tail -50

# Check external dependencies
curl -f https://api.openai.com/v1/models
curl -f https://api.anthropic.com/v1/messages
```

**Mitigation**
- Rollback recent changes if applicable
- Enable circuit breakers for external services
- Scale up resources if capacity issue

#### Database Connection Issues

**Symptoms**
- Database connection errors in logs
- Prometheus alert: `DatabaseConnectionFailures`
- Timeouts on database operations

**Investigation Steps**
```bash
# Check database status
docker-compose ps postgres
docker-compose logs postgres

# Check connection pool
# (Framework-specific commands)

# Check database disk space
docker exec postgres_container df -h
```

**Mitigation**
```bash
# Restart application (to reset connection pool)
docker-compose restart reflexion

# Restart database if needed
docker-compose restart postgres

# Increase connection pool size (if needed)
# Update configuration and restart
```

#### Memory Leak

**Symptoms**
- Gradually increasing memory usage
- OOM kills in container logs
- Performance degradation over time

**Investigation Steps**
```bash
# Monitor memory usage over time
docker stats --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Check for memory leaks in application
# (Use memory profiling tools)

# Review recent code changes
git log --grep="memory" --oneline -20
```

**Mitigation**
```bash
# Restart application to free memory
docker-compose restart reflexion

# Reduce memory-intensive operations
# (Application-specific changes)

# Increase memory limits temporarily
docker update --memory=4g reflexion_container
```

#### LLM API Failures

**Symptoms**
- LLM API timeout errors
- Rate limit exceeded errors
- Authentication failures

**Investigation Steps**
```bash
# Check API status
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Review API usage
grep -i "rate limit\|timeout\|unauthorized" /var/log/reflexion/*.log

# Check API key validity
# (API-specific verification)
```

**Mitigation**
```bash
# Switch to backup API provider
export FALLBACK_LLM_PROVIDER=anthropic
docker-compose restart reflexion

# Implement exponential backoff
# (Code change required)

# Request rate limit increase
# (Contact API provider)
```

### Post-Incident Activities

#### 1. Post-Incident Review (within 48 hours)

**Meeting Agenda**
- Timeline review
- Root cause analysis
- What went well
- What could be improved
- Action items

**Documentation**
- Update runbook based on learnings
- Document new monitoring/alerting needs
- Create tickets for follow-up work

#### 2. Follow-up Actions

**Immediate (within 1 week)**
- Fix root cause if not addressed during incident
- Implement additional monitoring
- Update documentation

**Short-term (within 1 month)**
- Improve alerting and detection
- Add automated recovery procedures
- Conduct chaos engineering tests

**Long-term (within 3 months)**
- Architecture improvements
- Redundancy and failover improvements
- Team training and process improvements

### Useful Commands and Tools

#### Monitoring Commands
```bash
# Check service health
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics

# Check logs
docker-compose logs reflexion --since 1h

# Monitor real-time metrics
watch -n 5 'curl -s http://localhost:8000/metrics | grep reflexion_requests_total'
```

#### Debug Information Collection
```bash
# Collect system information
uname -a
df -h
free -h
ps aux | grep reflexion

# Collect Docker information
docker version
docker-compose ps
docker stats --no-stream

# Collect application information
python --version
pip list | grep reflexion
```

### Contact Information

**On-Call Engineer**: [Contact details]
**Escalation Manager**: [Contact details]
**Technical Lead**: [Contact details]

**Emergency Contacts**
- Cloud Provider Support: [Contact]
- Database Administrator: [Contact]
- Security Team: [Contact]

### External Resources

- **Monitoring Dashboard**: http://grafana.your-org.com
- **Log Aggregation**: http://kibana.your-org.com
- **Status Page**: http://status.your-org.com
- **Documentation**: http://docs.your-org.com