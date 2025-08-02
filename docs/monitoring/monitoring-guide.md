# Monitoring Guide

This guide covers setting up and using monitoring and observability for the Reflexion Agent Boilerplate.

## 🎯 Monitoring Overview

### Monitoring Stack

The Reflexion Agent uses a comprehensive monitoring stack:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards  
- **Docker Compose**: Container health monitoring
- **Application Logging**: Structured logging with correlation IDs

### Architecture

```mermaid
graph TB
    A[Reflexion App] -->|metrics| B[Prometheus]
    C[PostgreSQL] -->|metrics| B
    D[Redis] -->|metrics| B
    E[Nginx] -->|metrics| B
    
    B -->|data| F[Grafana]
    B -->|alerts| G[Alertmanager]
    
    A -->|logs| H[Log Files]
    H -->|aggregation| I[Log Analysis]
    
    F -->|dashboards| J[Operations Team]
    G -->|notifications| K[On-Call Team]
```

## 🚀 Quick Start

### Start Monitoring Stack

```bash
# Start application with monitoring
docker-compose --profile full --profile monitoring up -d

# Or use deployment script
./scripts/deploy.sh -p monitoring deploy
```

### Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application Health**: http://localhost:8000/health

## 📊 Metrics Collection

### Application Metrics

The Reflexion Agent exposes metrics at `/metrics` endpoint:

```bash
# View all metrics
curl http://localhost:8000/metrics

# Filter specific metrics
curl http://localhost:8000/metrics | grep reflexion_
```

#### Key Application Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `reflexion_requests_total` | Counter | Total API requests |
| `reflexion_request_duration_seconds` | Histogram | Request duration |
| `reflexion_active_agents` | Gauge | Number of active agents |
| `reflexion_memory_episodes_total` | Counter | Total memory episodes |
| `reflexion_reflection_cycles_total` | Counter | Total reflection cycles |
| `reflexion_success_rate` | Gauge | Agent success rate |

### Infrastructure Metrics

#### Docker Container Metrics

```bash
# View container stats
docker stats --no-stream

# Monitor specific container
docker stats reflexion-app --no-stream
```

#### System Metrics (via Node Exporter)

```yaml
# Add to docker-compose.yml
node-exporter:
  image: prom/node-exporter:latest
  container_name: node-exporter
  ports:
    - "9100:9100"
  volumes:
    - /proc:/host/proc:ro
    - /sys:/host/sys:ro
    - /:/rootfs:ro
  command:
    - '--path.procfs=/host/proc'
    - '--path.rootfs=/rootfs'
    - '--path.sysfs=/host/sys'
```

### Database Metrics

#### PostgreSQL Exporter

```yaml
# Add to docker-compose.yml
postgres-exporter:
  image: prometheuscommunity/postgres-exporter
  container_name: postgres-exporter
  environment:
    DATA_SOURCE_NAME: "postgresql://reflexion:password@postgres:5432/reflexion?sslmode=disable"
  ports:
    - "9187:9187"
  depends_on:
    - postgres
```

#### Redis Exporter

```yaml
# Add to docker-compose.yml  
redis-exporter:
  image: oliver006/redis_exporter
  container_name: redis-exporter
  environment:
    REDIS_ADDR: "redis://redis:6379"
  ports:
    - "9121:9121"
  depends_on:
    - redis
```

## 🎛️ Prometheus Configuration

### Basic Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'reflexion-app'
    static_configs:
      - targets: ['reflexion-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
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
          description: "The Reflexion application has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(reflexion_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(reflexion_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: LowSuccessRate
        expr: reflexion_success_rate < 0.8
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low agent success rate"
          description: "Agent success rate is {{ $value }}"

      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes{name="reflexion-app"} / container_spec_memory_limit_bytes{name="reflexion-app"}) > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Container memory usage is {{ $value }}%"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "PostgreSQL database is unreachable"
```

## 📈 Grafana Dashboards

### Importing Dashboards

1. **Access Grafana**: http://localhost:3000
2. **Login**: admin/admin (change password on first login)
3. **Import**: Go to "+" → Import → Upload JSON file

### Pre-configured Dashboards

#### Application Dashboard (`monitoring/dashboards/reflexion-dashboard.json`)

Key panels:
- Request rate and response time
- Error rate and success rate  
- Active agents and memory episodes
- Reflection cycle metrics

#### Infrastructure Dashboard (`monitoring/dashboards/performance-dashboard.json`)

Key panels:
- CPU and memory usage
- Disk I/O and network traffic
- Container health status
- Database performance metrics

### Creating Custom Dashboards

#### Panel Examples

**Request Rate Panel**:
```promql
rate(reflexion_requests_total[5m])
```

**Response Time Panel**:
```promql
histogram_quantile(0.95, rate(reflexion_request_duration_seconds_bucket[5m]))
```

**Memory Usage Panel**:
```promql
container_memory_usage_bytes{name="reflexion-app"} / 1024 / 1024
```

**Success Rate Panel**:
```promql
reflexion_success_rate
```

## 🔔 Alerting Setup

### Alertmanager Configuration

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@yourcompany.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'team-notifications'

receivers:
  - name: 'team-notifications'
    email_configs:
      - to: 'team@yourcompany.com'
        subject: '[ALERT] {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: '[ALERT] {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}
```

### Webhook Notifications

```python
# Custom webhook handler
from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_alert():
    data = request.get_json()
    
    for alert in data.get('alerts', []):
        print(f"Alert: {alert['annotations']['summary']}")
        # Send to your notification system
        
    return 'OK'
```

## 📋 Logging

### Application Logging

The application uses structured logging with correlation IDs:

```python
import logging
import uuid
from contextvars import ContextVar

# Correlation ID context
correlation_id = ContextVar('correlation_id', default=None)

class CorrelationFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id.get() or str(uuid.uuid4())
        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.addFilter(CorrelationFilter())
```

### Log Aggregation

#### Using ELK Stack (Optional)

```yaml
# docker-compose.yml addition
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
  environment:
    - discovery.type=single-node
  ports:
    - "9200:9200"

logstash:
  image: docker.elastic.co/logstash/logstash:8.5.0
  volumes:
    - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
  depends_on:
    - elasticsearch

kibana:
  image: docker.elastic.co/kibana/kibana:8.5.0
  ports:
    - "5601:5601"
  depends_on:
    - elasticsearch
```

### Log Analysis

```bash
# View recent errors
grep -E "(ERROR|CRITICAL)" logs/reflexion.log | tail -20

# Analyze response times
awk '/response_time/ {sum+=$NF; count++} END {print "Average:", sum/count}' logs/access.log

# Find slow requests
awk '/response_time/ && $NF > 2 {print}' logs/access.log

# Count errors by type
grep "ERROR" logs/reflexion.log | awk '{print $5}' | sort | uniq -c | sort -nr
```

## 🔍 Distributed Tracing (Optional)

### Jaeger Setup

```yaml
# docker-compose.yml addition
jaeger:
  image: jaegertracing/all-in-one:latest
  ports:
    - "16686:16686"
    - "14268:14268"
  environment:
    - COLLECTOR_OTLP_ENABLED=true
```

### Application Instrumentation

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Use in code
@tracer.start_as_current_span("reflexion_process")
def process_reflexion(task):
    with tracer.start_as_current_span("llm_call"):
        # LLM processing
        pass
    
    with tracer.start_as_current_span("memory_update"):
        # Memory operations
        pass
```

## 📊 Custom Metrics

### Application Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define custom metrics
REQUEST_COUNT = Counter('reflexion_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('reflexion_request_duration_seconds', 'Request latency')
ACTIVE_AGENTS = Gauge('reflexion_active_agents', 'Number of active agents')
MEMORY_EPISODES = Counter('reflexion_memory_episodes_total', 'Total memory episodes')

# Use in application
@REQUEST_LATENCY.time()
def process_request():
    REQUEST_COUNT.labels(method='POST', endpoint='/reflect', status='200').inc()
    # Process request
    ACTIVE_AGENTS.set(get_active_agent_count())
```

### Business Metrics

```python
# Business-specific metrics
REFLEXION_SUCCESS_RATE = Gauge('reflexion_success_rate', 'Agent success rate')
USER_SATISFACTION = Histogram('reflexion_user_satisfaction', 'User satisfaction scores')
IMPROVEMENT_RATE = Gauge('reflexion_improvement_rate', 'Agent improvement rate')

# Update metrics
def update_business_metrics():
    success_rate = calculate_success_rate()
    REFLEXION_SUCCESS_RATE.set(success_rate)
    
    satisfaction = get_user_satisfaction()
    USER_SATISFACTION.observe(satisfaction)
```

## 🎛️ Health Checks

### Application Health Endpoint

```python
from flask import Flask, jsonify
import psutil
import redis
import psycopg2

app = Flask(__name__)

@app.route('/health')
def health_check():
    checks = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }
    
    # Database check
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        checks['checks']['database'] = 'healthy'
    except Exception as e:
        checks['checks']['database'] = f'unhealthy: {str(e)}'
        checks['status'] = 'unhealthy'
    
    # Redis check
    try:
        r = redis.Redis.from_url(REDIS_URL)
        r.ping()
        checks['checks']['redis'] = 'healthy'
    except Exception as e:
        checks['checks']['redis'] = f'unhealthy: {str(e)}'
        checks['status'] = 'unhealthy'
    
    # Memory check
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        checks['checks']['memory'] = f'warning: {memory.percent}% used'
    else:
        checks['checks']['memory'] = f'healthy: {memory.percent}% used'
    
    return jsonify(checks)

@app.route('/ready')
def readiness_check():
    # Check if application is ready to serve requests
    return jsonify({'status': 'ready'})
```

### Container Health Checks

```dockerfile
# Dockerfile health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

## 📈 Performance Monitoring

### Query Performance

```sql
-- PostgreSQL slow query monitoring
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
WHERE mean_time > 1000  -- queries taking more than 1 second
ORDER BY mean_time DESC
LIMIT 10;
```

### Application Performance

```python
import time
import functools
from prometheus_client import Histogram

FUNCTION_LATENCY = Histogram('function_duration_seconds', 'Function execution time', ['function_name'])

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            FUNCTION_LATENCY.labels(function_name=func.__name__).observe(duration)
    return wrapper
```

## 🔧 Troubleshooting

### Common Issues

1. **Metrics not appearing**:
   ```bash
   # Check if metrics endpoint is accessible
   curl http://localhost:8000/metrics
   
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   ```

2. **Grafana dashboard not loading**:
   ```bash
   # Check Grafana logs
   docker-compose logs grafana
   
   # Verify data source connection
   curl -H "Authorization: Bearer $GRAFANA_API_KEY" http://localhost:3000/api/datasources
   ```

3. **Alerts not firing**:
   ```bash
   # Check alert rules
   curl http://localhost:9090/api/v1/rules
   
   # Check alertmanager status
   curl http://localhost:9093/api/v1/status
   ```

### Debug Commands

```bash
# View all Prometheus metrics
curl -s http://localhost:9090/api/v1/label/__name__/values | jq '.data[]'

# Query specific metric
curl -s "http://localhost:9090/api/v1/query?query=up" | jq

# Check alertmanager alerts
curl -s http://localhost:9093/api/v1/alerts | jq
```

## 📚 Resources

### Documentation
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)

### Best Practices
- Use meaningful metric names and labels
- Set appropriate alert thresholds
- Implement proper health checks
- Monitor business metrics, not just technical ones
- Use correlation IDs for distributed tracing

---

For more advanced monitoring setups, see:
- [Incident Response Runbook](../runbooks/incident-response.md)
- [Maintenance Runbook](../runbooks/maintenance.md)
- [Security Guide](../security.md)