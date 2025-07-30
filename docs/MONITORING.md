# Monitoring and Observability

This document outlines monitoring, logging, and observability practices for the Reflexion Agent Boilerplate.

## Overview

Comprehensive monitoring is essential for production deployments of AI agents. This setup provides:

- **Application Performance Monitoring (APM)**
- **Structured Logging**
- **Custom Metrics**
- **Health Checks**
- **Alerting**

## Monitoring Stack

### Core Components
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **Elasticsearch/Fluentd/Kibana (EFK)**: Log aggregation

### Python Instrumentation
```python
from prometheus_client import Counter, Histogram, Gauge
import structlog
from opentelemetry import trace

# Metrics
reflexion_counter = Counter('reflexions_total', 'Total reflexions performed', ['status'])
reflexion_duration = Histogram('reflexion_duration_seconds', 'Reflexion execution time')
active_agents = Gauge('active_agents', 'Number of active agents')

# Structured logging
logger = structlog.get_logger()

# Tracing
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("reflexion_execution")
@reflexion_duration.time()
def execute_reflexion(task):
    reflexion_counter.labels(status='started').inc()
    
    try:
        with tracer.start_as_current_span("llm_call"):
            result = perform_reflexion(task)
        
        logger.info(
            "reflexion_completed",
            task_id=task.id,
            iterations=result.iterations,
            success=result.success,
            duration=result.duration
        )
        
        reflexion_counter.labels(status='success').inc()
        return result
        
    except Exception as e:
        logger.error(
            "reflexion_failed",
            task_id=task.id,
            error=str(e),
            exc_info=True
        )
        reflexion_counter.labels(status='error').inc()
        raise
```

## Key Metrics

### Application Metrics
```python
# Core metrics to track
CORE_METRICS = {
    # Performance
    'reflexion_duration_seconds': 'Time spent on reflexion cycles',
    'llm_api_duration_seconds': 'LLM API response time',
    'memory_retrieval_duration_seconds': 'Memory lookup time',
    
    # Volume
    'reflexions_total': 'Total number of reflexions',
    'tasks_processed_total': 'Total tasks processed',
    'memory_entries_total': 'Total memory entries stored',
    
    # Quality
    'reflexion_success_rate': 'Success rate of reflexions',
    'improvement_rate': 'Rate of improvement from reflexion',
    'memory_hit_rate': 'Memory cache hit rate',
    
    # Resources
    'active_agents': 'Number of active agents',
    'memory_usage_bytes': 'Memory consumption',
    'cpu_usage_percent': 'CPU utilization'
}
```

### Business Metrics
```python
# Business-focused metrics
BUSINESS_METRICS = {
    'task_completion_rate': 'Percentage of successfully completed tasks',
    'average_iterations_per_task': 'Average reflexion iterations needed',
    'user_satisfaction_score': 'User satisfaction ratings',
    'cost_per_task': 'Cost efficiency metrics',
    'time_to_solution': 'Time from task start to completion'
}
```

## Structured Logging

### Log Configuration
```python
import structlog
from structlog.stdlib import LoggerFactory

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
```

### Log Events
```python
# Standard log events
LOG_EVENTS = {
    # Lifecycle
    'agent_started': {'level': 'info', 'fields': ['agent_id', 'config']},
    'agent_stopped': {'level': 'info', 'fields': ['agent_id', 'duration']},
    
    # Task processing
    'task_received': {'level': 'info', 'fields': ['task_id', 'task_type']},
    'task_completed': {'level': 'info', 'fields': ['task_id', 'duration', 'success']},
    'task_failed': {'level': 'error', 'fields': ['task_id', 'error', 'stack_trace']},
    
    # Reflexion
    'reflexion_started': {'level': 'info', 'fields': ['task_id', 'iteration']},
    'reflexion_completed': {'level': 'info', 'fields': ['task_id', 'iteration', 'improvement']},
    
    # Memory
    'memory_stored': {'level': 'debug', 'fields': ['entry_id', 'type', 'size']},
    'memory_retrieved': {'level': 'debug', 'fields': ['query', 'results_count']},
    
    # External services
    'llm_api_call': {'level': 'debug', 'fields': ['model', 'tokens', 'duration']},
    'llm_api_error': {'level': 'error', 'fields': ['model', 'error', 'retry_count']},
}
```

## Health Checks

### Application Health
```python
from typing import Dict, Any
from datetime import datetime, timedelta

class HealthChecker:
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'memory_store': self._check_memory_store,
            'llm_api': self._check_llm_api,
            'disk_space': self._check_disk_space,
            'memory_usage': self._check_memory_usage,
        }
    
    async def check_all(self) -> Dict[str, Any]:
        results = {}
        overall_status = 'healthy'
        
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = result
                if result['status'] != 'healthy':
                    overall_status = 'degraded'
            except Exception as e:
                results[name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                overall_status = 'unhealthy'
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }
    
    async def _check_database(self) -> Dict[str, Any]:
        # Database connectivity check
        return {'status': 'healthy', 'response_time_ms': 50}
    
    async def _check_llm_api(self) -> Dict[str, Any]:
        # LLM API availability check
        return {'status': 'healthy', 'response_time_ms': 200}
```

### Deep Health Checks
```python
class DeepHealthChecker(HealthChecker):
    async def check_reflexion_pipeline(self) -> Dict[str, Any]:
        """Test complete reflexion pipeline with synthetic task."""
        try:
            start_time = datetime.utcnow()
            
            # Create test agent
            agent = ReflexionAgent(llm="test-model")
            
            # Run simple reflexion task
            result = await agent.run("Test task", timeout=30)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'status': 'healthy' if result.success else 'degraded',
                'duration_seconds': duration,
                'iterations': result.iterations,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
```

## Alerting Rules

### Prometheus Alerting Rules
```yaml
# prometheus-alerts.yml
groups:
  - name: reflexion-agent-alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(reflexions_total{status="error"}[5m]) / rate(reflexions_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
      
      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(reflexion_duration_seconds_bucket[5m])) > 10
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High reflexion latency"
          description: "95th percentile latency is {{ $value }}s"
      
      # Memory usage
      - alert: HighMemoryUsage
        expr: memory_usage_bytes / (1024*1024*1024) > 2
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
      
      # API availability
      - alert: LLMAPIDown
        expr: up{job="llm-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM API is down"
          description: "LLM API has been down for more than 1 minute"
```

## Dashboards

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Reflexion Agent Dashboard",
    "panels": [
      {
        "title": "Reflexion Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(reflexions_total[5m])",
            "legendFormat": "Reflexions/sec"
          }
        ]
      },
      {
        "title": "Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(reflexions_total{status=\"success\"}[5m]) / rate(reflexions_total[5m])",
            "legendFormat": "Success Rate"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(reflexion_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(reflexion_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## Distributed Tracing

### OpenTelemetry Setup
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

tracer = trace.get_tracer(__name__)

# Use in code
@tracer.start_as_current_span("reflexion_cycle")
def perform_reflexion(task):
    with tracer.start_as_current_span("generate_response") as span:
        span.set_attribute("task.id", task.id)
        span.set_attribute("task.type", task.type)
        
        response = llm_client.generate(task.prompt)
        
        span.set_attribute("response.tokens", len(response.tokens))
        span.set_attribute("response.model", response.model)
        
        return response
```

## Log Analysis

### Common Queries
```bash
# Find errors in the last hour
jq 'select(.level == "error" and .timestamp > (now - 3600))' logs.jsonl

# Calculate average reflexion duration
jq -r 'select(.event == "reflexion_completed") | .duration' logs.jsonl | awk '{sum+=$1; count++} END {print sum/count}'

# Find most common failure patterns
jq -r 'select(.event == "task_failed") | .error' logs.jsonl | sort | uniq -c | sort -nr

# Monitor memory growth
jq -r 'select(.event == "memory_stored") | .timestamp + " " + (.size | tostring)' logs.jsonl
```

## Performance Baselines

### Expected Performance Metrics
```yaml
# Performance baselines for monitoring
baselines:
  response_time:
    p50: "< 2s"
    p95: "< 5s" 
    p99: "< 10s"
  
  throughput:
    peak: "> 100 req/min"
    sustained: "> 50 req/min"
  
  success_rate: "> 95%"
  
  resource_usage:
    cpu: "< 70%"
    memory: "< 2GB"
    disk: "< 10GB"
  
  availability: "> 99.9%"
```