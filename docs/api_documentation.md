# Autonomous SDLC API Documentation

This document provides comprehensive API documentation for the Autonomous SDLC system, including all endpoints, request/response formats, and integration examples.

## Base URL

```
https://your-domain.com/api/v1
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Obtaining a Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## Research Orchestrator API

### Start Research Cycle

Initiates a new autonomous research cycle with specified parameters.

```http
POST /research/cycle
Authorization: Bearer <token>
Content-Type: application/json

{
  "research_focus": "neural_optimization",
  "max_concurrent_studies": 3,
  "cycle_duration_hours": 24,
  "hypothesis_generation_params": {
    "creativity_level": 0.8,
    "focus_domains": ["machine_learning", "optimization"],
    "max_hypotheses": 10
  },
  "experiment_config": {
    "timeout_minutes": 60,
    "resource_limits": {
      "cpu_cores": 4,
      "memory_gb": 8
    },
    "validation_criteria": {
      "min_accuracy": 0.85,
      "significance_level": 0.05
    }
  }
}
```

Response:
```json
{
  "cycle_id": "rc_001_2025011501",
  "status": "started",
  "estimated_completion": "2025-01-16T01:00:00Z",
  "initial_hypotheses": [
    {
      "hypothesis_id": "hyp_001",
      "description": "Neural attention mechanisms improve reflexion accuracy",
      "confidence": 0.75,
      "domains": ["neural_networks", "attention_mechanisms"]
    }
  ],
  "experiments_scheduled": 5,
  "message": "Research cycle started successfully"
}
```

### Get Research Status

Retrieves the current status and progress of a research cycle.

```http
GET /research/cycle/{cycle_id}
Authorization: Bearer <token>
```

Response:
```json
{
  "cycle_id": "rc_001_2025011501",
  "status": "running",
  "progress": {
    "completed_experiments": 3,
    "total_experiments": 5,
    "success_rate": 0.67,
    "current_phase": "validation"
  },
  "results": {
    "validated_hypotheses": 2,
    "rejected_hypotheses": 1,
    "promising_discoveries": [
      {
        "discovery": "Quantum-inspired attention improves performance by 15%",
        "confidence": 0.92,
        "supporting_experiments": ["exp_001", "exp_003"]
      }
    ]
  },
  "estimated_completion": "2025-01-15T18:30:00Z",
  "resource_usage": {
    "cpu_hours": 156.5,
    "memory_gb_hours": 1024.8,
    "experiments_completed": 3
  }
}
```

### List Research Cycles

Retrieves a list of research cycles with optional filtering.

```http
GET /research/cycles?status=running&limit=10&offset=0
Authorization: Bearer <token>
```

Response:
```json
{
  "cycles": [
    {
      "cycle_id": "rc_001_2025011501",
      "status": "running",
      "created_at": "2025-01-15T01:00:00Z",
      "focus": "neural_optimization",
      "progress": 60
    }
  ],
  "total_count": 25,
  "has_more": true
}
```

### Stop Research Cycle

Gracefully stops a running research cycle.

```http
POST /research/cycle/{cycle_id}/stop
Authorization: Bearer <token>
Content-Type: application/json

{
  "reason": "Resource constraints",
  "save_progress": true
}
```

Response:
```json
{
  "cycle_id": "rc_001_2025011501",
  "status": "stopped",
  "final_results": {
    "completed_experiments": 3,
    "validated_hypotheses": 2,
    "total_runtime_hours": 12.5
  },
  "message": "Research cycle stopped successfully"
}
```

## Distributed Processing API

### Submit Task

Submits a task to the distributed processing engine.

```http
POST /distributed/tasks
Authorization: Bearer <token>
Content-Type: application/json

{
  "task_type": "reflexion_analysis",
  "input_data": {
    "text": "The system should optimize performance",
    "context": "performance_optimization",
    "parameters": {
      "max_iterations": 5,
      "quality_threshold": 0.8
    }
  },
  "priority": "normal",
  "timeout_seconds": 300,
  "required_capabilities": ["reflexion", "analysis"],
  "retry_policy": {
    "max_attempts": 3,
    "backoff_multiplier": 2
  }
}
```

Response:
```json
{
  "task_id": "task_abc123def456",
  "status": "queued",
  "priority": "normal",
  "estimated_start_time": "2025-01-15T10:05:00Z",
  "estimated_completion_time": "2025-01-15T10:10:00Z",
  "assigned_node": null,
  "queue_position": 3
}
```

### Get Task Status

Retrieves the status and results of a submitted task.

```http
GET /distributed/tasks/{task_id}
Authorization: Bearer <token>
```

Response:
```json
{
  "task_id": "task_abc123def456",
  "status": "completed",
  "priority": "normal",
  "created_at": "2025-01-15T10:00:00Z",
  "started_at": "2025-01-15T10:03:00Z",
  "completed_at": "2025-01-15T10:08:00Z",
  "assigned_node": "node_worker_001",
  "result": {
    "task": "The system should optimize performance",
    "output": "Optimized system performance through cache implementation and query optimization",
    "success": true,
    "iterations": 3,
    "reflections": [
      {
        "task": "Initial analysis",
        "output": "Identified bottlenecks in database queries",
        "success": true,
        "score": 0.85,
        "issues": [],
        "improvements": ["Add database indexing", "Implement query caching"],
        "confidence": 0.9
      }
    ],
    "total_time": 5.2,
    "metadata": {
      "performance_improvement": "35%",
      "optimization_techniques": ["caching", "indexing", "query_optimization"]
    }
  },
  "execution_stats": {
    "cpu_time_seconds": 4.1,
    "memory_peak_mb": 256,
    "network_io_kb": 12
  }
}
```

### List Processing Nodes

Retrieves information about available processing nodes.

```http
GET /distributed/nodes
Authorization: Bearer <token>
```

Response:
```json
{
  "nodes": [
    {
      "node_id": "node_worker_001",
      "status": "active",
      "address": "192.168.1.100",
      "port": 8081,
      "capabilities": ["reflexion", "analysis", "optimization"],
      "current_load": {
        "active_tasks": 2,
        "max_capacity": 10,
        "cpu_usage": 45.2,
        "memory_usage": 68.1
      },
      "health": {
        "last_heartbeat": "2025-01-15T10:09:30Z",
        "uptime_hours": 168.5,
        "tasks_completed": 1247,
        "success_rate": 0.984
      }
    }
  ],
  "total_nodes": 5,
  "active_nodes": 4,
  "total_capacity": 50,
  "current_utilization": 0.32
}
```

### Scale Cluster

Dynamically scales the processing cluster up or down.

```http
POST /distributed/scale
Authorization: Bearer <token>
Content-Type: application/json

{
  "action": "scale_up",
  "target_nodes": 8,
  "node_config": {
    "capacity": 10,
    "capabilities": ["reflexion", "analysis"],
    "resource_limits": {
      "cpu_cores": 4,
      "memory_gb": 8
    }
  },
  "reason": "High queue load"
}
```

Response:
```json
{
  "scaling_operation_id": "scale_op_789",
  "action": "scale_up",
  "current_nodes": 5,
  "target_nodes": 8,
  "estimated_completion": "2025-01-15T10:15:00Z",
  "nodes_to_add": 3,
  "status": "in_progress"
}
```

## Monitoring and Metrics API

### System Health

Retrieves comprehensive system health information.

```http
GET /monitoring/health
Authorization: Bearer <token>
```

Response:
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-01-15T10:10:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 23,
      "connection_pool": {
        "active": 8,
        "idle": 12,
        "max": 20
      }
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 2,
      "memory_usage_mb": 156
    },
    "research_orchestrator": {
      "status": "healthy",
      "active_cycles": 2,
      "queue_size": 5
    },
    "distributed_engine": {
      "status": "healthy",
      "active_nodes": 4,
      "queued_tasks": 12,
      "processing_tasks": 8
    }
  },
  "alerts": {
    "total_active": 1,
    "critical": 0,
    "warning": 1,
    "info": 0
  },
  "performance": {
    "avg_response_time_ms": 157,
    "requests_per_minute": 245,
    "error_rate": 0.02
  }
}
```

### Get Metrics

Retrieves specific metrics or metrics summaries.

```http
GET /monitoring/metrics?metric_name=system.cpu.percent&duration=3600
Authorization: Bearer <token>
```

Response:
```json
{
  "metric_name": "system.cpu.percent",
  "metric_type": "gauge",
  "description": "CPU usage percentage",
  "duration_seconds": 3600,
  "data_points": [
    {
      "timestamp": 1705315800.0,
      "value": 45.2,
      "tags": {}
    }
  ],
  "statistics": {
    "count": 60,
    "mean": 42.1,
    "median": 41.8,
    "min": 15.2,
    "max": 78.4,
    "std": 12.3,
    "p95": 65.8,
    "p99": 72.1
  }
}
```

### Get Active Alerts

Retrieves currently active system alerts.

```http
GET /monitoring/alerts?severity=warning&component=system
Authorization: Bearer <token>
```

Response:
```json
{
  "alerts": [
    {
      "alert_id": "alert_cpu_high_001",
      "severity": "warning",
      "component": "system",
      "metric_name": "system.cpu.percent",
      "message": "CPU usage high: system.cpu.percent > 80 (actual: 82.5)",
      "threshold_value": 80.0,
      "actual_value": 82.5,
      "timestamp": "2025-01-15T10:05:00Z",
      "duration_seconds": 300,
      "resolved": false
    }
  ],
  "total_count": 1
}
```

### Export Metrics

Exports metrics in various formats (JSON, Prometheus, CSV).

```http
GET /monitoring/export?format=prometheus&duration_hours=24
Authorization: Bearer <token>
```

Response (Prometheus format):
```
# HELP system_cpu_percent CPU usage percentage
# TYPE system_cpu_percent gauge
system_cpu_percent 45.2

# HELP reflexion_operations_total Total reflexion operations
# TYPE reflexion_operations_total counter
reflexion_operations_total 1247

# HELP reflexion_response_time_seconds Reflexion response time
# TYPE reflexion_response_time_seconds histogram
reflexion_response_time_seconds_bucket{le="0.1"} 234
reflexion_response_time_seconds_bucket{le="0.5"} 567
reflexion_response_time_seconds_bucket{le="1.0"} 890
reflexion_response_time_seconds_bucket{le="+Inf"} 1000
```

## Error Recovery API

### Get Recovery Status

Retrieves the current status of error recovery systems.

```http
GET /recovery/status
Authorization: Bearer <token>
```

Response:
```json
{
  "recovery_system_status": "active",
  "circuit_breakers": [
    {
      "component": "database",
      "status": "closed",
      "failure_count": 0,
      "success_count": 1247,
      "last_failure": null,
      "next_attempt": null
    },
    {
      "component": "external_api",
      "status": "half_open",
      "failure_count": 3,
      "success_count": 12,
      "last_failure": "2025-01-15T09:45:00Z",
      "next_attempt": "2025-01-15T10:15:00Z"
    }
  ],
  "self_healing": {
    "active": true,
    "last_healing_attempt": "2025-01-15T08:30:00Z",
    "successful_healings": 15,
    "failed_healings": 2
  },
  "fallback_activations": {
    "total_activations": 8,
    "recent_activations": 1,
    "success_rate": 0.875
  }
}
```

### Trigger Manual Recovery

Manually triggers recovery procedures for a specific component.

```http
POST /recovery/trigger
Authorization: Bearer <token>
Content-Type: application/json

{
  "component": "database",
  "recovery_type": "circuit_breaker_reset",
  "reason": "Manual intervention after maintenance"
}
```

Response:
```json
{
  "recovery_id": "recovery_001",
  "component": "database",
  "recovery_type": "circuit_breaker_reset",
  "status": "initiated",
  "timestamp": "2025-01-15T10:10:00Z",
  "estimated_completion": "2025-01-15T10:12:00Z"
}
```

## Configuration API

### Get Configuration

Retrieves current system configuration.

```http
GET /config
Authorization: Bearer <token>
```

Response:
```json
{
  "region": "north_america",
  "language": "en",
  "timezone": "America/New_York",
  "compliance_standards": ["sox", "soc2"],
  "security": {
    "encryption_at_rest": true,
    "encryption_in_transit": true,
    "audit_logging": false,
    "data_residency": false
  },
  "performance": {
    "max_concurrent_tasks": 100,
    "task_timeout_seconds": 300,
    "memory_limit_mb": 2048,
    "cpu_limit_cores": 2.0
  },
  "monitoring": {
    "log_level": "INFO",
    "metrics_retention_days": 30,
    "health_check_interval": 60
  }
}
```

### Update Configuration

Updates system configuration (requires admin privileges).

```http
PUT /config
Authorization: Bearer <admin-token>
Content-Type: application/json

{
  "performance": {
    "max_concurrent_tasks": 150,
    "task_timeout_seconds": 600
  },
  "monitoring": {
    "log_level": "DEBUG",
    "metrics_retention_days": 60
  }
}
```

Response:
```json
{
  "updated_fields": ["performance.max_concurrent_tasks", "performance.task_timeout_seconds", "monitoring.log_level", "monitoring.metrics_retention_days"],
  "restart_required": false,
  "message": "Configuration updated successfully"
}
```

## Language and Internationalization API

### Set Language

Changes the system language for the current session.

```http
POST /i18n/language
Authorization: Bearer <token>
Content-Type: application/json

{
  "language": "es"
}
```

Response:
```json
{
  "language": "es",
  "language_name": "Spanish",
  "native_name": "Español",
  "rtl": false,
  "message": "Language updated successfully"
}
```

### Get Available Languages

Retrieves list of supported languages.

```http
GET /i18n/languages
Authorization: Bearer <token>
```

Response:
```json
{
  "languages": [
    {
      "code": "en",
      "name": "English",
      "native_name": "English",
      "rtl": false
    },
    {
      "code": "es",
      "name": "Spanish",
      "native_name": "Español",
      "rtl": false
    },
    {
      "code": "fr",
      "name": "French",
      "native_name": "Français",
      "rtl": false
    }
  ],
  "current_language": "en"
}
```

## WebSocket API

### Real-time Updates

Connect to WebSocket for real-time system updates.

```javascript
const ws = new WebSocket('wss://your-domain.com/ws?token=your-jwt-token');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'task_completed':
            console.log('Task completed:', data.payload);
            break;
        case 'alert_triggered':
            console.log('Alert triggered:', data.payload);
            break;
        case 'system_health_update':
            console.log('Health update:', data.payload);
            break;
    }
};
```

### Subscribe to Events

```javascript
// Subscribe to specific event types
ws.send(JSON.stringify({
    action: 'subscribe',
    events: ['task_completed', 'alert_triggered', 'research_cycle_update']
}));
```

## Error Responses

All API endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "max_concurrent_studies",
      "issue": "Value must be between 1 and 10"
    },
    "request_id": "req_123456789",
    "timestamp": "2025-01-15T10:10:00Z"
  }
}
```

### Common Error Codes

- `AUTHENTICATION_REQUIRED` (401): Missing or invalid authentication token
- `AUTHORIZATION_DENIED` (403): Insufficient permissions
- `VALIDATION_ERROR` (400): Invalid request parameters
- `RESOURCE_NOT_FOUND` (404): Requested resource doesn't exist
- `RATE_LIMIT_EXCEEDED` (429): Too many requests
- `INTERNAL_SERVER_ERROR` (500): Unexpected server error
- `SERVICE_UNAVAILABLE` (503): Service temporarily unavailable

## Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Standard endpoints**: 1000 requests per hour per user
- **Computation endpoints**: 100 requests per hour per user
- **Admin endpoints**: 500 requests per hour per admin user

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1705319400
```

## SDKs and Client Libraries

### Python SDK

```python
from autonomous_sdlc_client import Client

client = Client(
    base_url="https://your-domain.com/api/v1",
    auth_token="your-jwt-token"
)

# Start research cycle
cycle = await client.research.start_cycle(
    research_focus="neural_optimization",
    max_concurrent_studies=3
)

# Submit distributed task
task = await client.distributed.submit_task(
    task_type="reflexion_analysis",
    input_data={"text": "Optimize performance"}
)

# Get system health
health = await client.monitoring.get_health()
```

### JavaScript SDK

```javascript
import { AutonomousSDLCClient } from 'autonomous-sdlc-js';

const client = new AutonomousSDLCClient({
    baseURL: 'https://your-domain.com/api/v1',
    authToken: 'your-jwt-token'
});

// Start research cycle
const cycle = await client.research.startCycle({
    researchFocus: 'neural_optimization',
    maxConcurrentStudies: 3
});

// Submit distributed task
const task = await client.distributed.submitTask({
    taskType: 'reflexion_analysis',
    inputData: { text: 'Optimize performance' }
});
```

This API documentation provides comprehensive coverage of all endpoints and integration patterns for the Autonomous SDLC system. Use this reference for building applications and integrations with the platform.