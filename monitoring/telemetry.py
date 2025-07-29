"""Telemetry and metrics collection for Reflexion agents."""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import logging

# Initialize Prometheus registry
REGISTRY = CollectorRegistry()

# Define Prometheus metrics
REFLECTION_ATTEMPTS = Counter(
    'reflexion_attempts_total',
    'Total number of reflection attempts',
    ['task_type', 'agent_id'],
    registry=REGISTRY
)

REFLECTION_SUCCESS = Counter(
    'reflexion_success_total',
    'Total number of successful reflections',
    ['task_type', 'agent_id'],
    registry=REGISTRY
)

REFLECTION_FAILURE = Counter(
    'reflexion_failure_total',
    'Total number of failed reflections',
    ['task_type', 'agent_id', 'error_type'],
    registry=REGISTRY
)

REFLECTION_DURATION = Histogram(
    'reflexion_execution_duration_seconds',
    'Time spent on reflection execution',
    ['task_type', 'agent_id'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)

REFLECTION_ITERATIONS = Histogram(
    'reflexion_iterations_total',
    'Number of iterations per reflection cycle',
    ['task_type'],
    buckets=[1, 2, 3, 4, 5, 10, 20],
    registry=REGISTRY
)

REFLECTION_QUALITY = Gauge(
    'reflexion_quality_score',
    'Quality score of reflection output',
    ['task_type', 'agent_id'],
    registry=REGISTRY
)

REFLECTION_IMPROVEMENT = Gauge(
    'reflexion_improvement_score',
    'Improvement score between iterations',
    ['task_type', 'agent_id'],
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'reflexion_memory_usage_bytes',
    'Memory usage in bytes',
    ['component'],
    registry=REGISTRY
)

MEMORY_EPISODES = Gauge(
    'reflexion_memory_episodes_count',
    'Number of episodes stored in memory',
    ['memory_type'],
    registry=REGISTRY
)

LLM_REQUEST_DURATION = Histogram(
    'reflexion_llm_request_duration_seconds',
    'Duration of LLM API requests',
    ['llm_provider', 'model'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

TOKENS_CONSUMED = Counter(
    'reflexion_tokens_consumed_total',
    'Total tokens consumed by LLM requests',
    ['llm_provider', 'model', 'token_type'],
    registry=REGISTRY
)

ACTIVE_AGENTS = Gauge(
    'reflexion_active_agents',
    'Number of currently active agents',
    registry=REGISTRY
)

CACHE_REQUESTS = Counter(
    'reflexion_cache_requests_total',
    'Total cache requests',
    ['cache_type'],
    registry=REGISTRY
)

CACHE_HITS = Counter(
    'reflexion_cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=REGISTRY
)

ERRORS = Counter(
    'reflexion_errors_total',
    'Total errors by type',
    ['error_type', 'component'],
    registry=REGISTRY
)

DEPLOYMENTS = Counter(
    'reflexion_deployments_total',
    'Total deployments',
    ['version', 'environment'],
    registry=REGISTRY
)


class TelemetryCollector:
    """Collects and exports telemetry data for Reflexion agents."""
    
    def __init__(self, enabled: bool = True, export_interval: int = 30):
        """Initialize telemetry collector.
        
        Args:
            enabled: Whether telemetry collection is enabled
            export_interval: Interval in seconds for metric exports
        """
        self.enabled = enabled
        self.export_interval = export_interval
        self.logger = logging.getLogger(__name__)
        
    def record_reflection_attempt(self, task_type: str, agent_id: str):
        """Record a reflection attempt."""
        if not self.enabled:
            return
            
        REFLECTION_ATTEMPTS.labels(
            task_type=task_type,
            agent_id=agent_id
        ).inc()
        
    def record_reflection_success(self, task_type: str, agent_id: str, 
                                quality_score: float, iterations: int,
                                duration: float):
        """Record a successful reflection."""
        if not self.enabled:
            return
            
        REFLECTION_SUCCESS.labels(
            task_type=task_type,
            agent_id=agent_id
        ).inc()
        
        REFLECTION_QUALITY.labels(
            task_type=task_type,
            agent_id=agent_id
        ).set(quality_score)
        
        REFLECTION_ITERATIONS.labels(
            task_type=task_type
        ).observe(iterations)
        
        REFLECTION_DURATION.labels(
            task_type=task_type,
            agent_id=agent_id
        ).observe(duration)
        
    def record_reflection_failure(self, task_type: str, agent_id: str,
                                error_type: str, duration: float):
        """Record a failed reflection."""
        if not self.enabled:
            return
            
        REFLECTION_FAILURE.labels(
            task_type=task_type,
            agent_id=agent_id,
            error_type=error_type
        ).inc()
        
        REFLECTION_DURATION.labels(
            task_type=task_type,
            agent_id=agent_id
        ).observe(duration)
        
    def record_improvement_score(self, task_type: str, agent_id: str,
                               improvement_score: float):
        """Record improvement score between iterations."""
        if not self.enabled:
            return
            
        REFLECTION_IMPROVEMENT.labels(
            task_type=task_type,
            agent_id=agent_id
        ).set(improvement_score)
        
    def record_memory_usage(self, component: str, bytes_used: int):
        """Record memory usage for a component."""
        if not self.enabled:
            return
            
        MEMORY_USAGE.labels(component=component).set(bytes_used)
        
    def record_memory_episodes(self, memory_type: str, episode_count: int):
        """Record number of episodes in memory."""
        if not self.enabled:
            return
            
        MEMORY_EPISODES.labels(memory_type=memory_type).set(episode_count)
        
    def record_llm_request(self, llm_provider: str, model: str,
                          duration: float, input_tokens: int, 
                          output_tokens: int):
        """Record LLM API request metrics."""
        if not self.enabled:
            return
            
        LLM_REQUEST_DURATION.labels(
            llm_provider=llm_provider,
            model=model
        ).observe(duration)
        
        TOKENS_CONSUMED.labels(
            llm_provider=llm_provider,
            model=model,
            token_type='input'
        ).inc(input_tokens)
        
        TOKENS_CONSUMED.labels(
            llm_provider=llm_provider,
            model=model,
            token_type='output'
        ).inc(output_tokens)
        
    def record_active_agents(self, count: int):
        """Record number of active agents."""
        if not self.enabled:
            return
            
        ACTIVE_AGENTS.set(count)
        
    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache access metrics."""
        if not self.enabled:
            return
            
        CACHE_REQUESTS.labels(cache_type=cache_type).inc()
        
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
            
    def record_error(self, error_type: str, component: str):
        """Record an error occurrence."""
        if not self.enabled:
            return
            
        ERRORS.labels(
            error_type=error_type,
            component=component
        ).inc()
        
    def record_deployment(self, version: str, environment: str):
        """Record a deployment event."""
        if not self.enabled:
            return
            
        DEPLOYMENTS.labels(
            version=version,
            environment=environment
        ).inc()


class ReflectionMetricsContext:
    """Context manager for tracking reflection metrics."""
    
    def __init__(self, collector: TelemetryCollector, task_type: str, 
                 agent_id: str):
        """Initialize metrics context.
        
        Args:
            collector: Telemetry collector instance
            task_type: Type of task being executed
            agent_id: Unique agent identifier
        """
        self.collector = collector
        self.task_type = task_type
        self.agent_id = agent_id
        self.start_time = None
        self.iterations = 0
        self.success = False
        self.error_type = None
        
    def __enter__(self):
        """Enter context and start timing."""
        self.start_time = time.time()
        self.collector.record_reflection_attempt(self.task_type, self.agent_id)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record final metrics."""
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            # Exception occurred
            error_type = exc_type.__name__
            self.collector.record_reflection_failure(
                self.task_type, self.agent_id, error_type, duration
            )
        elif self.success:
            # Successful completion
            self.collector.record_reflection_success(
                self.task_type, self.agent_id, 
                getattr(self, 'quality_score', 0.0),
                self.iterations, duration
            )
        else:
            # Failed without exception
            self.collector.record_reflection_failure(
                self.task_type, self.agent_id, 
                self.error_type or 'unknown', duration
            )
            
    def add_iteration(self):
        """Increment iteration counter."""
        self.iterations += 1
        
    def set_success(self, quality_score: float = 0.0):
        """Mark as successful with quality score."""
        self.success = True
        self.quality_score = quality_score
        
    def set_failure(self, error_type: str):
        """Mark as failed with error type."""
        self.success = False
        self.error_type = error_type


# Global telemetry collector instance
telemetry = TelemetryCollector()


def track_reflection(task_type: str, agent_id: str):
    """Decorator/context manager for tracking reflection metrics.
    
    Args:
        task_type: Type of task being executed
        agent_id: Unique agent identifier
        
    Returns:
        ReflectionMetricsContext: Context manager for metrics tracking
    """
    return ReflectionMetricsContext(telemetry, task_type, agent_id)


def configure_telemetry(enabled: bool = None, export_interval: int = None):
    """Configure global telemetry settings.
    
    Args:
        enabled: Enable/disable telemetry collection
        export_interval: Interval for metric exports
    """
    global telemetry
    
    if enabled is not None:
        telemetry.enabled = enabled
        
    if export_interval is not None:
        telemetry.export_interval = export_interval
        
    logging.getLogger(__name__).info(
        f"Telemetry configured: enabled={telemetry.enabled}, "
        f"export_interval={telemetry.export_interval}s"
    )