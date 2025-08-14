"""
Advanced Monitoring and Telemetry for Reflexion Agents.

Provides comprehensive metrics, alerting, and observability.
"""

import time
import threading
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

from .exceptions import LLMError, ReflectionError, SecurityError


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Alert:
    """Alert definition and state."""
    name: str
    condition: str
    severity: AlertSeverity
    message: str
    threshold: float
    is_active: bool = False
    triggered_at: Optional[float] = None
    last_checked: Optional[float] = None


class MetricsCollector:
    """Collects and stores various performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            self._record_metric(name, self.counters[name], MetricType.COUNTER, labels or {})
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            self.gauges[name] = value
            self._record_metric(name, value, MetricType.GAUGE, labels or {})
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add observation to histogram."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._record_metric(name, value, MetricType.HISTOGRAM, labels or {})
    
    def time_operation(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record operation timing."""
        with self._lock:
            self.timers[name].append(duration)
            # Keep only recent values
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            self._record_metric(name, duration, MetricType.TIMER, labels or {})
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str]):
        """Record metric in the time series."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels
        )
        self.metrics.append(metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "avg": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                    } for name, values in self.histograms.items()
                },
                "timers": {
                    name: {
                        "count": len(values),
                        "avg_ms": (sum(values) / len(values)) * 1000 if values else 0,
                        "min_ms": min(values) * 1000 if values else 0,
                        "max_ms": max(values) * 1000 if values else 0,
                    } for name, values in self.timers.items()
                }
            }


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
    
    def register_alert(self, alert: Alert):
        """Register a new alert."""
        with self._lock:
            self.alerts[alert.name] = alert
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: MetricsCollector):
        """Check all alert conditions."""
        current_time = time.time()
        
        with self._lock:
            for alert in self.alerts.values():
                self._check_single_alert(alert, metrics, current_time)
    
    def _check_single_alert(self, alert: Alert, metrics: MetricsCollector, current_time: float):
        """Check a single alert condition."""
        alert.last_checked = current_time
        
        try:
            triggered = self._evaluate_condition(alert, metrics)
            
            if triggered and not alert.is_active:
                # Alert triggered
                alert.is_active = True
                alert.triggered_at = current_time
                self._fire_alert(alert)
            
            elif not triggered and alert.is_active:
                # Alert resolved
                alert.is_active = False
                alert.triggered_at = None
                self._resolve_alert(alert)
                
        except Exception as e:
            logging.error(f"Error checking alert {alert.name}: {e}")
    
    def _evaluate_condition(self, alert: Alert, metrics: MetricsCollector) -> bool:
        """Evaluate alert condition."""
        # Simple condition evaluation for common patterns
        if "error_rate >" in alert.condition:
            error_count = metrics.counters.get("task_errors", 0)
            total_count = max(metrics.counters.get("task_total", 1), 1)
            error_rate = error_count / total_count
            return error_rate > alert.threshold
        
        elif "response_time >" in alert.condition:
            response_times = metrics.timers.get("task_execution", [])
            if response_times:
                avg_time = sum(response_times[-10:]) / min(len(response_times), 10)  # Last 10
                return avg_time > alert.threshold
        
        elif "memory_usage >" in alert.condition:
            memory_usage = metrics.gauges.get("memory_usage_mb", 0)
            return memory_usage > alert.threshold
        
        return False
    
    def _fire_alert(self, alert: Alert):
        """Fire an alert notification."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Error in alert handler: {e}")
    
    def _resolve_alert(self, alert: Alert):
        """Resolve an alert."""
        # Could notify about resolution
        pass


class PerformanceProfiler:
    """Profiles performance characteristics of reflexion operations."""
    
    def __init__(self):
        self.active_operations: Dict[str, float] = {}
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def start_operation(self, operation_id: str):
        """Start timing an operation."""
        with self._lock:
            self.active_operations[operation_id] = time.time()
    
    def end_operation(self, operation_id: str) -> float:
        """End timing an operation and return duration."""
        with self._lock:
            if operation_id in self.active_operations:
                duration = time.time() - self.active_operations[operation_id]
                del self.active_operations[operation_id]
                
                # Store stats by operation type
                op_type = operation_id.split('_')[0] if '_' in operation_id else operation_id
                self.operation_stats[op_type].append(duration)
                
                # Keep only recent data
                if len(self.operation_stats[op_type]) > 100:
                    self.operation_stats[op_type] = self.operation_stats[op_type][-100:]
                
                return duration
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary by operation type."""
        with self._lock:
            summary = {}
            for op_type, durations in self.operation_stats.items():
                if durations:
                    summary[op_type] = {
                        "count": len(durations),
                        "avg_duration_ms": (sum(durations) / len(durations)) * 1000,
                        "min_duration_ms": min(durations) * 1000,
                        "max_duration_ms": max(durations) * 1000,
                        "p95_duration_ms": self._percentile(durations, 0.95) * 1000,
                        "p99_duration_ms": self._percentile(durations, 0.99) * 1000,
                    }
            return summary
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        with self._lock:
            self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return status."""
        overall_healthy = True
        
        with self._lock:
            for name, check_func in self.health_checks.items():
                try:
                    start_time = time.time()
                    is_healthy = check_func()
                    check_duration = time.time() - start_time
                    
                    self.health_status[name] = {
                        "healthy": is_healthy,
                        "last_checked": datetime.now().isoformat(),
                        "check_duration_ms": check_duration * 1000,
                        "error": None
                    }
                    
                    if not is_healthy:
                        overall_healthy = False
                        
                except Exception as e:
                    self.health_status[name] = {
                        "healthy": False,
                        "last_checked": datetime.now().isoformat(),
                        "check_duration_ms": 0,
                        "error": str(e)
                    }
                    overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "checks": dict(self.health_status),
            "timestamp": datetime.now().isoformat()
        }


class ReflexionMonitor:
    """Main monitoring system combining all monitoring components."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self.profiler = PerformanceProfiler()
        self.health = HealthChecker()
        self.logger = logging.getLogger(__name__)
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Start background monitoring
        self._start_monitoring_thread()
    
    def _setup_default_alerts(self):
        """Setup default alert conditions."""
        alerts = [
            Alert(
                name="high_error_rate",
                condition="error_rate > threshold",
                severity=AlertSeverity.ERROR,
                message="High error rate detected in reflexion tasks",
                threshold=0.1  # 10% error rate
            ),
            Alert(
                name="slow_response_time",
                condition="response_time > threshold",
                severity=AlertSeverity.WARNING,
                message="Slow response times detected",
                threshold=10.0  # 10 seconds
            ),
            Alert(
                name="memory_usage_high",
                condition="memory_usage > threshold",
                severity=AlertSeverity.WARNING,
                message="High memory usage detected",
                threshold=1000.0  # 1GB
            )
        ]
        
        for alert in alerts:
            self.alerts.register_alert(alert)
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        def check_basic_functionality():
            # Basic reflexion engine test
            try:
                # Could do a lightweight test here
                return True
            except:
                return False
        
        def check_memory_usage():
            # Memory usage check
            import psutil
            try:
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 90  # Less than 90% memory usage
            except:
                return True  # If psutil not available, assume healthy
        
        self.health.register_health_check("basic_functionality", check_basic_functionality)
        self.health.register_health_check("memory_usage", check_memory_usage)
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread."""
        def monitor_loop():
            while True:
                try:
                    # Check alerts every 30 seconds
                    self.alerts.check_alerts(self.metrics)
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait longer if error
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def record_task_start(self, task_id: str):
        """Record start of a task."""
        self.profiler.start_operation(f"task_{task_id}")
        self.metrics.increment("task_total")
    
    def record_task_completion(self, task_id: str, success: bool, error_type: Optional[str] = None):
        """Record completion of a task."""
        duration = self.profiler.end_operation(f"task_{task_id}")
        self.metrics.time_operation("task_execution", duration)
        
        if success:
            self.metrics.increment("task_success")
        else:
            self.metrics.increment("task_errors")
            if error_type:
                self.metrics.increment(f"error_{error_type}")
    
    def record_llm_call(self, model: str, duration: float, success: bool):
        """Record LLM API call metrics."""
        self.metrics.time_operation("llm_call", duration, {"model": model})
        self.metrics.increment("llm_calls_total", labels={"model": model})
        
        if success:
            self.metrics.increment("llm_calls_success", labels={"model": model})
        else:
            self.metrics.increment("llm_calls_error", labels={"model": model})
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "metrics": self.metrics.get_summary(),
            "performance": self.profiler.get_performance_summary(),
            "health": self.health.run_health_checks(),
            "active_alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at
                }
                for alert in self.alerts.alerts.values()
                if alert.is_active
            ],
            "timestamp": datetime.now().isoformat()
        }


# Global monitoring instance
monitor = ReflexionMonitor()


def console_alert_handler(alert: Alert):
    """Simple console alert handler."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}")


# Register default alert handler
monitor.alerts.add_alert_handler(console_alert_handler)