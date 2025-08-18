"""Comprehensive Monitoring System V2.0 for Production Reflexion Systems.

This module provides enterprise-grade monitoring, alerting, performance tracking,
and observability for reflexion agent systems with real-time analytics.
"""

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from statistics import mean, median, stdev
import hashlib
# Optional psutil import for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Mock psutil functions for compatibility
    class psutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 45.0  # Mock CPU usage
        
        @staticmethod
        def virtual_memory():
            class MockMemory:
                total = 8 * 1024 * 1024 * 1024  # 8GB
                available = 4 * 1024 * 1024 * 1024  # 4GB
                percent = 50.0
                used = 4 * 1024 * 1024 * 1024  # 4GB
            return MockMemory()
        
        @staticmethod
        def disk_usage(path):
            class MockDisk:
                total = 100 * 1024 * 1024 * 1024  # 100GB
                used = 50 * 1024 * 1024 * 1024   # 50GB
                free = 50 * 1024 * 1024 * 1024   # 50GB
            return MockDisk()
        
        @staticmethod
        def net_io_counters():
            class MockNetwork:
                bytes_sent = 1000000
                bytes_recv = 2000000
                packets_sent = 1000
                packets_recv = 2000
            return MockNetwork()
        
        @staticmethod
        def pids():
            return list(range(100))  # Mock 100 processes
        
        @staticmethod
        def cpu_count():
            return 4  # Mock 4 CPU cores
        
        @staticmethod
        def getloadavg():
            return (1.0, 1.5, 2.0)  # Mock load averages
        
        class Process:
            def __init__(self):
                pass
            
            def memory_info(self):
                class MockMemInfo:
                    rss = 50 * 1024 * 1024  # 50MB
                return MockMemInfo()
            
            def pid(self):
                return 1234
            
            def num_threads(self):
                return 8
            
            def open_files(self):
                return []

from .logging_config import logger


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringStatus(Enum):
    """Monitoring system status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points."""
    name: str
    metric_type: MetricType
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    def add_point(self, value: Union[int, float], tags: Dict[str, str] = None):
        """Add a new metric point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
    
    def get_recent_values(self, duration_seconds: int = 60) -> List[float]:
        """Get recent metric values within specified duration."""
        cutoff_time = time.time() - duration_seconds
        return [p.value for p in self.points if p.timestamp >= cutoff_time]
    
    def get_statistics(self, duration_seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary of recent values."""
        values = self.get_recent_values(duration_seconds)
        
        if not values:
            return {"count": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
        
        return {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "std": stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "p95": sorted(values)[int(len(values) * 0.95)] if values else 0,
            "p99": sorted(values)[int(len(values) * 0.99)] if values else 0
        }


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    component: str
    metric_name: str
    message: str
    threshold_value: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_timestamp = datetime.now()


@dataclass
class ThresholdRule:
    """Alert threshold rule."""
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: AlertSeverity
    duration_seconds: int = 60  # How long condition must persist
    component: str = ""
    description: str = ""
    enabled: bool = True


class PerformanceProfiler:
    """Advanced performance profiler for reflexion operations."""
    
    def __init__(self):
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.completed_operations: deque = deque(maxlen=5000)
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0,
            "min_time": float('inf'),
            "max_time": 0,
            "errors": 0,
            "success_rate": 0
        })
        
    def start_operation(self, operation_id: str, operation_type: str, metadata: Dict[str, Any] = None):
        """Start tracking an operation."""
        self.active_operations[operation_id] = {
            "operation_type": operation_type,
            "start_time": time.time(),
            "metadata": metadata or {},
            "thread_id": threading.get_ident(),
            "memory_start": psutil.Process().memory_info().rss
        }
    
    def end_operation(self, operation_id: str, success: bool = True, result_metadata: Dict[str, Any] = None):
        """End tracking an operation."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations.pop(operation_id)
        end_time = time.time()
        duration = end_time - operation["start_time"]
        
        # Calculate memory delta
        memory_end = psutil.Process().memory_info().rss
        memory_delta = memory_end - operation["memory_start"]
        
        # Record completed operation
        completed_op = {
            "operation_id": operation_id,
            "operation_type": operation["operation_type"],
            "duration": duration,
            "success": success,
            "start_time": operation["start_time"],
            "end_time": end_time,
            "thread_id": operation["thread_id"],
            "memory_delta": memory_delta,
            "metadata": operation["metadata"],
            "result_metadata": result_metadata or {}
        }
        
        self.completed_operations.append(completed_op)
        
        # Update statistics
        op_type = operation["operation_type"]
        stats = self.operation_stats[op_type]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        
        if not success:
            stats["errors"] += 1
        
        stats["success_rate"] = (stats["count"] - stats["errors"]) / stats["count"] * 100
    
    def get_operation_statistics(self, operation_type: str = None) -> Dict[str, Any]:
        """Get statistics for operations."""
        if operation_type:
            if operation_type not in self.operation_stats:
                return {}
            
            stats = self.operation_stats[operation_type].copy()
            if stats["count"] > 0:
                stats["avg_time"] = stats["total_time"] / stats["count"]
                stats["min_time"] = stats["min_time"] if stats["min_time"] != float('inf') else 0
            return stats
        
        # Return all operation statistics
        result = {}
        for op_type, stats in self.operation_stats.items():
            result[op_type] = stats.copy()
            if stats["count"] > 0:
                result[op_type]["avg_time"] = stats["total_time"] / stats["count"]
                result[op_type]["min_time"] = stats["min_time"] if stats["min_time"] != float('inf') else 0
        
        return result
    
    def get_recent_operations(self, operation_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent completed operations."""
        operations = list(self.completed_operations)
        
        if operation_type:
            operations = [op for op in operations if op["operation_type"] == operation_type]
        
        return operations[-limit:]


class SystemResourceMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread = None
        self.resource_history: Dict[str, deque] = {
            "cpu_percent": deque(maxlen=1440),  # 24 hours at 1-minute intervals
            "memory_percent": deque(maxlen=1440),
            "memory_available": deque(maxlen=1440),
            "disk_usage": deque(maxlen=1440),
            "network_io": deque(maxlen=1440),
            "process_count": deque(maxlen=1440)
        }
        self.last_network_io = None
        
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"Started system resource monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped system resource monitoring")
    
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                timestamp = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.resource_history["cpu_percent"].append((timestamp, cpu_percent))
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.resource_history["memory_percent"].append((timestamp, memory.percent))
                self.resource_history["memory_available"].append((timestamp, memory.available))
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.resource_history["disk_usage"].append((timestamp, disk_percent))
                
                # Network I/O
                network = psutil.net_io_counters()
                if self.last_network_io:
                    bytes_sent_delta = network.bytes_sent - self.last_network_io.bytes_sent
                    bytes_recv_delta = network.bytes_recv - self.last_network_io.bytes_recv
                    total_delta = bytes_sent_delta + bytes_recv_delta
                    self.resource_history["network_io"].append((timestamp, total_delta))
                
                self.last_network_io = network
                
                # Process count
                process_count = len(psutil.pids())
                self.resource_history["process_count"].append((timestamp, process_count))
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "processes": len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"Error getting current resources: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def get_resource_trends(self, duration_minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """Get resource usage trends over specified duration."""
        cutoff_time = time.time() - (duration_minutes * 60)
        trends = {}
        
        for resource_name, history in self.resource_history.items():
            recent_points = [(ts, val) for ts, val in history if ts >= cutoff_time]
            
            if recent_points:
                values = [val for _, val in recent_points]
                trends[resource_name] = {
                    "current": values[-1] if values else 0,
                    "avg": mean(values),
                    "min": min(values),
                    "max": max(values),
                    "std": stdev(values) if len(values) > 1 else 0,
                    "trend": self._calculate_trend(recent_points)
                }
            else:
                trends[resource_name] = {
                    "current": 0, "avg": 0, "min": 0, "max": 0, "std": 0, "trend": 0
                }
        
        return trends
    
    def _calculate_trend(self, points: List[Tuple[float, float]]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(points) < 2:
            return 0
        
        # Simple linear trend calculation
        n = len(points)
        sum_x = sum(i for i, _ in enumerate(points))
        sum_y = sum(val for _, val in points)
        sum_xy = sum(i * val for i, (_, val) in enumerate(points))
        sum_x2 = sum(i * i for i, _ in enumerate(points))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize slope to -1 to 1 range
        return max(-1, min(1, slope / 10))


class ComprehensiveMonitoringSystem:
    """Enterprise-grade monitoring system for reflexion agents."""
    
    def __init__(self):
        self.metrics: Dict[str, MetricSeries] = {}
        self.alerts: Dict[str, Alert] = {}
        self.threshold_rules: List[ThresholdRule] = []
        self.profiler = PerformanceProfiler()
        self.resource_monitor = SystemResourceMonitor()
        
        # Monitoring state
        self.monitoring_active = False
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Built-in metrics
        self._setup_default_metrics()
        self._setup_default_thresholds()
        
        # Start background monitoring
        self.start_monitoring()
    
    def _setup_default_metrics(self):
        """Setup default system metrics."""
        default_metrics = [
            ("system.cpu.percent", MetricType.GAUGE, "CPU usage percentage"),
            ("system.memory.percent", MetricType.GAUGE, "Memory usage percentage"),
            ("system.disk.percent", MetricType.GAUGE, "Disk usage percentage"),
            ("reflexion.operations.total", MetricType.COUNTER, "Total reflexion operations"),
            ("reflexion.operations.success", MetricType.COUNTER, "Successful reflexion operations"),
            ("reflexion.operations.errors", MetricType.COUNTER, "Failed reflexion operations"),
            ("reflexion.response.time", MetricType.TIMER, "Reflexion response time"),
            ("reflexion.iterations.count", MetricType.HISTOGRAM, "Number of reflexion iterations"),
            ("reflexion.quality.score", MetricType.GAUGE, "Reflexion quality score"),
            ("reflexion.memory.usage", MetricType.GAUGE, "Memory usage during reflexion"),
            ("alerts.active.count", MetricType.GAUGE, "Number of active alerts"),
            ("errors.recovery.success", MetricType.COUNTER, "Successful error recoveries"),
            ("errors.recovery.failures", MetricType.COUNTER, "Failed error recoveries"),
            ("cache.hit.rate", MetricType.RATE, "Cache hit rate"),
            ("api.requests.rate", MetricType.RATE, "API request rate")
        ]
        
        for name, metric_type, description in default_metrics:
            self.create_metric(name, metric_type, description=description)
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        default_thresholds = [
            ThresholdRule("system.cpu.percent", ">", 80, AlertSeverity.WARNING, 120, "system", "CPU usage high"),
            ThresholdRule("system.cpu.percent", ">", 95, AlertSeverity.CRITICAL, 60, "system", "CPU usage critical"),
            ThresholdRule("system.memory.percent", ">", 85, AlertSeverity.WARNING, 120, "system", "Memory usage high"),
            ThresholdRule("system.memory.percent", ">", 95, AlertSeverity.CRITICAL, 60, "system", "Memory usage critical"),
            ThresholdRule("system.disk.percent", ">", 90, AlertSeverity.WARNING, 300, "system", "Disk usage high"),
            ThresholdRule("reflexion.quality.score", "<", 0.6, AlertSeverity.WARNING, 180, "reflexion", "Quality score low"),
            ThresholdRule("reflexion.quality.score", "<", 0.4, AlertSeverity.ERROR, 120, "reflexion", "Quality score very low"),
            ThresholdRule("errors.recovery.failures", ">", 10, AlertSeverity.ERROR, 300, "recovery", "High recovery failure rate"),
            ThresholdRule("alerts.active.count", ">", 20, AlertSeverity.WARNING, 60, "monitoring", "Too many active alerts")
        ]
        
        self.threshold_rules.extend(default_thresholds)
    
    def create_metric(self, name: str, metric_type: MetricType, tags: Dict[str, str] = None, description: str = ""):
        """Create a new metric series."""
        if name in self.metrics:
            logger.warning(f"Metric {name} already exists")
            return
        
        self.metrics[name] = MetricSeries(
            name=name,
            metric_type=metric_type,
            tags=tags or {},
            description=description
        )
        logger.debug(f"Created metric: {name} ({metric_type.value})")
    
    def record_metric(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Record a metric value."""
        if name not in self.metrics:
            logger.warning(f"Metric {name} not found, creating as gauge")
            self.create_metric(name, MetricType.GAUGE)
        
        self.metrics[name].add_point(value, tags)
        
        # Check for threshold violations
        self._check_thresholds(name, value)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        if name not in self.metrics:
            self.create_metric(name, MetricType.COUNTER)
        
        # For counters, we track the increment
        last_value = self.metrics[name].points[-1].value if self.metrics[name].points else 0
        self.record_metric(name, last_value + value, tags)
    
    def set_gauge(self, name: str, value: Union[int, float], tags: Dict[str, str] = None):
        """Set a gauge metric value."""
        if name not in self.metrics:
            self.create_metric(name, MetricType.GAUGE)
        
        self.record_metric(name, value, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        if name not in self.metrics:
            self.create_metric(name, MetricType.TIMER)
        
        self.record_metric(name, duration, tags)
    
    def add_threshold_rule(self, rule: ThresholdRule):
        """Add a new threshold rule."""
        self.threshold_rules.append(rule)
        logger.info(f"Added threshold rule: {rule.metric_name} {rule.operator} {rule.threshold}")
    
    def _check_thresholds(self, metric_name: str, value: float):
        """Check if metric value violates any thresholds."""
        for rule in self.threshold_rules:
            if rule.metric_name != metric_name or not rule.enabled:
                continue
            
            violated = False
            
            if rule.operator == ">" and value > rule.threshold:
                violated = True
            elif rule.operator == "<" and value < rule.threshold:
                violated = True
            elif rule.operator == ">=" and value >= rule.threshold:
                violated = True
            elif rule.operator == "<=" and value <= rule.threshold:
                violated = True
            elif rule.operator == "==" and value == rule.threshold:
                violated = True
            elif rule.operator == "!=" and value != rule.threshold:
                violated = True
            
            if violated:
                # Check if condition persists for required duration
                if self._condition_persists(rule, value):
                    self._trigger_alert(rule, value)
    
    def _condition_persists(self, rule: ThresholdRule, current_value: float) -> bool:
        """Check if threshold violation persists for required duration."""
        metric = self.metrics.get(rule.metric_name)
        if not metric:
            return False
        
        # Get values from the last duration_seconds
        recent_values = metric.get_recent_values(rule.duration_seconds)
        
        if len(recent_values) < 3:  # Need some data points
            return False
        
        # Check if majority of recent values violate the threshold
        violations = 0
        for value in recent_values:
            if rule.operator == ">" and value > rule.threshold:
                violations += 1
            elif rule.operator == "<" and value < rule.threshold:
                violations += 1
            elif rule.operator == ">=" and value >= rule.threshold:
                violations += 1
            elif rule.operator == "<=" and value <= rule.threshold:
                violations += 1
        
        # Require at least 70% of samples to violate threshold
        return violations / len(recent_values) >= 0.7
    
    def _trigger_alert(self, rule: ThresholdRule, actual_value: float):
        """Trigger an alert for threshold violation."""
        alert_id = hashlib.md5(f"{rule.metric_name}_{rule.threshold}_{rule.operator}".encode()).hexdigest()
        
        # Check if alert already exists and is not resolved
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return  # Don't duplicate active alerts
        
        alert = Alert(
            alert_id=alert_id,
            severity=rule.severity,
            component=rule.component or "unknown",
            metric_name=rule.metric_name,
            message=f"{rule.description or 'Threshold violation'}: {rule.metric_name} {rule.operator} {rule.threshold} (actual: {actual_value})",
            threshold_value=rule.threshold,
            actual_value=actual_value,
            timestamp=datetime.now()
        )
        
        self.alerts[alert_id] = alert
        
        # Update alert count metric
        active_alerts = len([a for a in self.alerts.values() if not a.resolved])
        self.set_gauge("alerts.active.count", active_alerts)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolve()
            
            # Update alert count
            active_alerts = len([a for a in self.alerts.values() if not a.resolved])
            self.set_gauge("alerts.active.count", active_alerts)
            
            logger.info(f"Alert {alert_id} resolved")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.resource_monitor.start_monitoring(60)  # Monitor every minute
        
        # Start background metrics collection
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
        logger.info("Comprehensive monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        self.resource_monitor.stop_monitoring()
        logger.info("Comprehensive monitoring system stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                resources = self.resource_monitor.get_current_resources()
                
                if "error" not in resources:
                    self.set_gauge("system.cpu.percent", resources["cpu"]["percent"])
                    self.set_gauge("system.memory.percent", resources["memory"]["percent"])
                    self.set_gauge("system.disk.percent", resources["disk"]["percent"])
                
                time.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def get_metric_statistics(self, metric_name: str, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get statistical summary for a metric."""
        if metric_name not in self.metrics:
            return {"error": f"Metric {metric_name} not found"}
        
        return self.metrics[metric_name].get_statistics(duration_seconds)
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for name, metric in self.metrics.items():
            stats = metric.get_statistics()
            summary[name] = {
                "type": metric.metric_type.value,
                "description": metric.description,
                "current_value": stats["mean"],
                "data_points": stats["count"],
                "last_updated": max((p.timestamp for p in metric.points), default=0)
            }
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [
            {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "component": alert.component,
                "metric_name": alert.metric_name,
                "message": alert.message,
                "threshold_value": alert.threshold_value,
                "actual_value": alert.actual_value,
                "timestamp": alert.timestamp.isoformat(),
                "duration": (datetime.now() - alert.timestamp).total_seconds()
            }
            for alert in self.alerts.values() if not alert.resolved
        ]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        active_alerts = [a for a in self.alerts.values() if not a.resolved]
        
        # Determine overall health status
        health_status = MonitoringStatus.HEALTHY
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]
        warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
        
        if critical_alerts:
            health_status = MonitoringStatus.CRITICAL
        elif error_alerts:
            health_status = MonitoringStatus.CRITICAL
        elif warning_alerts:
            health_status = MonitoringStatus.WARNING
        
        # Get resource trends
        resource_trends = self.resource_monitor.get_resource_trends(60)
        
        # Performance summary
        perf_stats = self.profiler.get_operation_statistics()
        
        return {
            "overall_status": health_status.value,
            "timestamp": datetime.now().isoformat(),
            "alerts": {
                "total_active": len(active_alerts),
                "critical": len(critical_alerts),
                "error": len(error_alerts),
                "warning": len(warning_alerts),
                "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO])
            },
            "system_resources": resource_trends,
            "performance": {
                "operations_tracked": len(perf_stats),
                "total_operations": sum(stats["count"] for stats in perf_stats.values()),
                "avg_success_rate": mean([stats["success_rate"] for stats in perf_stats.values()]) if perf_stats else 100
            },
            "monitoring": {
                "active": self.monitoring_active,
                "metrics_tracked": len(self.metrics),
                "threshold_rules": len(self.threshold_rules)
            }
        }
    
    def export_metrics(self, format_type: str = "json", duration_hours: int = 24) -> str:
        """Export metrics data."""
        cutoff_time = time.time() - (duration_hours * 3600)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "duration_hours": duration_hours,
            "metrics": {}
        }
        
        for name, metric in self.metrics.items():
            recent_points = [
                {"timestamp": p.timestamp, "value": p.value, "tags": p.tags}
                for p in metric.points if p.timestamp >= cutoff_time
            ]
            
            export_data["metrics"][name] = {
                "type": metric.metric_type.value,
                "description": metric.description,
                "tags": metric.tags,
                "data_points": recent_points,
                "statistics": metric.get_statistics(duration_hours * 3600)
            }
        
        if format_type == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    # Context manager for operation profiling
    def profile_operation(self, operation_type: str, metadata: Dict[str, Any] = None):
        """Context manager for profiling operations."""
        class OperationProfiler:
            def __init__(self, monitoring_system, op_type, meta):
                self.monitoring_system = monitoring_system
                self.operation_type = op_type
                self.metadata = meta
                self.operation_id = None
            
            def __enter__(self):
                self.operation_id = f"{self.operation_type}_{time.time()}_{id(self)}"
                self.monitoring_system.profiler.start_operation(
                    self.operation_id, self.operation_type, self.metadata
                )
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                success = exc_type is None
                result_metadata = {"exception": str(exc_val)} if exc_val else {}
                
                self.monitoring_system.profiler.end_operation(
                    self.operation_id, success, result_metadata
                )
                
                # Record metrics
                self.monitoring_system.increment_counter("reflexion.operations.total")
                if success:
                    self.monitoring_system.increment_counter("reflexion.operations.success")
                else:
                    self.monitoring_system.increment_counter("reflexion.operations.errors")
        
        return OperationProfiler(self, operation_type, metadata)


# Global monitoring system instance
monitoring_system = ComprehensiveMonitoringSystem()


# Decorators for automatic monitoring
def monitor_performance(operation_type: str = None, track_memory: bool = True):
    """Decorator for automatic performance monitoring."""
    def decorator(func):
        op_type = operation_type or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with monitoring_system.profile_operation(op_type, {"track_memory": track_memory}):
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        monitoring_system.record_timer(f"{op_type}.duration", duration)
                        return result
                    except Exception as e:
                        monitoring_system.increment_counter(f"{op_type}.errors")
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with monitoring_system.profile_operation(op_type, {"track_memory": track_memory}):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        monitoring_system.record_timer(f"{op_type}.duration", duration)
                        return result
                    except Exception as e:
                        monitoring_system.increment_counter(f"{op_type}.errors")
                        raise
            return sync_wrapper
    
    return decorator


# Example alert handler
def log_alert(alert: Alert):
    """Example alert handler that logs alerts."""
    logger.warning(f"ALERT: {alert.severity.value.upper()} - {alert.message}")


# Setup default alert handler
monitoring_system.add_alert_callback(log_alert)


async def test_monitoring_system():
    """Test the comprehensive monitoring system."""
    logger.info("Testing Comprehensive Monitoring System")
    
    # Record some test metrics
    monitoring_system.increment_counter("test.operations", 5)
    monitoring_system.set_gauge("test.temperature", 75.5)
    monitoring_system.record_timer("test.processing_time", 1.25)
    
    # Test operation profiling
    with monitoring_system.profile_operation("test_operation", {"test": True}):
        await asyncio.sleep(0.1)  # Simulate work
    
    # Get system health
    health = monitoring_system.get_system_health()
    logger.info(f"System health: {health['overall_status']}")
    
    # Get metrics summary
    summary = monitoring_system.get_all_metrics_summary()
    logger.info(f"Tracking {len(summary)} metrics")
    
    return monitoring_system


if __name__ == "__main__":
    # Run test
    asyncio.run(test_monitoring_system())