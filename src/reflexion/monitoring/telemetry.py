"""Advanced telemetry and monitoring for reflexion agents."""

import asyncio
import json
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import socket
import platform

from ..core.types import ReflexionResult, Reflection
from ..core.logging_config import logger


@dataclass
class TelemetryEvent:
    """Telemetry event data structure."""
    event_type: str
    timestamp: str
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    process_count: int
    thread_count: int
    file_descriptors: int
    uptime_seconds: float


@dataclass
class ReflexionMetrics:
    """Reflexion-specific performance metrics."""
    timestamp: str
    total_tasks_executed: int
    successful_tasks: int
    failed_tasks: int
    avg_execution_time: float
    avg_iterations: float
    avg_reflections_per_task: float
    cache_hit_rate: float
    algorithm_distribution: Dict[str, int]
    error_types: Dict[str, int]
    performance_trends: Dict[str, float]


class MetricsCollector:
    """Collects and aggregates metrics from various sources."""
    
    def __init__(self, collection_interval: float = 30.0):
        self.collection_interval = collection_interval
        self.system_metrics_history = deque(maxlen=1000)
        self.reflexion_metrics_history = deque(maxlen=1000)
        self.custom_metrics = defaultdict(deque)
        
        self.is_collecting = False
        self.collection_thread = None
        
        # Reflexion-specific counters
        self.task_counters = defaultdict(int)
        self.execution_times = deque(maxlen=1000)
        self.reflection_counts = deque(maxlen=1000)
        self.algorithm_usage = defaultdict(int)
        self.error_tracking = defaultdict(int)
        
        self.start_time = time.time()
        
    def start_collection(self):
        """Start periodic metrics collection."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.is_collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect reflexion metrics
                reflexion_metrics = self._collect_reflexion_metrics()
                self.reflexion_metrics_history.append(reflexion_metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv
            }
            
            # Process info
            process = psutil.Process()
            process_info = process.as_dict(['num_threads', 'num_fds'])
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / 1024 / 1024,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_io_bytes=network_io,
                process_count=len(psutil.pids()),
                thread_count=process_info.get('num_threads', 0),
                file_descriptors=process_info.get('num_fds', 0),
                uptime_seconds=time.time() - self.start_time
            )
            
        except ImportError:
            # Fallback when psutil is not available
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_io_bytes={"bytes_sent": 0, "bytes_recv": 0},
                process_count=0,
                thread_count=0,
                file_descriptors=0,
                uptime_seconds=time.time() - self.start_time
            )
    
    def _collect_reflexion_metrics(self) -> ReflexionMetrics:
        """Collect reflexion-specific metrics."""
        total_tasks = self.task_counters['total']
        successful_tasks = self.task_counters['successful']
        failed_tasks = self.task_counters['failed']
        
        # Calculate averages
        avg_execution_time = 0.0
        if self.execution_times:
            avg_execution_time = sum(self.execution_times) / len(self.execution_times)
        
        avg_reflections = 0.0
        if self.reflection_counts:
            avg_reflections = sum(self.reflection_counts) / len(self.reflection_counts)
        
        # Calculate cache hit rate (placeholder)
        cache_hit_rate = 0.0  # Would be updated by cache system
        
        # Performance trends (simplified)
        recent_times = list(self.execution_times)[-10:]
        older_times = list(self.execution_times)[-20:-10] if len(self.execution_times) >= 20 else []
        
        performance_trends = {}
        if recent_times and older_times:
            recent_avg = sum(recent_times) / len(recent_times)
            older_avg = sum(older_times) / len(older_times)
            performance_trends["execution_time_trend"] = (recent_avg - older_avg) / older_avg
        
        return ReflexionMetrics(
            timestamp=datetime.now().isoformat(),
            total_tasks_executed=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            avg_execution_time=avg_execution_time,
            avg_iterations=0.0,  # Would be calculated from iteration tracking
            avg_reflections_per_task=avg_reflections,
            cache_hit_rate=cache_hit_rate,
            algorithm_distribution=dict(self.algorithm_usage),
            error_types=dict(self.error_tracking),
            performance_trends=performance_trends
        )
    
    def record_task_execution(self, result: ReflexionResult, algorithm: str = "classic"):
        """Record task execution metrics."""
        self.task_counters['total'] += 1
        
        if result.success:
            self.task_counters['successful'] += 1
        else:
            self.task_counters['failed'] += 1
        
        self.execution_times.append(result.total_time)
        self.reflection_counts.append(len(result.reflections))
        self.algorithm_usage[algorithm] += 1
        
        # Track errors
        if not result.success and result.metadata and 'error' in result.metadata:
            error_type = result.metadata.get('error_type', 'unknown')
            self.error_tracking[error_type] += 1
    
    def record_custom_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record custom metric with optional tags."""
        metric_entry = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {}
        }
        
        self.custom_metrics[name].append(metric_entry)
        
        # Maintain reasonable history size
        if len(self.custom_metrics[name]) > 1000:
            self.custom_metrics[name] = deque(list(self.custom_metrics[name])[-500:], maxlen=1000)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest collected metrics."""
        return {
            "system": asdict(self.system_metrics_history[-1]) if self.system_metrics_history else None,
            "reflexion": asdict(self.reflexion_metrics_history[-1]) if self.reflexion_metrics_history else None,
            "custom": {name: list(metrics)[-1] for name, metrics in self.custom_metrics.items()},
            "collection_info": {
                "collection_interval": self.collection_interval,
                "is_collecting": self.is_collecting,
                "uptime_seconds": time.time() - self.start_time
            }
        }


class AlertManager:
    """Manages alerts and notifications for system health."""
    
    def __init__(self):
        self.alert_rules = []
        self.alert_history = deque(maxlen=1000)
        self.notification_handlers = []
        self.active_alerts = {}
        
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                      severity: str = "warning", cooldown_seconds: float = 300.0):
        """Add alert rule with condition function."""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "cooldown_seconds": cooldown_seconds,
            "last_triggered": 0.0
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def add_notification_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    # Check cooldown period
                    time_since_last = current_time - rule["last_triggered"]
                    if time_since_last >= rule["cooldown_seconds"]:
                        self._trigger_alert(rule, metrics)
                        rule["last_triggered"] = current_time
                        
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger alert and send notifications."""
        alert = {
            "rule_name": rule["name"],
            "severity": rule["severity"],
            "timestamp": datetime.now().isoformat(),
            "metrics_snapshot": metrics,
            "message": f"Alert triggered: {rule['name']}"
        }
        
        self.alert_history.append(alert)
        self.active_alerts[rule["name"]] = alert
        
        logger.warning(f"Alert triggered: {rule['name']} (severity: {rule['severity']})")
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def resolve_alert(self, rule_name: str):
        """Manually resolve an active alert."""
        if rule_name in self.active_alerts:
            del self.active_alerts[rule_name]
            logger.info(f"Alert resolved: {rule_name}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return list(self.active_alerts.values())


class TelemetryExporter:
    """Exports telemetry data to external systems."""
    
    def __init__(self):
        self.export_handlers = []
        self.export_queue = deque(maxlen=10000)
        self.export_thread = None
        self.is_exporting = False
        
    def add_export_handler(self, handler: Callable[[List[TelemetryEvent]], None]):
        """Add export handler for telemetry events."""
        self.export_handlers.append(handler)
    
    def export_event(self, event: TelemetryEvent):
        """Queue event for export."""
        self.export_queue.append(event)
    
    def start_export_worker(self, batch_size: int = 100, export_interval: float = 10.0):
        """Start background export worker."""
        if self.is_exporting:
            return
        
        def export_worker():
            while self.is_exporting:
                try:
                    # Collect batch of events
                    events = []
                    for _ in range(min(batch_size, len(self.export_queue))):
                        if self.export_queue:
                            events.append(self.export_queue.popleft())
                    
                    # Export batch if not empty
                    if events:
                        for handler in self.export_handlers:
                            try:
                                handler(events)
                            except Exception as e:
                                logger.error(f"Export handler failed: {e}")
                    
                    time.sleep(export_interval)
                    
                except Exception as e:
                    logger.error(f"Export worker error: {e}")
                    time.sleep(5)
        
        self.is_exporting = True
        self.export_thread = threading.Thread(target=export_worker, daemon=True)
        self.export_thread.start()
        logger.info("Telemetry export worker started")
    
    def stop_export_worker(self):
        """Stop export worker."""
        self.is_exporting = False
        if self.export_thread:
            self.export_thread.join(timeout=5.0)
        logger.info("Telemetry export worker stopped")


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.health_checks = {}
        self.health_history = deque(maxlen=100)
        
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_func()
                duration = time.time() - start_time
                
                check_result = {
                    "healthy": result.get("healthy", True),
                    "message": result.get("message", "OK"),
                    "details": result.get("details", {}),
                    "duration_ms": duration * 1000,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not check_result["healthy"]:
                    overall_healthy = False
                
                results[name] = check_result
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = {
                    "healthy": False,
                    "message": f"Health check failed: {str(e)}",
                    "details": {"error": str(e)},
                    "duration_ms": 0,
                    "timestamp": datetime.now().isoformat()
                }
                overall_healthy = False
        
        # Record health check result
        health_summary = {
            "overall_healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }
        
        self.health_history.append(health_summary)
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        if not self.health_history:
            return {"overall_healthy": True, "message": "No health checks performed yet"}
        
        latest = self.health_history[-1]
        
        # Calculate health trends
        if len(self.health_history) >= 5:
            recent_health = [h["overall_healthy"] for h in list(self.health_history)[-5:]]
            health_trend = sum(recent_health) / len(recent_health)
        else:
            health_trend = 1.0 if latest["overall_healthy"] else 0.0
        
        return {
            "overall_healthy": latest["overall_healthy"],
            "health_trend": health_trend,
            "last_check": latest["timestamp"],
            "total_checks": len(self.health_history),
            "failing_checks": [
                name for name, result in latest["checks"].items() 
                if not result["healthy"]
            ]
        }


class TelemetryManager:
    """Main telemetry management system."""
    
    def __init__(self, collection_interval: float = 30.0):
        self.metrics_collector = MetricsCollector(collection_interval)
        self.alert_manager = AlertManager()
        self.telemetry_exporter = TelemetryExporter()
        self.health_checker = HealthChecker()
        
        self.setup_default_alerts()
        self.setup_default_health_checks()
        
        logger.info("Telemetry manager initialized")
    
    def setup_default_alerts(self):
        """Setup default alert rules."""
        # High CPU usage alert
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            lambda m: m.get("system", {}).get("cpu_usage_percent", 0) > 80,
            severity="warning",
            cooldown_seconds=300
        )
        
        # High memory usage alert
        self.alert_manager.add_alert_rule(
            "high_memory_usage", 
            lambda m: m.get("system", {}).get("memory_usage_percent", 0) > 85,
            severity="critical",
            cooldown_seconds=300
        )
        
        # Low success rate alert
        self.alert_manager.add_alert_rule(
            "low_success_rate",
            lambda m: (
                m.get("reflexion", {}).get("successful_tasks", 0) / 
                max(m.get("reflexion", {}).get("total_tasks_executed", 1), 1)
            ) < 0.5,
            severity="warning",
            cooldown_seconds=600
        )
        
        # High execution time alert
        self.alert_manager.add_alert_rule(
            "high_execution_time",
            lambda m: m.get("reflexion", {}).get("avg_execution_time", 0) > 120,
            severity="warning",
            cooldown_seconds=300
        )
    
    def setup_default_health_checks(self):
        """Setup default health checks."""
        def memory_health_check():
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    "healthy": memory.percent < 90,
                    "message": f"Memory usage: {memory.percent:.1f}%",
                    "details": {"memory_percent": memory.percent}
                }
            except ImportError:
                return {"healthy": True, "message": "Memory check not available"}
        
        def disk_health_check():
            try:
                import psutil
                disk = psutil.disk_usage('/')
                return {
                    "healthy": disk.percent < 90,
                    "message": f"Disk usage: {disk.percent:.1f}%",
                    "details": {"disk_percent": disk.percent}
                }
            except ImportError:
                return {"healthy": True, "message": "Disk check not available"}
        
        def process_health_check():
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                return {
                    "healthy": memory_mb < 1024,  # < 1GB
                    "message": f"Process memory: {memory_mb:.1f}MB",
                    "details": {"process_memory_mb": memory_mb}
                }
            except ImportError:
                return {"healthy": True, "message": "Process check not available"}
        
        self.health_checker.register_health_check("memory", memory_health_check)
        self.health_checker.register_health_check("disk", disk_health_check)
        self.health_checker.register_health_check("process", process_health_check)
    
    def start(self):
        """Start all telemetry components."""
        self.metrics_collector.start_collection()
        self.telemetry_exporter.start_export_worker()
        logger.info("Telemetry system started")
    
    def stop(self):
        """Stop all telemetry components."""
        self.metrics_collector.stop_collection()
        self.telemetry_exporter.stop_export_worker()
        logger.info("Telemetry system stopped")
    
    def record_task_execution(self, result: ReflexionResult, algorithm: str = "classic"):
        """Record task execution for monitoring."""
        self.metrics_collector.record_task_execution(result, algorithm)
        
        # Create telemetry event
        event = TelemetryEvent(
            event_type="task_execution",
            timestamp=datetime.now().isoformat(),
            source="reflexion_agent",
            data={
                "success": result.success,
                "execution_time": result.total_time,
                "iterations": result.iterations,
                "reflection_count": len(result.reflections),
                "algorithm": algorithm
            },
            metadata=result.metadata or {}
        )
        
        self.telemetry_exporter.export_event(event)
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        health_results = self.health_checker.run_health_checks()
        
        # Check alerts
        current_metrics = self.metrics_collector.get_latest_metrics()
        self.alert_manager.check_alerts(current_metrics)
        
        # Combine health check results with alerts
        health_summary = self.health_checker.get_health_summary()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "health_checks": health_results,
            "health_summary": health_summary,
            "active_alerts": active_alerts,
            "metrics": current_metrics
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "system_info": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "timestamp": datetime.now().isoformat()
            },
            "metrics": self.metrics_collector.get_latest_metrics(),
            "health": self.health_checker.get_health_summary(),
            "alerts": {
                "active": self.alert_manager.get_active_alerts(),
                "total_rules": len(self.alert_manager.alert_rules)
            },
            "telemetry": {
                "export_queue_size": len(self.telemetry_exporter.export_queue),
                "export_handlers": len(self.telemetry_exporter.export_handlers)
            }
        }


# Global telemetry manager instance
telemetry_manager = TelemetryManager()