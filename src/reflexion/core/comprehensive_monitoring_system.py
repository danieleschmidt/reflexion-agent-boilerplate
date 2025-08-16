"""
Comprehensive Monitoring System v4.0
Advanced monitoring, observability, and intelligent alerting
"""

import asyncio
import json
import time
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import logging
from collections import defaultdict, deque
import weakref

from .types import ReflectionType, ReflexionResult
from .autonomous_sdlc_engine import QualityMetrics


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringScope(Enum):
    """Scope of monitoring"""
    SYSTEM = "system"
    APPLICATION = "application"
    COMPONENT = "component"
    OPERATION = "operation"
    USER = "user"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    type: MetricType
    description: str
    unit: str
    labels: List[str] = field(default_factory=list)
    aggregation_window: int = 300  # seconds
    retention_period: int = 86400  # seconds (24 hours)


@dataclass
class MetricValue:
    """A metric value with metadata"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition and state"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    evaluation_window: int = 300  # seconds
    cooldown_period: int = 600  # seconds
    last_triggered: Optional[datetime] = None
    active: bool = False
    suppressed: bool = False
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MonitoringEvent:
    """Monitoring event record"""
    id: str
    timestamp: datetime
    event_type: str
    scope: MonitoringScope
    component: str
    severity: AlertSeverity
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class MetricCollector(ABC):
    """Abstract base class for metric collectors"""
    
    @abstractmethod
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect metrics from the source"""
        pass
    
    @abstractmethod
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get metric definitions supported by this collector"""
        pass


class SystemMetricsCollector(MetricCollector):
    """Collector for system-level metrics"""
    
    def __init__(self):
        self.metric_definitions = [
            MetricDefinition(
                name="cpu_usage_percent",
                type=MetricType.GAUGE,
                description="CPU usage percentage",
                unit="percent"
            ),
            MetricDefinition(
                name="memory_usage_bytes",
                type=MetricType.GAUGE,
                description="Memory usage in bytes",
                unit="bytes"
            ),
            MetricDefinition(
                name="disk_usage_percent",
                type=MetricType.GAUGE,
                description="Disk usage percentage",
                unit="percent"
            ),
            MetricDefinition(
                name="network_io_bytes",
                type=MetricType.COUNTER,
                description="Network I/O in bytes",
                unit="bytes",
                labels=["direction"]  # in/out
            )
        ]
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect system metrics"""
        try:
            import psutil
        except ImportError:
            # Return mock metrics if psutil not available
            timestamp = datetime.now()
            return [
                MetricValue(name="cpu_usage_percent", value=50.0, timestamp=timestamp),
                MetricValue(name="memory_usage_bytes", value=1000000, timestamp=timestamp),
                MetricValue(name="disk_usage_percent", value=60.0, timestamp=timestamp),
                MetricValue(name="network_io_bytes", value=5000, timestamp=timestamp, labels={"direction": "out"}),
                MetricValue(name="network_io_bytes", value=3000, timestamp=timestamp, labels={"direction": "in"})
            ]
        
        metrics = []
        timestamp = datetime.now()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(MetricValue(
            name="cpu_usage_percent",
            value=cpu_percent,
            timestamp=timestamp
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics.append(MetricValue(
            name="memory_usage_bytes",
            value=memory.used,
            timestamp=timestamp
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(MetricValue(
            name="disk_usage_percent",
            value=disk_percent,
            timestamp=timestamp
        ))
        
        # Network I/O
        net_io = psutil.net_io_counters()
        metrics.extend([
            MetricValue(
                name="network_io_bytes",
                value=net_io.bytes_sent,
                timestamp=timestamp,
                labels={"direction": "out"}
            ),
            MetricValue(
                name="network_io_bytes",
                value=net_io.bytes_recv,
                timestamp=timestamp,
                labels={"direction": "in"}
            )
        ])
        
        return metrics
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get system metric definitions"""
        return self.metric_definitions


class ApplicationMetricsCollector(MetricCollector):
    """Collector for application-level metrics"""
    
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        self.metric_definitions = [
            MetricDefinition(
                name="requests_total",
                type=MetricType.COUNTER,
                description="Total number of requests",
                unit="count",
                labels=["method", "endpoint", "status"]
            ),
            MetricDefinition(
                name="response_time_seconds",
                type=MetricType.HISTOGRAM,
                description="Response time in seconds",
                unit="seconds",
                labels=["method", "endpoint"]
            ),
            MetricDefinition(
                name="errors_total",
                type=MetricType.COUNTER,
                description="Total number of errors",
                unit="count",
                labels=["type", "component"]
            ),
            MetricDefinition(
                name="active_connections",
                type=MetricType.GAUGE,
                description="Number of active connections",
                unit="count"
            )
        ]
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect application metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # Simulate application metrics
        import random
        
        # Request metrics
        for endpoint in ["/api/health", "/api/users", "/api/data"]:
            count = random.randint(10, 100)
            metrics.append(MetricValue(
                name="requests_total",
                value=count,
                timestamp=timestamp,
                labels={"method": "GET", "endpoint": endpoint, "status": "200"}
            ))
            
            # Response times
            response_time = random.uniform(0.1, 2.0)
            metrics.append(MetricValue(
                name="response_time_seconds",
                value=response_time,
                timestamp=timestamp,
                labels={"method": "GET", "endpoint": endpoint}
            ))
        
        # Error metrics
        error_count = random.randint(0, 5)
        metrics.append(MetricValue(
            name="errors_total",
            value=error_count,
            timestamp=timestamp,
            labels={"type": "timeout", "component": "database"}
        ))
        
        # Connection metrics
        active_connections = random.randint(50, 200)
        metrics.append(MetricValue(
            name="active_connections",
            value=active_connections,
            timestamp=timestamp
        ))
        
        return metrics
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get application metric definitions"""
        return self.metric_definitions


class ReflexionMetricsCollector(MetricCollector):
    """Collector for reflexion-specific metrics"""
    
    def __init__(self):
        self.metric_definitions = [
            MetricDefinition(
                name="reflexion_iterations_total",
                type=MetricType.COUNTER,
                description="Total number of reflexion iterations",
                unit="count",
                labels=["agent_type", "success"]
            ),
            MetricDefinition(
                name="reflexion_success_rate",
                type=MetricType.GAUGE,
                description="Reflexion success rate",
                unit="percent"
            ),
            MetricDefinition(
                name="reflexion_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Duration of reflexion process",
                unit="seconds",
                labels=["agent_type"]
            ),
            MetricDefinition(
                name="memory_operations_total",
                type=MetricType.COUNTER,
                description="Total memory operations",
                unit="count",
                labels=["operation_type"]
            ),
            MetricDefinition(
                name="quality_score",
                type=MetricType.GAUGE,
                description="Overall quality score",
                unit="score",
                labels=["component"]
            )
        ]
        
        self.reflexion_stats = {
            'total_iterations': 0,
            'successful_iterations': 0,
            'total_duration': 0.0,
            'memory_operations': defaultdict(int)
        }
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect reflexion metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # Simulate reflexion metrics
        import random
        
        # Reflexion iterations
        iterations = random.randint(1, 10)
        success = random.choice([True, False])
        
        metrics.append(MetricValue(
            name="reflexion_iterations_total",
            value=iterations,
            timestamp=timestamp,
            labels={"agent_type": "quantum", "success": str(success).lower()}
        ))
        
        # Success rate
        success_rate = random.uniform(0.7, 0.95) * 100
        metrics.append(MetricValue(
            name="reflexion_success_rate",
            value=success_rate,
            timestamp=timestamp
        ))
        
        # Duration
        duration = random.uniform(1.0, 30.0)
        metrics.append(MetricValue(
            name="reflexion_duration_seconds",
            value=duration,
            timestamp=timestamp,
            labels={"agent_type": "quantum"}
        ))
        
        # Memory operations
        for op_type in ["store", "recall", "pattern_extract"]:
            count = random.randint(5, 50)
            metrics.append(MetricValue(
                name="memory_operations_total",
                value=count,
                timestamp=timestamp,
                labels={"operation_type": op_type}
            ))
        
        # Quality scores
        for component in ["test_coverage", "security", "performance"]:
            score = random.uniform(0.7, 0.95)
            metrics.append(MetricValue(
                name="quality_score",
                value=score,
                timestamp=timestamp,
                labels={"component": component}
            ))
        
        return metrics
    
    def get_metric_definitions(self) -> List[MetricDefinition]:
        """Get reflexion metric definitions"""
        return self.metric_definitions


class ComprehensiveMonitoringSystem:
    """
    Comprehensive Monitoring System with advanced observability,
    intelligent alerting, and real-time analytics.
    """
    
    def __init__(
        self,
        collection_interval: int = 30,
        retention_period: int = 86400,
        enable_alerting: bool = True
    ):
        self.collection_interval = collection_interval
        self.retention_period = retention_period
        self.enable_alerting = enable_alerting
        
        # Storage
        self.metrics_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.events_store: List[MonitoringEvent] = []
        self.alerts: Dict[str, Alert] = {}
        
        # Collectors
        self.collectors: List[MetricCollector] = []
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # State
        self.monitoring_active = False
        self.collection_task: Optional[asyncio.Task] = None
        self.alert_evaluation_task: Optional[asyncio.Task] = None
        
        # Analytics
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default collectors and alerts
        self._initialize_default_setup()
    
    def _initialize_default_setup(self):
        """Initialize default collectors and alerts"""
        # Add default collectors
        self.add_collector(SystemMetricsCollector())
        self.add_collector(ApplicationMetricsCollector())
        self.add_collector(ReflexionMetricsCollector())
        
        # Add default alerts
        self._setup_default_alerts()
    
    def add_collector(self, collector: MetricCollector):
        """Add a metric collector"""
        self.collectors.append(collector)
        
        # Register metric definitions
        for metric_def in collector.get_metric_definitions():
            self.metric_definitions[metric_def.name] = metric_def
            
        self.logger.info(f"Added collector: {collector.__class__.__name__}")
    
    def _setup_default_alerts(self):
        """Setup default alerts"""
        default_alerts = [
            Alert(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is above 80%",
                severity=AlertSeverity.WARNING,
                condition="cpu_usage_percent > 80",
                threshold=80.0,
                evaluation_window=300
            ),
            Alert(
                id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage is above 85%",
                severity=AlertSeverity.WARNING,
                condition="memory_usage_percent > 85",
                threshold=85.0,
                evaluation_window=300
            ),
            Alert(
                id="high_error_rate",
                name="High Error Rate",
                description="Error rate is above 5%",
                severity=AlertSeverity.ERROR,
                condition="error_rate > 0.05",
                threshold=0.05,
                evaluation_window=600
            ),
            Alert(
                id="low_reflexion_success_rate",
                name="Low Reflexion Success Rate",
                description="Reflexion success rate is below 70%",
                severity=AlertSeverity.CRITICAL,
                condition="reflexion_success_rate < 70",
                threshold=70.0,
                evaluation_window=900
            ),
            Alert(
                id="high_response_time",
                name="High Response Time",
                description="Average response time is above 2 seconds",
                severity=AlertSeverity.WARNING,
                condition="avg_response_time > 2.0",
                threshold=2.0,
                evaluation_window=300
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.id] = alert
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start metric collection
        self.collection_task = asyncio.create_task(self._metric_collection_loop())
        
        # Start alert evaluation
        if self.enable_alerting:
            self.alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
        
        self.logger.info("📊 Comprehensive Monitoring System started")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        
        # Cancel tasks
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.alert_evaluation_task:
            self.alert_evaluation_task.cancel()
            try:
                await self.alert_evaluation_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🛑 Monitoring system stopped")
    
    async def _metric_collection_loop(self):
        """Main metric collection loop"""
        while self.monitoring_active:
            try:
                await self._collect_all_metrics()
                await self._cleanup_old_metrics()
                await self._update_aggregated_metrics()
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_all_metrics(self):
        """Collect metrics from all collectors"""
        collection_tasks = []
        
        for collector in self.collectors:
            task = asyncio.create_task(collector.collect_metrics())
            collection_tasks.append(task)
        
        try:
            results = await asyncio.gather(*collection_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Collector {i} failed: {result}")
                    continue
                
                # Store metrics
                for metric in result:
                    self._store_metric(metric)
                    
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
    
    def _store_metric(self, metric: MetricValue):
        """Store a metric value"""
        metric_key = self._get_metric_key(metric)
        self.metrics_store[metric_key].append(metric)
    
    def _get_metric_key(self, metric: MetricValue) -> str:
        """Generate a unique key for the metric"""
        labels_str = ",".join(f"{k}={v}" for k, v in sorted(metric.labels.items()))
        return f"{metric.name}:{labels_str}" if labels_str else metric.name
    
    async def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = datetime.now() - timedelta(seconds=self.retention_period)
        
        for metric_key, metric_queue in self.metrics_store.items():
            # Remove old metrics
            while metric_queue and metric_queue[0].timestamp < cutoff_time:
                metric_queue.popleft()
    
    async def _update_aggregated_metrics(self):
        """Update aggregated metrics for analytics"""
        for metric_key, metric_queue in self.metrics_store.items():
            if not metric_queue:
                continue
            
            # Calculate aggregations
            values = [m.value for m in metric_queue]
            
            if values:
                self.aggregated_metrics[metric_key] = {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'p50': statistics.median(values),
                    'p95': self._percentile(values, 0.95),
                    'p99': self._percentile(values, 0.99)
                }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    async def _alert_evaluation_loop(self):
        """Alert evaluation loop"""
        while self.monitoring_active:
            try:
                await self._evaluate_all_alerts()
                await asyncio.sleep(60)  # Evaluate alerts every minute
                
            except Exception as e:
                self.logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_all_alerts(self):
        """Evaluate all configured alerts"""
        for alert_id, alert in self.alerts.items():
            if alert.suppressed:
                continue
            
            try:
                should_trigger = await self._evaluate_alert_condition(alert)
                
                if should_trigger and not alert.active:
                    await self._trigger_alert(alert)
                elif not should_trigger and alert.active:
                    await self._resolve_alert(alert)
                    
            except Exception as e:
                self.logger.error(f"Failed to evaluate alert {alert_id}: {e}")
    
    async def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate alert condition"""
        # Simplified condition evaluation
        # In a real implementation, this would parse and evaluate complex conditions
        
        if "cpu_usage_percent" in alert.condition:
            cpu_metrics = self._get_recent_metrics("cpu_usage_percent", alert.evaluation_window)
            if cpu_metrics:
                avg_cpu = statistics.mean([m.value for m in cpu_metrics])
                return avg_cpu > alert.threshold
        
        elif "memory_usage_percent" in alert.condition:
            # Calculate memory usage percentage
            memory_metrics = self._get_recent_metrics("memory_usage_bytes", alert.evaluation_window)
            if memory_metrics:
                # Simplified calculation - in reality would need total memory
                avg_memory_bytes = statistics.mean([m.value for m in memory_metrics])
                # Assume 8GB total memory for calculation
                memory_percent = (avg_memory_bytes / (8 * 1024 * 1024 * 1024)) * 100
                return memory_percent > alert.threshold
        
        elif "error_rate" in alert.condition:
            # Calculate error rate from request and error metrics
            # Simplified implementation
            return False  # Would need actual error rate calculation
        
        elif "reflexion_success_rate" in alert.condition:
            success_metrics = self._get_recent_metrics("reflexion_success_rate", alert.evaluation_window)
            if success_metrics:
                avg_success_rate = statistics.mean([m.value for m in success_metrics])
                return avg_success_rate < alert.threshold
        
        elif "avg_response_time" in alert.condition:
            response_metrics = self._get_recent_metrics("response_time_seconds", alert.evaluation_window)
            if response_metrics:
                avg_response_time = statistics.mean([m.value for m in response_metrics])
                return avg_response_time > alert.threshold
        
        return False
    
    def _get_recent_metrics(self, metric_name: str, window_seconds: int) -> List[MetricValue]:
        """Get recent metrics within time window"""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_metrics = []
        
        for metric_key, metric_queue in self.metrics_store.items():
            if metric_key.startswith(metric_name):
                recent_metrics.extend([
                    m for m in metric_queue 
                    if m.timestamp >= cutoff_time
                ])
        
        return recent_metrics
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        alert.active = True
        alert.last_triggered = datetime.now()
        
        # Create monitoring event
        event = MonitoringEvent(
            id=f"alert_{alert.id}_{int(time.time())}",
            timestamp=datetime.now(),
            event_type="alert_triggered",
            scope=MonitoringScope.SYSTEM,
            component="monitoring_system",
            severity=alert.severity,
            message=f"Alert triggered: {alert.name} - {alert.description}",
            context={"alert_id": alert.id, "condition": alert.condition}
        )
        
        self.events_store.append(event)
        
        self.logger.warning(f"🚨 ALERT: {alert.name} - {alert.description}")
        
        # Send alert notification (in real implementation)
        await self._send_alert_notification(alert, event)
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        alert.active = False
        
        # Create monitoring event
        event = MonitoringEvent(
            id=f"alert_resolved_{alert.id}_{int(time.time())}",
            timestamp=datetime.now(),
            event_type="alert_resolved",
            scope=MonitoringScope.SYSTEM,
            component="monitoring_system",
            severity=AlertSeverity.INFO,
            message=f"Alert resolved: {alert.name}",
            context={"alert_id": alert.id}
        )
        
        self.events_store.append(event)
        
        self.logger.info(f"✅ RESOLVED: {alert.name}")
    
    async def _send_alert_notification(self, alert: Alert, event: MonitoringEvent):
        """Send alert notification"""
        # In real implementation, this would send notifications via:
        # - Email
        # - Slack
        # - PagerDuty
        # - SMS
        # - Webhooks
        pass
    
    def record_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a custom metric"""
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        self._store_metric(metric)
    
    def record_event(
        self,
        event_type: str,
        component: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        scope: MonitoringScope = MonitoringScope.APPLICATION,
        context: Dict[str, Any] = None
    ):
        """Record a monitoring event"""
        event = MonitoringEvent(
            id=f"event_{int(time.time())}_{hash(message)}",
            timestamp=datetime.now(),
            event_type=event_type,
            scope=scope,
            component=component,
            severity=severity,
            message=message,
            context=context or {}
        )
        
        self.events_store.append(event)
        self.logger.info(f"📝 Event recorded: {event_type} - {message}")
    
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive metrics dashboard"""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "system_overview": {
                "total_metrics": sum(len(queue) for queue in self.metrics_store.values()),
                "active_alerts": len([a for a in self.alerts.values() if a.active]),
                "collectors_count": len(self.collectors),
                "monitoring_active": self.monitoring_active
            },
            "recent_metrics": {},
            "aggregated_metrics": dict(self.aggregated_metrics),
            "active_alerts": [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None
                }
                for alert in self.alerts.values() if alert.active
            ],
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type,
                    "component": event.component,
                    "severity": event.severity.value,
                    "message": event.message
                }
                for event in self.events_store[-10:]  # Last 10 events
            ]
        }
        
        # Add recent metrics for key indicators
        key_metrics = ["cpu_usage_percent", "memory_usage_bytes", "reflexion_success_rate"]
        for metric_name in key_metrics:
            recent_metrics = self._get_recent_metrics(metric_name, 300)  # Last 5 minutes
            if recent_metrics:
                dashboard_data["recent_metrics"][metric_name] = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "value": m.value,
                        "labels": m.labels
                    }
                    for m in recent_metrics[-20:]  # Last 20 data points
                ]
        
        return dashboard_data
    
    def create_custom_alert(
        self,
        alert_id: str,
        name: str,
        description: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING
    ):
        """Create a custom alert"""
        alert = Alert(
            id=alert_id,
            name=name,
            description=description,
            severity=severity,
            condition=condition,
            threshold=threshold
        )
        
        self.alerts[alert_id] = alert
        self.logger.info(f"Created custom alert: {name}")
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        # Calculate overall health score
        health_factors = []
        
        # CPU health
        cpu_metrics = self._get_recent_metrics("cpu_usage_percent", 300)
        if cpu_metrics:
            avg_cpu = statistics.mean([m.value for m in cpu_metrics])
            cpu_health = max(0, 1 - (avg_cpu / 100))
            health_factors.append(cpu_health)
        
        # Memory health
        memory_metrics = self._get_recent_metrics("memory_usage_bytes", 300)
        if memory_metrics:
            # Simplified memory health calculation
            memory_health = 0.8  # Placeholder
            health_factors.append(memory_health)
        
        # Reflexion health
        reflexion_metrics = self._get_recent_metrics("reflexion_success_rate", 300)
        if reflexion_metrics:
            avg_success = statistics.mean([m.value for m in reflexion_metrics])
            reflexion_health = avg_success / 100
            health_factors.append(reflexion_health)
        
        # Calculate overall health score
        overall_health = statistics.mean(health_factors) if health_factors else 0.5
        
        return {
            "monitoring_system_health": {
                "overall_health_score": overall_health,
                "health_status": self._get_health_status(overall_health),
                "active_alerts_count": len([a for a in self.alerts.values() if a.active]),
                "critical_alerts_count": len([
                    a for a in self.alerts.values() 
                    if a.active and a.severity == AlertSeverity.CRITICAL
                ]),
                "metrics_collected_last_hour": len([
                    m for queue in self.metrics_store.values()
                    for m in queue
                    if m.timestamp > datetime.now() - timedelta(hours=1)
                ]),
                "system_uptime": self.monitoring_active,
                "component_health": {
                    "cpu": self._calculate_component_health("cpu"),
                    "memory": self._calculate_component_health("memory"),
                    "reflexion": self._calculate_component_health("reflexion"),
                    "application": self._calculate_component_health("application")
                }
            }
        }
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on score"""
        if score >= 0.9:
            return "EXCELLENT"
        elif score >= 0.8:
            return "GOOD"
        elif score >= 0.7:
            return "FAIR"
        elif score >= 0.5:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _calculate_component_health(self, component: str) -> Dict[str, Any]:
        """Calculate health for a specific component"""
        # Simplified component health calculation
        return {
            "status": "HEALTHY",
            "score": 0.85,
            "last_updated": datetime.now().isoformat()
        }


# Global monitoring functions
async def create_monitoring_system(
    collection_interval: int = 30,
    enable_alerting: bool = True
) -> ComprehensiveMonitoringSystem:
    """Create and start monitoring system"""
    system = ComprehensiveMonitoringSystem(
        collection_interval=collection_interval,
        enable_alerting=enable_alerting
    )
    await system.start_monitoring()
    return system


def monitor_operation(monitoring_system: ComprehensiveMonitoringSystem):
    """Decorator to monitor operation performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            operation_name = func.__name__
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                monitoring_system.record_custom_metric(
                    f"operation_duration_seconds",
                    duration,
                    {"operation": operation_name, "status": "success"}
                )
                
                monitoring_system.record_event(
                    "operation_completed",
                    operation_name,
                    f"Operation {operation_name} completed successfully in {duration:.2f}s"
                )
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                monitoring_system.record_custom_metric(
                    f"operation_duration_seconds",
                    duration,
                    {"operation": operation_name, "status": "error"}
                )
                
                monitoring_system.record_event(
                    "operation_failed",
                    operation_name,
                    f"Operation {operation_name} failed: {str(e)}",
                    severity=AlertSeverity.ERROR
                )
                
                raise
        
        return wrapper
    return decorator