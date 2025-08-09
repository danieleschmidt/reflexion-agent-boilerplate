"""Auto-scaling capabilities for reflexion agent deployments."""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging
import statistics


class ScalingDirection(Enum):
    """Scaling direction indicators."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class MetricType(Enum):
    """Types of metrics for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class ScalingMetrics:
    """Current system metrics for scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    avg_response_time: float
    error_rate: float
    throughput: float
    active_connections: int
    timestamp: str


@dataclass
class ScalingThreshold:
    """Defines thresholds for scaling decisions."""
    metric_type: MetricType
    scale_up_threshold: float
    scale_down_threshold: float
    evaluation_periods: int
    cooldown_seconds: int


@dataclass
class ScalingPolicy:
    """Defines auto-scaling policy."""
    name: str
    description: str
    thresholds: List[ScalingThreshold]
    min_instances: int
    max_instances: int
    scale_up_increment: int
    scale_down_increment: int
    warmup_time: int  # seconds
    evaluation_interval: int  # seconds
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Records a scaling event."""
    event_id: str
    timestamp: str
    direction: ScalingDirection
    trigger_metric: str
    trigger_value: float
    threshold: float
    instances_before: int
    instances_after: int
    reason: str
    success: bool


class MetricsCollector:
    """Collects system metrics for scaling decisions."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.logger = logging.getLogger(__name__)
        self.custom_collectors: Dict[str, Callable[[], float]] = {}
    
    def register_custom_collector(
        self, 
        metric_name: str, 
        collector_func: Callable[[], float]
    ):
        """Register custom metric collector."""
        self.custom_collectors[metric_name] = collector_func
        self.logger.info(f"Registered custom metric collector: {metric_name}")
    
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        
        # In production, these would use actual system monitoring
        try:
            # Simulate metric collection
            cpu_util = await self._get_cpu_utilization()
            memory_util = await self._get_memory_utilization()
            queue_length = await self._get_queue_length()
            response_time = await self._get_avg_response_time()
            error_rate = await self._get_error_rate()
            throughput = await self._get_throughput()
            active_connections = await self._get_active_connections()
            
            metrics = ScalingMetrics(
                cpu_utilization=cpu_util,
                memory_utilization=memory_util,
                queue_length=queue_length,
                avg_response_time=response_time,
                error_rate=error_rate,
                throughput=throughput,
                active_connections=active_connections,
                timestamp=datetime.now().isoformat()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            # Return safe defaults
            return ScalingMetrics(
                cpu_utilization=0.0,
                memory_utilization=0.0,
                queue_length=0,
                avg_response_time=0.0,
                error_rate=0.0,
                throughput=0.0,
                active_connections=0,
                timestamp=datetime.now().isoformat()
            )
    
    async def _get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            # Simulate CPU utilization
            import random
            return random.uniform(20, 80)
    
    async def _get_memory_utilization(self) -> float:
        """Get memory utilization percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Simulate memory utilization
            import random
            return random.uniform(30, 70)
    
    async def _get_queue_length(self) -> int:
        """Get current queue length."""
        # This would integrate with actual queue monitoring
        import random
        return random.randint(0, 50)
    
    async def _get_avg_response_time(self) -> float:
        """Get average response time in seconds."""
        # This would integrate with actual response time monitoring
        import random
        return random.uniform(0.1, 2.0)
    
    async def _get_error_rate(self) -> float:
        """Get error rate percentage."""
        # This would integrate with actual error monitoring
        import random
        return random.uniform(0.0, 5.0)
    
    async def _get_throughput(self) -> float:
        """Get requests per second."""
        # This would integrate with actual throughput monitoring
        import random
        return random.uniform(10, 100)
    
    async def _get_active_connections(self) -> int:
        """Get number of active connections."""
        # This would integrate with actual connection monitoring
        import random
        return random.randint(5, 200)


class ScalingDecisionEngine:
    """Makes scaling decisions based on metrics and policies."""
    
    def __init__(self):
        """Initialize scaling decision engine."""
        self.logger = logging.getLogger(__name__)
        self.metric_history: List[ScalingMetrics] = []
        self.scaling_events: List[ScalingEvent] = []
        self.last_scaling_time: Dict[str, datetime] = {}
    
    def should_scale(
        self,
        current_metrics: ScalingMetrics,
        policy: ScalingPolicy,
        current_instances: int
    ) -> Dict[str, Any]:
        """Determine if scaling is needed."""
        
        # Store metrics history
        self.metric_history.append(current_metrics)
        
        # Keep only recent history
        max_history = 100
        if len(self.metric_history) > max_history:
            self.metric_history = self.metric_history[-max_history:]
        
        scaling_decision = {
            "should_scale": False,
            "direction": ScalingDirection.STABLE,
            "instances_change": 0,
            "trigger_metric": None,
            "trigger_value": None,
            "threshold": None,
            "reason": "No scaling needed",
            "confidence": 0.0
        }
        
        if not policy.enabled:
            scaling_decision["reason"] = "Scaling policy disabled"
            return scaling_decision
        
        # Check each threshold
        for threshold in policy.thresholds:
            decision = self._evaluate_threshold(
                current_metrics, threshold, policy, current_instances
            )
            
            # If any threshold triggers scaling, use it
            if decision["should_scale"]:
                scaling_decision.update(decision)
                break
        
        return scaling_decision
    
    def _evaluate_threshold(
        self,
        metrics: ScalingMetrics,
        threshold: ScalingThreshold,
        policy: ScalingPolicy,
        current_instances: int
    ) -> Dict[str, Any]:
        """Evaluate a specific scaling threshold."""
        
        # Get metric value
        metric_value = self._get_metric_value(metrics, threshold.metric_type)
        
        # Check cooldown period
        cooldown_key = f"{policy.name}_{threshold.metric_type.value}"
        if cooldown_key in self.last_scaling_time:
            time_since_last = datetime.now() - self.last_scaling_time[cooldown_key]
            if time_since_last.total_seconds() < threshold.cooldown_seconds:
                return {
                    "should_scale": False,
                    "reason": f"Cooldown period active ({threshold.cooldown_seconds}s)"
                }
        
        # Get historical data for evaluation periods
        recent_metrics = self.metric_history[-threshold.evaluation_periods:]
        
        if len(recent_metrics) < threshold.evaluation_periods:
            return {
                "should_scale": False,
                "reason": f"Insufficient metrics history ({len(recent_metrics)}/{threshold.evaluation_periods})"
            }
        
        # Calculate average over evaluation periods
        recent_values = [
            self._get_metric_value(m, threshold.metric_type) 
            for m in recent_metrics
        ]
        avg_value = statistics.mean(recent_values)
        
        # Determine scaling direction
        if avg_value > threshold.scale_up_threshold and current_instances < policy.max_instances:
            return {
                "should_scale": True,
                "direction": ScalingDirection.UP,
                "instances_change": policy.scale_up_increment,
                "trigger_metric": threshold.metric_type.value,
                "trigger_value": avg_value,
                "threshold": threshold.scale_up_threshold,
                "reason": f"{threshold.metric_type.value} above scale-up threshold",
                "confidence": min(1.0, (avg_value - threshold.scale_up_threshold) / threshold.scale_up_threshold)
            }
        
        elif avg_value < threshold.scale_down_threshold and current_instances > policy.min_instances:
            return {
                "should_scale": True,
                "direction": ScalingDirection.DOWN,
                "instances_change": -policy.scale_down_increment,
                "trigger_metric": threshold.metric_type.value,
                "trigger_value": avg_value,
                "threshold": threshold.scale_down_threshold,
                "reason": f"{threshold.metric_type.value} below scale-down threshold",
                "confidence": min(1.0, (threshold.scale_down_threshold - avg_value) / threshold.scale_down_threshold)
            }
        
        return {"should_scale": False}
    
    def _get_metric_value(self, metrics: ScalingMetrics, metric_type: MetricType) -> float:
        """Get specific metric value from metrics object."""
        metric_map = {
            MetricType.CPU_UTILIZATION: metrics.cpu_utilization,
            MetricType.MEMORY_UTILIZATION: metrics.memory_utilization,
            MetricType.QUEUE_LENGTH: float(metrics.queue_length),
            MetricType.RESPONSE_TIME: metrics.avg_response_time,
            MetricType.ERROR_RATE: metrics.error_rate,
            MetricType.THROUGHPUT: metrics.throughput
        }
        
        return metric_map.get(metric_type, 0.0)
    
    def record_scaling_event(
        self,
        direction: ScalingDirection,
        trigger_metric: str,
        trigger_value: float,
        threshold: float,
        instances_before: int,
        instances_after: int,
        reason: str,
        success: bool
    ) -> str:
        """Record a scaling event."""
        
        event_id = f"scale_{int(time.time() * 1000)}"
        
        event = ScalingEvent(
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            direction=direction,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            threshold=threshold,
            instances_before=instances_before,
            instances_after=instances_after,
            reason=reason,
            success=success
        )
        
        self.scaling_events.append(event)
        
        # Update last scaling time
        self.last_scaling_time[trigger_metric] = datetime.now()
        
        self.logger.info(
            f"Scaling event recorded: {direction.value} from {instances_before} to {instances_after} "
            f"due to {trigger_metric}={trigger_value:.2f}"
        )
        
        return event_id


class AutoScaler:
    """Automatic scaling system for reflexion agents."""
    
    def __init__(
        self,
        scaling_policies: List[ScalingPolicy],
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize auto-scaler.
        
        Args:
            scaling_policies: List of scaling policies to evaluate
            metrics_collector: Metrics collector instance
        """
        self.policies = {policy.name: policy for policy in scaling_policies}
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.decision_engine = ScalingDecisionEngine()
        self.logger = logging.getLogger(__name__)
        
        # State
        self.current_instances = 1
        self.running = False
        self.scaling_callbacks: List[Callable] = []
    
    def register_scaling_callback(self, callback: Callable[[int, int], None]):
        """Register callback for scaling events.
        
        Args:
            callback: Function called with (old_instances, new_instances)
        """
        self.scaling_callbacks.append(callback)
        self.logger.info("Registered scaling callback")
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring and auto-scaling."""
        self.running = True
        self.logger.info(f"Started auto-scaling with {interval_seconds}s interval")
        
        try:
            while self.running:
                await self._scaling_cycle()
                await asyncio.sleep(interval_seconds)
        except Exception as e:
            self.logger.error(f"Auto-scaling monitoring failed: {e}")
        finally:
            self.running = False
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.running = False
        self.logger.info("Stopped auto-scaling monitoring")
    
    async def _scaling_cycle(self):
        """Execute one scaling evaluation cycle."""
        try:
            # Collect current metrics
            metrics = await self.metrics_collector.collect_metrics()
            
            # Evaluate each policy
            for policy_name, policy in self.policies.items():
                if not policy.enabled:
                    continue
                
                decision = self.decision_engine.should_scale(
                    metrics, policy, self.current_instances
                )
                
                if decision["should_scale"]:
                    await self._execute_scaling(decision, policy, metrics)
                    break  # Only execute one scaling action per cycle
            
        except Exception as e:
            self.logger.error(f"Scaling cycle failed: {e}")
    
    async def _execute_scaling(
        self,
        decision: Dict[str, Any],
        policy: ScalingPolicy,
        metrics: ScalingMetrics
    ):
        """Execute scaling action."""
        
        old_instances = self.current_instances
        new_instances = max(
            policy.min_instances,
            min(policy.max_instances, old_instances + decision["instances_change"])
        )
        
        if new_instances == old_instances:
            self.logger.info("Scaling decision resulted in no change")
            return
        
        self.logger.info(
            f"Executing scaling: {old_instances} -> {new_instances} instances "
            f"due to {decision['trigger_metric']}={decision['trigger_value']:.2f}"
        )
        
        try:
            # Execute scaling callbacks
            for callback in self.scaling_callbacks:
                await self._call_scaling_callback(callback, old_instances, new_instances)
            
            # Update current instance count
            self.current_instances = new_instances
            
            # Record successful scaling event
            self.decision_engine.record_scaling_event(
                direction=decision["direction"],
                trigger_metric=decision["trigger_metric"],
                trigger_value=decision["trigger_value"],
                threshold=decision["threshold"],
                instances_before=old_instances,
                instances_after=new_instances,
                reason=decision["reason"],
                success=True
            )
            
            # Wait for warmup time if scaling up
            if new_instances > old_instances and policy.warmup_time > 0:
                self.logger.info(f"Waiting {policy.warmup_time}s for warmup")
                await asyncio.sleep(policy.warmup_time)
            
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
            
            # Record failed scaling event
            self.decision_engine.record_scaling_event(
                direction=decision["direction"],
                trigger_metric=decision["trigger_metric"],
                trigger_value=decision["trigger_value"],
                threshold=decision["threshold"],
                instances_before=old_instances,
                instances_after=old_instances,  # No change
                reason=f"Scaling failed: {str(e)}",
                success=False
            )
    
    async def _call_scaling_callback(
        self,
        callback: Callable,
        old_instances: int,
        new_instances: int
    ):
        """Safely call scaling callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(old_instances, new_instances)
            else:
                callback(old_instances, new_instances)
        except Exception as e:
            self.logger.error(f"Scaling callback failed: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        recent_events = self.decision_engine.scaling_events[-10:]  # Last 10 events
        recent_metrics = self.decision_engine.metric_history[-1:] if self.decision_engine.metric_history else []
        
        return {
            "running": self.running,
            "current_instances": self.current_instances,
            "active_policies": len([p for p in self.policies.values() if p.enabled]),
            "recent_events": [asdict(event) for event in recent_events],
            "current_metrics": [asdict(metrics) for metrics in recent_metrics],
            "scaling_callbacks": len(self.scaling_callbacks)
        }
    
    def create_default_policy(
        self,
        name: str = "default_policy",
        min_instances: int = 1,
        max_instances: int = 10
    ) -> ScalingPolicy:
        """Create a default scaling policy with reasonable defaults."""
        
        thresholds = [
            ScalingThreshold(
                metric_type=MetricType.CPU_UTILIZATION,
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                evaluation_periods=3,
                cooldown_seconds=300
            ),
            ScalingThreshold(
                metric_type=MetricType.MEMORY_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=40.0,
                evaluation_periods=3,
                cooldown_seconds=300
            ),
            ScalingThreshold(
                metric_type=MetricType.QUEUE_LENGTH,
                scale_up_threshold=20.0,
                scale_down_threshold=5.0,
                evaluation_periods=2,
                cooldown_seconds=180
            )
        ]
        
        return ScalingPolicy(
            name=name,
            description="Default auto-scaling policy",
            thresholds=thresholds,
            min_instances=min_instances,
            max_instances=max_instances,
            scale_up_increment=2,
            scale_down_increment=1,
            warmup_time=60,
            evaluation_interval=30,
            enabled=True
        )