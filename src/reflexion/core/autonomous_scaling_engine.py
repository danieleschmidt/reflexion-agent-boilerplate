"""
Autonomous scaling engine with predictive load balancing and resource optimization.
"""

import asyncio
import time
import math
import json
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading
import logging
from statistics import mean, median
import heapq

from .intelligent_monitoring import intelligent_monitor, MetricType
from .logging_config import logger


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "scale_up"
    DOWN = "scale_down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    CONCURRENT_TASKS = "concurrent_tasks"
    CACHE_SIZE = "cache_size"
    THREAD_POOL = "thread_pool"
    CONNECTION_POOL = "connection_pool"


class LoadPattern(Enum):
    """Recognized load patterns for predictive scaling."""
    STEADY = "steady"
    BURSTY = "bursty"
    CYCLICAL = "cyclical"
    TRENDING = "trending"
    UNPREDICTABLE = "unpredictable"


@dataclass
class ScalingEvent:
    """Record of a scaling operation."""
    timestamp: datetime
    resource_type: ResourceType
    direction: ScalingDirection
    old_value: Union[int, float]
    new_value: Union[int, float]
    trigger_metric: str
    trigger_value: Union[int, float]
    success: bool = True
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type.value,
            "direction": self.direction.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "trigger_metric": self.trigger_metric,
            "trigger_value": self.trigger_value,
            "success": self.success,
            "reason": self.reason
        }


@dataclass
class ResourceConfiguration:
    """Configuration for a scalable resource."""
    resource_type: ResourceType
    current_value: Union[int, float]
    min_value: Union[int, float]
    max_value: Union[int, float]
    step_size: Union[int, float]
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period: int  # seconds
    last_scaling_time: Optional[datetime] = None
    target_utilization: float = 0.7
    
    def can_scale(self) -> bool:
        """Check if resource can be scaled (cooldown check)."""
        if self.last_scaling_time is None:
            return True
        
        time_since_last = (datetime.now() - self.last_scaling_time).total_seconds()
        return time_since_last >= self.cooldown_period
    
    def scale_up(self) -> bool:
        """Scale resource up."""
        if self.current_value >= self.max_value:
            return False
        
        new_value = min(self.max_value, self.current_value + self.step_size)
        self.current_value = new_value
        self.last_scaling_time = datetime.now()
        return True
    
    def scale_down(self) -> bool:
        """Scale resource down."""
        if self.current_value <= self.min_value:
            return False
        
        new_value = max(self.min_value, self.current_value - self.step_size)
        self.current_value = new_value
        self.last_scaling_time = datetime.now()
        return True


class PredictiveLoadAnalyzer:
    """Analyzes load patterns for predictive scaling."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.load_history = deque(maxlen=history_size)
        self.pattern_cache = {}
        self.prediction_accuracy = deque(maxlen=100)
        
    def record_load(self, load_value: float, timestamp: Optional[datetime] = None):
        """Record load measurement."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.load_history.append((timestamp, load_value))
    
    def detect_pattern(self) -> LoadPattern:
        """Detect current load pattern."""
        if len(self.load_history) < 50:
            return LoadPattern.UNPREDICTABLE
        
        recent_loads = [load for _, load in list(self.load_history)[-50:]]
        
        # Calculate pattern metrics
        mean_load = mean(recent_loads)
        variance = sum((x - mean_load) ** 2 for x in recent_loads) / len(recent_loads)
        coefficient_of_variation = (variance ** 0.5) / mean_load if mean_load > 0 else 0
        
        # Detect cyclical patterns
        if self._is_cyclical():
            return LoadPattern.CYCLICAL
        
        # Detect trending patterns
        if self._is_trending():
            return LoadPattern.TRENDING
        
        # Classify based on variation
        if coefficient_of_variation < 0.1:
            return LoadPattern.STEADY
        elif coefficient_of_variation > 0.5:
            return LoadPattern.BURSTY
        else:
            return LoadPattern.UNPREDICTABLE
    
    def predict_load(self, horizon_minutes: int = 5) -> List[Tuple[datetime, float]]:
        """Predict future load values."""
        if len(self.load_history) < 20:
            # Not enough data for prediction
            current_load = self.load_history[-1][1] if self.load_history else 0.5
            base_time = datetime.now()
            return [(base_time + timedelta(minutes=i), current_load) 
                   for i in range(1, horizon_minutes + 1)]
        
        pattern = self.detect_pattern()
        
        if pattern == LoadPattern.CYCLICAL:
            return self._predict_cyclical(horizon_minutes)
        elif pattern == LoadPattern.TRENDING:
            return self._predict_trending(horizon_minutes)
        elif pattern == LoadPattern.STEADY:
            return self._predict_steady(horizon_minutes)
        else:
            return self._predict_default(horizon_minutes)
    
    def _is_cyclical(self) -> bool:
        """Check if load pattern is cyclical."""
        if len(self.load_history) < 100:
            return False
        
        # Simple autocorrelation check
        loads = [load for _, load in list(self.load_history)[-100:]]
        
        # Check for hourly cycles (assuming 1-minute intervals)
        if len(loads) >= 60:
            correlation_60 = self._autocorrelation(loads, 60)
            if correlation_60 > 0.7:
                return True
        
        # Check for daily cycles (1440 minutes, but check with available data)
        cycle_length = min(len(loads) // 2, 720)  # Up to 12 hours
        if cycle_length >= 30:
            correlation_cycle = self._autocorrelation(loads, cycle_length)
            if correlation_cycle > 0.6:
                return True
        
        return False
    
    def _is_trending(self) -> bool:
        """Check if load pattern is trending."""
        if len(self.load_history) < 30:
            return False
        
        loads = [load for _, load in list(self.load_history)[-30:]]
        
        # Calculate trend using linear regression
        n = len(loads)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(loads)
        sum_xy = sum(x * y for x, y in zip(x_values, loads))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return False
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Significant trend if slope is substantial relative to mean
        mean_load = mean(loads)
        relative_slope = abs(slope) / max(mean_load, 0.001)
        
        return relative_slope > 0.01  # 1% change per time unit
    
    def _autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if len(data) <= lag:
            return 0.0
        
        n = len(data) - lag
        mean_val = mean(data)
        
        numerator = sum((data[i] - mean_val) * (data[i + lag] - mean_val) for i in range(n))
        denominator = sum((x - mean_val) ** 2 for x in data)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _predict_cyclical(self, horizon_minutes: int) -> List[Tuple[datetime, float]]:
        """Predict load for cyclical pattern."""
        # Use historical cycle to predict
        loads = [load for _, load in list(self.load_history)[-720:]]  # Last 12 hours
        
        if len(loads) < horizon_minutes:
            return self._predict_default(horizon_minutes)
        
        predictions = []
        base_time = datetime.now()
        
        for i in range(1, horizon_minutes + 1):
            # Use cycle position to predict
            cycle_position = i % len(loads)
            predicted_load = loads[cycle_position]
            
            # Add some noise for realism
            noise = 0.05 * predicted_load * (hash(str(i)) % 100 - 50) / 100
            predicted_load += noise
            
            predictions.append((base_time + timedelta(minutes=i), max(0, predicted_load)))
        
        return predictions
    
    def _predict_trending(self, horizon_minutes: int) -> List[Tuple[datetime, float]]:
        """Predict load for trending pattern."""
        loads = [load for _, load in list(self.load_history)[-50:]]
        
        # Calculate trend
        n = len(loads)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(loads)
        sum_xy = sum(x * y for x, y in zip(x_values, loads))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return self._predict_default(horizon_minutes)
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Project trend forward
        predictions = []
        base_time = datetime.now()
        
        for i in range(1, horizon_minutes + 1):
            predicted_load = intercept + slope * (n + i)
            predictions.append((base_time + timedelta(minutes=i), max(0, predicted_load)))
        
        return predictions
    
    def _predict_steady(self, horizon_minutes: int) -> List[Tuple[datetime, float]]:
        """Predict load for steady pattern."""
        recent_loads = [load for _, load in list(self.load_history)[-20:]]
        avg_load = mean(recent_loads)
        
        predictions = []
        base_time = datetime.now()
        
        for i in range(1, horizon_minutes + 1):
            # Steady state with minimal variation
            predicted_load = avg_load + 0.02 * avg_load * math.sin(i * 0.1)
            predictions.append((base_time + timedelta(minutes=i), max(0, predicted_load)))
        
        return predictions
    
    def _predict_default(self, horizon_minutes: int) -> List[Tuple[datetime, float]]:
        """Default prediction for unpredictable patterns."""
        if not self.load_history:
            current_load = 0.5
        else:
            current_load = self.load_history[-1][1]
        
        predictions = []
        base_time = datetime.now()
        
        for i in range(1, horizon_minutes + 1):
            # Gradual return to mean with some variability
            mean_load = 0.5  # Assume 50% as normal load
            predicted_load = current_load + 0.1 * (mean_load - current_load)
            
            # Add some randomness
            noise = 0.1 * predicted_load * (hash(str(i)) % 100 - 50) / 100
            predicted_load += noise
            
            predictions.append((base_time + timedelta(minutes=i), max(0, predicted_load)))
            current_load = predicted_load
        
        return predictions


class ResourceOptimizer:
    """Optimizes resource allocation based on performance metrics."""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=500)
        self.performance_baselines = {}
        self.optimization_rules = []
        
    def analyze_resource_efficiency(self, resources: Dict[ResourceType, ResourceConfiguration]) -> Dict[str, Any]:
        """Analyze current resource efficiency."""
        
        efficiency_scores = {}
        recommendations = []
        
        for resource_type, config in resources.items():
            # Calculate utilization efficiency
            current_utilization = self._get_current_utilization(resource_type)
            target_utilization = config.target_utilization
            
            efficiency = 1.0 - abs(current_utilization - target_utilization)
            efficiency_scores[resource_type.value] = efficiency
            
            # Generate recommendations
            if current_utilization > target_utilization + 0.1:
                recommendations.append(f"Scale up {resource_type.value} - utilization {current_utilization:.1%} > target {target_utilization:.1%}")
            elif current_utilization < target_utilization - 0.1:
                recommendations.append(f"Scale down {resource_type.value} - utilization {current_utilization:.1%} < target {target_utilization:.1%}")
        
        overall_efficiency = mean(efficiency_scores.values()) if efficiency_scores else 0.0
        
        return {
            "overall_efficiency": overall_efficiency,
            "resource_efficiencies": efficiency_scores,
            "recommendations": recommendations,
            "optimization_opportunities": self._identify_optimization_opportunities(resources)
        }
    
    def _get_current_utilization(self, resource_type: ResourceType) -> float:
        """Get current resource utilization."""
        # This would integrate with actual system metrics
        # For now, return simulated values
        
        utilization_map = {
            ResourceType.CPU: 0.65,
            ResourceType.MEMORY: 0.72,
            ResourceType.CONCURRENT_TASKS: 0.58,
            ResourceType.CACHE_SIZE: 0.45,
            ResourceType.THREAD_POOL: 0.68,
            ResourceType.CONNECTION_POOL: 0.55
        }
        
        base_utilization = utilization_map.get(resource_type, 0.5)
        
        # Add some variation based on current time
        variation = 0.1 * math.sin(time.time() / 600)  # 10-minute cycle
        return max(0.0, min(1.0, base_utilization + variation))
    
    def _identify_optimization_opportunities(self, resources: Dict[ResourceType, ResourceConfiguration]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        
        opportunities = []
        
        # Resource correlation analysis
        cpu_util = self._get_current_utilization(ResourceType.CPU)
        memory_util = self._get_current_utilization(ResourceType.MEMORY)
        
        if cpu_util > 0.8 and memory_util < 0.5:
            opportunities.append({
                "type": "resource_imbalance",
                "description": "High CPU utilization with low memory usage - consider CPU-intensive optimization",
                "priority": "high"
            })
        
        # Underutilized resources
        for resource_type, config in resources.items():
            utilization = self._get_current_utilization(resource_type)
            if utilization < 0.3:
                opportunities.append({
                    "type": "underutilization",
                    "resource": resource_type.value,
                    "description": f"{resource_type.value} is significantly underutilized ({utilization:.1%})",
                    "priority": "medium"
                })
        
        return opportunities


class AutonomousScalingEngine:
    """
    Advanced autonomous scaling engine with:
    - Predictive load analysis
    - Multi-dimensional resource optimization
    - Pattern-aware scaling decisions
    - Cost-performance optimization
    """
    
    def __init__(self):
        self.resources = self._initialize_resources()
        self.load_analyzer = PredictiveLoadAnalyzer()
        self.resource_optimizer = ResourceOptimizer()
        
        self.scaling_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(deque)
        
        self.scaling_enabled = True
        self.aggressive_scaling = False
        self.cost_optimization_mode = False
        
        # Background processing
        self.scaling_thread = None
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Start autonomous operations
        self._start_autonomous_scaling()
    
    def _initialize_resources(self) -> Dict[ResourceType, ResourceConfiguration]:
        """Initialize default resource configurations."""
        
        return {
            ResourceType.CPU: ResourceConfiguration(
                resource_type=ResourceType.CPU,
                current_value=4,  # CPU cores
                min_value=2,
                max_value=16,
                step_size=2,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                cooldown_period=300,  # 5 minutes
                target_utilization=0.7
            ),
            
            ResourceType.MEMORY: ResourceConfiguration(
                resource_type=ResourceType.MEMORY,
                current_value=8,  # GB
                min_value=4,
                max_value=32,
                step_size=4,
                scale_up_threshold=0.85,
                scale_down_threshold=0.4,
                cooldown_period=300,
                target_utilization=0.75
            ),
            
            ResourceType.CONCURRENT_TASKS: ResourceConfiguration(
                resource_type=ResourceType.CONCURRENT_TASKS,
                current_value=10,
                min_value=5,
                max_value=100,
                step_size=5,
                scale_up_threshold=0.9,
                scale_down_threshold=0.5,
                cooldown_period=60,  # 1 minute
                target_utilization=0.8
            ),
            
            ResourceType.CACHE_SIZE: ResourceConfiguration(
                resource_type=ResourceType.CACHE_SIZE,
                current_value=256,  # MB
                min_value=64,
                max_value=2048,
                step_size=128,
                scale_up_threshold=0.9,
                scale_down_threshold=0.3,
                cooldown_period=600,  # 10 minutes
                target_utilization=0.6
            ),
            
            ResourceType.THREAD_POOL: ResourceConfiguration(
                resource_type=ResourceType.THREAD_POOL,
                current_value=20,
                min_value=10,
                max_value=200,
                step_size=10,
                scale_up_threshold=0.85,
                scale_down_threshold=0.4,
                cooldown_period=120,  # 2 minutes
                target_utilization=0.7
            ),
            
            ResourceType.CONNECTION_POOL: ResourceConfiguration(
                resource_type=ResourceType.CONNECTION_POOL,
                current_value=50,
                min_value=10,
                max_value=500,
                step_size=25,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                cooldown_period=180,  # 3 minutes
                target_utilization=0.6
            )
        }
    
    def _start_autonomous_scaling(self):
        """Start autonomous scaling and monitoring threads."""
        
        def scaling_loop():
            while not self.shutdown_event.is_set():
                try:
                    if self.scaling_enabled:
                        self._autonomous_scaling_cycle()
                    
                    time.sleep(30)  # Run every 30 seconds
                except Exception as e:
                    self.logger.error(f"Autonomous scaling error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        def monitoring_loop():
            while not self.shutdown_event.is_set():
                try:
                    self._collect_performance_metrics()
                    time.sleep(10)  # Collect metrics every 10 seconds
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {e}")
                    time.sleep(30)
        
        self.scaling_thread = threading.Thread(target=scaling_loop, daemon=True)
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        
        self.scaling_thread.start()
        self.monitoring_thread.start()
        
        self.logger.info("Autonomous scaling engine started")
    
    def _autonomous_scaling_cycle(self):
        """Run one cycle of autonomous scaling decisions."""
        
        # Collect current metrics
        current_load = self._get_system_load()
        self.load_analyzer.record_load(current_load)
        
        # Detect load pattern
        pattern = self.load_analyzer.detect_pattern()
        
        # Get load predictions
        predictions = self.load_analyzer.predict_load(horizon_minutes=10)
        predicted_peak_load = max(load for _, load in predictions)
        
        # Make scaling decisions for each resource
        for resource_type, config in self.resources.items():
            if not config.can_scale():
                continue  # Still in cooldown
            
            current_utilization = self._get_resource_utilization(resource_type)
            
            # Determine scaling need
            scaling_decision = self._make_scaling_decision(
                resource_type, config, current_utilization, predicted_peak_load, pattern
            )
            
            if scaling_decision != ScalingDirection.STABLE:
                self._execute_scaling(resource_type, config, scaling_decision, current_utilization)
    
    def _get_system_load(self) -> float:
        """Get overall system load metric."""
        # Composite load metric combining multiple factors
        
        cpu_util = self._get_resource_utilization(ResourceType.CPU)
        memory_util = self._get_resource_utilization(ResourceType.MEMORY)
        task_util = self._get_resource_utilization(ResourceType.CONCURRENT_TASKS)
        
        # Weighted combination
        system_load = (cpu_util * 0.4 + memory_util * 0.3 + task_util * 0.3)
        
        # Record in monitoring system
        intelligent_monitor.record_metric("system_load", system_load, MetricType.GAUGE)
        
        return system_load
    
    def _get_resource_utilization(self, resource_type: ResourceType) -> float:
        """Get current utilization for a resource type."""
        # This would integrate with actual system monitoring
        # For demonstration, return simulated values with realistic patterns
        
        base_utilizations = {
            ResourceType.CPU: 0.65,
            ResourceType.MEMORY: 0.72,
            ResourceType.CONCURRENT_TASKS: 0.58,
            ResourceType.CACHE_SIZE: 0.45,
            ResourceType.THREAD_POOL: 0.68,
            ResourceType.CONNECTION_POOL: 0.55
        }
        
        base_util = base_utilizations.get(resource_type, 0.5)
        
        # Add time-based variation to simulate realistic load patterns
        time_factor = time.time() / 3600  # Hour-based cycle
        daily_variation = 0.2 * math.sin(time_factor * 2 * math.pi / 24)  # Daily cycle
        hourly_variation = 0.1 * math.sin(time_factor * 2 * math.pi)  # Hourly variation
        
        utilization = base_util + daily_variation + hourly_variation
        
        # Add some randomness
        import random
        noise = random.uniform(-0.05, 0.05)
        utilization += noise
        
        # Ensure bounds
        utilization = max(0.0, min(1.0, utilization))
        
        # Record metric
        intelligent_monitor.record_metric(
            f"{resource_type.value}_utilization", 
            utilization, 
            MetricType.GAUGE
        )
        
        return utilization
    
    def _make_scaling_decision(
        self,
        resource_type: ResourceType,
        config: ResourceConfiguration,
        current_utilization: float,
        predicted_peak_load: float,
        load_pattern: LoadPattern
    ) -> ScalingDirection:
        """Make intelligent scaling decision based on multiple factors."""
        
        # Base decision on current thresholds
        if current_utilization > config.scale_up_threshold:
            base_decision = ScalingDirection.UP
        elif current_utilization < config.scale_down_threshold:
            base_decision = ScalingDirection.DOWN
        else:
            base_decision = ScalingDirection.STABLE
        
        # Adjust based on predictions
        if predicted_peak_load > config.scale_up_threshold * 0.8:
            if base_decision == ScalingDirection.DOWN:
                base_decision = ScalingDirection.STABLE  # Don't scale down if spike predicted
            elif base_decision == ScalingDirection.STABLE and predicted_peak_load > config.scale_up_threshold:
                base_decision = ScalingDirection.UP  # Proactive scaling
        
        # Adjust based on load pattern
        if load_pattern == LoadPattern.BURSTY and base_decision == ScalingDirection.UP:
            # More aggressive scaling for bursty loads
            return ScalingDirection.UP
        elif load_pattern == LoadPattern.STEADY and base_decision == ScalingDirection.DOWN:
            # More conservative scaling for steady loads
            if current_utilization < config.scale_down_threshold * 0.7:
                return ScalingDirection.DOWN
            else:
                return ScalingDirection.STABLE
        
        # Cost optimization mode
        if self.cost_optimization_mode and base_decision == ScalingDirection.UP:
            if current_utilization < config.scale_up_threshold * 1.1:
                return ScalingDirection.STABLE  # Delay scaling up for cost savings
        
        return base_decision
    
    def _execute_scaling(
        self,
        resource_type: ResourceType,
        config: ResourceConfiguration,
        direction: ScalingDirection,
        current_utilization: float
    ):
        """Execute scaling operation."""
        
        old_value = config.current_value
        success = False
        
        try:
            if direction == ScalingDirection.UP:
                success = config.scale_up()
            elif direction == ScalingDirection.DOWN:
                success = config.scale_down()
            
            if success:
                new_value = config.current_value
                
                # Record scaling event
                scaling_event = ScalingEvent(
                    timestamp=datetime.now(),
                    resource_type=resource_type,
                    direction=direction,
                    old_value=old_value,
                    new_value=new_value,
                    trigger_metric=f"{resource_type.value}_utilization",
                    trigger_value=current_utilization,
                    success=True,
                    reason=f"Utilization {current_utilization:.2f} triggered {direction.value}"
                )
                
                self.scaling_history.append(scaling_event)
                
                # Apply scaling in system (would integrate with actual resource management)
                self._apply_resource_scaling(resource_type, old_value, new_value)
                
                # Record metric
                intelligent_monitor.record_metric(
                    f"{resource_type.value}_scaling",
                    new_value - old_value,
                    MetricType.GAUGE
                )
                
                self.logger.info(
                    f"Scaled {resource_type.value} {direction.value}: "
                    f"{old_value} -> {new_value} (utilization: {current_utilization:.2f})"
                )
            
        except Exception as e:
            scaling_event = ScalingEvent(
                timestamp=datetime.now(),
                resource_type=resource_type,
                direction=direction,
                old_value=old_value,
                new_value=old_value,
                trigger_metric=f"{resource_type.value}_utilization",
                trigger_value=current_utilization,
                success=False,
                reason=f"Scaling failed: {str(e)}"
            )
            
            self.scaling_history.append(scaling_event)
            self.logger.error(f"Failed to scale {resource_type.value}: {e}")
    
    def _apply_resource_scaling(self, resource_type: ResourceType, old_value: Union[int, float], new_value: Union[int, float]):
        """Apply resource scaling to actual system (integration point)."""
        
        # This would integrate with actual resource management systems
        # For now, just log the scaling operation
        
        scaling_operations = {
            ResourceType.CPU: lambda old, new: self._scale_cpu_resources(old, new),
            ResourceType.MEMORY: lambda old, new: self._scale_memory_resources(old, new),
            ResourceType.CONCURRENT_TASKS: lambda old, new: self._scale_task_concurrency(old, new),
            ResourceType.CACHE_SIZE: lambda old, new: self._scale_cache_size(old, new),
            ResourceType.THREAD_POOL: lambda old, new: self._scale_thread_pool(old, new),
            ResourceType.CONNECTION_POOL: lambda old, new: self._scale_connection_pool(old, new)
        }
        
        operation = scaling_operations.get(resource_type)
        if operation:
            try:
                operation(old_value, new_value)
            except Exception as e:
                self.logger.error(f"Failed to apply {resource_type.value} scaling: {e}")
    
    def _scale_cpu_resources(self, old_value: Union[int, float], new_value: Union[int, float]):
        """Scale CPU resources."""
        self.logger.info(f"CPU scaling: {old_value} -> {new_value} cores")
        # Integration point for actual CPU scaling
    
    def _scale_memory_resources(self, old_value: Union[int, float], new_value: Union[int, float]):
        """Scale memory resources."""
        self.logger.info(f"Memory scaling: {old_value} -> {new_value} GB")
        # Integration point for actual memory scaling
    
    def _scale_task_concurrency(self, old_value: Union[int, float], new_value: Union[int, float]):
        """Scale concurrent task limit."""
        self.logger.info(f"Task concurrency scaling: {old_value} -> {new_value} tasks")
        # Integration point for task pool scaling
    
    def _scale_cache_size(self, old_value: Union[int, float], new_value: Union[int, float]):
        """Scale cache size."""
        self.logger.info(f"Cache scaling: {old_value} -> {new_value} MB")
        # Integration point for cache scaling
    
    def _scale_thread_pool(self, old_value: Union[int, float], new_value: Union[int, float]):
        """Scale thread pool size."""
        self.logger.info(f"Thread pool scaling: {old_value} -> {new_value} threads")
        # Integration point for thread pool scaling
    
    def _scale_connection_pool(self, old_value: Union[int, float], new_value: Union[int, float]):
        """Scale connection pool size."""
        self.logger.info(f"Connection pool scaling: {old_value} -> {new_value} connections")
        # Integration point for connection pool scaling
    
    def _collect_performance_metrics(self):
        """Collect performance metrics for optimization."""
        
        timestamp = datetime.now()
        
        # Collect utilization metrics
        for resource_type in ResourceType:
            utilization = self._get_resource_utilization(resource_type)
            self.performance_metrics[f"{resource_type.value}_utilization"].append(
                (timestamp, utilization)
            )
        
        # Collect system-wide metrics
        system_load = self._get_system_load()
        response_time = self._get_average_response_time()
        throughput = self._get_system_throughput()
        
        self.performance_metrics["system_load"].append((timestamp, system_load))
        self.performance_metrics["response_time"].append((timestamp, response_time))
        self.performance_metrics["throughput"].append((timestamp, throughput))
        
        # Record in monitoring system
        intelligent_monitor.record_metric("response_time", response_time, MetricType.TIMER)
        intelligent_monitor.record_metric("throughput", throughput, MetricType.RATE)
    
    def _get_average_response_time(self) -> float:
        """Get average response time metric."""
        # Simulate response time based on current load
        system_load = self._get_system_load()
        base_response_time = 50  # ms
        
        # Response time increases exponentially with load
        response_time = base_response_time * (1 + system_load ** 2)
        
        # Add some randomness
        import random
        noise = random.uniform(0.8, 1.2)
        response_time *= noise
        
        return response_time
    
    def _get_system_throughput(self) -> float:
        """Get system throughput metric."""
        # Simulate throughput based on resources and load
        cpu_util = self._get_resource_utilization(ResourceType.CPU)
        concurrent_tasks = self.resources[ResourceType.CONCURRENT_TASKS].current_value
        
        # Throughput increases with resources but decreases with high utilization
        base_throughput = concurrent_tasks * 10  # requests per minute
        utilization_factor = max(0.1, 1 - cpu_util ** 2)
        
        throughput = base_throughput * utilization_factor
        
        return throughput
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report."""
        
        # Recent scaling activity
        recent_events = [event.to_dict() for event in list(self.scaling_history)[-20:]]
        
        # Scaling frequency by resource
        scaling_frequency = defaultdict(int)
        for event in self.scaling_history:
            scaling_frequency[event.resource_type.value] += 1
        
        # Current resource status
        resource_status = {}
        for resource_type, config in self.resources.items():
            current_utilization = self._get_resource_utilization(resource_type)
            resource_status[resource_type.value] = {
                "current_value": config.current_value,
                "utilization": current_utilization,
                "min_value": config.min_value,
                "max_value": config.max_value,
                "target_utilization": config.target_utilization,
                "can_scale": config.can_scale()
            }
        
        # Load pattern analysis
        current_pattern = self.load_analyzer.detect_pattern()
        load_predictions = self.load_analyzer.predict_load(horizon_minutes=15)
        
        # Resource optimization analysis
        optimization_analysis = self.resource_optimizer.analyze_resource_efficiency(self.resources)
        
        return {
            "scaling_status": {
                "scaling_enabled": self.scaling_enabled,
                "aggressive_scaling": self.aggressive_scaling,
                "cost_optimization_mode": self.cost_optimization_mode,
                "total_scaling_events": len(self.scaling_history)
            },
            "current_resources": resource_status,
            "recent_scaling_events": recent_events,
            "scaling_frequency": dict(scaling_frequency),
            "load_analysis": {
                "current_pattern": current_pattern.value,
                "predictions": [(ts.isoformat(), load) for ts, load in load_predictions],
                "pattern_confidence": 0.8  # Would be calculated based on prediction accuracy
            },
            "optimization_analysis": optimization_analysis,
            "performance_summary": {
                "current_system_load": self._get_system_load(),
                "average_response_time": self._get_average_response_time(),
                "current_throughput": self._get_system_throughput()
            }
        }
    
    def configure_scaling(
        self,
        resource_type: ResourceType,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        scale_up_threshold: Optional[float] = None,
        scale_down_threshold: Optional[float] = None,
        target_utilization: Optional[float] = None
    ):
        """Configure scaling parameters for a resource."""
        
        if resource_type not in self.resources:
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        config = self.resources[resource_type]
        
        if min_value is not None:
            config.min_value = min_value
        if max_value is not None:
            config.max_value = max_value
        if scale_up_threshold is not None:
            config.scale_up_threshold = scale_up_threshold
        if scale_down_threshold is not None:
            config.scale_down_threshold = scale_down_threshold
        if target_utilization is not None:
            config.target_utilization = target_utilization
        
        self.logger.info(f"Updated scaling configuration for {resource_type.value}")
    
    def enable_cost_optimization(self, enabled: bool = True):
        """Enable or disable cost optimization mode."""
        self.cost_optimization_mode = enabled
        mode_str = "enabled" if enabled else "disabled"
        self.logger.info(f"Cost optimization mode {mode_str}")
    
    def set_aggressive_scaling(self, enabled: bool = True):
        """Enable or disable aggressive scaling mode."""
        self.aggressive_scaling = enabled
        mode_str = "enabled" if enabled else "disabled"
        self.logger.info(f"Aggressive scaling mode {mode_str}")
    
    def stop(self):
        """Stop the autonomous scaling engine."""
        
        self.scaling_enabled = False
        self.shutdown_event.set()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Autonomous scaling engine stopped")


# Global scaling engine instance
autonomous_scaling_engine = AutonomousScalingEngine()