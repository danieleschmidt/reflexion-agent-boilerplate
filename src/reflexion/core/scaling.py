"""Auto-scaling and load distribution for reflexion systems."""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import queue
from concurrent.futures import ThreadPoolExecutor

from .types import ReflexionResult
from .health import HealthStatus
from .exceptions import ResourceExhaustedError


class ScalingEvent(Enum):
    """Scaling events."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    LOAD_SPIKE = "load_spike"
    RESOURCE_PRESSURE = "resource_pressure"
    RECOVERY = "recovery"


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME_BASED = "response_time_based"
    RESOURCE_BASED = "resource_based"


@dataclass
class WorkerNode:
    """Represents a worker node in the scaling system."""
    node_id: str
    capacity: int
    current_load: int = 0
    health_status: HealthStatus = HealthStatus.HEALTHY
    average_response_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    last_health_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def utilization(self) -> float:
        """Current utilization percentage."""
        if self.capacity == 0:
            return 0.0
        return self.current_load / self.capacity
    
    @property
    def success_rate(self) -> float:
        """Request success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def is_available(self) -> bool:
        """Check if node is available for work."""
        return (self.health_status == HealthStatus.HEALTHY and 
                self.current_load < self.capacity)
    
    def assign_work(self) -> bool:
        """Assign work to this node."""
        if self.current_load >= self.capacity:
            return False
        self.current_load += 1
        self.total_requests += 1
        return True
    
    def complete_work(self, success: bool = True, response_time: float = 0.0):
        """Mark work as completed."""
        if self.current_load > 0:
            self.current_load -= 1
        
        if success:
            self.successful_requests += 1
        
        # Update average response time with exponential moving average
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.average_response_time
            )


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    average_response_time: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class TaskQueue:
    """Thread-safe task queue with priority support."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.stats = {
            "enqueued": 0,
            "dequeued": 0,
            "dropped": 0,
            "current_size": 0
        }
    
    def put(self, task: Dict[str, Any], priority: int = 5) -> bool:
        """Add task to queue with priority (lower number = higher priority)."""
        try:
            task_item = (priority, time.time(), task)
            self.queue.put_nowait(task_item)
            self.stats["enqueued"] += 1
            self.stats["current_size"] = self.queue.qsize()
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get next task from queue."""
        try:
            priority, timestamp, task = self.queue.get(timeout=timeout)
            self.stats["dequeued"] += 1
            self.stats["current_size"] = self.queue.qsize()
            return task
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Current queue size."""
        return self.queue.qsize()
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.queue.full()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return self.stats.copy()


class LoadBalancer:
    """Advanced load balancer for distributing tasks across workers."""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.workers: Dict[str, WorkerNode] = {}
        self.round_robin_index = 0
        self.logger = logging.getLogger(__name__)
    
    def add_worker(self, worker: WorkerNode):
        """Add worker node to load balancer."""
        self.workers[worker.node_id] = worker
        self.logger.info(f"Added worker {worker.node_id} with capacity {worker.capacity}")
    
    def remove_worker(self, node_id: str):
        """Remove worker node from load balancer."""
        if node_id in self.workers:
            del self.workers[node_id]
            self.logger.info(f"Removed worker {node_id}")
    
    def select_worker(self, task_metadata: Optional[Dict] = None) -> Optional[WorkerNode]:
        """Select optimal worker based on load balancing strategy."""
        available_workers = [
            worker for worker in self.workers.values()
            if worker.is_available()
        ]
        
        if not available_workers:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_workers)
        
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(available_workers)
        
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(available_workers)
        
        elif self.strategy == LoadBalanceStrategy.RESPONSE_TIME_BASED:
            return self._select_by_response_time(available_workers)
        
        elif self.strategy == LoadBalanceStrategy.RESOURCE_BASED:
            return self._select_by_resources(available_workers)
        
        else:
            return available_workers[0]  # Default to first available
    
    def _select_round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin selection."""
        if self.round_robin_index >= len(workers):
            self.round_robin_index = 0
        
        worker = workers[self.round_robin_index]
        self.round_robin_index += 1
        return worker
    
    def _select_least_connections(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least current load."""
        return min(workers, key=lambda w: w.current_load)
    
    def _select_weighted_round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted selection based on capacity."""
        # Simple implementation: probability proportional to available capacity
        available_capacities = [w.capacity - w.current_load for w in workers]
        total_capacity = sum(available_capacities)
        
        if total_capacity == 0:
            return workers[0]
        
        # Find worker with highest available capacity
        return max(workers, key=lambda w: w.capacity - w.current_load)
    
    def _select_by_response_time(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with best response time."""
        return min(workers, key=lambda w: w.average_response_time or float('inf'))
    
    def _select_by_resources(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on resource utilization."""
        return min(workers, key=lambda w: w.utilization)
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across workers."""
        total_capacity = sum(w.capacity for w in self.workers.values())
        total_load = sum(w.current_load for w in self.workers.values())
        
        worker_stats = {}
        for worker_id, worker in self.workers.items():
            worker_stats[worker_id] = {
                "capacity": worker.capacity,
                "current_load": worker.current_load,
                "utilization": worker.utilization,
                "health": worker.health_status.value,
                "success_rate": worker.success_rate,
                "avg_response_time": worker.average_response_time
            }
        
        return {
            "total_capacity": total_capacity,
            "total_load": total_load,
            "overall_utilization": total_load / total_capacity if total_capacity > 0 else 0,
            "strategy": self.strategy.value,
            "workers": worker_stats,
            "available_workers": len([w for w in self.workers.values() if w.is_available()])
        }


class AutoScaler:
    """Intelligent auto-scaling system for reflexion workers."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: float = 60.0  # seconds
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.last_scaling_action = 0.0
        self.scaling_history: List[Dict[str, Any]] = []
        self.metrics_history: List[ScalingMetrics] = []
        
        self.logger = logging.getLogger(__name__)
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale up."""
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return False
        
        if metrics.active_workers >= self.max_workers:
            return False
        
        # Scale up conditions
        conditions = [
            metrics.cpu_utilization > self.scale_up_threshold,
            metrics.memory_utilization > self.scale_up_threshold,
            metrics.queue_depth > metrics.active_workers * 2,
            metrics.average_response_time > 5.0,  # 5 second threshold
            metrics.error_rate > 0.1  # 10% error rate
        ]
        
        # Scale up if any 2 conditions are met
        return sum(conditions) >= 2
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale down."""
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return False
        
        if metrics.active_workers <= self.min_workers:
            return False
        
        # Scale down conditions (all must be met for conservative scaling)
        conditions = [
            metrics.cpu_utilization < self.scale_down_threshold,
            metrics.memory_utilization < self.scale_down_threshold,
            metrics.queue_depth < metrics.active_workers,
            metrics.average_response_time < 1.0,
            metrics.error_rate < 0.01  # 1% error rate
        ]
        
        return all(conditions)
    
    def calculate_target_workers(self, metrics: ScalingMetrics) -> int:
        """Calculate optimal number of workers based on metrics."""
        if metrics.requests_per_second == 0:
            return self.min_workers
        
        # Estimate required workers based on utilization
        target_workers = max(
            self.min_workers,
            int(metrics.active_workers * (metrics.cpu_utilization / self.target_utilization))
        )
        
        # Consider queue depth
        if metrics.queue_depth > 0:
            queue_based_workers = int(metrics.queue_depth / 5) + metrics.active_workers
            target_workers = max(target_workers, queue_based_workers)
        
        return min(target_workers, self.max_workers)
    
    def record_scaling_action(self, action: ScalingEvent, details: Dict[str, Any]):
        """Record scaling action for history."""
        self.scaling_history.append({
            "timestamp": time.time(),
            "action": action.value,
            "details": details
        })
        
        self.last_scaling_action = time.time()
        
        # Keep only recent history
        one_hour_ago = time.time() - 3600
        self.scaling_history = [
            entry for entry in self.scaling_history
            if entry["timestamp"] > one_hour_ago
        ]
    
    def get_scaling_recommendations(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Get scaling recommendations without taking action."""
        recommendations = {
            "current_workers": metrics.active_workers,
            "target_workers": self.calculate_target_workers(metrics),
            "should_scale_up": self.should_scale_up(metrics),
            "should_scale_down": self.should_scale_down(metrics),
            "cooldown_remaining": max(0, self.cooldown_period - (time.time() - self.last_scaling_action)),
            "utilization_metrics": {
                "cpu": metrics.cpu_utilization,
                "memory": metrics.memory_utilization,
                "target": self.target_utilization
            }
        }
        
        return recommendations


class ScalingManager:
    """Comprehensive scaling management system."""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 20,
        worker_capacity: int = 10
    ):
        self.task_queue = TaskQueue(max_size=1000)
        self.load_balancer = LoadBalancer(LoadBalanceStrategy.LEAST_CONNECTIONS)
        self.auto_scaler = AutoScaler(min_workers, max_workers)
        
        self.worker_capacity = worker_capacity
        self.worker_counter = 0
        self.metrics = ScalingMetrics()
        
        # Initialize minimum workers
        for i in range(min_workers):
            self._create_worker()
        
        # Start background scaling monitor
        self.scaling_thread = threading.Thread(target=self._scaling_monitor_loop, daemon=True)
        self.scaling_active = True
        self.scaling_thread.start()
        
        self.logger = logging.getLogger(__name__)
    
    def _create_worker(self) -> WorkerNode:
        """Create a new worker node."""
        self.worker_counter += 1
        worker = WorkerNode(
            node_id=f"worker_{self.worker_counter}",
            capacity=self.worker_capacity,
            health_status=HealthStatus.HEALTHY
        )
        
        self.load_balancer.add_worker(worker)
        return worker
    
    def _remove_worker(self, worker_id: str):
        """Remove a worker node."""
        self.load_balancer.remove_worker(worker_id)
    
    def _scaling_monitor_loop(self):
        """Background scaling monitor loop."""
        while self.scaling_active:
            try:
                self._update_metrics()
                self._apply_auto_scaling()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Scaling monitor error: {str(e)}")
                time.sleep(30)  # Wait longer on error
    
    def _update_metrics(self):
        """Update current system metrics."""
        load_dist = self.load_balancer.get_load_distribution()
        queue_stats = self.task_queue.get_stats()
        
        self.metrics = ScalingMetrics(
            cpu_utilization=min(1.0, load_dist["overall_utilization"] * 1.5),  # Simulated
            memory_utilization=load_dist["overall_utilization"] * 0.8,  # Simulated
            queue_depth=self.task_queue.size(),
            active_workers=load_dist["available_workers"],
            average_response_time=0.5,  # Simulated
            requests_per_second=queue_stats["enqueued"] / 10,  # Rough estimate
            error_rate=0.02  # Simulated 2% error rate
        )
    
    def _apply_auto_scaling(self):
        """Apply auto-scaling decisions."""
        if self.auto_scaler.should_scale_up(self.metrics):
            self._scale_up()
        elif self.auto_scaler.should_scale_down(self.metrics):
            self._scale_down()
    
    def _scale_up(self):
        """Scale up by adding workers."""
        if len(self.load_balancer.workers) >= self.auto_scaler.max_workers:
            return
        
        worker = self._create_worker()
        self.auto_scaler.record_scaling_action(
            ScalingEvent.SCALE_UP,
            {"new_worker": worker.node_id, "total_workers": len(self.load_balancer.workers)}
        )
        
        self.logger.info(f"Scaled up: added {worker.node_id}")
    
    def _scale_down(self):
        """Scale down by removing workers."""
        if len(self.load_balancer.workers) <= self.auto_scaler.min_workers:
            return
        
        # Find least utilized worker to remove
        workers = list(self.load_balancer.workers.values())
        worker_to_remove = min(workers, key=lambda w: w.current_load)
        
        # Only remove if worker has no current load
        if worker_to_remove.current_load == 0:
            self._remove_worker(worker_to_remove.node_id)
            self.auto_scaler.record_scaling_action(
                ScalingEvent.SCALE_DOWN,
                {"removed_worker": worker_to_remove.node_id, "total_workers": len(self.load_balancer.workers)}
            )
            
            self.logger.info(f"Scaled down: removed {worker_to_remove.node_id}")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using load balancing."""
        # Add to queue
        if not self.task_queue.put(task):
            raise ResourceExhaustedError(
                "Task queue is full",
                resource_type="queue",
                retry_count=0,
                max_retries=3
            )
        
        # Select worker
        worker = self.load_balancer.select_worker()
        if not worker:
            raise ResourceExhaustedError(
                "No available workers",
                resource_type="workers",
                retry_count=0,
                max_retries=3
            )
        
        # Assign work
        if not worker.assign_work():
            raise ResourceExhaustedError(
                f"Worker {worker.node_id} at capacity",
                resource_type="worker_capacity",
                retry_count=0,
                max_retries=3
            )
        
        try:
            # Simulate task processing
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            processing_time = time.time() - start_time
            
            # Mark work complete
            worker.complete_work(success=True, response_time=processing_time)
            
            return {
                "task_id": task.get("id", "unknown"),
                "status": "completed",
                "worker_id": worker.node_id,
                "processing_time": processing_time
            }
            
        except Exception as e:
            worker.complete_work(success=False)
            raise e
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        return {
            "workers": self.load_balancer.get_load_distribution(),
            "queue": self.task_queue.get_stats(),
            "metrics": {
                "cpu_utilization": self.metrics.cpu_utilization,
                "memory_utilization": self.metrics.memory_utilization,
                "queue_depth": self.metrics.queue_depth,
                "active_workers": self.metrics.active_workers
            },
            "auto_scaler": self.auto_scaler.get_scaling_recommendations(self.metrics),
            "scaling_history": self.auto_scaler.scaling_history[-10:]  # Last 10 events
        }
    
    def shutdown(self):
        """Shutdown scaling manager."""
        self.scaling_active = False
        if self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5.0)
        
        self.logger.info("Scaling manager shutdown completed")


# Global scaling manager instance
scaling_manager = ScalingManager()