"""Distributed Reflexion Engine for Horizontal Scaling.

This module implements a distributed reflexion processing system that can scale
across multiple nodes, handle high concurrency, and provide fault tolerance.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import deque, defaultdict
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.types import ReflexionResult, Reflection
from ..core.logging_config import logger


class NodeStatus(Enum):
    """Status of processing nodes."""
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class DistributionStrategy(Enum):
    """Strategies for distributing tasks."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    GEOGRAPHIC = "geographic"
    SPECIALTY = "specialty"  # Route based on node capabilities


@dataclass
class ProcessingNode:
    """Represents a processing node in the distributed system."""
    node_id: str
    address: str
    port: int
    status: NodeStatus = NodeStatus.UNKNOWN
    capabilities: Set[str] = field(default_factory=set)
    current_load: float = 0.0
    max_capacity: int = 10
    active_tasks: int = 0
    total_processed: int = 0
    success_rate: float = 100.0
    avg_processing_time: float = 0.0
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.last_heartbeat = datetime.now()
    
    @property
    def utilization_percent(self) -> float:
        """Get current utilization percentage."""
        return (self.active_tasks / self.max_capacity) * 100 if self.max_capacity > 0 else 0
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for new tasks."""
        return (self.status == NodeStatus.ACTIVE and 
                self.active_tasks < self.max_capacity and
                self.last_heartbeat and
                datetime.now() - self.last_heartbeat < timedelta(minutes=2))
    
    def update_metrics(self, processing_time: float, success: bool):
        """Update node performance metrics."""
        self.total_processed += 1
        
        # Update average processing time (exponential moving average)
        if self.avg_processing_time == 0:
            self.avg_processing_time = processing_time
        else:
            alpha = 0.1  # Smoothing factor
            self.avg_processing_time = (alpha * processing_time + 
                                      (1 - alpha) * self.avg_processing_time)
        
        # Update success rate (exponential moving average)
        if success:
            success_score = 100.0
        else:
            success_score = 0.0
        
        alpha = 0.1
        self.success_rate = (alpha * success_score + (1 - alpha) * self.success_rate)
        
        # Update load based on utilization
        self.current_load = self.utilization_percent / 100.0


@dataclass
class ReflexionTask:
    """Represents a reflexion task for distributed processing."""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    required_capabilities: Set[str] = field(default_factory=set)
    max_retries: int = 3
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.now)
    assigned_node: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    result: Optional[ReflexionResult] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.started_at:
            return datetime.now() - self.started_at > timedelta(seconds=self.timeout_seconds)
        return datetime.now() - self.created_at > timedelta(seconds=self.timeout_seconds * 2)
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired


class ConsistentHashRing:
    """Consistent hash ring for distributed task assignment."""
    
    def __init__(self, replicas: int = 150):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        self.nodes = set()
    
    def add_node(self, node_id: str):
        """Add a node to the hash ring."""
        if node_id in self.nodes:
            return
        
        self.nodes.add(node_id)
        for i in range(self.replicas):
            key = self._hash(f"{node_id}:{i}")
            self.ring[key] = node_id
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node_id: str):
        """Remove a node from the hash ring."""
        if node_id not in self.nodes:
            return
        
        self.nodes.remove(node_id)
        for i in range(self.replicas):
            key = self._hash(f"{node_id}:{i}")
            if key in self.ring:
                del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a given key."""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find the first node with a key >= hash_key
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                return self.ring[ring_key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class TaskQueue:
    """Priority-based task queue with distributed capabilities."""
    
    def __init__(self):
        self.queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.pending_tasks: Dict[str, ReflexionTask] = {}
        self.completed_tasks: Dict[str, ReflexionTask] = {}
        self.lock = threading.Lock()
        
        # Task metrics
        self.queue_metrics = {
            "total_enqueued": 0,
            "total_completed": 0,
            "total_failed": 0,
            "avg_queue_time": 0.0,
            "avg_processing_time": 0.0
        }
    
    def enqueue(self, task: ReflexionTask):
        """Add a task to the appropriate priority queue."""
        with self.lock:
            self.queues[task.priority].append(task)
            self.pending_tasks[task.task_id] = task
            self.queue_metrics["total_enqueued"] += 1
    
    def dequeue(self, required_capabilities: Set[str] = None) -> Optional[ReflexionTask]:
        """Get the next highest priority task that matches capabilities."""
        with self.lock:
            # Check queues from highest to lowest priority
            for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
                queue = self.queues[priority]
                
                # Find a compatible task
                for i, task in enumerate(queue):
                    if required_capabilities is None or task.required_capabilities.issubset(required_capabilities):
                        # Remove from queue and return
                        queue.remove(task)
                        return task
            
            return None
    
    def complete_task(self, task: ReflexionTask, success: bool = True):
        """Mark a task as completed."""
        with self.lock:
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
            
            task.completed_at = datetime.now()
            self.completed_tasks[task.task_id] = task
            
            # Update metrics
            if success:
                self.queue_metrics["total_completed"] += 1
            else:
                self.queue_metrics["total_failed"] += 1
            
            # Calculate timing metrics
            if task.started_at:
                processing_time = (task.completed_at - task.started_at).total_seconds()
                self._update_avg_metric("avg_processing_time", processing_time)
            
            queue_time = ((task.started_at or task.completed_at) - task.created_at).total_seconds()
            self._update_avg_metric("avg_queue_time", queue_time)
    
    def requeue_task(self, task: ReflexionTask):
        """Requeue a failed task for retry."""
        with self.lock:
            if task.can_retry:
                task.retry_count += 1
                task.assigned_node = None
                task.started_at = None
                task.error = None
                
                # Add back to appropriate queue
                self.queues[task.priority].appendleft(task)  # Priority for retries
                self.pending_tasks[task.task_id] = task
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        with self.lock:
            queue_lengths = {
                priority.name: len(queue) 
                for priority, queue in self.queues.items()
            }
            
            total_pending = sum(queue_lengths.values())
            
            return {
                "queue_lengths": queue_lengths,
                "total_pending": total_pending,
                "total_completed": len(self.completed_tasks),
                "metrics": self.queue_metrics.copy()
            }
    
    def _update_avg_metric(self, metric_name: str, new_value: float, alpha: float = 0.1):
        """Update average metric using exponential moving average."""
        current = self.queue_metrics[metric_name]
        self.queue_metrics[metric_name] = alpha * new_value + (1 - alpha) * current


class LoadBalancer:
    """Intelligent load balancer for distributing reflexion tasks."""
    
    def __init__(self, strategy: DistributionStrategy = DistributionStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.hash_ring = ConsistentHashRing()
        self.node_performance_history = defaultdict(list)
        
    def select_node(self, nodes: List[ProcessingNode], task: ReflexionTask) -> Optional[ProcessingNode]:
        """Select the best node for a task based on the distribution strategy."""
        available_nodes = [node for node in nodes if node.is_available]
        
        if not available_nodes:
            return None
        
        # Filter by required capabilities
        if task.required_capabilities:
            compatible_nodes = [
                node for node in available_nodes 
                if task.required_capabilities.issubset(node.capabilities)
            ]
            if compatible_nodes:
                available_nodes = compatible_nodes
        
        # Apply distribution strategy
        if self.strategy == DistributionStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_nodes)
        elif self.strategy == DistributionStrategy.LEAST_LOADED:
            return self._least_loaded_selection(available_nodes)
        elif self.strategy == DistributionStrategy.RANDOM:
            return random.choice(available_nodes)
        elif self.strategy == DistributionStrategy.CONSISTENT_HASH:
            return self._consistent_hash_selection(available_nodes, task)
        elif self.strategy == DistributionStrategy.SPECIALTY:
            return self._specialty_selection(available_nodes, task)
        else:
            return self._least_loaded_selection(available_nodes)  # Default
    
    def _round_robin_selection(self, nodes: List[ProcessingNode]) -> ProcessingNode:
        """Round-robin node selection."""
        # Simple implementation using hash of current time
        index = int(time.time()) % len(nodes)
        return nodes[index]
    
    def _least_loaded_selection(self, nodes: List[ProcessingNode]) -> ProcessingNode:
        """Select node with least current load."""
        return min(nodes, key=lambda node: (node.current_load, node.active_tasks))
    
    def _consistent_hash_selection(self, nodes: List[ProcessingNode], task: ReflexionTask) -> Optional[ProcessingNode]:
        """Consistent hash-based selection."""
        # Ensure all nodes are in the hash ring
        for node in nodes:
            self.hash_ring.add_node(node.node_id)
        
        # Get node for this task
        selected_node_id = self.hash_ring.get_node(task.task_id)
        
        # Find the actual node object
        for node in nodes:
            if node.node_id == selected_node_id:
                return node
        
        # Fallback to least loaded
        return self._least_loaded_selection(nodes)
    
    def _specialty_selection(self, nodes: List[ProcessingNode], task: ReflexionTask) -> ProcessingNode:
        """Select node based on specialization and performance."""
        # Score nodes based on capability match and performance
        node_scores = []
        
        for node in nodes:
            score = 0
            
            # Capability matching bonus
            if task.required_capabilities.issubset(node.capabilities):
                score += 10
                
                # Extra points for exact capability match
                if task.required_capabilities == node.capabilities:
                    score += 5
            
            # Performance bonus
            score += node.success_rate / 10  # Max 10 points for 100% success rate
            score += max(0, (5 - node.avg_processing_time))  # Bonus for faster processing
            
            # Load penalty
            score -= node.current_load * 5  # Penalty for high load
            
            # Historical performance bonus
            if node.node_id in self.node_performance_history:
                history = self.node_performance_history[node.node_id]
                if history:
                    avg_historical_score = sum(history[-10:]) / len(history[-10:])  # Last 10 tasks
                    score += avg_historical_score / 20  # Small historical bonus
            
            node_scores.append((node, score))
        
        # Select highest scoring node
        return max(node_scores, key=lambda x: x[1])[0]
    
    def record_task_result(self, node_id: str, task: ReflexionTask, success: bool, processing_time: float):
        """Record task result for improving future selections."""
        # Calculate performance score
        score = 100 if success else 0
        
        # Adjust score based on processing time relative to task priority
        time_bonus = max(0, 10 - processing_time)  # Bonus for fast processing
        if task.priority == TaskPriority.URGENT:
            time_bonus *= 2  # Double time bonus for urgent tasks
        
        final_score = score + time_bonus
        
        # Store in history (limited to last 100 results per node)
        history = self.node_performance_history[node_id]
        history.append(final_score)
        if len(history) > 100:
            history.pop(0)


class DistributedReflexionEngine:
    """Main distributed reflexion processing engine."""
    
    def __init__(self, 
                 node_id: str = None,
                 max_concurrent_tasks: int = 10,
                 distribution_strategy: DistributionStrategy = DistributionStrategy.LEAST_LOADED):
        
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Core components
        self.task_queue = TaskQueue()
        self.load_balancer = LoadBalancer(distribution_strategy)
        self.processing_nodes: Dict[str, ProcessingNode] = {}
        
        # Local node (this instance)
        self.local_node = ProcessingNode(
            node_id=self.node_id,
            address="localhost",
            port=8080,
            status=NodeStatus.ACTIVE,
            max_capacity=max_concurrent_tasks,
            capabilities={"reflexion", "analysis", "optimization"}
        )
        self.processing_nodes[self.node_id] = self.local_node
        
        # Processing state
        self.active_tasks: Dict[str, ReflexionTask] = {}
        self.task_futures: Dict[str, asyncio.Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Monitoring and metrics
        self.engine_metrics = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_task_time": 0.0,
            "nodes_active": 1,
            "total_capacity": max_concurrent_tasks,
            "current_utilization": 0.0
        }
        
        # Background tasks
        self.monitoring_active = False
        self.heartbeat_interval = 30  # seconds
        
        self.logger = logging.getLogger(__name__)
        
    def add_processing_node(self, node: ProcessingNode):
        """Add a processing node to the cluster."""
        self.processing_nodes[node.node_id] = node
        self.load_balancer.hash_ring.add_node(node.node_id)
        
        # Update metrics
        self.engine_metrics["nodes_active"] = len([
            n for n in self.processing_nodes.values() 
            if n.status == NodeStatus.ACTIVE
        ])
        self.engine_metrics["total_capacity"] = sum(
            n.max_capacity for n in self.processing_nodes.values()
        )
        
        self.logger.info(f"Added processing node: {node.node_id}")
    
    def remove_processing_node(self, node_id: str):
        """Remove a processing node from the cluster."""
        if node_id in self.processing_nodes:
            del self.processing_nodes[node_id]
            self.load_balancer.hash_ring.remove_node(node_id)
            
            # Update metrics
            self.engine_metrics["nodes_active"] = len([
                n for n in self.processing_nodes.values() 
                if n.status == NodeStatus.ACTIVE
            ])
            self.engine_metrics["total_capacity"] = sum(
                n.max_capacity for n in self.processing_nodes.values()
            )
            
            self.logger.info(f"Removed processing node: {node_id}")
    
    async def submit_task(self, 
                         task_type: str,
                         input_data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         required_capabilities: Set[str] = None,
                         timeout_seconds: int = 300) -> str:
        """Submit a new reflexion task for processing."""
        
        task = ReflexionTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            input_data=input_data,
            priority=priority,
            required_capabilities=required_capabilities or set(),
            timeout_seconds=timeout_seconds
        )
        
        # Add to queue
        self.task_queue.enqueue(task)
        
        self.logger.info(f"Submitted task {task.task_id} with priority {priority.name}")
        
        # Start processing if not already running
        if not self.monitoring_active:
            await self.start_processing()
        
        return task.task_id
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Optional[ReflexionResult]:
        """Get the result of a submitted task."""
        start_time = time.time()
        
        while True:
            # Check if task is completed
            if task_id in self.task_queue.completed_tasks:
                task = self.task_queue.completed_tasks[task_id]
                return task.result
            
            # Check if task is still pending or active
            if (task_id not in self.task_queue.pending_tasks and 
                task_id not in self.active_tasks):
                return None  # Task not found
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            await asyncio.sleep(0.1)  # Brief pause before checking again
    
    async def start_processing(self):
        """Start the distributed processing engine."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start background tasks
        asyncio.create_task(self._task_processor())
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._cleanup_expired_tasks())
        
        self.logger.info("Started distributed reflexion engine")
    
    async def stop_processing(self):
        """Stop the distributed processing engine."""
        self.monitoring_active = False
        
        # Cancel active tasks
        for future in self.task_futures.values():
            if not future.done():
                future.cancel()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("Stopped distributed reflexion engine")
    
    async def _task_processor(self):
        """Main task processing loop."""
        while self.monitoring_active:
            try:
                # Check for available capacity
                active_count = len(self.active_tasks)
                if active_count >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task from queue
                available_nodes = [node for node in self.processing_nodes.values() if node.is_available]
                if not available_nodes:
                    await asyncio.sleep(1)
                    continue
                
                # Find a compatible task
                task = None
                for node in available_nodes:
                    task = self.task_queue.dequeue(node.capabilities)
                    if task:
                        break
                
                if not task:
                    await asyncio.sleep(0.5)
                    continue
                
                # Select best node for the task
                selected_node = self.load_balancer.select_node(available_nodes, task)
                if not selected_node:
                    # Requeue task if no suitable node
                    self.task_queue.requeue_task(task)
                    await asyncio.sleep(1)
                    continue
                
                # Assign task to node
                task.assigned_node = selected_node.node_id
                task.started_at = datetime.now()
                self.active_tasks[task.task_id] = task
                
                # Update node status
                selected_node.active_tasks += 1
                if selected_node.active_tasks >= selected_node.max_capacity:
                    selected_node.status = NodeStatus.BUSY
                
                # Process task
                if selected_node.node_id == self.node_id:
                    # Process locally
                    future = asyncio.create_task(self._process_local_task(task))
                    self.task_futures[task.task_id] = future
                else:
                    # Process remotely (would require network communication)
                    future = asyncio.create_task(self._process_remote_task(task, selected_node))
                    self.task_futures[task.task_id] = future
                
                self.logger.debug(f"Started processing task {task.task_id} on node {selected_node.node_id}")
                
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_local_task(self, task: ReflexionTask) -> ReflexionResult:
        """Process a task locally."""
        try:
            start_time = time.time()
            
            # Simulate reflexion processing based on task type
            result = await self._execute_reflexion_task(task)
            
            processing_time = time.time() - start_time
            success = result is not None
            
            # Update task
            task.result = result
            task.completed_at = datetime.now()
            
            # Update node metrics
            node = self.processing_nodes[self.node_id]
            node.update_metrics(processing_time, success)
            node.active_tasks -= 1
            
            if node.active_tasks < node.max_capacity and node.status == NodeStatus.BUSY:
                node.status = NodeStatus.ACTIVE
            
            # Update engine metrics
            self.engine_metrics["total_tasks_processed"] += 1
            if success:
                self.engine_metrics["successful_tasks"] += 1
            else:
                self.engine_metrics["failed_tasks"] += 1
            
            # Update average task time
            current_avg = self.engine_metrics["avg_task_time"]
            total_processed = self.engine_metrics["total_tasks_processed"]
            self.engine_metrics["avg_task_time"] = (
                (current_avg * (total_processed - 1) + processing_time) / total_processed
            )
            
            # Record result with load balancer
            self.load_balancer.record_task_result(
                self.node_id, task, success, processing_time
            )
            
            # Complete task
            self.task_queue.complete_task(task, success)
            
            # Cleanup
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            if task.task_id in self.task_futures:
                del self.task_futures[task.task_id]
            
            self.logger.debug(f"Completed task {task.task_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            
            # Handle task failure
            task.error = str(e)
            node = self.processing_nodes[self.node_id]
            node.active_tasks -= 1
            
            # Try to requeue if possible
            if task.can_retry:
                self.task_queue.requeue_task(task)
                self.logger.info(f"Requeued failed task {task.task_id} (attempt {task.retry_count + 1})")
            else:
                self.task_queue.complete_task(task, success=False)
                self.engine_metrics["failed_tasks"] += 1
            
            # Cleanup
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            if task.task_id in self.task_futures:
                del self.task_futures[task.task_id]
            
            return None
    
    async def _process_remote_task(self, task: ReflexionTask, node: ProcessingNode) -> ReflexionResult:
        """Process a task on a remote node (simulation)."""
        # In a real implementation, this would make network calls
        # For now, we simulate remote processing
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate network latency
        
        # Simulate success/failure
        success_rate = node.success_rate / 100.0
        success = random.random() < success_rate
        
        if success:
            # Create mock result
            result = ReflexionResult(
                output=f"Remote result from {node.node_id}",
                iterations=random.randint(1, 5),
                reflections=[],
                success=True,
                metadata={"processed_by": node.node_id}
            )
        else:
            result = None
        
        processing_time = random.uniform(1.0, 5.0)
        
        # Update task
        task.result = result
        task.completed_at = datetime.now()
        
        # Update node metrics
        node.update_metrics(processing_time, success)
        node.active_tasks -= 1
        
        # Record result
        self.load_balancer.record_task_result(node.node_id, task, success, processing_time)
        
        # Complete or requeue task
        if success:
            self.task_queue.complete_task(task, success=True)
        elif task.can_retry:
            self.task_queue.requeue_task(task)
        else:
            self.task_queue.complete_task(task, success=False)
        
        # Cleanup
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        if task.task_id in self.task_futures:
            del self.task_futures[task.task_id]
        
        return result
    
    async def _execute_reflexion_task(self, task: ReflexionTask) -> Optional[ReflexionResult]:
        """Execute a reflexion task based on its type."""
        task_type = task.task_type
        input_data = task.input_data
        
        try:
            # Simulate different types of reflexion tasks
            if task_type == "basic_reflexion":
                return await self._basic_reflexion(input_data)
            elif task_type == "advanced_analysis":
                return await self._advanced_analysis(input_data)
            elif task_type == "optimization":
                return await self._optimization_task(input_data)
            elif task_type == "research":
                return await self._research_task(input_data)
            else:
                # Generic reflexion
                return await self._generic_reflexion(input_data)
                
        except Exception as e:
            self.logger.error(f"Error executing reflexion task {task_type}: {e}")
            return None
    
    async def _basic_reflexion(self, input_data: Dict[str, Any]) -> ReflexionResult:
        """Execute basic reflexion task."""
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate processing
        
        return ReflexionResult(
            task=input_data.get('task', 'basic_reflexion'),
            output=f"Basic reflexion completed: {input_data.get('task', 'unknown')}",
            success=True,
            iterations=random.randint(1, 3),
            reflections=[
                Reflection(
                    task="basic_reflexion",
                    output="Initial analysis completed",
                    success=True,
                    score=0.8,
                    issues=[],
                    improvements=["enhance_accuracy", "improve_efficiency"],
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                )
            ],
            total_time=random.uniform(0.5, 2.0),
            metadata={"task_type": "basic_reflexion", "processing_node": self.node_id}
        )
    
    async def _advanced_analysis(self, input_data: Dict[str, Any]) -> ReflexionResult:
        """Execute advanced analysis task."""
        await asyncio.sleep(random.uniform(2.0, 5.0))  # Longer processing
        
        return ReflexionResult(
            task=input_data.get('subject', 'advanced_analysis'),
            output=f"Advanced analysis completed: {input_data.get('subject', 'unknown')}",
            success=True,
            iterations=random.randint(3, 7),
            reflections=[
                Reflection(
                    task="advanced_analysis",
                    output="Deep analysis performed",
                    success=True,
                    score=0.9,
                    issues=[],
                    improvements=["statistical_validation", "cross_reference_data", "peer_review"],
                    confidence=0.9,
                    timestamp=datetime.now().isoformat()
                ),
                Reflection(
                    task="quality_check",
                    output="Quality assurance check",
                    success=True,
                    score=0.85,
                    issues=[],
                    improvements=["methodology_review", "bias_correction"],
                    confidence=0.85,
                    timestamp=datetime.now().isoformat()
                )
            ],
            total_time=random.uniform(2.0, 5.0),
            metadata={
                "task_type": "advanced_analysis",
                "processing_node": self.node_id,
                "analysis_depth": "comprehensive"
            }
        )
    
    async def _optimization_task(self, input_data: Dict[str, Any]) -> ReflexionResult:
        """Execute optimization task."""
        await asyncio.sleep(random.uniform(1.5, 4.0))
        
        return ReflexionResult(
            task=input_data.get('target', 'optimization'),
            output=f"Optimization completed: {input_data.get('target', 'performance')}",
            success=True,
            iterations=random.randint(2, 5),
            reflections=[
                Reflection(
                    task="optimization",
                    output="Performance bottlenecks identified",
                    success=True,
                    score=0.85,
                    issues=[],
                    improvements=["algorithm_efficiency", "resource_utilization"],
                    confidence=0.85,
                    timestamp=datetime.now().isoformat()
                ),
                Reflection(
                    task="optimization_strategies",
                    output="Optimization strategies applied",
                    success=True,
                    score=0.9,
                    issues=[],
                    improvements=["caching_layer", "parallel_processing"],
                    confidence=0.9,
                    timestamp=datetime.now().isoformat()
                )
            ],
            total_time=random.uniform(1.5, 4.0),
            metadata={
                "task_type": "optimization",
                "processing_node": self.node_id,
                "optimization_target": input_data.get("target", "performance")
            }
        )
    
    async def _research_task(self, input_data: Dict[str, Any]) -> ReflexionResult:
        """Execute research task."""
        await asyncio.sleep(random.uniform(3.0, 8.0))  # Longest processing
        
        return ReflexionResult(
            task=input_data.get('topic', 'research'),
            output=f"Research completed: {input_data.get('topic', 'general')}",
            success=True,
            iterations=random.randint(5, 10),
            reflections=[
                Reflection(
                    task="literature_review",
                    output="Literature review completed",
                    success=True,
                    score=0.8,
                    issues=[],
                    improvements=["expand_source_base", "include_recent_studies"],
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                ),
                Reflection(
                    task="hypothesis_formation",
                    output="Hypothesis formation",
                    success=True,
                    score=0.85,
                    issues=[],
                    improvements=["statistical_validation", "experimental_design"],
                    confidence=0.85,
                    timestamp=datetime.now().isoformat()
                ),
                Reflection(
                    task="results_synthesis",
                    output="Results synthesis",
                    success=True,
                    score=0.9,
                    issues=[],
                    improvements=["peer_review", "replication_study"],
                    confidence=0.9,
                    timestamp=datetime.now().isoformat()
                )
            ],
            total_time=random.uniform(3.0, 8.0),
            metadata={
                "task_type": "research",
                "processing_node": self.node_id,
                "research_topic": input_data.get("topic", "general"),
                "methodology": "systematic_review"
            }
        )
    
    async def _generic_reflexion(self, input_data: Dict[str, Any]) -> ReflexionResult:
        """Execute generic reflexion task."""
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        return ReflexionResult(
            task="generic_reflexion",
            output=f"Generic reflexion completed: {input_data}",
            success=True,
            iterations=random.randint(1, 4),
            reflections=[
                Reflection(
                    task="generic_analysis",
                    output="Generic analysis performed",
                    success=True,
                    score=0.75,
                    issues=[],
                    improvements=["improve_specificity", "add_context"],
                    confidence=0.75,
                    timestamp=datetime.now().isoformat()
                )
            ],
            total_time=random.uniform(1.0, 3.0),
            metadata={"task_type": "generic", "processing_node": self.node_id}
        )
    
    async def _heartbeat_monitor(self):
        """Monitor node heartbeats and health."""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                # Update local node heartbeat
                self.local_node.last_heartbeat = current_time
                
                # Check other nodes for stale heartbeats
                stale_nodes = []
                for node_id, node in self.processing_nodes.items():
                    if node_id == self.node_id:
                        continue
                    
                    if (node.last_heartbeat and 
                        current_time - node.last_heartbeat > timedelta(minutes=3)):
                        stale_nodes.append(node_id)
                        node.status = NodeStatus.FAILED
                
                # Remove failed nodes
                for node_id in stale_nodes:
                    self.logger.warning(f"Node {node_id} failed heartbeat check, removing from cluster")
                    self.remove_processing_node(node_id)
                
                # Update utilization metrics
                active_nodes = [n for n in self.processing_nodes.values() if n.status == NodeStatus.ACTIVE]
                if active_nodes:
                    total_active_tasks = sum(n.active_tasks for n in active_nodes)
                    total_capacity = sum(n.max_capacity for n in active_nodes)
                    self.engine_metrics["current_utilization"] = (total_active_tasks / total_capacity * 100) if total_capacity > 0 else 0
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _cleanup_expired_tasks(self):
        """Clean up expired tasks."""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                expired_tasks = []
                
                # Check active tasks for expiration
                for task_id, task in self.active_tasks.items():
                    if task.is_expired:
                        expired_tasks.append(task_id)
                
                # Handle expired tasks
                for task_id in expired_tasks:
                    task = self.active_tasks[task_id]
                    self.logger.warning(f"Task {task_id} expired after {task.timeout_seconds} seconds")
                    
                    # Cancel future if exists
                    if task_id in self.task_futures:
                        future = self.task_futures[task_id]
                        if not future.done():
                            future.cancel()
                        del self.task_futures[task_id]
                    
                    # Update node status
                    if task.assigned_node and task.assigned_node in self.processing_nodes:
                        node = self.processing_nodes[task.assigned_node]
                        node.active_tasks = max(0, node.active_tasks - 1)
                    
                    # Try to requeue if possible
                    if task.can_retry:
                        self.task_queue.requeue_task(task)
                    else:
                        task.error = "Task expired"
                        self.task_queue.complete_task(task, success=False)
                    
                    del self.active_tasks[task_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        active_nodes = [n for n in self.processing_nodes.values() if n.status == NodeStatus.ACTIVE]
        
        node_details = []
        for node in self.processing_nodes.values():
            node_details.append({
                "node_id": node.node_id,
                "status": node.status.value,
                "utilization": node.utilization_percent,
                "active_tasks": node.active_tasks,
                "max_capacity": node.max_capacity,
                "success_rate": node.success_rate,
                "avg_processing_time": node.avg_processing_time,
                "total_processed": node.total_processed,
                "capabilities": list(node.capabilities),
                "last_heartbeat": node.last_heartbeat.isoformat() if node.last_heartbeat else None
            })
        
        queue_stats = self.task_queue.get_queue_stats()
        
        return {
            "cluster_overview": {
                "total_nodes": len(self.processing_nodes),
                "active_nodes": len(active_nodes),
                "total_capacity": sum(n.max_capacity for n in active_nodes),
                "current_utilization": self.engine_metrics["current_utilization"],
                "distribution_strategy": self.load_balancer.strategy.value
            },
            "task_statistics": {
                "active_tasks": len(self.active_tasks),
                "queue_stats": queue_stats,
                "engine_metrics": self.engine_metrics.copy()
            },
            "nodes": node_details,
            "system_health": {
                "monitoring_active": self.monitoring_active,
                "avg_task_completion_time": self.engine_metrics["avg_task_time"],
                "overall_success_rate": (
                    (self.engine_metrics["successful_tasks"] / 
                     max(1, self.engine_metrics["total_tasks_processed"])) * 100
                )
            }
        }
    
    async def scale_cluster(self, target_nodes: int, node_template: Dict[str, Any] = None):
        """Automatically scale the cluster up or down."""
        current_active_nodes = len([n for n in self.processing_nodes.values() if n.status == NodeStatus.ACTIVE])
        
        if target_nodes > current_active_nodes:
            # Scale up
            for i in range(target_nodes - current_active_nodes):
                new_node = ProcessingNode(
                    node_id=f"auto_node_{uuid.uuid4().hex[:8]}",
                    address="auto-scaled",
                    port=8080 + i,
                    status=NodeStatus.ACTIVE,
                    max_capacity=self.max_concurrent_tasks,
                    capabilities={"reflexion", "analysis"} if not node_template else set(node_template.get("capabilities", []))
                )
                
                self.add_processing_node(new_node)
                self.logger.info(f"Scaled up: Added node {new_node.node_id}")
        
        elif target_nodes < current_active_nodes:
            # Scale down (remove least utilized nodes)
            active_nodes = [n for n in self.processing_nodes.values() if n.status == NodeStatus.ACTIVE and n.node_id != self.node_id]
            active_nodes.sort(key=lambda n: n.utilization_percent)
            
            nodes_to_remove = current_active_nodes - target_nodes
            for node in active_nodes[:nodes_to_remove]:
                # Gracefully remove node (wait for current tasks to complete)
                node.status = NodeStatus.MAINTENANCE
                
                # Wait for tasks to complete or timeout
                timeout = 300  # 5 minutes
                start_time = time.time()
                while node.active_tasks > 0 and (time.time() - start_time) < timeout:
                    await asyncio.sleep(5)
                
                self.remove_processing_node(node.node_id)
                self.logger.info(f"Scaled down: Removed node {node.node_id}")


# Example usage and testing
async def test_distributed_engine():
    """Test the distributed reflexion engine."""
    logger.info("Testing Distributed Reflexion Engine")
    
    # Create engine
    engine = DistributedReflexionEngine(
        node_id="primary_node",
        max_concurrent_tasks=5,
        distribution_strategy=DistributionStrategy.LEAST_LOADED
    )
    
    # Add additional nodes
    for i in range(3):
        node = ProcessingNode(
            node_id=f"worker_{i+1}",
            address=f"worker{i+1}.cluster",
            port=8080,
            status=NodeStatus.ACTIVE,
            max_capacity=8,
            capabilities={"reflexion", "analysis", "optimization"}
        )
        engine.add_processing_node(node)
    
    # Start processing
    await engine.start_processing()
    
    # Submit test tasks
    task_ids = []
    for i in range(10):
        task_id = await engine.submit_task(
            task_type="basic_reflexion" if i % 2 == 0 else "advanced_analysis",
            input_data={"task": f"test_task_{i}", "complexity": random.choice(["low", "medium", "high"])},
            priority=random.choice(list(TaskPriority))
        )
        task_ids.append(task_id)
    
    logger.info(f"Submitted {len(task_ids)} tasks")
    
    # Wait for some tasks to complete
    await asyncio.sleep(10)
    
    # Check results
    completed_count = 0
    for task_id in task_ids:
        result = await engine.get_task_result(task_id, timeout=1.0)
        if result:
            completed_count += 1
            logger.info(f"Task {task_id} completed: {result.success}")
    
    logger.info(f"Completed {completed_count}/{len(task_ids)} tasks")
    
    # Get cluster status
    status = engine.get_cluster_status()
    logger.info(f"Cluster status: {json.dumps(status, indent=2, default=str)}")
    
    # Test scaling
    logger.info("Testing auto-scaling")
    await engine.scale_cluster(6)
    
    # Stop processing
    await engine.stop_processing()
    
    return engine


if __name__ == "__main__":
    # Run test
    asyncio.run(test_distributed_engine())