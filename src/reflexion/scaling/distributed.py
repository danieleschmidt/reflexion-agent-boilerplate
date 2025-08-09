"""Distributed processing capabilities for reflexion agents."""

import asyncio
import json
import hashlib
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from enum import Enum
import logging

from ..core.types import ReflexionResult, ReflectionType
from ..core.agent import ReflexionAgent


class NodeStatus(Enum):
    """Status of worker nodes."""
    HEALTHY = "healthy"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    node_id: str
    host: str
    port: int
    status: NodeStatus
    capacity: int
    current_load: int
    capabilities: Set[str]
    last_heartbeat: str
    started_at: str
    metadata: Dict[str, Any]


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    requirements: Set[str]
    created_at: str
    assigned_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    status: TaskStatus
    assigned_node: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    retry_count: int
    max_retries: int


@dataclass
class NodeHealthMetrics:
    """Health metrics for a worker node."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    last_updated: str


class TaskQueue:
    """Priority queue for distributed tasks."""
    
    def __init__(self):
        """Initialize task queue."""
        self.pending_tasks: List[DistributedTask] = []
        self.assigned_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task: DistributedTask):
        """Add task to the queue."""
        self.pending_tasks.append(task)
        # Sort by priority (higher priority first)
        self.pending_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        self.logger.info(f"Added task {task.task_id} with priority {task.priority.value}")
    
    def get_next_task(self, node_capabilities: Set[str]) -> Optional[DistributedTask]:
        """Get next available task for a node with given capabilities."""
        
        for i, task in enumerate(self.pending_tasks):
            # Check if node has required capabilities
            if task.requirements and not task.requirements.issubset(node_capabilities):
                continue
            
            # Remove from pending and return
            return self.pending_tasks.pop(i)
        
        return None
    
    def assign_task(self, task: DistributedTask, node_id: str):
        """Mark task as assigned to a node."""
        task.status = TaskStatus.ASSIGNED
        task.assigned_node = node_id
        task.assigned_at = datetime.now().isoformat()
        
        self.assigned_tasks[task.task_id] = task
        self.logger.info(f"Assigned task {task.task_id} to node {node_id}")
    
    def start_task(self, task_id: str):
        """Mark task as started."""
        if task_id in self.assigned_tasks:
            task = self.assigned_tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            self.logger.info(f"Started task {task_id}")
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed with result."""
        if task_id in self.assigned_tasks:
            task = self.assigned_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = result
            
            self.completed_tasks[task_id] = task
            self.logger.info(f"Completed task {task_id}")
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        if task_id in self.assigned_tasks:
            task = self.assigned_tasks.pop(task_id)
            task.status = TaskStatus.FAILED
            task.error = error
            task.retry_count += 1
            
            # Retry if under limit
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                task.assigned_node = None
                task.assigned_at = None
                self.add_task(task)
                self.logger.info(f"Retrying task {task_id} (attempt {task.retry_count + 1})")
            else:
                self.failed_tasks[task_id] = task
                self.logger.error(f"Failed task {task_id} permanently: {error}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "assigned_tasks": len(self.assigned_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks": (
                len(self.pending_tasks) + len(self.assigned_tasks) + 
                len(self.completed_tasks) + len(self.failed_tasks)
            )
        }


class NodeRegistry:
    """Registry of worker nodes in the distributed system."""
    
    def __init__(self, heartbeat_timeout: int = 60):
        """Initialize node registry.
        
        Args:
            heartbeat_timeout: Seconds before considering node offline
        """
        self.nodes: Dict[str, WorkerNode] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.logger = logging.getLogger(__name__)
    
    def register_node(self, node: WorkerNode):
        """Register a new worker node."""
        self.nodes[node.node_id] = node
        self.logger.info(f"Registered node {node.node_id} at {node.host}:{node.port}")
    
    def update_heartbeat(self, node_id: str, metrics: NodeHealthMetrics):
        """Update node heartbeat with health metrics."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_heartbeat = datetime.now().isoformat()
            
            # Update status based on metrics
            node.status = self._determine_node_status(metrics)
            node.current_load = metrics.active_tasks
            
            self.logger.debug(f"Updated heartbeat for node {node_id}")
        else:
            self.logger.warning(f"Heartbeat received from unknown node: {node_id}")
    
    def _determine_node_status(self, metrics: NodeHealthMetrics) -> NodeStatus:
        """Determine node status based on health metrics."""
        
        # Check for critical resource usage
        if (metrics.cpu_usage > 90 or 
            metrics.memory_usage > 90 or 
            metrics.disk_usage > 95):
            return NodeStatus.OVERLOADED
        
        # Check for high but manageable load
        if (metrics.cpu_usage > 75 or 
            metrics.memory_usage > 75 or 
            metrics.active_tasks > 10):
            return NodeStatus.BUSY
        
        # Check for network issues
        if metrics.network_latency > 1000:  # 1 second
            return NodeStatus.UNHEALTHY
        
        return NodeStatus.HEALTHY
    
    def get_available_nodes(self) -> List[WorkerNode]:
        """Get nodes available for task assignment."""
        available = []
        current_time = datetime.now()
        
        for node in self.nodes.values():
            # Check if node is still alive
            last_heartbeat = datetime.fromisoformat(node.last_heartbeat)
            if (current_time - last_heartbeat).total_seconds() > self.heartbeat_timeout:
                node.status = NodeStatus.OFFLINE
                continue
            
            # Only return healthy or busy nodes (not overloaded/unhealthy)
            if node.status in [NodeStatus.HEALTHY, NodeStatus.BUSY]:
                available.append(node)
        
        return available
    
    def get_best_node(self, requirements: Set[str] = None) -> Optional[WorkerNode]:
        """Get the best available node for task assignment."""
        available_nodes = self.get_available_nodes()
        
        # Filter by requirements
        if requirements:
            available_nodes = [
                node for node in available_nodes
                if requirements.issubset(node.capabilities)
            ]
        
        if not available_nodes:
            return None
        
        # Sort by load (ascending) and status priority
        def node_score(node: WorkerNode) -> float:
            status_priority = {
                NodeStatus.HEALTHY: 0,
                NodeStatus.BUSY: 1,
                NodeStatus.OVERLOADED: 2,
                NodeStatus.UNHEALTHY: 3,
                NodeStatus.OFFLINE: 4
            }
            
            load_factor = node.current_load / max(1, node.capacity)
            status_factor = status_priority.get(node.status, 4)
            
            return load_factor + status_factor
        
        available_nodes.sort(key=node_score)
        return available_nodes[0]
    
    def remove_node(self, node_id: str):
        """Remove node from registry."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Removed node {node_id} from registry")


class TaskDistribution:
    """Handles distribution of tasks across worker nodes."""
    
    def __init__(
        self,
        task_queue: TaskQueue,
        node_registry: NodeRegistry
    ):
        """Initialize task distribution.
        
        Args:
            task_queue: Task queue for managing tasks
            node_registry: Registry of worker nodes
        """
        self.task_queue = task_queue
        self.node_registry = node_registry
        self.logger = logging.getLogger(__name__)
        
        # Distribution strategies
        self.strategies = {
            "round_robin": self._round_robin_strategy,
            "least_loaded": self._least_loaded_strategy,
            "capability_based": self._capability_based_strategy
        }
        
        self.current_strategy = "least_loaded"
    
    async def distribute_tasks(self):
        """Distribute pending tasks to available nodes."""
        
        while True:
            # Get available nodes
            available_nodes = self.node_registry.get_available_nodes()
            
            if not available_nodes:
                await asyncio.sleep(1)  # No nodes available
                continue
            
            # Get next task
            for node in available_nodes:
                if node.current_load >= node.capacity:
                    continue  # Node is at capacity
                
                task = self.task_queue.get_next_task(node.capabilities)
                if not task:
                    break  # No more tasks
                
                # Assign task to node
                await self._assign_task_to_node(task, node)
            
            await asyncio.sleep(0.1)  # Small delay between distribution cycles
    
    async def _assign_task_to_node(self, task: DistributedTask, node: WorkerNode):
        """Assign a task to a specific node."""
        try:
            # Mark task as assigned
            self.task_queue.assign_task(task, node.node_id)
            
            # Send task to node (in production, this would be a network call)
            await self._send_task_to_node(task, node)
            
            # Update node load
            node.current_load += 1
            
            self.logger.info(f"Assigned task {task.task_id} to node {node.node_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task.task_id} to node {node.node_id}: {e}")
            # Return task to queue
            self.task_queue.fail_task(task.task_id, str(e))
    
    async def _send_task_to_node(self, task: DistributedTask, node: WorkerNode):
        """Send task to worker node (simulated)."""
        # In a real implementation, this would send the task over the network
        # For now, we'll just simulate the network call
        await asyncio.sleep(0.01)  # Simulate network latency
        
        # Simulate task execution on the node
        asyncio.create_task(self._simulate_task_execution(task, node))
    
    async def _simulate_task_execution(self, task: DistributedTask, node: WorkerNode):
        """Simulate task execution on a worker node."""
        try:
            # Mark task as started
            self.task_queue.start_task(task.task_id)
            
            # Simulate task execution time
            import random
            execution_time = random.uniform(1, 5)
            await asyncio.sleep(execution_time)
            
            # Simulate task result
            if task.task_type == "reflexion_task":
                # Create a mock ReflexionAgent and execute
                agent_config = task.payload.get("agent_config", {})
                task_description = task.payload.get("task", "Default task")
                
                # Simulate reflexion execution
                result = {
                    "task": task_description,
                    "output": f"Completed {task_description} on node {node.node_id}",
                    "success": random.choice([True, True, True, False]),  # 75% success rate
                    "iterations": random.randint(1, 4),
                    "reflections": [],
                    "total_time": execution_time,
                    "metadata": {
                        "node_id": node.node_id,
                        "execution_time": execution_time
                    }
                }
            else:
                result = {
                    "status": "completed",
                    "output": f"Task {task.task_id} completed on {node.node_id}",
                    "execution_time": execution_time
                }
            
            # Complete the task
            self.task_queue.complete_task(task.task_id, result)
            
            # Update node load
            node.current_load = max(0, node.current_load - 1)
            
        except Exception as e:
            # Fail the task
            self.task_queue.fail_task(task.task_id, str(e))
            node.current_load = max(0, node.current_load - 1)
    
    def _round_robin_strategy(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round-robin node selection."""
        # Simple implementation - in production would track last used node
        import random
        return random.choice(nodes)
    
    def _least_loaded_strategy(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least load."""
        return min(nodes, key=lambda n: n.current_load / max(1, n.capacity))
    
    def _capability_based_strategy(self, nodes: List[WorkerNode], requirements: Set[str] = None) -> WorkerNode:
        """Select node based on capabilities and load."""
        if requirements:
            capable_nodes = [n for n in nodes if requirements.issubset(n.capabilities)]
            if capable_nodes:
                return self._least_loaded_strategy(capable_nodes)
        
        return self._least_loaded_strategy(nodes)


class DistributedReflexionManager:
    """Manages distributed reflexion agent execution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize distributed reflexion manager.
        
        Args:
            config: Configuration for distributed processing
        """
        self.config = config or {}
        self.task_queue = TaskQueue()
        self.node_registry = NodeRegistry()
        self.task_distribution = TaskDistribution(self.task_queue, self.node_registry)
        self.logger = logging.getLogger(__name__)
        
        # Start distribution task
        self._distribution_task = None
        self.running = False
    
    async def start(self):
        """Start the distributed system."""
        if self.running:
            return
        
        self.running = True
        self._distribution_task = asyncio.create_task(
            self.task_distribution.distribute_tasks()
        )
        
        self.logger.info("Started distributed reflexion manager")
    
    async def stop(self):
        """Stop the distributed system."""
        if not self.running:
            return
        
        self.running = False
        
        if self._distribution_task:
            self._distribution_task.cancel()
            try:
                await self._distribution_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped distributed reflexion manager")
    
    def add_worker_node(
        self,
        node_id: str,
        host: str,
        port: int,
        capacity: int = 10,
        capabilities: Set[str] = None
    ) -> WorkerNode:
        """Add a worker node to the system."""
        
        node = WorkerNode(
            node_id=node_id,
            host=host,
            port=port,
            status=NodeStatus.HEALTHY,
            capacity=capacity,
            current_load=0,
            capabilities=capabilities or {"reflexion", "general"},
            last_heartbeat=datetime.now().isoformat(),
            started_at=datetime.now().isoformat(),
            metadata={}
        )
        
        self.node_registry.register_node(node)
        return node
    
    async def submit_reflexion_task(
        self,
        task: str,
        agent_config: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        requirements: Set[str] = None,
        max_retries: int = 3
    ) -> str:
        """Submit a reflexion task for distributed execution."""
        
        task_id = str(uuid.uuid4())
        
        distributed_task = DistributedTask(
            task_id=task_id,
            task_type="reflexion_task",
            priority=priority,
            payload={
                "task": task,
                "agent_config": agent_config
            },
            requirements=requirements or {"reflexion"},
            created_at=datetime.now().isoformat(),
            assigned_at=None,
            started_at=None,
            completed_at=None,
            status=TaskStatus.PENDING,
            assigned_node=None,
            result=None,
            error=None,
            retry_count=0,
            max_retries=max_retries
        )
        
        self.task_queue.add_task(distributed_task)
        
        self.logger.info(f"Submitted reflexion task {task_id}")
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task."""
        
        # Check completed tasks
        if task_id in self.task_queue.completed_tasks:
            task = self.task_queue.completed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "completed",
                "result": task.result,
                "execution_time": self._calculate_execution_time(task)
            }
        
        # Check failed tasks
        if task_id in self.task_queue.failed_tasks:
            task = self.task_queue.failed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "failed",
                "error": task.error,
                "retry_count": task.retry_count
            }
        
        # Check running/assigned tasks
        if task_id in self.task_queue.assigned_tasks:
            task = self.task_queue.assigned_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status.value,
                "assigned_node": task.assigned_node,
                "started_at": task.started_at
            }
        
        # Check pending tasks
        for task in self.task_queue.pending_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "queue_position": self.task_queue.pending_tasks.index(task) + 1
                }
        
        return None  # Task not found
    
    def _calculate_execution_time(self, task: DistributedTask) -> float:
        """Calculate total execution time for a task."""
        if not task.started_at or not task.completed_at:
            return 0.0
        
        start_time = datetime.fromisoformat(task.started_at)
        end_time = datetime.fromisoformat(task.completed_at)
        
        return (end_time - start_time).total_seconds()
    
    async def wait_for_task_completion(
        self,
        task_id: str,
        timeout: float = 300
    ) -> Dict[str, Any]:
        """Wait for a task to complete with timeout."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.get_task_result(task_id)
            
            if not result:
                raise ValueError(f"Task {task_id} not found")
            
            if result["status"] in ["completed", "failed"]:
                return result
            
            await asyncio.sleep(1)  # Check every second
        
        raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout}s")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Node statistics
        all_nodes = list(self.node_registry.nodes.values())
        node_stats = {
            "total_nodes": len(all_nodes),
            "healthy_nodes": len([n for n in all_nodes if n.status == NodeStatus.HEALTHY]),
            "busy_nodes": len([n for n in all_nodes if n.status == NodeStatus.BUSY]),
            "overloaded_nodes": len([n for n in all_nodes if n.status == NodeStatus.OVERLOADED]),
            "offline_nodes": len([n for n in all_nodes if n.status == NodeStatus.OFFLINE])
        }
        
        # Queue statistics
        queue_stats = self.task_queue.get_queue_status()
        
        # System capacity
        total_capacity = sum(node.capacity for node in all_nodes)
        current_load = sum(node.current_load for node in all_nodes)
        
        return {
            "running": self.running,
            "node_statistics": node_stats,
            "queue_statistics": queue_stats,
            "system_capacity": {
                "total_capacity": total_capacity,
                "current_load": current_load,
                "utilization": (current_load / max(1, total_capacity)) * 100
            },
            "timestamp": datetime.now().isoformat()
        }