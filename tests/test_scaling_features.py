"""Tests for scaling and distributed processing features."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from reflexion.scaling.auto_scaler import (
    AutoScaler, ScalingPolicy, ScalingThreshold, MetricsCollector,
    ScalingMetrics, MetricType, ScalingDirection
)
from reflexion.scaling.distributed import (
    DistributedReflexionManager, WorkerNode, DistributedTask,
    NodeStatus, TaskPriority, TaskStatus
)


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector."""
        return MetricsCollector()
    
    def test_register_custom_collector(self, metrics_collector):
        """Test custom metric collector registration."""
        def custom_metric():
            return 42.0
        
        metrics_collector.register_custom_collector("custom_metric", custom_metric)
        
        assert "custom_metric" in metrics_collector.custom_collectors
        assert metrics_collector.custom_collectors["custom_metric"]() == 42.0
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, metrics_collector):
        """Test metrics collection."""
        metrics = await metrics_collector.collect_metrics()
        
        assert isinstance(metrics, ScalingMetrics)
        assert metrics.cpu_utilization >= 0
        assert metrics.memory_utilization >= 0
        assert metrics.queue_length >= 0
        assert metrics.avg_response_time >= 0
        assert metrics.error_rate >= 0
        assert metrics.throughput >= 0
        assert metrics.timestamp


class TestAutoScaler:
    """Test auto-scaling functionality."""
    
    @pytest.fixture
    def scaling_policy(self):
        """Create test scaling policy."""
        thresholds = [
            ScalingThreshold(
                metric_type=MetricType.CPU_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=20.0,
                evaluation_periods=2,
                cooldown_seconds=60
            )
        ]
        
        return ScalingPolicy(
            name="test_policy",
            description="Test scaling policy",
            thresholds=thresholds,
            min_instances=1,
            max_instances=5,
            scale_up_increment=1,
            scale_down_increment=1,
            warmup_time=30,
            evaluation_interval=30
        )
    
    @pytest.fixture
    def auto_scaler(self, scaling_policy):
        """Create auto-scaler with test policy."""
        return AutoScaler([scaling_policy])
    
    def test_auto_scaler_initialization(self, auto_scaler):
        """Test auto-scaler initialization."""
        assert len(auto_scaler.policies) == 1
        assert auto_scaler.current_instances == 1
        assert auto_scaler.running == False
    
    def test_register_scaling_callback(self, auto_scaler):
        """Test scaling callback registration."""
        callback_called = []
        
        def callback(old, new):
            callback_called.append((old, new))
        
        auto_scaler.register_scaling_callback(callback)
        
        assert len(auto_scaler.scaling_callbacks) == 1
    
    def test_create_default_policy(self, auto_scaler):
        """Test default policy creation."""
        policy = auto_scaler.create_default_policy(
            name="default_test",
            min_instances=2,
            max_instances=8
        )
        
        assert policy.name == "default_test"
        assert policy.min_instances == 2
        assert policy.max_instances == 8
        assert len(policy.thresholds) > 0
        assert policy.enabled == True
    
    def test_should_scale_up(self, auto_scaler, scaling_policy):
        """Test scale-up decision."""
        # Create metrics that should trigger scale-up
        high_cpu_metrics = ScalingMetrics(
            cpu_utilization=85.0,  # Above threshold
            memory_utilization=50.0,
            queue_length=5,
            avg_response_time=1.0,
            error_rate=2.0,
            throughput=50.0,
            active_connections=20,
            timestamp=datetime.now().isoformat()
        )
        
        # Add metrics to history (need enough for evaluation periods)
        auto_scaler.decision_engine.metric_history = [high_cpu_metrics, high_cpu_metrics]
        
        decision = auto_scaler.decision_engine.should_scale(
            high_cpu_metrics, scaling_policy, 1
        )
        
        assert decision["should_scale"] == True
        assert decision["direction"] == ScalingDirection.UP
        assert decision["instances_change"] == 1
    
    def test_should_scale_down(self, auto_scaler, scaling_policy):
        """Test scale-down decision."""
        # Create metrics that should trigger scale-down
        low_cpu_metrics = ScalingMetrics(
            cpu_utilization=15.0,  # Below threshold
            memory_utilization=30.0,
            queue_length=0,
            avg_response_time=0.1,
            error_rate=0.0,
            throughput=10.0,
            active_connections=5,
            timestamp=datetime.now().isoformat()
        )
        
        # Add metrics to history
        auto_scaler.decision_engine.metric_history = [low_cpu_metrics, low_cpu_metrics]
        
        decision = auto_scaler.decision_engine.should_scale(
            low_cpu_metrics, scaling_policy, 3  # Current instances > min
        )
        
        assert decision["should_scale"] == True
        assert decision["direction"] == ScalingDirection.DOWN
        assert decision["instances_change"] == -1
    
    def test_should_not_scale_within_thresholds(self, auto_scaler, scaling_policy):
        """Test no scaling when metrics are within thresholds."""
        normal_metrics = ScalingMetrics(
            cpu_utilization=50.0,  # Between thresholds
            memory_utilization=50.0,
            queue_length=5,
            avg_response_time=0.5,
            error_rate=1.0,
            throughput=30.0,
            active_connections=15,
            timestamp=datetime.now().isoformat()
        )
        
        decision = auto_scaler.decision_engine.should_scale(
            normal_metrics, scaling_policy, 2
        )
        
        assert decision["should_scale"] == False
    
    def test_get_scaling_status(self, auto_scaler):
        """Test scaling status retrieval."""
        status = auto_scaler.get_scaling_status()
        
        assert "running" in status
        assert "current_instances" in status
        assert "active_policies" in status
        assert "recent_events" in status
        assert "current_metrics" in status


class TestDistributedReflexionManager:
    """Test distributed processing functionality."""
    
    @pytest.fixture
    def dist_manager(self):
        """Create distributed reflexion manager."""
        return DistributedReflexionManager()
    
    def test_initialization(self, dist_manager):
        """Test distributed manager initialization."""
        assert dist_manager.task_queue
        assert dist_manager.node_registry
        assert dist_manager.task_distribution
        assert dist_manager.running == False
    
    def test_add_worker_node(self, dist_manager):
        """Test worker node addition."""
        node = dist_manager.add_worker_node(
            node_id="test_node",
            host="localhost",
            port=8000,
            capacity=5,
            capabilities={"reflexion", "analysis"}
        )
        
        assert node.node_id == "test_node"
        assert node.host == "localhost"
        assert node.port == 8000
        assert node.capacity == 5
        assert "reflexion" in node.capabilities
        
        # Check node is registered
        assert "test_node" in dist_manager.node_registry.nodes
    
    @pytest.mark.asyncio
    async def test_submit_reflexion_task(self, dist_manager):
        """Test reflexion task submission."""
        task_id = await dist_manager.submit_reflexion_task(
            task="Test task",
            agent_config={"llm": "gpt-4", "max_iterations": 2},
            priority=TaskPriority.HIGH
        )
        
        assert task_id
        
        # Check task is in queue
        assert len(dist_manager.task_queue.pending_tasks) == 1
        task = dist_manager.task_queue.pending_tasks[0]
        assert task.task_id == task_id
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_task_result_pending(self, dist_manager):
        """Test getting result of pending task."""
        task_id = await dist_manager.submit_reflexion_task(
            "Test task", {"llm": "gpt-4"}
        )
        
        result = await dist_manager.get_task_result(task_id)
        
        assert result["task_id"] == task_id
        assert result["status"] == "pending"
        assert "queue_position" in result
    
    @pytest.mark.asyncio
    async def test_get_task_result_not_found(self, dist_manager):
        """Test getting result of non-existent task."""
        result = await dist_manager.get_task_result("nonexistent")
        assert result is None
    
    def test_get_system_status(self, dist_manager):
        """Test system status retrieval."""
        # Add a worker node
        dist_manager.add_worker_node("node1", "localhost", 8000, 3)
        
        status = dist_manager.get_system_status()
        
        assert "running" in status
        assert "node_statistics" in status
        assert "queue_statistics" in status
        assert "system_capacity" in status
        
        assert status["node_statistics"]["total_nodes"] == 1
        assert status["system_capacity"]["total_capacity"] == 3


class TestTaskQueue:
    """Test task queue functionality."""
    
    @pytest.fixture
    def task_queue(self):
        """Create task queue."""
        from reflexion.scaling.distributed import TaskQueue
        return TaskQueue()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample distributed task."""
        return DistributedTask(
            task_id="test_task",
            task_type="reflexion_task",
            priority=TaskPriority.NORMAL,
            payload={"task": "Test task"},
            requirements={"reflexion"},
            created_at=datetime.now().isoformat(),
            assigned_at=None,
            started_at=None,
            completed_at=None,
            status=TaskStatus.PENDING,
            assigned_node=None,
            result=None,
            error=None,
            retry_count=0,
            max_retries=3
        )
    
    def test_add_task(self, task_queue, sample_task):
        """Test task addition."""
        task_queue.add_task(sample_task)
        
        assert len(task_queue.pending_tasks) == 1
        assert task_queue.pending_tasks[0].task_id == "test_task"
    
    def test_get_next_task(self, task_queue, sample_task):
        """Test getting next available task."""
        task_queue.add_task(sample_task)
        
        # Get task with matching capabilities
        next_task = task_queue.get_next_task({"reflexion", "general"})
        
        assert next_task is not None
        assert next_task.task_id == "test_task"
        assert len(task_queue.pending_tasks) == 0  # Task removed from pending
    
    def test_get_next_task_no_capabilities(self, task_queue, sample_task):
        """Test getting next task without required capabilities."""
        task_queue.add_task(sample_task)
        
        # Try to get task without required capabilities
        next_task = task_queue.get_next_task({"other_capability"})
        
        assert next_task is None
        assert len(task_queue.pending_tasks) == 1  # Task remains in pending
    
    def test_assign_task(self, task_queue, sample_task):
        """Test task assignment."""
        task_queue.assign_task(sample_task, "worker_node_1")
        
        assert sample_task.status == TaskStatus.ASSIGNED
        assert sample_task.assigned_node == "worker_node_1"
        assert sample_task.assigned_at is not None
        assert "test_task" in task_queue.assigned_tasks
    
    def test_complete_task(self, task_queue, sample_task):
        """Test task completion."""
        # First assign the task
        task_queue.assign_task(sample_task, "worker_1")
        
        # Then complete it
        result = {"output": "Task completed", "success": True}
        task_queue.complete_task("test_task", result)
        
        assert "test_task" not in task_queue.assigned_tasks
        assert "test_task" in task_queue.completed_tasks
        
        completed_task = task_queue.completed_tasks["test_task"]
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result == result
    
    def test_fail_task_with_retry(self, task_queue, sample_task):
        """Test task failure with retry."""
        task_queue.assign_task(sample_task, "worker_1")
        
        # Fail the task (should retry since retry_count < max_retries)
        task_queue.fail_task("test_task", "Simulated failure")
        
        assert "test_task" not in task_queue.assigned_tasks
        assert len(task_queue.pending_tasks) == 1  # Task back in pending for retry
        
        retry_task = task_queue.pending_tasks[0]
        assert retry_task.retry_count == 1
        assert retry_task.status == TaskStatus.PENDING
    
    def test_fail_task_max_retries(self, task_queue, sample_task):
        """Test task failure after max retries."""
        # Set task to max retries
        sample_task.retry_count = sample_task.max_retries
        task_queue.assign_task(sample_task, "worker_1")
        
        # Fail the task (should not retry)
        task_queue.fail_task("test_task", "Final failure")
        
        assert "test_task" not in task_queue.assigned_tasks
        assert len(task_queue.pending_tasks) == 0  # No retry
        assert "test_task" in task_queue.failed_tasks
        
        failed_task = task_queue.failed_tasks["test_task"]
        assert failed_task.status == TaskStatus.FAILED
        assert failed_task.error == "Final failure"
    
    def test_get_queue_status(self, task_queue, sample_task):
        """Test queue status retrieval."""
        # Add tasks in different states
        task_queue.add_task(sample_task)
        
        task2 = DistributedTask(
            task_id="task2", task_type="test", priority=TaskPriority.HIGH,
            payload={}, requirements=set(), created_at=datetime.now().isoformat(),
            assigned_at=None, started_at=None, completed_at=None,
            status=TaskStatus.PENDING, assigned_node=None, result=None,
            error=None, retry_count=0, max_retries=3
        )
        task_queue.assign_task(task2, "worker_1")
        
        status = task_queue.get_queue_status()
        
        assert status["pending_tasks"] == 1
        assert status["assigned_tasks"] == 1
        assert status["completed_tasks"] == 0
        assert status["failed_tasks"] == 0
        assert status["total_tasks"] == 2


@pytest.mark.integration
class TestScalingIntegration:
    """Integration tests for scaling features."""
    
    @pytest.mark.asyncio
    async def test_auto_scaler_with_distributed_system(self):
        """Test auto-scaler integration with distributed system."""
        # Create distributed system
        dist_manager = DistributedReflexionManager()
        
        # Add worker nodes
        dist_manager.add_worker_node("worker_1", "localhost", 8001, 3)
        dist_manager.add_worker_node("worker_2", "localhost", 8002, 3)
        
        # Create auto-scaler with callback that manages worker nodes
        scaling_events = []
        
        def scaling_callback(old_instances, new_instances):
            scaling_events.append({
                "old": old_instances,
                "new": new_instances,
                "timestamp": datetime.now().isoformat()
            })
            
            # In real implementation, this would add/remove worker nodes
            if new_instances > old_instances:
                # Scale up - add more nodes
                for i in range(old_instances, new_instances):
                    node_id = f"auto_worker_{i+1}"
                    dist_manager.add_worker_node(
                        node_id, "localhost", 8100 + i, 2
                    )
            elif new_instances < old_instances:
                # Scale down - remove nodes (simplified)
                pass
        
        auto_scaler = AutoScaler([])
        policy = auto_scaler.create_default_policy(min_instances=2, max_instances=6)
        auto_scaler.policies["test_policy"] = policy
        auto_scaler.register_scaling_callback(scaling_callback)
        
        # Test scaling decision
        high_load_metrics = ScalingMetrics(
            cpu_utilization=85.0, memory_utilization=80.0, queue_length=20,
            avg_response_time=2.0, error_rate=1.0, throughput=100.0,
            active_connections=50, timestamp=datetime.now().isoformat()
        )
        
        # Simulate scaling decision
        auto_scaler.decision_engine.metric_history = [high_load_metrics] * 3
        decision = auto_scaler.decision_engine.should_scale(
            high_load_metrics, policy, 2
        )
        
        if decision["should_scale"] and decision["direction"] == ScalingDirection.UP:
            # Simulate scaling execution
            old_instances = auto_scaler.current_instances
            new_instances = old_instances + decision["instances_change"]
            scaling_callback(old_instances, new_instances)
            auto_scaler.current_instances = new_instances
        
        # Verify scaling occurred
        assert len(scaling_events) >= 1
        
        # Verify worker nodes were added
        initial_nodes = 2  # worker_1 and worker_2
        current_nodes = len(dist_manager.node_registry.nodes)
        assert current_nodes >= initial_nodes
    
    @pytest.mark.asyncio
    async def test_distributed_system_under_load(self):
        """Test distributed system behavior under load."""
        dist_manager = DistributedReflexionManager()
        
        # Add multiple worker nodes
        for i in range(3):
            dist_manager.add_worker_node(
                f"worker_{i+1}", "localhost", 8000 + i, 2
            )
        
        await dist_manager.start()
        
        try:
            # Submit multiple tasks
            task_ids = []
            for i in range(6):  # More tasks than total capacity
                task_id = await dist_manager.submit_reflexion_task(
                    f"Load test task {i+1}",
                    {"llm": "gpt-4", "max_iterations": 1},
                    TaskPriority.NORMAL
                )
                task_ids.append(task_id)
            
            # Wait a moment for task distribution
            await asyncio.sleep(0.5)
            
            # Check system status
            status = dist_manager.get_system_status()
            
            # Should have distributed tasks across nodes
            assert status["queue_statistics"]["pending_tasks"] + status["queue_statistics"]["assigned_tasks"] > 0
            assert status["system_capacity"]["utilization"] > 0
            
        finally:
            await dist_manager.stop()