"""Unit tests for optimized reflexion agents."""

import pytest
import asyncio
from unittest.mock import Mock, patch
import time

from src.reflexion import OptimizedReflexionAgent, AutoScalingReflexionAgent, ReflectionType


class TestOptimizedReflexionAgent:
    """Test cases for OptimizedReflexionAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            reflection_type=ReflectionType.BINARY,
            success_threshold=0.7,
            enable_caching=True,
            enable_parallel_reflection=True
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.llm == "gpt-4"
        assert self.agent.max_iterations == 2
        assert self.agent.enable_caching is True
        assert self.agent.enable_parallel_reflection is True
        assert hasattr(self.agent.engine, 'enable_caching')
    
    def test_basic_execution(self):
        """Test basic task execution."""
        result = self.agent.run("Test task")
        
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'output')
        assert hasattr(result, 'total_time')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'metadata')
    
    def test_caching_behavior(self):
        """Test that caching improves performance."""
        task = "Create a unique test task for caching"
        
        # First execution
        start_time = time.time()
        result1 = self.agent.run(task)
        first_time = time.time() - start_time
        
        # Second execution (should hit cache)
        start_time = time.time()
        result2 = self.agent.run(task)
        second_time = time.time() - start_time
        
        # Both should succeed
        assert result1.success
        assert result2.success
        
        # Second should be faster or same (due to caching)
        assert second_time <= first_time + 0.01  # Allow small tolerance
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        # Run a task to generate stats
        self.agent.run("Test task for stats")
        
        stats = self.agent.get_performance_stats()
        
        assert "engine_stats" in stats
        assert "cache_stats" in stats
        assert "throttling_stats" in stats
        assert "optimizations_enabled" in stats
        
        # Check cache stats structure
        cache_stats = stats["cache_stats"]
        assert "hit_rate" in cache_stats
        assert "size" in cache_stats
        assert "max_size" in cache_stats
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Run task to populate cache
        self.agent.run("Test task for cache clearing")
        
        # Get initial cache size
        initial_stats = self.agent.get_performance_stats()
        initial_size = initial_stats["cache_stats"]["size"]
        
        # Clear cache
        self.agent.clear_cache()
        
        # Check cache is cleared
        final_stats = self.agent.get_performance_stats()
        final_size = final_stats["cache_stats"]["size"]
        
        assert final_size == 0
    
    def test_optimization_modes(self):
        """Test different optimization modes."""
        # Test throughput optimization
        self.agent.optimize_for_throughput()
        assert self.agent.engine.enable_caching is True
        assert self.agent.engine.enable_parallel_reflection is True
        
        # Test accuracy optimization
        self.agent.optimize_for_accuracy()
        assert self.agent.engine.enable_parallel_reflection is True
        
        # Test cost optimization
        self.agent.optimize_for_cost()
        assert self.agent.engine.enable_caching is True
        assert self.agent.engine.enable_parallel_reflection is False
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=1
        ) as agent:
            result = agent.run("Test context manager")
            assert result is not None
        
        # Should complete without errors
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing functionality."""
        tasks = [
            "Task 1: Create a function",
            "Task 2: Write documentation", 
            "Task 3: Add error handling"
        ]
        
        results = await self.agent.run_batch(tasks)
        
        assert len(results) == len(tasks)
        for result in results:
            assert "task" in result
            assert "success" in result


class TestAutoScalingReflexionAgent:
    """Test cases for AutoScalingReflexionAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = AutoScalingReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            reflection_type=ReflectionType.BINARY
        )
    
    def test_initialization(self):
        """Test auto-scaling agent initialization."""
        assert hasattr(self.agent, 'load_metrics')
        assert hasattr(self.agent, 'scale_up_threshold')
        assert hasattr(self.agent, 'scale_down_threshold')
        assert hasattr(self.agent, 'min_workers')
        assert hasattr(self.agent, 'max_workers')
    
    def test_load_metrics_initialization(self):
        """Test load metrics are properly initialized."""
        metrics = self.agent.load_metrics
        
        assert "active_tasks" in metrics
        assert "queue_size" in metrics
        assert "avg_response_time" in metrics
        assert "success_rate" in metrics
        
        # Check initial values
        assert metrics["active_tasks"] == 0
        assert metrics["success_rate"] == 1.0
    
    def test_scaling_stats(self):
        """Test scaling statistics."""
        stats = self.agent.get_scaling_stats()
        
        assert "current_workers" in stats
        assert "min_workers" in stats
        assert "max_workers" in stats
        assert "load_metrics" in stats
        assert "scale_up_threshold" in stats
        assert "scale_down_threshold" in stats
    
    @pytest.mark.asyncio
    async def test_autoscaling_execution(self):
        """Test execution with auto-scaling."""
        result = await self.agent.run_with_autoscaling("Test auto-scaling task")
        
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'output')
    
    def test_load_metrics_update(self):
        """Test load metrics update mechanism."""
        initial_response_time = self.agent.load_metrics["avg_response_time"]
        initial_success_rate = self.agent.load_metrics["success_rate"]
        
        # Simulate metric update
        self.agent._update_load_metrics(1.5, True)
        
        # Check metrics were updated
        assert self.agent.load_metrics["avg_response_time"] != initial_response_time
        # Success rate should remain high with successful execution
        assert self.agent.load_metrics["success_rate"] >= initial_success_rate * 0.9


class TestPerformanceIntegration:
    """Integration tests for performance features."""
    
    def test_performance_comparison(self):
        """Compare performance between basic and optimized agents."""
        from src.reflexion import ReflexionAgent
        
        # Create both agent types
        basic_agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            reflection_type=ReflectionType.BINARY
        )
        
        optimized_agent = OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            reflection_type=ReflectionType.BINARY,
            enable_caching=True
        )
        
        task = "Performance comparison test task"
        
        # Run same task on both agents
        basic_result = basic_agent.run(task)
        optimized_result = optimized_agent.run(task)
        
        # Both should succeed
        assert basic_result.success
        assert optimized_result.success
        
        # Optimized should have additional metadata
        assert "optimization_enabled" in optimized_result.metadata
        assert "cache_enabled" in optimized_result.metadata
    
    def test_concurrent_execution_safety(self):
        """Test thread safety with concurrent execution."""
        import threading
        import queue
        
        agent = OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=1,
            enable_caching=True
        )
        
        results_queue = queue.Queue()
        
        def run_task(task_id):
            result = agent.run(f"Concurrent task {task_id}")
            results_queue.put((task_id, result.success))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 5
        # All tasks should succeed
        for task_id, success in results:
            assert success, f"Task {task_id} failed"


@pytest.fixture
def mock_performance_cache():
    """Mock performance cache for testing."""
    with patch('src.reflexion.core.performance.performance_cache') as mock_cache:
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.put.return_value = None
        mock_cache.get_stats.return_value = {
            "hit_rate": 0.25,
            "size": 3,
            "max_size": 1000,
            "hit_count": 1,
            "miss_count": 3,
            "utilization": 0.003
        }
        yield mock_cache


def test_with_mocked_cache(mock_performance_cache):
    """Test agent behavior with mocked cache."""
    agent = OptimizedReflexionAgent(
        llm="gpt-4",
        max_iterations=1,
        enable_caching=True
    )
    
    result = agent.run("Test with mocked cache")
    
    assert result.success
    # Verify cache interactions
    mock_performance_cache.get.assert_called()
    mock_performance_cache.put.assert_called()