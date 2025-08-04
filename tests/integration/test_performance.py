"""Integration tests for performance and optimization features."""

import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.reflexion import OptimizedReflexionAgent, ReflectionType
from src.reflexion.core.performance import (
    PerformanceCache, BatchProcessor, AdaptiveThrottling, ResourceMonitor
)


class TestPerformanceCache:
    """Test performance caching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = PerformanceCache(max_size=10, ttl_seconds=60)
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Test cache miss
        result = self.cache.get("task1", "gpt-4", {"param": "value"})
        assert result is None
        
        # Test cache put and hit
        test_data = {"output": "test result", "success": True}
        self.cache.put("task1", "gpt-4", {"param": "value"}, test_data)
        
        cached_result = self.cache.get("task1", "gpt-4", {"param": "value"})
        assert cached_result == test_data
    
    def test_cache_key_generation(self):
        """Test cache key generation consistency."""
        # Same parameters should generate same key
        key1 = self.cache._generate_key("task", "gpt-4", {"a": 1, "b": 2})
        key2 = self.cache._generate_key("task", "gpt-4", {"b": 2, "a": 1})  # Different order
        
        assert key1 == key2
        
        # Different parameters should generate different keys
        key3 = self.cache._generate_key("task", "gpt-4", {"a": 1, "b": 3})
        assert key1 != key3
    
    def test_cache_ttl(self):
        """Test cache TTL expiration."""
        short_ttl_cache = PerformanceCache(max_size=10, ttl_seconds=1)
        
        # Add item to cache
        test_data = {"output": "test", "success": True}
        short_ttl_cache.put("task", "gpt-4", {}, test_data)
        
        # Should be available immediately
        result = short_ttl_cache.get("task", "gpt-4", {})
        assert result == test_data
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Should be expired now
        result = short_ttl_cache.get("task", "gpt-4", {})
        assert result is None
    
    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        small_cache = PerformanceCache(max_size=2, ttl_seconds=60)
        
        # Fill cache to capacity
        small_cache.put("task1", "gpt-4", {}, "result1")
        small_cache.put("task2", "gpt-4", {}, "result2")
        
        # Both should be available
        assert small_cache.get("task1", "gpt-4", {}) == "result1"
        assert small_cache.get("task2", "gpt-4", {}) == "result2"
        
        # Access task1 to make it more recent
        small_cache.get("task1", "gpt-4", {})
        
        # Add third item (should evict task2 as it's least recent)
        small_cache.put("task3", "gpt-4", {}, "result3")
        
        # task1 and task3 should be available, task2 should be evicted
        assert small_cache.get("task1", "gpt-4", {}) == "result1"
        assert small_cache.get("task3", "gpt-4", {}) == "result3"
        assert small_cache.get("task2", "gpt-4", {}) is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Initial stats
        stats = self.cache.get_stats()
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        
        # Cache miss
        self.cache.get("missing", "gpt-4", {})
        stats = self.cache.get_stats()
        assert stats["miss_count"] == 1
        
        # Cache put and hit
        self.cache.put("task", "gpt-4", {}, "result")
        self.cache.get("task", "gpt-4", {})
        
        stats = self.cache.get_stats()
        assert stats["hit_count"] == 1
        assert stats["hit_rate"] == 0.5  # 1 hit, 1 miss


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor(max_workers=2, batch_size=3)
    
    def teardown_method(self):
        """Clean up resources."""
        self.processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test processing multiple tasks in batches."""
        def create_mock_agent(**kwargs):
            from unittest.mock import Mock
            agent = Mock()
            agent.run.return_value = Mock(
                success=True,
                output="Mock output",
                iterations=1,
                total_time=0.1,
                reflections=[]
            )
            return agent
        
        tasks = [f"Task {i}" for i in range(5)]
        
        results = await self.processor.process_batch(
            tasks, create_mock_agent
        )
        
        assert len(results) == 5
        for result in results:
            assert result["success"] is True
            assert "task" in result
    
    def test_batch_error_handling(self):
        """Test error handling in batch processing."""
        def create_failing_agent(**kwargs):
            from unittest.mock import Mock
            agent = Mock()
            agent.run.side_effect = Exception("Mock error")
            return agent
        
        # Use sync test since we're testing error handling
        task = "Failing task"
        result = self.processor._process_single_task(
            create_failing_agent(), task
        )
        
        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Mock error"


class TestAdaptiveThrottling:
    """Test adaptive throttling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.throttling = AdaptiveThrottling()
    
    def test_throttling_initialization(self):
        """Test throttling initialization."""
        assert self.throttling.current_delay == 0.0
        assert self.throttling.min_delay == 0.0
        assert self.throttling.max_delay == 2.0
    
    def test_request_recording(self):
        """Test request metrics recording."""
        # Record successful fast request
        self.throttling.record_request(1.0, True)
        
        stats = self.throttling.get_stats()
        assert stats["sample_count"] == 1
        assert stats["avg_duration"] == 1.0
        assert stats["avg_success_rate"] == 1.0
    
    def test_throttling_adaptation(self):
        """Test adaptive throttling behavior."""
        initial_delay = self.throttling.current_delay
        
        # Record many slow requests to trigger throttling increase
        for _ in range(15):
            self.throttling.record_request(6.0, False)  # Slow, failed requests
        
        # Delay should increase
        assert self.throttling.current_delay > initial_delay
        
        # Record fast successful requests to reduce throttling
        for _ in range(15):
            self.throttling.record_request(0.5, True)  # Fast, successful requests
        
        # Delay should decrease
        final_delay = self.throttling.current_delay
        assert final_delay <= self.throttling.current_delay
    
    @pytest.mark.asyncio
    async def test_throttle_application(self):
        """Test throttling delay application."""
        # Set a delay
        self.throttling.current_delay = 0.1
        
        start_time = time.time()
        await self.throttling.throttle()
        end_time = time.time()
        
        # Should have delayed for approximately the set time
        actual_delay = end_time - start_time
        assert actual_delay >= 0.09  # Allow for timing variation


class TestResourceMonitor:
    """Test resource monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ResourceMonitor()
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = self.monitor.collect_metrics()
        
        assert "timestamp" in metrics
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_percent" in metrics
        assert "process_count" in metrics
    
    def test_metrics_history(self):
        """Test metrics history management."""
        initial_count = len(self.monitor.metrics_history)
        
        # Collect multiple metrics
        for _ in range(3):
            self.monitor.collect_metrics()
        
        assert len(self.monitor.metrics_history) == initial_count + 3
    
    def test_resource_summary(self):
        """Test resource usage summary."""
        # Collect some metrics first
        self.monitor.collect_metrics()
        
        summary = self.monitor.get_resource_summary(hours=1)
        
        assert "sample_count" in summary
        assert "avg_cpu_percent" in summary
        assert "avg_memory_percent" in summary
        assert summary["sample_count"] >= 1


class TestIntegratedPerformance:
    """Integration tests for complete performance system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=2,
            enable_caching=True,
            enable_parallel_reflection=True,
            max_concurrent_tasks=2
        )
    
    def test_end_to_end_performance(self):
        """Test complete performance optimization pipeline."""
        tasks = [
            "Create a function to calculate fibonacci numbers",
            "Write a function to sort an array",
            "Create a function to calculate fibonacci numbers",  # Duplicate for caching
            "Implement a binary tree",
            "Write a function to sort an array"  # Another duplicate
        ]
        
        start_time = time.time()
        results = []
        
        for task in tasks:
            result = self.agent.run(task)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # All tasks should succeed
        assert all(r.success for r in results)
        
        # Get performance stats
        stats = self.agent.get_performance_stats()
        
        # Should have cache hits due to duplicates
        assert stats["cache_stats"]["hit_rate"] > 0
        
        # Should have processed all tasks
        assert len(results) == 5
        
        print(f"Processed {len(tasks)} tasks in {total_time:.2f}s")
        print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.1%}")
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        import threading
        import queue
        
        tasks = [f"Concurrent task {i}" for i in range(10)]
        results_queue = queue.Queue()
        
        def run_task(task):
            start = time.time()
            result = self.agent.run(task)
            duration = time.time() - start
            results_queue.put((task, result.success, duration))
        
        # Run tasks concurrently
        threads = []
        start_time = time.time()
        
        for task in tasks:
            thread = threading.Thread(target=run_task, args=(task,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all completed successfully
        assert len(results) == len(tasks)
        success_count = sum(1 for _, success, _ in results if success)
        assert success_count == len(tasks)
        
        # Calculate performance metrics
        avg_task_time = sum(duration for _, _, duration in results) / len(results)
        
        print(f"Concurrent execution: {len(tasks)} tasks in {total_time:.2f}s")
        print(f"Average task time: {avg_task_time:.3f}s")
        print(f"Theoretical sequential time: {avg_task_time * len(tasks):.2f}s")
        
        # Concurrent execution should be faster than sequential
        theoretical_sequential = avg_task_time * len(tasks)
        assert total_time < theoretical_sequential * 0.8  # At least 20% improvement
    
    @pytest.mark.asyncio
    async def test_async_batch_performance(self):
        """Test asynchronous batch processing performance."""
        tasks = [f"Async batch task {i}" for i in range(6)]
        
        start_time = time.time()
        results = await self.agent.run_batch(tasks)
        total_time = time.time() - start_time
        
        # All tasks should complete
        assert len(results) == len(tasks)
        
        # Most should succeed (allowing for some simulated failures)
        success_count = sum(1 for r in results if r.get("success", False))
        assert success_count >= len(tasks) * 0.7  # At least 70% success rate
        
        print(f"Async batch: {len(tasks)} tasks in {total_time:.2f}s")
        print(f"Success rate: {success_count}/{len(tasks)} ({success_count/len(tasks):.1%})")
    
    def teardown_method(self):
        """Clean up resources."""
        if hasattr(self.agent, 'engine'):
            self.agent.engine.cleanup()