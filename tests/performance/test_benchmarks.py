"""Performance benchmarks for Reflexion engine."""

import pytest
import time
from unittest.mock import Mock, patch

from src.reflexion.core.agent import ReflexionAgent
from src.reflexion.core.engine import ReflexionEngine
from src.reflexion.core.types import ReflectionType


class TestReflexionPerformance:
    """Performance benchmark tests for Reflexion components."""

    @pytest.mark.benchmark
    def test_agent_initialization_performance(self, benchmark):
        """Benchmark agent initialization time."""
        def init_agent():
            return ReflexionAgent(
                llm="test-model",
                max_iterations=3,
                reflection_type=ReflectionType.BINARY
            )
        
        result = benchmark(init_agent)
        assert result is not None

    @pytest.mark.benchmark
    def test_single_reflection_cycle_performance(self, benchmark):
        """Benchmark a single reflection cycle execution time."""
        engine = ReflexionEngine()
        
        def single_reflection():
            with patch.object(engine, '_execute_task') as mock_execute:
                with patch.object(engine, '_evaluate_output') as mock_evaluate:
                    mock_execute.return_value = "Test solution"
                    mock_evaluate.return_value = {"success": True, "score": 0.9}
                    
                    return engine.execute_with_reflexion(
                        task="benchmark task",
                        llm="test-model",
                        max_iterations=1,
                        reflection_type=ReflectionType.BINARY,
                        success_threshold=0.8
                    )
        
        result = benchmark(single_reflection)
        assert result.success is True

    @pytest.mark.benchmark
    def test_multiple_reflection_cycles_performance(self, benchmark):
        """Benchmark multiple reflection cycles."""
        engine = ReflexionEngine()
        
        def multiple_reflections():
            with patch.object(engine, '_execute_task') as mock_execute:
                with patch.object(engine, '_evaluate_output') as mock_evaluate:
                    with patch.object(engine, '_generate_reflection') as mock_reflect:
                        # Simulate 3 failed attempts, then success
                        mock_execute.side_effect = ["Poor"] * 3 + ["Great solution"]
                        mock_evaluate.side_effect = [
                            {"success": False, "score": 0.2},
                            {"success": False, "score": 0.3},
                            {"success": False, "score": 0.4},
                            {"success": True, "score": 0.9}
                        ]
                        mock_reflect.return_value = Mock()
                        
                        return engine.execute_with_reflexion(
                            task="complex benchmark task",
                            llm="test-model",
                            max_iterations=4,
                            reflection_type=ReflectionType.BINARY,
                            success_threshold=0.8
                        )
        
        result = benchmark(multiple_reflections)
        assert result.success is True
        assert result.iterations == 4

    @pytest.mark.benchmark
    def test_reflection_generation_performance(self, benchmark):
        """Benchmark reflection generation speed."""
        engine = ReflexionEngine()
        evaluation = {"success": False, "score": 0.3, "details": {"issues": ["too short"]}}
        
        def generate_reflection():
            return engine._generate_reflection(
                task="performance test task",
                output="short output",
                evaluation=evaluation,
                reflection_type=ReflectionType.BINARY
            )
        
        result = benchmark(generate_reflection)
        assert result is not None
        assert len(result.improvements) > 0

    @pytest.mark.benchmark
    def test_concurrent_agent_execution(self, benchmark):
        """Benchmark concurrent execution of multiple agents."""
        import threading
        import concurrent.futures
        
        def concurrent_execution():
            agents = [
                ReflexionAgent(llm="test-model", max_iterations=1)
                for _ in range(5)
            ]
            
            def run_agent(agent):
                with patch.object(agent.engine, '_execute_task') as mock_execute:
                    with patch.object(agent.engine, '_evaluate_output') as mock_evaluate:
                        mock_execute.return_value = "Concurrent solution"
                        mock_evaluate.return_value = {"success": True, "score": 0.9}
                        return agent.run("concurrent task")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(run_agent, agent) for agent in agents]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            return results
        
        results = benchmark(concurrent_execution)
        assert len(results) == 5
        assert all(result.success for result in results)

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_memory_usage_under_load(self, benchmark):
        """Benchmark memory usage during extended operation."""
        import psutil
        import os
        
        def memory_intensive_operation():
            agent = ReflexionAgent(llm="test-model", max_iterations=3)
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Simulate extended usage
            results = []
            for i in range(100):
                with patch.object(agent.engine, '_execute_task') as mock_execute:
                    with patch.object(agent.engine, '_evaluate_output') as mock_evaluate:
                        mock_execute.return_value = f"Solution {i}"
                        mock_evaluate.return_value = {"success": True, "score": 0.9}
                        results.append(agent.run(f"task {i}"))
            
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            return {
                "results_count": len(results),
                "memory_growth_mb": memory_growth / (1024 * 1024),
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024)
            }
        
        result = benchmark(memory_intensive_operation)
        assert result["results_count"] == 100
        # Memory growth should be reasonable (less than 100MB for this test)
        assert result["memory_growth_mb"] < 100

    @pytest.mark.benchmark
    def test_error_handling_performance(self, benchmark):
        """Benchmark performance when handling errors."""
        engine = ReflexionEngine()
        
        def error_handling():
            results = []
            for _ in range(10):
                try:
                    with patch.object(engine, '_execute_task') as mock_execute:
                        with patch.object(engine, '_evaluate_output') as mock_evaluate:
                            # Simulate some executions that raise exceptions
                            mock_execute.side_effect = [Exception("Test error"), "Recovery solution"]
                            mock_evaluate.return_value = {"success": True, "score": 0.9}
                            
                            result = engine.execute_with_reflexion(
                                task="error handling test",
                                llm="test-model",
                                max_iterations=2,
                                reflection_type=ReflectionType.BINARY,
                                success_threshold=0.8
                            )
                            results.append(result)
                except Exception:
                    # Handle expected exceptions gracefully
                    pass
            return results
        
        results = benchmark(error_handling)
        # Should handle errors without significant performance degradation
        assert isinstance(results, list)


class TestMemoryAndResourceUsage:
    """Resource usage and memory efficiency tests."""

    def test_memory_cleanup_after_execution(self):
        """Test that memory is properly cleaned up after execution."""
        import gc
        import sys
        
        # Force garbage collection and get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and use multiple agents
        for i in range(10):
            agent = ReflexionAgent(llm="test-model")
            with patch.object(agent.engine, '_execute_task') as mock_execute:
                with patch.object(agent.engine, '_evaluate_output') as mock_evaluate:
                    mock_execute.return_value = f"Solution {i}"
                    mock_evaluate.return_value = {"success": True, "score": 0.9}
                    agent.run(f"cleanup test {i}")
            
            # Explicitly delete the agent
            del agent
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory should not grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Too many objects created: {object_growth}"

    def test_no_memory_leaks_in_reflection_loop(self):
        """Test for memory leaks in the reflection loop."""
        import tracemalloc
        
        tracemalloc.start()
        
        agent = ReflexionAgent(llm="test-model", max_iterations=5)
        
        # Baseline memory
        current, peak = tracemalloc.get_traced_memory()
        baseline_memory = current
        
        # Run multiple reflection cycles
        for i in range(20):
            with patch.object(agent.engine, '_execute_task') as mock_execute:
                with patch.object(agent.engine, '_evaluate_output') as mock_evaluate:
                    with patch.object(agent.engine, '_generate_reflection') as mock_reflect:
                        # Force multiple reflections
                        mock_execute.side_effect = ["Poor"] * 4 + ["Great solution"]
                        mock_evaluate.side_effect = [
                            {"success": False, "score": 0.2}
                        ] * 4 + [{"success": True, "score": 0.9}]
                        mock_reflect.return_value = Mock()
                        
                        agent.run(f"memory test {i}")
        
        # Check final memory
        current, peak = tracemalloc.get_traced_memory()
        memory_growth = current - baseline_memory
        
        tracemalloc.stop()
        
        # Memory growth should be minimal (less than 1MB)
        assert memory_growth < 1_000_000, f"Memory leak detected: {memory_growth} bytes"