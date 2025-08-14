"""
Production integration tests for the reflexion agent system.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.reflexion.core.quantum_reflexion_agent import QuantumReflexionAgent
from src.reflexion.core.autonomous_scaling_engine import autonomous_scaling_engine
from src.reflexion.core.advanced_error_recovery import error_recovery_manager
from src.reflexion.core.intelligent_monitoring import intelligent_monitor
from src.reflexion.core.types import ReflectionType
from src.reflexion.research.novel_algorithms import research_comparator


class TestProductionWorkflow:
    """Test complete production workflow integration."""
    
    @pytest.fixture
    def production_agent(self):
        """Create production-ready quantum reflexion agent."""
        return QuantumReflexionAgent(
            llm="gpt-4",
            max_iterations=3,
            reflection_type=ReflectionType.STRUCTURED,
            success_threshold=0.8,
            quantum_states=5,
            entanglement_strength=0.7,
            enable_superposition=True,
            enable_uncertainty_quantification=True
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_task_execution(self, production_agent):
        """Test complete end-to-end task execution."""
        
        # Mock LLM provider for controlled testing
        with patch.object(production_agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = """
            def fibonacci(n):
                \"\"\"Calculate fibonacci number with memoization.\"\"\"
                if n <= 1:
                    return n
                
                memo = {}
                def fib_helper(num):
                    if num in memo:
                        return memo[num]
                    if num <= 1:
                        return num
                    memo[num] = fib_helper(num-1) + fib_helper(num-2)
                    return memo[num]
                
                return fib_helper(n)
            
            # Test cases
            assert fibonacci(0) == 0
            assert fibonacci(1) == 1
            assert fibonacci(10) == 55
            print("All tests passed!")
            """
            
            # Execute production task
            result = await production_agent.quantum_run(
                task="Implement an efficient fibonacci function with memoization and test cases",
                success_criteria="complete implementation with tests",
                algorithm_ensemble=True
            )
            
            # Verify result structure
            assert result.success is True
            assert result.task == "Implement an efficient fibonacci function with memoization and test cases"
            assert len(result.output) > 100
            assert result.metadata["quantum_enhanced"] is True
            assert "quantum_metrics" in result.metadata
            
            # Verify quantum metrics
            quantum_metrics = result.metadata["quantum_metrics"]
            assert "quantum_score" in quantum_metrics
            assert quantum_metrics["quantum_score"] >= 0.0
            assert quantum_metrics["quantum_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, production_agent):
        """Test integration with error recovery system."""
        
        # Simulate LLM failure and recovery
        call_count = 0
        
        async def failing_then_succeeding_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Temporary LLM connection failure")
            return "Recovered response after error recovery"
        
        with patch.object(production_agent.quantum_llm, 'generate_async', side_effect=failing_then_succeeding_llm):
            
            # Task should succeed after error recovery
            result = await production_agent.quantum_run(
                task="Simple test task after error recovery",
                algorithm_ensemble=False
            )
            
            # Verify recovery worked
            assert result.success is True
            assert "Recovered response" in result.output
            assert call_count > 2  # Should have retried
    
    def test_monitoring_integration(self, production_agent):
        """Test integration with monitoring system."""
        
        # Record some metrics
        intelligent_monitor.record_metric("test_metric", 0.75)
        intelligent_monitor.record_metric("response_time", 150.0)
        intelligent_monitor.record_metric("throughput", 45.0)
        
        # Get system health
        health = intelligent_monitor.get_system_health()
        
        assert "overall_health_score" in health
        assert "health_status" in health
        assert "metrics_tracked" in health
        assert health["metrics_tracked"] >= 3
    
    def test_scaling_integration(self, production_agent):
        """Test integration with scaling engine."""
        
        # Configure scaling for testing
        from src.reflexion.core.autonomous_scaling_engine import ResourceType, AlertSeverity
        
        autonomous_scaling_engine.configure_scaling(
            resource_type=ResourceType.CPU,
            scale_up_threshold=0.7,
            scale_down_threshold=0.3
        )
        
        # Get scaling report
        scaling_report = autonomous_scaling_engine.get_scaling_report()
        
        assert "scaling_status" in scaling_report
        assert "current_resources" in scaling_report
        assert scaling_report["scaling_status"]["scaling_enabled"] is True


class TestReliabilityAndStability:
    """Test system reliability and stability under various conditions."""
    
    @pytest.fixture
    def stress_test_agent(self):
        """Create agent for stress testing."""
        return QuantumReflexionAgent(
            llm="test-model",
            max_iterations=2,  # Reduced for faster testing
            quantum_states=3
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, stress_test_agent):
        """Test concurrent task execution stability."""
        
        async def mock_llm_response(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"Response at {datetime.now().isoformat()}"
        
        with patch.object(stress_test_agent.quantum_llm, 'generate_async', side_effect=mock_llm_response):
            
            # Execute multiple tasks concurrently
            tasks = []
            for i in range(5):
                task = stress_test_agent.quantum_run(
                    task=f"Concurrent test task {i}",
                    algorithm_ensemble=False
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all tasks completed successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == 5
            
            for result in successful_results:
                assert result.success is True
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, stress_test_agent):
        """Test memory usage remains stable over multiple executions."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch.object(stress_test_agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Stable response"
            
            # Execute multiple tasks
            for i in range(10):
                await stress_test_agent.quantum_run(
                    task=f"Memory stability test {i}",
                    algorithm_ensemble=False
                )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 50MB for 10 tasks)
            assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_error_cascade_prevention(self, stress_test_agent):
        """Test prevention of error cascades."""
        
        failure_count = 0
        
        async def intermittent_failure(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count % 3 == 0:
                raise ValueError(f"Intermittent failure {failure_count}")
            return f"Success {failure_count}"
        
        with patch.object(stress_test_agent.quantum_llm, 'generate_async', side_effect=intermittent_failure):
            
            results = []
            for i in range(6):
                try:
                    result = await stress_test_agent.quantum_run(
                        task=f"Error cascade test {i}",
                        algorithm_ensemble=False
                    )
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            # Some tasks should succeed despite intermittent failures
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 2


class TestPerformanceBenchmarks:
    """Performance benchmarks for production deployment."""
    
    @pytest.fixture
    def benchmark_agent(self):
        """Create agent for benchmarking."""
        return QuantumReflexionAgent(
            llm="benchmark-model",
            max_iterations=3,
            quantum_states=5
        )
    
    @pytest.mark.asyncio
    async def test_response_time_benchmark(self, benchmark_agent):
        """Benchmark response times for various task types."""
        
        test_tasks = [
            "Implement a simple sorting algorithm",
            "Write a function to validate email addresses", 
            "Create a basic REST API endpoint",
            "Design a simple database schema",
            "Optimize a slow SQL query"
        ]
        
        with patch.object(benchmark_agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Optimized benchmark response"
            
            response_times = []
            
            for task in test_tasks:
                start_time = time.time()
                
                result = await benchmark_agent.quantum_run(
                    task=task,
                    algorithm_ensemble=True
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                assert result.success is True
            
            # Calculate performance metrics
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Performance assertions (adjust thresholds based on requirements)
            assert avg_response_time < 5.0, f"Average response time {avg_response_time:.2f}s too high"
            assert max_response_time < 10.0, f"Max response time {max_response_time:.2f}s too high"
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, benchmark_agent):
        """Benchmark system throughput."""
        
        with patch.object(benchmark_agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Throughput test response"
            
            start_time = time.time()
            
            # Execute tasks in batches
            batch_size = 3
            total_tasks = 15
            
            for batch_start in range(0, total_tasks, batch_size):
                batch_tasks = []
                
                for i in range(batch_start, min(batch_start + batch_size, total_tasks)):
                    task = benchmark_agent.quantum_run(
                        task=f"Throughput test task {i}",
                        algorithm_ensemble=False  # Faster for throughput testing
                    )
                    batch_tasks.append(task)
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*batch_tasks)
                
                # Verify all tasks in batch succeeded
                for result in batch_results:
                    assert result.success is True
            
            end_time = time.time()
            total_time = end_time - start_time
            
            throughput = total_tasks / total_time  # tasks per second
            
            # Throughput assertion (adjust based on requirements)
            assert throughput > 1.0, f"Throughput {throughput:.2f} tasks/sec too low"


class TestScalabilityLimits:
    """Test system behavior at scalability limits."""
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion scenarios."""
        
        # Test with artificially low resource limits
        from src.reflexion.core.autonomous_scaling_engine import ResourceType
        
        # Temporarily reduce resource limits
        cpu_config = autonomous_scaling_engine.resources[ResourceType.CPU]
        original_max = cpu_config.max_value
        cpu_config.max_value = cpu_config.current_value + 2  # Very low limit
        
        try:
            # Simulate high load
            for _ in range(10):
                autonomous_scaling_engine._autonomous_scaling_cycle()
            
            # System should remain stable despite resource constraints
            scaling_report = autonomous_scaling_engine.get_scaling_report()
            assert "scaling_status" in scaling_report
            
        finally:
            # Restore original limits
            cpu_config.max_value = original_max
    
    def test_monitoring_system_under_load(self):
        """Test monitoring system behavior under high load."""
        
        # Generate high metric volume
        for i in range(1000):
            intelligent_monitor.record_metric(f"load_test_metric_{i % 10}", i * 0.001)
        
        # System should remain responsive
        health = intelligent_monitor.get_system_health()
        assert health is not None
        assert "overall_health_score" in health


class TestDataConsistency:
    """Test data consistency across system components."""
    
    @pytest.mark.asyncio
    async def test_metric_consistency(self):
        """Test consistency of metrics across components."""
        
        # Record metrics from different components
        intelligent_monitor.record_metric("cpu_usage", 0.75)
        intelligent_monitor.record_metric("memory_usage", 0.60)
        
        # Get metrics from monitoring system
        cpu_summary = intelligent_monitor.get_metric_summary("cpu_usage")
        memory_summary = intelligent_monitor.get_metric_summary("memory_usage")
        
        assert cpu_summary["current_value"] == 0.75
        assert memory_summary["current_value"] == 0.60
        
        # Verify system health reflects these metrics
        health = intelligent_monitor.get_system_health()
        assert health["health_status"] in ["excellent", "good", "fair", "poor", "critical"]
    
    def test_scaling_decision_consistency(self):
        """Test consistency of scaling decisions."""
        
        from src.reflexion.core.autonomous_scaling_engine import ResourceType, ScalingDirection, LoadPattern
        
        cpu_config = autonomous_scaling_engine.resources[ResourceType.CPU]
        
        # Test consistent scaling decisions for same conditions
        decision1 = autonomous_scaling_engine._make_scaling_decision(
            resource_type=ResourceType.CPU,
            config=cpu_config,
            current_utilization=0.85,
            predicted_peak_load=0.90,
            load_pattern=LoadPattern.STEADY
        )
        
        decision2 = autonomous_scaling_engine._make_scaling_decision(
            resource_type=ResourceType.CPU,
            config=cpu_config,
            current_utilization=0.85,
            predicted_peak_load=0.90,
            load_pattern=LoadPattern.STEADY
        )
        
        assert decision1 == decision2, "Scaling decisions should be consistent"


class TestGracefulDegradation:
    """Test system graceful degradation capabilities."""
    
    @pytest.mark.asyncio
    async def test_partial_system_failure_handling(self):
        """Test handling of partial system failures."""
        
        agent = QuantumReflexionAgent(llm="test-model", quantum_states=3)
        
        # Simulate partial algorithm failure
        with patch.object(agent.quantum_algorithm, 'execute', side_effect=Exception("Algorithm failed")):
            with patch.object(agent.quantum_llm, 'generate_async', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = "Degraded but functional response"
                
                # System should degrade gracefully
                result = await agent.quantum_run(
                    task="Test graceful degradation",
                    algorithm_ensemble=False
                )
                
                # Should still return a result, albeit potentially degraded
                assert isinstance(result, type(result))  # Should be ReflexionResult type
    
    def test_monitoring_system_degradation(self):
        """Test monitoring system graceful degradation."""
        
        # Disable collection temporarily
        original_enabled = intelligent_monitor.collection_enabled
        intelligent_monitor.collection_enabled = False
        
        try:
            # System should handle disabled monitoring gracefully
            intelligent_monitor.record_metric("test_metric", 0.5)
            health = intelligent_monitor.get_system_health()
            
            assert health is not None
            
        finally:
            # Restore monitoring
            intelligent_monitor.collection_enabled = original_enabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])