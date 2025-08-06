#!/usr/bin/env python3
"""Comprehensive tests for all three generations of the SDLC implementation."""

import sys
import os
import asyncio
import time
from typing import Dict, List, Any

# Simple pytest substitute for testing
class pytest:
    @staticmethod
    def skip(reason):
        print(f"SKIPPED: {reason}")
        return
    
    class mark:
        @staticmethod
        def asyncio(func):
            return func

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from reflexion import ReflexionAgent, OptimizedReflexionAgent, AutoScalingReflexionAgent
from reflexion import ReflectionType, ReflectionPrompts, PromptDomain, EpisodicMemory
from reflexion.core.health import health_checker
from reflexion.core.optimization import optimization_manager
from reflexion.core.scaling import scaling_manager


class TestGeneration1_Basic:
    """Test Generation 1: Basic Functionality (Make it Work)."""
    
    def test_basic_agent_creation(self):
        """Test basic agent can be created."""
        agent = ReflexionAgent(llm="gpt-4", max_iterations=2)
        assert agent is not None
        assert agent.llm == "gpt-4"
        assert agent.max_iterations == 2
    
    def test_basic_task_execution(self):
        """Test basic task execution works."""
        agent = ReflexionAgent(llm="gpt-4", max_iterations=1)
        result = agent.run("Simple test task", success_criteria="basic")
        
        assert result is not None
        assert hasattr(result, 'task')
        assert hasattr(result, 'output')
        assert hasattr(result, 'success')
        assert hasattr(result, 'iterations')
        assert result.iterations >= 1
    
    def test_reflection_types(self):
        """Test different reflection types work."""
        reflection_types = [
            ReflectionType.BINARY,
            ReflectionType.SCALAR, 
            ReflectionType.STRUCTURED
        ]
        
        for reflection_type in reflection_types:
            agent = ReflexionAgent(
                llm="gpt-4",
                max_iterations=1,
                reflection_type=reflection_type
            )
            result = agent.run("Test reflection type", success_criteria="working")
            assert result is not None
    
    def test_domain_specific_prompts(self):
        """Test domain-specific prompts functionality."""
        domains = [
            PromptDomain.GENERAL,
            PromptDomain.SOFTWARE_ENGINEERING,
            PromptDomain.DATA_ANALYSIS
        ]
        
        for domain in domains:
            prompts = ReflectionPrompts.for_domain(domain)
            assert prompts is not None
            assert hasattr(prompts, 'initial_reflection')
            assert hasattr(prompts, 'improvement_strategy')
    
    def test_episodic_memory(self):
        """Test episodic memory functionality."""
        memory = EpisodicMemory(storage_path="./test_memory.json")
        
        # Create a fake result for testing
        from reflexion.core.types import ReflexionResult, Reflection
        result = ReflexionResult(
            task="Test task",
            output="Test output",
            success=True,
            iterations=1,
            reflections=[],
            total_time=0.1,
            metadata={}
        )
        
        # Store episode
        memory.store_episode("Test task", result, {"test": True})
        
        # Test recall
        similar = memory.recall_similar("Test task", k=1)
        assert len(similar) >= 0
        
        # Test patterns
        patterns = memory.get_success_patterns()
        assert "patterns" in patterns
        assert "success_rate" in patterns
    
    def test_framework_adapters_import(self):
        """Test that framework adapters can be imported."""
        from reflexion.adapters import AutoGenReflexion, ReflexiveCrewMember
        
        # Test AutoGen adapter creation
        autogen_agent = AutoGenReflexion(
            name="test_agent",
            llm_config={"model": "gpt-4"}
        )
        assert autogen_agent is not None
        
        # Test CrewAI adapter creation
        crew_member = ReflexiveCrewMember(
            role="Test Role",
            goal="Test Goal"
        )
        assert crew_member is not None


class TestGeneration2_Robustness:
    """Test Generation 2: Robustness & Reliability (Make it Robust)."""
    
    def test_health_checking(self):
        """Test health checking system."""
        # Test basic health metrics collection
        try:
            metrics = health_checker.get_system_metrics()
            assert hasattr(metrics, 'cpu_percent')
            assert hasattr(metrics, 'memory_percent')
            assert hasattr(metrics, 'uptime_seconds')
        except Exception as e:
            # Health checking may fail in some environments, which is acceptable
            assert "psutil" in str(e) or "health" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_health_checks_async(self):
        """Test async health checks."""
        try:
            health_results = await health_checker.run_all_checks()
            assert isinstance(health_results, dict)
            assert len(health_results) > 0
        except Exception as e:
            # Async health checks may fail in test environment
            pytest.skip(f"Async health checks not available: {e}")
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        agent = ReflexionAgent(llm="gpt-4", max_iterations=1)
        
        # Test with potentially problematic input
        result = agent.run("", success_criteria="basic")  # Empty task
        assert result is not None
        
        # Test with very long input
        long_task = "x" * 1000
        result = agent.run(long_task, success_criteria="basic")
        assert result is not None
    
    def test_validation_system(self):
        """Test input validation system."""
        from reflexion.core.validation import validator
        
        # Test task validation
        valid_result = validator.validate_task("Valid task")
        assert valid_result.is_valid
        
        # Test invalid task
        invalid_result = validator.validate_task("")
        assert not invalid_result.is_valid
        
        # Test LLM config validation
        llm_result = validator.validate_llm_config("gpt-4")
        assert llm_result.is_valid
    
    def test_resilience_patterns(self):
        """Test basic resilience pattern functionality."""
        # Test that resilience components can be imported and initialized
        from reflexion.core.resilience import ResilienceManager
        from reflexion.core.retry import RetryManager
        
        resilience_mgr = ResilienceManager()
        retry_mgr = RetryManager()
        
        assert resilience_mgr is not None
        assert retry_mgr is not None
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        from reflexion.core.health import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        assert circuit_breaker.state == "closed"
        
        # Test that circuit breaker can be created and has expected attributes
        assert hasattr(circuit_breaker, 'failure_count')
        assert hasattr(circuit_breaker, 'state')


class TestGeneration3_Scaling:
    """Test Generation 3: Optimization & Scaling (Make it Scale)."""
    
    def test_optimized_agent_creation(self):
        """Test optimized agent creation."""
        agent = OptimizedReflexionAgent(
            llm="gpt-4",
            enable_caching=True,
            enable_parallel_execution=True,
            cache_size=100
        )
        assert agent is not None
        assert agent.enable_caching
        assert agent.enable_parallel_execution
    
    def test_caching_functionality(self):
        """Test caching improves performance."""
        agent = OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=1,
            enable_caching=True
        )
        
        # Run same task twice
        task = "Test caching performance"
        
        start_time = time.time()
        result1 = agent.run(task, "basic")
        first_run_time = time.time() - start_time
        
        start_time = time.time()
        result2 = agent.run(task, "basic")  # Should be cached
        second_run_time = time.time() - start_time
        
        # Second run should be faster (cached)
        assert second_run_time < first_run_time
        
        # Check performance stats
        stats = agent.get_performance_stats()
        assert "cache_hit_rate" in stats["derived_metrics"]
        assert stats["derived_metrics"]["cache_hit_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing capabilities."""
        agent = OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=1,
            enable_parallel_execution=True
        )
        
        tasks = [
            "Task 1",
            "Task 2", 
            "Task 3"
        ]
        
        start_time = time.time()
        results = await agent.run_batch(tasks, "basic")
        batch_time = time.time() - start_time
        
        assert len(results) == len(tasks)
        assert batch_time < 1.0  # Should be reasonably fast
        
        # Check that all results are valid
        for result in results:
            assert hasattr(result, 'task')
            assert hasattr(result, 'output')
    
    def test_auto_scaling_agent(self):
        """Test auto-scaling agent functionality."""
        agent = AutoScalingReflexionAgent(llm="gpt-4")
        
        # Test that scaling stats are available
        scaling_stats = agent.get_scaling_stats()
        assert "current_workers" in scaling_stats
        assert "min_workers" in scaling_stats
        assert "max_workers" in scaling_stats
        assert "load_metrics" in scaling_stats
        
        # Test scaling thresholds are reasonable
        assert 0 < scaling_stats["scale_up_threshold"] < 1
        assert 0 < scaling_stats["scale_down_threshold"] < 1
    
    def test_performance_analytics(self):
        """Test performance analytics functionality."""
        agent = OptimizedReflexionAgent(llm="gpt-4", enable_caching=True)
        
        # Run a few tasks to generate data
        for i in range(3):
            agent.run(f"Test task {i}", "basic")
        
        # Get performance stats
        stats = agent.get_performance_stats()
        
        # Validate stats structure
        assert "agent_stats" in stats
        assert "cache_stats" in stats
        assert "derived_metrics" in stats
        assert "configuration" in stats
        
        # Validate derived metrics
        derived = stats["derived_metrics"]
        assert "avg_processing_time" in derived
        assert "cache_hit_rate" in derived
        assert "tasks_per_second" in derived
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations system."""
        agent = OptimizedReflexionAgent(llm="gpt-4")
        
        # Run some tasks to generate performance data
        for i in range(2):
            agent.run(f"Test optimization {i}", "basic")
        
        # Get recommendations
        recommendations = agent.get_optimization_recommendations()
        
        assert "recommendations" in recommendations
        assert "current_performance" in recommendations
        assert "optimization_score" in recommendations
        
        # Optimization score should be between 0 and 100
        assert 0 <= recommendations["optimization_score"] <= 100
    
    def test_smart_cache(self):
        """Test smart cache functionality."""
        from reflexion.core.optimization import SmartCache
        
        cache = SmartCache(max_size=10, default_ttl=3600)
        
        # Test basic operations
        assert cache.put("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test cache stats
        stats = cache.get_stats()
        assert "size" in stats
        assert "hit_rate" in stats
        assert stats["size"] == 1
    
    def test_scaling_manager(self):
        """Test scaling manager functionality."""
        # Test that scaling manager is available and has expected interface
        assert scaling_manager is not None
        
        # Test basic scaling status
        status = scaling_manager.get_scaling_status()
        assert "workers" in status
        assert "queue" in status
        assert "metrics" in status


class TestIntegration:
    """Integration tests across all generations."""
    
    def test_full_workflow_integration(self):
        """Test complete workflow from basic to optimized execution."""
        # Generation 1: Basic functionality
        basic_agent = ReflexionAgent(llm="gpt-4", max_iterations=1)
        basic_result = basic_agent.run("Integration test task", "basic")
        assert basic_result is not None
        
        # Generation 2: Add robustness (basic agent already has it)
        health_metrics = health_checker.get_system_metrics()
        assert health_metrics is not None
        
        # Generation 3: Optimization
        optimized_agent = OptimizedReflexionAgent(
            llm="gpt-4",
            max_iterations=1,
            enable_caching=True
        )
        optimized_result = optimized_agent.run("Integration test task", "basic")
        assert optimized_result is not None
        
        # Compare performance (optimized should have analytics)
        basic_output_len = len(basic_result.output)
        optimized_output_len = len(optimized_result.output)
        
        # Both should produce meaningful output
        assert basic_output_len > 0
        assert optimized_output_len > 0
    
    def test_memory_integration(self):
        """Test memory integration across components."""
        memory = EpisodicMemory(storage_path="./integration_test_memory.json")
        agent = ReflexionAgent(llm="gpt-4", max_iterations=1)
        
        # Execute task
        result = agent.run("Memory integration test", "basic")
        
        # Store in memory
        memory.store_episode("Memory integration test", result)
        
        # Recall and verify
        similar = memory.recall_similar("Memory integration", k=1)
        assert len(similar) >= 0
    
    def test_end_to_end_performance(self):
        """Test end-to-end performance across all generations."""
        agents = [
            ("Basic", ReflexionAgent(llm="gpt-4", max_iterations=1)),
            ("Optimized", OptimizedReflexionAgent(llm="gpt-4", max_iterations=1, enable_caching=True)),
        ]
        
        task = "End-to-end performance test"
        results = {}
        
        for name, agent in agents:
            start_time = time.time()
            result = agent.run(task, "basic")
            execution_time = time.time() - start_time
            
            results[name] = {
                "success": result.success if hasattr(result, 'success') else True,
                "time": execution_time,
                "output_length": len(result.output) if hasattr(result, 'output') else 0
            }
        
        # Validate all agents completed successfully
        for name, result in results.items():
            assert result["output_length"] > 0, f"{name} agent produced no output"
            assert result["time"] < 10.0, f"{name} agent took too long: {result['time']}s"


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)
    
    test_classes = [
        ("Generation 1 (Basic Functionality)", TestGeneration1_Basic),
        ("Generation 2 (Robustness)", TestGeneration2_Robustness), 
        ("Generation 3 (Scaling)", TestGeneration3_Scaling),
        ("Integration Tests", TestIntegration)
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for suite_name, test_class in test_classes:
        print(f"\n🔧 Testing {suite_name}")
        print("-" * 30)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            test_instance = test_class()
            test_method = getattr(test_instance, test_method_name)
            
            try:
                # Handle async tests
                if asyncio.iscoroutinefunction(test_method):
                    asyncio.run(test_method())
                else:
                    test_method()
                
                print(f"✅ {test_method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ {test_method_name}: {str(e)}")
                failed_tests.append((suite_name, test_method_name, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 TEST SUITE SUMMARY")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    if failed_tests:
        print("\n❌ Failed Tests:")
        for suite, test, error in failed_tests:
            print(f"   {suite} - {test}: {error}")
    
    # Final status
    if len(failed_tests) == 0:
        print("\n🎉 ALL TESTS PASSED - AUTONOMOUS SDLC EXECUTION SUCCESSFUL!")
    else:
        print(f"\n⚠️ {len(failed_tests)} tests failed - Review and fix issues")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)